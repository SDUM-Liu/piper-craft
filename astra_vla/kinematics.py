"""Piper FK/IK and collision preview; imported only in the existing pika environment."""
import math
import numpy as np
import pinocchio as pin
import hppfcl
from astra_vla.teleop import rate_rejection


class ContactSetTruncated(RuntimeError):
    """An incomplete contact query must be retried, never used as a depth bound."""


def world_bounds(shape, pose):
    """Conservative world AABB of the entire transformed collision shape."""
    shape.computeLocalAABB()
    lo, hi = shape.aabb_local.min_, shape.aabb_local.max_
    corners = np.array([[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]]
                        for z in [lo[2], hi[2]]])
    world = corners @ pose[:3, :3].T + pose[:3, 3]
    return world, (world.min(axis=0), world.max(axis=0))


class Kinematics:
    @staticmethod
    def state_pose_matrix(state):
        result = np.eye(4)
        result[:3, :3] = pin.rpy.rpyToMatrix(np.deg2rad(state["ee_pose"]["rpy_deg"]))
        result[:3, 3] = state["ee_pose"]["position_m"]
        return result

    def __init__(self, config):
        limits = config.get("limits", {})
        self.disabled_start_tolerance = math.radians(float(limits.get("disabled_start_joint_tolerance_deg", 3)))
        if not 0 <= self.disabled_start_tolerance <= math.radians(3):
            raise ValueError("disabled start joint tolerance must be between 0 and 3 degrees")
        self.disabled_start_joints = set(limits.get("disabled_start_tolerance_joints", [2, 3]))
        if not self.disabled_start_joints <= {2, 3}:
            raise ValueError("disabled start tolerance is limited to gravity-loaded J2/J3")
        self.gripper_model = config["gripper_model"]
        self.robot = pin.RobotWrapper.BuildFromURDF(config["urdf"], config["package_dirs"])
        self.model, self.data = self.robot.model, self.robot.data
        self.regularized_teleop_ik = None
        self.geometry = self.robot.collision_model
        self.frame = self.model.getFrameId("link6")
        if self.frame >= self.model.nframes or self.model.nq != 8:
            raise RuntimeError("Piper URDF must have joint1..8 and link6")
        self.geometry.addAllCollisionPairs()
        pairs = [(p.first, p.second) for p in self.geometry.collisionPairs]
        for first, second in pairs:
            a = self.geometry.geometryObjects[first].parentJoint
            b = self.geometry.geometryObjects[second].parentJoint
            if a == b or self.model.parents[a] == b or self.model.parents[b] == a:
                self.geometry.removeCollisionPair(pin.CollisionPair(first, second))
        self.gdata = pin.GeometryData(self.geometry)

    def configuration(self, state):
        q = pin.neutral(self.model)
        q[:6] = state["joints_rad"]
        width = float(state["gripper"]["width_m"])
        if self.gripper_model == "piper":
            # Use the measured stroke; clipping to 70 mm underestimated the
            # configured right jaw extent during bottle handling.
            q[6] = max(0, width / 2)
            q[7] = -q[6]
        elif self.gripper_model == "pika":
            q[6] = -min(.05, max(0, (.10 - width) / 2))
            q[7] = -q[6]
        else:
            raise ValueError("unknown gripper geometry model")
        return q

    def placement(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame].copy()

    def feedback_pose(self, state):
        """Controller-reported pose is authoritative; local FK is model-only.

        Validate shape/finiteness without comparing or replacing SDK feedback.
        Freshness, controller health and joint/path guards belong to the worker.
        """
        position = np.asarray(state['ee_pose']['position_m'], dtype=float)
        angles = np.asarray(state['ee_pose']['rpy_deg'], dtype=float)
        if (position.shape != (3,) or angles.shape != (3,) or
                not np.isfinite(position).all() or not np.isfinite(angles).all()):
            raise ValueError('invalid controller pose feedback')
        return dict(state)


    def joint_limit_violations(self, state):
        """Report every invalid measured joint without clipping feedback or changing limits."""
        joints = np.asarray(state["joints_rad"], dtype=float)
        if joints.shape != (6,):
            raise ValueError("measured joint feedback must contain six joint angles")
        result = []
        for index, value in enumerate(joints):
            lower = float(self.model.lowerPositionLimit[index])
            upper = float(self.model.upperPositionLimit[index])
            if not math.isfinite(value):
                reason = "nonfinite"
            elif value < lower:
                reason = "below_lower_limit"
            elif value > upper:
                reason = "above_upper_limit"
            else:
                continue
            measured_deg = math.degrees(value)
            result.append({"joint": index + 1,
                           "measured_deg": measured_deg if math.isfinite(measured_deg) else None,
                           "lower_deg": math.degrees(lower), "upper_deg": math.degrees(upper),
                           "reason": reason})
        return result

    def check_hold(self, state):
        """Validate finite feedback and self collision; angles are controller-owned."""
        joints = np.asarray(state['joints_rad'], dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError('invalid teleop joint feedback')
        q = self.configuration(state)
        if self.contacts(q):
            raise ValueError("measured pose has URDF self collision")

    def start_limit_report(self, state, operator_recovery=False):
        """Allow bounded measured sag only in fully disabled standby; never change feedback."""
        control = state.get("controller", {})
        enabled = control.get("motors_enabled", [])
        disabled = control.get("ctrl_mode") == 0 and len(enabled) == 6 and not any(enabled)
        supported_recovery = (operator_recovery and control.get("ctrl_mode") == 1 and
                              len(enabled) == 6 and all(enabled))
        tolerated, blocking = [], []
        for item in self.joint_limit_violations(state):
            value = item["measured_deg"]
            excess = (max(item["lower_deg"] - value, value - item["upper_deg"], 0)
                      if value is not None else None)
            accepted = ((disabled or supported_recovery) and item["joint"] in self.disabled_start_joints and
                        excess is not None and excess <= math.degrees(self.disabled_start_tolerance) + 1e-10)
            (tolerated if accepted else blocking).append({**item, "excess_deg": excess})
        return {"tolerated": tolerated, "blocking": blocking,
                "disabled_start_tolerance_deg": math.degrees(self.disabled_start_tolerance)}

    def joint_limit_excess(self, q):
        return np.maximum(np.maximum(self.model.lowerPositionLimit[:6] - q[:6],
                                     q[:6] - self.model.upperPositionLimit[:6]), 0)

    def objects(self, q, base_transform=None):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.geometry, self.gdata)
        transform = np.eye(4) if base_transform is None else np.array(base_transform)
        result = []
        for index, obj in enumerate(self.geometry.geometryObjects):
            pose = transform @ self.gdata.oMg[index].homogeneous
            result.append((obj, hppfcl.CollisionObject(obj.geometry, hppfcl.Transform3f(
                pose[:3, :3], pose[:3, 3])), pose))
        return result

    def contacts(self, q):
        # Self-collision only needs Pinocchio's geometry placements. objects()
        # also constructs a second set of world CollisionObjects for table /
        # other-arm distance queries, which are unused here. Repeating those
        # allocations at every asynchronous finger width exhausted teleop's
        # deadline. Keep the exact same collision pairs and contact filtering.
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.geometry, self.gdata)
        pin.computeCollisions(self.geometry, self.gdata, False)
        return {(p.first, p.second): max(float(c.penetration_depth) for c in r.getContacts())
                for p, r in zip(self.geometry.collisionPairs, self.gdata.collisionResults)
                if r.isCollision() and not self.closed_finger_contact(p, r)}

    def closed_finger_contact(self, pair, result):
        joints = {self.geometry.geometryObjects[i].parentJoint for i in (pair.first, pair.second)}
        return joints == {7, 8} and max(float(c.penetration_depth) for c in result.getContacts()) <= 1e-4

    def recovery_contacts(self, q, maximum=None):
        """Exhaust contacts before comparing penetration along a recovery path.

        Pinocchio defaults to one contact per mesh pair. That first triangle
        can change between poses and is not a stable maximum-depth measure.
        Restore the fast boolean-query limits used by ordinary motion checks.
        """
        requests = list(self.gdata.collisionRequests)
        original = [request.num_max_contacts for request in requests]
        try:
            for limit in (1024, 4096, 16384, 65536):
                for request in requests:
                    request.num_max_contacts = limit
                try:
                    contacts = self.bounded_gripper_contacts(q, maximum, contact_limit=limit)
                except ContactSetTruncated:
                    continue
                if all(not result.isCollision() or result.numContacts() < limit
                       for result in self.gdata.collisionResults):
                    return contacts
            raise ValueError('recovery contact set could not be exhausted')
        finally:
            for request, limit in zip(requests, original):
                request.num_max_contacts = limit

    def bounded_gripper_contacts(self, q, maximum=None, *, contact_limit=1):
        """Self collision with a conservative box sweeping every finger opening.

        The Piper fingers translate without rotating. The union of the local
        mesh AABBs at both stroke endpoints therefore contains the entire
        physical stroke. No encoder zero or finger command is inferred here.
        """
        if maximum is None:
            return self.contacts(q)
        if self.gripper_model != 'piper' or not math.isfinite(maximum) or not 0 < maximum <= .1:
            raise ValueError('unsupported unverified gripper envelope')
        cache = getattr(self, '_gripper_envelopes', {})
        if maximum not in cache:
            endpoints = []
            for width in (0., maximum):
                neutral = self.configuration({'joints_rad': [0.]*6, 'gripper': {'width_m': width}})
                endpoints.append(self.objects(neutral))
            shapes = {}
            for index, obj in enumerate(self.geometry.geometryObjects):
                if obj.parentJoint not in (7, 8):
                    continue
                origin = endpoints[0][index][2]
                relative = np.linalg.inv(origin) @ endpoints[1][index][2]
                if not np.allclose(relative[:3, :3], np.eye(3), atol=1e-10):
                    raise ValueError('gripper envelope requires prismatic fingers')
                corners, _ = world_bounds(obj.geometry, np.eye(4))
                corners = np.vstack((corners, corners + relative[:3, 3]))
                lower, upper = corners.min(axis=0), corners.max(axis=0)
                shapes[index] = (hppfcl.Box(*(upper-lower)), (lower+upper)/2)
            if len(shapes) != 2:
                raise ValueError('gripper envelope requires two Piper finger geometries')
            cache[maximum] = shapes
            self._gripper_envelopes = cache
        shapes = cache[maximum]
        closed = np.array(q, copy=True)
        closed[6:8] = 0.
        contacts = self.contacts(closed)
        contacts = {pair: depth for pair, depth in contacts.items() if not set(pair) & shapes.keys()}
        objects = []
        for index, obj in enumerate(self.geometry.geometryObjects):
            transform = self.gdata.oMg[index]
            if index in shapes:
                shape, center = shapes[index]
                translation = transform.translation + transform.rotation @ center
            else:
                shape, translation = obj.geometry, transform.translation
            objects.append(hppfcl.CollisionObject(shape, hppfcl.Transform3f(transform.rotation, translation)))
        for pair in self.geometry.collisionPairs:
            first, second = pair.first, pair.second
            if not {first, second} & shapes.keys() or {first, second} <= shapes.keys():
                # The two opposing sweep volumes overlap, but the coupled
                # physical fingers cannot occupy different openings at once.
                continue
            result = hppfcl.CollisionResult()
            request = hppfcl.CollisionRequest()
            request.num_max_contacts = contact_limit
            hppfcl.collide(objects[first], objects[second], request, result)
            if contact_limit > 1 and result.numContacts() >= contact_limit:
                raise ContactSetTruncated('recovery finger contact set truncated')
            if result.isCollision():
                contacts[first, second] = max(float(c.penetration_depth) for c in result.getContacts())
        return contacts

    def bounded_gripper_collision(self, q, maximum):
        """Boolean equivalent of the swept-finger contact query, in Pinocchio.

        Reuse the same conservative boxes and collision pairs. Dedicated cached
        geometry avoids constructing Python FCL objects at every route corner;
        the contact/depth query used by recovery remains separate.
        """
        if maximum is None:
            return bool(self.contacts(q))
        if maximum not in getattr(self, '_gripper_envelopes', {}):
            self.bounded_gripper_contacts(q, maximum)
        cache = getattr(self, '_gripper_collision_models', {})
        if maximum not in cache:
            shapes = self._gripper_envelopes[maximum]
            geometry = pin.GeometryModel(self.geometry)
            for index, (shape, center) in shapes.items():
                obj = geometry.geometryObjects[index]
                obj.geometry = shape
                obj.placement = obj.placement * pin.SE3(np.eye(3), center)
            for pair in list(geometry.collisionPairs):
                # As in bounded_gripper_contacts, opposing swept volumes may
                # overlap although the coupled physical fingers cannot.
                if pair.first in shapes and pair.second in shapes:
                    geometry.removeCollisionPair(pair)
            cache[maximum] = (geometry, pin.GeometryData(geometry))
            self._gripper_collision_models = cache
        geometry, data = cache[maximum]
        closed = np.array(q, copy=True)
        closed[6:8] = 0.
        pin.forwardKinematics(self.model, self.data, closed)
        pin.updateGeometryPlacements(self.model, self.data, geometry, data)
        return bool(pin.computeCollisions(geometry, data, True))


    def prepare_teleop_ik(self):
        if self.regularized_teleop_ik is None:
            from astra_vla.teleop_ik import RegularizedTeleopIK
            solver = RegularizedTeleopIK(self.model)
            solver.prime()
            self.regularized_teleop_ik = solver
        return self.regularized_teleop_ik

    def teleop_step(self, state, other_state, target, geometry, width, policy, previous_velocity,
                    previous_goal=None, validated_goal=None):
        """Solve and check exact MOVE J integers, acceleration and following budgets.

        This is an internal, single-use calculation. It exports no motion permission
        and never connects CAN. The owning worker rechecks feedback/input before TX.
        """
        if geometry.get('scope') != 'tcp_jog':
            raise ValueError('PIPER CRAFT requires robot-local geometry')
        joints = np.asarray(state['joints_rad'], dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError('invalid teleop joint feedback')
        current = self.configuration(state)
        actual = self.state_pose_matrix(state)
        target = np.asarray(target)
        if target.shape != (4, 4) or not np.isfinite(target).all() or not math.isfinite(width):
            raise ValueError('invalid teleop target')
        dt = 1 / policy['control_hz']
        # Encoder lag is a distance, not velocity over a single control tick.
        # Bound tracking lead and acceleration against the last sent target.
        # Joint speed is handled by the controller, with no local speed cap.
        lead = math.radians(policy['joint_lead_deg'])
        previous_goal = current[:6] if previous_goal is None else np.asarray(previous_goal)
        if previous_goal.shape != (6,) or not np.isfinite(previous_goal).all():
            raise ValueError('invalid previous teleop joint target')
        seed = current.copy()
        seed[:6] = previous_goal
        acceleration_step = math.radians(policy['joint_acceleration_deg_s2']) * dt
        joint_jog = 'joint_jog_direction' in geometry
        if joint_jog:
            direction = np.asarray(geometry['joint_jog_direction'], dtype=float)
            gain = geometry.get('joint_jog_gain')
            if (policy.get('ik_solver') != 'casadi_ipopt' or
                    direction.shape != (6,) or not np.isfinite(direction).all() or
                    np.count_nonzero(direction) != 1 or not np.isin(direction, (-1., 0., 1.)).all() or
                    isinstance(gain, bool) or not isinstance(gain, (int, float)) or not 0 < gain <= 1):
                raise ValueError('invalid joint jog intent')
        if validated_goal is not None:
            # Fresh feedback rechecks the SAME quantized command. Solving
            # IK again changes the endpoint and costs another full solve.
            integers = validated_goal.get('joint_sdk')
            if (policy.get('ik_solver') != 'casadi_ipopt' or
                    validated_goal.get('ik', {}).get('solver') != ('joint_jog' if joint_jog else 'casadi_ipopt') or
                    not isinstance(integers, list) or len(integers) != 6 or
                    any(isinstance(v, bool) or not isinstance(v, int) for v in integers)):
                raise ValueError('invalid teleop revalidation target')
            goal = seed.copy()
            goal[:6] = np.deg2rad(np.asarray(integers) / 1000)
        elif joint_jog:
            # The chosen axis moves directly in joint space. Leave braking
            # distance inside the same feedback lead used by regularized IK.
            acceleration = acceleration_step / dt
            positive_room = np.maximum(0., current[:6]+lead-previous_goal)
            negative_room = np.maximum(0., previous_goal-current[:6]+lead)
            rounding_speed = math.radians(.00051)/dt
            positive_speed = np.maximum(0., np.sqrt(acceleration_step**2+2*acceleration*positive_room)
                                        - acceleration_step-rounding_speed)
            negative_speed = np.maximum(0., np.sqrt(acceleration_step**2+2*acceleration*negative_room)
                                        - acceleration_step-rounding_speed)
            desired = direction*math.radians(policy['rotation_deg_s'])*gain
            desired = np.clip(desired, -negative_speed, positive_speed)
            packet_acceleration_step = max(0., acceleration_step-rounding_speed)
            velocity = np.clip(desired, np.asarray(previous_velocity)-packet_acceleration_step,
                               np.asarray(previous_velocity)+packet_acceleration_step)
            velocity[direction == 0] = 0.
            goal = seed.copy()
            goal[:6] += velocity*dt
        elif policy.get('ik_solver') == 'casadi_ipopt':
            solver = self.prepare_teleop_ik()
            tool = np.asarray(geometry['tcp'])
            # Hard local bounds keep optimization on this command's branch.
            velocity = np.asarray(previous_velocity)
            # Leave braking distance inside the existing tracking envelope.
            # A target at the hard lead boundary with nonzero velocity has
            # no admissible next acceleration step and causes wait/burst
            # cycling. v*dt + v^2/(2*a) must fit the remaining lead.
            acceleration = acceleration_step / dt
            # Brake inside the feedback tracking envelope only. Absolute
            # joint-angle limits and margins are handled by the controller.
            positive_room = np.maximum(0., current[:6] + lead - previous_goal)
            negative_room = np.maximum(0., previous_goal - current[:6] + lead)
            positive_speed = np.sqrt(acceleration_step**2 + 2*acceleration*positive_room) - acceleration_step
            negative_speed = np.sqrt(acceleration_step**2 + 2*acceleration*negative_room) - acceleration_step
            # Rounding can increase a command by half an SDK millidegree.
            rounding_speed = math.radians(.00051) / dt
            positive_speed = np.maximum(0., positive_speed - rounding_speed)
            negative_speed = np.maximum(0., negative_speed - rounding_speed)
            lower = np.maximum.reduce([current[:6]-lead,
                                       previous_goal-negative_speed*dt,
                                       previous_goal+(velocity-acceleration_step)*dt])
            upper = np.minimum.reduce([current[:6]+lead,
                                       previous_goal+positive_speed*dt,
                                       previous_goal+(velocity+acceleration_step)*dt])
            if np.any(lower >= upper):
                # Feedback lag can temporarily leave no intersection with
                # the acceleration box. Keep the previous validated goal;
                # this is tracking saturation, not a malformed IK problem.
                raise ValueError('teleop joint tracking envelope exhausted')
            goal = seed.copy()
            goal[:6] = solver.solve(np.asarray(target) @ tool, tool, seed[:6], lower, upper,
                                   rotation_only=bool(geometry.get('rotation_only')))
            target = np.asarray(solver.last['accepted_tcp_target']) @ np.linalg.inv(tool)
        else:
            raise ValueError('PIPER CRAFT requires casadi_ipopt')
        integers = np.rint(np.rad2deg(goal[:6]) * 1000).astype(np.int64)
        goal[:6] = np.deg2rad(integers.astype(float) / 1000)
        tool = np.asarray(geometry['tcp'])
        actual_goal = self.placement(goal).homogeneous @ tool
        if joint_jog:
            target = self.placement(goal).homogeneous.copy()
        target_tcp = np.asarray(target) @ tool
        error = pin.log6(pin.SE3(actual_goal).actInv(pin.SE3(target_tcp))).vector
        if (np.linalg.norm(actual_goal[:3, 3] - target_tcp[:3, 3]) > .00005 or
                np.linalg.norm(error[3:]) > math.radians(.01)):
            raise ValueError('quantized teleop target loses TCP accuracy')
        velocity = (goal[:6] - previous_goal) / dt
        if np.max(np.abs(goal[:6] - current[:6])) > lead + 1e-10:
            raise ValueError('teleop joint tracking envelope exhausted')
        if np.max(np.abs(velocity - np.asarray(previous_velocity))) > math.radians(
            policy['joint_acceleration_deg_s2']
        ) * dt + 1e-10:
            raise rate_rejection('teleop joint acceleration exceeds bound', velocity,
                                 previous_velocity, acceleration_step)
        tcp = np.asarray(geometry['tcp'])
        actual_tcp, target_tcp = actual @ tcp, target @ tcp
        if not joint_jog and (np.linalg.norm(target_tcp[:3, 3] - actual_tcp[:3, 3]) > policy['position_lead_m'] or
                np.linalg.norm(pin.log3(actual_tcp[:3, :3].T @ target_tcp[:3, :3])) >
                math.radians(policy['rotation_lead_deg'])):
            raise ValueError('teleop TCP target exceeds tracking lead')
        actual_width = float(state['gripper']['width_m'])
        if not math.isfinite(actual_width):
            raise ValueError('invalid teleop gripper feedback')
        actual_width = max(0., actual_width)
        # Interpolation only supports feedback-progress accounting. Per user
        # instruction it performs no path or finger-sweep collision queries.
        count = max(10, math.ceil(np.max(np.abs(goal[:6] - current[:6])) / math.radians(.1)))
        path_joints = []
        for fraction in np.linspace(0, 1, count + 1):
            candidate = pin.interpolate(self.model, current, goal, float(fraction))
            path_joints.append(candidate[:6].tolist())
        return {'reachable': True, 'joint_sdk': integers.tolist(), 'velocity': velocity.tolist(),
                'tcp_target': target_tcp.tolist(),
                # Advisory for endpoint replanning, not an ordinary-jog angle
                # rejection. Keep the controller responsible for normal limits.
                'detour_recommended': not joint_jog and bool(np.any(goal[:6]+velocity < self.model.lowerPositionLimit[:6])
                                           or np.any(goal[:6]+velocity > self.model.upperPositionLimit[:6])),
                'joint_limit_status': [], 'joint_position_limits': 'controller',
                'ik': dict(validated_goal['ik'], fresh_target_rechecked=True) if validated_goal is not None else
                      ({'solver': 'joint_jog', 'progress_fraction': float(
                          np.max(np.abs(goal[:6]-previous_goal)) > 1e-10)} if joint_jog else
                       dict(self.regularized_teleop_ik.last)),
                'goal_joints_rad': goal[:6].tolist(), 'samples': count + 1,
                'path_precheck': 'none', 'path_collision_check': False,
                'gripper_envelope_mode': 'none', 'gripper_envelope_max_m': None,
                'path_joints_rad': path_joints,
                'stroke_bounds_m': [min(actual_width, width), max(actual_width, width)]}
