"""Cancellable TCP endpoint detours. Geometry functions have no SDK/CAN access."""
from dataclasses import dataclass
import itertools
import json
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from astra_vla.config import ROOT
from astra_vla.teleop import Integrator, pose


@dataclass(frozen=True)
class DetourPolicy:
    enabled: bool = True
    low_progress: float = .2
    trigger_s: float = .6
    goal_horizon_s: float = 1.
    planning_timeout_s: float = 8.
    execution_timeout_s: float = 120.
    search_seeds: int = 16
    speed_percent: int = 100
    joint_lead_deg: float = 6.

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('invalid detour enable flag')
        for name, maximum in [('low_progress', .5), ('trigger_s', 5.), ('goal_horizon_s', 1.),
                              ('planning_timeout_s', 15.), ('execution_timeout_s', 180.)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= maximum:
                raise ValueError('invalid detour policy: '+name)
        if type(self.search_seeds) is not int or not 3 <= self.search_seeds <= 32:
            raise ValueError('invalid detour seed count')
        if type(self.speed_percent) is not int or not 1 <= self.speed_percent <= 100:
            raise ValueError('invalid detour SDK speed percentage')
        if (isinstance(self.joint_lead_deg, bool) or not isinstance(self.joint_lead_deg, (int, float))
                or not math.isfinite(self.joint_lead_deg) or not 0 < self.joint_lead_deg <= 6):
            raise ValueError('invalid detour joint following budget')

    @classmethod
    def load(cls):
        path = ROOT/'config/teleop_detour.json'
        return cls(**json.loads(path.read_text())) if path.exists() else cls(enabled=False)


def endpoint_intent(state, tcp, policy, keys, maximum_width, horizon):
    """One bounded intent horizon, frozen at SDK feedback; no accumulated debt."""
    integrator = Integrator(state, tcp, policy)
    integrator.speed_fraction = 1.
    proposal = integrator.propose(state, keys, 1/policy.control_hz, maximum_width, keys)
    if proposal is None or not proposal['arm_motion'] or proposal['gripper_motion']:
        raise ValueError('detour requires arm-only movement keys')
    start = pose(state) @ np.asarray(tcp)
    goal = start.copy()
    factor = policy.control_hz*horizon
    goal[:3, 3] += factor*(proposal['tcp_target'][:3, 3]-start[:3, 3])
    turn = Rotation.from_matrix(proposal['tcp_target'][:3, :3] @ start[:3, :3].T).as_rotvec()
    goal[:3, :3] = Rotation.from_rotvec(factor*turn).as_matrix() @ start[:3, :3]
    return goal


def route_samples(start, goal, step_deg=.1):
    count = max(1, math.ceil(float(np.max(np.abs(np.rad2deg(goal-start))))/step_deg))
    for fraction in np.linspace(0, 1, count+1):
        yield start+fraction*(goal-start)


def check_joint_route(model, state, start, goal, envelope, *, deadline=None, corners=True):
    """Sample self geometry and independent-axis corners; external world is unmodeled."""
    checked = 0
    def check(joints):
        nonlocal checked
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('detour geometry deadline exceeded')
        checked += 1
        q = model.configuration(dict(state, joints_rad=joints.tolist()))
        if model.bounded_gripper_collision(q, envelope):
            raise ValueError('detour path has URDF self collision')
    for joints in route_samples(start, goal):
        check(joints)
    if corners:
        count = max(1, math.ceil(float(np.max(np.abs(np.rad2deg(goal-start))))))
        for index in range(count):
            first = start+(goal-start)*index/count
            second = start+(goal-start)*(index+1)/count
            active = np.flatnonzero(np.abs(second-first) > 1e-12)
            for mask in itertools.product((False, True), repeat=len(active)):
                corner = first.copy()
                selected = active[np.asarray(mask, dtype=bool)]
                corner[selected] = second[selected]
                check(corner)
    return checked


class EndpointSolver:
    """Full-target multi-start IPOPT; never returns optimizer debug iterates."""
    def __init__(self, model):
        import casadi as ca
        local = model.prepare_teleop_ik()
        self.local = local
        self.opti = ca.Opti()
        self.q = self.opti.variable(6)
        self.target = self.opti.parameter(4, 4)
        self.tool = self.opti.parameter(4, 4)
        self.seed = self.opti.parameter(6)
        self.lower = self.opti.parameter(6)
        self.upper = self.opti.parameter(6)
        error = local.error(self.q, self.target, self.tool)
        self.opti.minimize(ca.sumsqr(self.q-self.seed))
        self.opti.subject_to(self.opti.bounded(self.lower, self.q, self.upper))
        self.opti.subject_to(self.opti.bounded(-1, error[:3]/(10e-6/math.sqrt(3)), 1))
        self.opti.subject_to(self.opti.bounded(-1, error[3:]/(math.radians(.002)/3), 1))
        self.opti.solver('ipopt', {'print_time': False},
                         {'print_level': 0, 'sb': 'yes', 'max_iter': 100, 'max_cpu_time': .15,
                          'tol': 1e-8, 'constr_viol_tol': 1e-10, 'bound_relax_factor': 0})

    def solve(self, target, tool, seed, lower, upper):
        for parameter, value in [(self.target, target), (self.tool, tool), (self.seed, seed),
                                 (self.lower, lower), (self.upper, upper)]:
            self.opti.set_value(parameter, value)
        self.opti.set_initial(self.q, np.clip(seed, lower, upper))
        try:
            solution = self.opti.solve()
        except RuntimeError:
            return None
        q = np.asarray(solution.value(self.q)).reshape(6)
        if not np.isfinite(q).all() or np.any(q < lower) or np.any(q > upper):
            return None
        return q


def plan_detour(model, state, target_tcp, geometry, options):
    config = DetourPolicy(**options)
    deadline = time.monotonic()+config.planning_timeout_s
    if not hasattr(model, '_detour_endpoint_solver'):
        model._detour_endpoint_solver = EndpointSolver(model)
    solver = model._detour_endpoint_solver
    start = np.asarray(state['joints_rad'], dtype=float)
    tool, target = np.asarray(geometry['tcp']), np.asarray(target_tcp)
    # Endpoint search domain only. Ordinary jog angle handling remains with
    # the controller. Alternative branches must use the robot's physical range.
    lower = model.model.lowerPositionLimit[:6]+math.radians(.001)
    upper = model.model.upperPositionLimit[:6]-math.radians(.001)
    seeds = [np.clip(start, lower, upper)]
    for shift, wrist_shift in itertools.product((-math.pi, math.pi), repeat=2):
        seed = start.copy()
        seed[3] += shift
        seed[4] *= -1
        seed[5] += wrist_shift
        seeds.append(np.clip(seed, lower, upper))
    seeds = seeds[:config.search_seeds]
    rng = np.random.default_rng(20261020)
    seeds += [rng.uniform(lower, upper) for _ in range(config.search_seeds-len(seeds))]
    envelope = max(state['gripper']['width_m'], geometry.get('unverified_gripper_max_m') or 0)
    envelope = min(.1, max(.001, math.ceil(envelope*1000)/1000))
    candidates, seen = [], []
    for seed in seeds:
        if time.monotonic() >= deadline:
            break
        goal = solver.solve(target, tool, seed, lower, upper)
        if goal is None:
            continue
        goal = np.deg2rad(np.rint(np.rad2deg(goal)*1000)/1000)
        if np.any(goal < lower) or np.any(goal > upper) or any(np.max(np.abs(goal-q)) < .001 for q in seen):
            continue
        seen.append(goal)
        q = model.configuration(dict(state, joints_rad=goal.tolist()))
        endpoint = model.placement(q).homogeneous @ tool
        angle = Rotation.from_matrix(endpoint[:3, :3].T @ target[:3, :3]).magnitude()
        if np.linalg.norm(endpoint[:3, 3]-target[:3, 3]) > 50e-6 or angle > math.radians(.01):
            continue
        candidates.append((float(np.linalg.norm(goal-start)), goal))
    for _, goal in sorted(candidates, key=lambda item: item[0]):
        try:
            samples = check_joint_route(model, state, start, goal, envelope, deadline=deadline)
        except ValueError:
            continue
        return {'start_joints_rad': start.tolist(), 'goal_joints_rad': goal.tolist(),
                'target_tcp': target.tolist(), 'envelope_max_m': envelope, 'samples': samples,
                'solver': 'casadi_ipopt', 'scope': 'robot_only',
                'environment_guard': False, 'other_arm_guard': False}
    raise ValueError('没有找到终点可达且自身几何检查通过的绕行路径')


def detour_joint_step(model, state, plan, policy, previous_goal, previous_velocity, validated=None):
    """Advance a fixed joint-space route with existing acceleration/lead guards."""
    dt = 1/policy['control_hz']
    current = np.asarray(state['joints_rad'])
    previous = np.asarray(current if previous_goal is None else previous_goal)
    velocity = np.asarray(previous_velocity)
    goal = np.asarray(plan['goal_joints_rad'])
    acceleration_step = math.radians(policy['joint_acceleration_deg_s2'])*dt
    lead = math.radians(policy['joint_lead_deg'])
    if validated is None:
        delta = goal-previous
        acceleration = acceleration_step/dt
        positive_room = np.maximum(0, current+lead-previous)
        negative_room = np.maximum(0, previous-current+lead)
        positive_speed = np.maximum(0, np.sqrt(acceleration_step**2+2*acceleration*positive_room)-acceleration_step-math.radians(.00051)/dt)
        negative_speed = np.maximum(0, np.sqrt(acceleration_step**2+2*acceleration*negative_room)-acceleration_step-math.radians(.00051)/dt)
        rounding_speed = math.radians(.00051)/dt
        low = np.maximum(-negative_speed, velocity-acceleration_step+rounding_speed)
        high = np.minimum(positive_speed, velocity+acceleration_step-rounding_speed)
        if np.any(low > high):
            return None
        # Preserve incoming momentum. Steer each joint toward its endpoint
        # through the existing acceleration box instead of resetting velocity
        # or requiring a stationary handoff. Fresh collision checking covers
        # the actual blended step, even when it departs from the preview line.
        desired = np.sign(delta)*(np.sqrt(acceleration_step**2+2*acceleration*np.abs(delta))-acceleration_step)
        desired = np.sign(desired)*np.minimum(np.abs(desired), np.abs(delta)/dt)
        new_velocity = np.clip(desired, low, high)
        target = np.deg2rad(np.rint(np.rad2deg(previous+new_velocity*dt)*1000)/1000)
    else:
        integers = validated.get('joint_sdk')
        if not isinstance(integers, list) or len(integers) != 6 or any(type(v) is not int for v in integers):
            raise ValueError('invalid detour integer target')
        target = np.deg2rad(np.asarray(integers)/1000)
    new_velocity = (target-previous)/dt
    if np.max(np.abs(target-current)) > lead+1e-10:
        raise ValueError('detour tracking lead exceeded')
    if np.max(np.abs(new_velocity-velocity)) > acceleration_step+1e-10:
        raise ValueError('detour acceleration exceeded')
    # Validate the fresh feedback-to-command path, including independent axes.
    samples = check_joint_route(model, state, current, target, plan['envelope_max_m'],
                               deadline=time.monotonic()+policy['geometry_timeout_s'])
    q = model.configuration(dict(state, joints_rad=target.tolist()))
    tcp = model.placement(q).homogeneous
    return {'joint_sdk': np.rint(np.rad2deg(target)*1000).astype(int).tolist(),
            'velocity': new_velocity.tolist(), 'samples': samples, 'tcp_target': tcp.tolist()}
