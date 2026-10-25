"""Regularized Pinocchio/CasADi/IPOPT IK, constructed only in the geometry worker.

No SDK, ROS, cameras or CAN access. Pose fitting is an objective; local joint
bounds and TCP accuracy remain constraints, with independent post-solve checks.
"""
import math
import time

import numpy as np


class RegularizedTeleopIK:
    POSITION_TOLERANCE_M = 20e-6
    ROTATION_TOLERANCE_RAD = math.radians(.005)

    def __init__(self, model):
        import casadi as ca
        import pinocchio as pin
        from pinocchio import casadi as cpin

        locked = [model.getJointId(name) for name in ('joint7', 'joint8')]
        self.model = pin.buildReducedModel(model, locked, pin.neutral(model))
        if self.model.nq != 6:
            raise RuntimeError('regularized teleop IK requires six arm joints')
        frame = self.model.getFrameId('link6')
        cm = cpin.Model(self.model)
        cd = cm.createData()
        cq = ca.SX.sym('q', 6)
        tool = ca.SX.sym('tool', 4, 4)
        target = ca.SX.sym('target', 4, 4)
        cpin.framesForwardKinematics(cm, cd, cq)
        ee = cd.oMf[frame]
        tcp = cpin.SE3(ee.rotation @ tool[:3, :3], ee.translation + ee.rotation @ tool[:3, 3])
        position = tcp.translation - target[:3, 3]
        # Matrix residual has finite derivatives at zero rotation. Symbolically
        # differentiating log6 at identity produced NaNs with this installed
        # Pinocchio/CasADi build. Frobenius/sqrt(2) equals 2*sin(angle/2).
        orientation = ca.vec(tcp.rotation - target[:3, :3]) / math.sqrt(2)
        error = ca.vertcat(position, orientation)
        self.error = ca.Function('teleop_error', [cq, target, tool], [error])
        self.position = ca.Function('teleop_position_error', [cq, target, tool], [position])

        self.opti = ca.Opti()
        self.q = self.opti.variable(6)
        self.progress = self.opti.variable()
        self.target = self.opti.parameter(4, 4)
        self.start = self.opti.parameter(4, 4)
        self.axis_angle = self.opti.parameter(4)
        self.pin_position = self.opti.parameter()
        self.tool = self.opti.parameter(4, 4)
        self.reference = self.opti.parameter(6)
        self.lower = self.opti.parameter(6)
        self.upper = self.opti.parameter(6)
        axis = self.axis_angle[:3]
        skew = ca.vertcat(ca.horzcat(0, -axis[2], axis[1]),
                          ca.horzcat(axis[2], 0, -axis[0]),
                          ca.horzcat(-axis[1], axis[0], 0))
        angle = self.progress * self.axis_angle[3]
        rotation = self.start[:3, :3] @ (ca.MX.eye(3) + ca.sin(angle) * skew
                                       + (1-ca.cos(angle)) * (skew @ skew))
        interpolated = self.start[:3, 3] + self.progress * (self.target[:3, 3] - self.start[:3, 3])
        point = (1-self.pin_position) * interpolated + self.pin_position * self.target[:3, 3]
        selected = ca.vertcat(ca.horzcat(rotation, point), ca.DM([[0, 0, 0, 1]]))
        residual = self.error(self.q, selected, self.tool)
        position = self.position(self.q, selected, self.tool)
        # Same weighted pose-fit architecture as Pika. Regularize relative to
        # the preceding command, rather than pulling every tiny jog to joint zero.
        self.opti.minimize(20 * (ca.sumsqr(residual[:3]) + ca.sumsqr(.1 * residual[3:]))
                           + .01 * ca.sumsqr(self.q - self.reference) + (1-self.progress)**2)
        self.opti.subject_to(self.opti.bounded(0, self.progress, 1))
        self.opti.subject_to(self.opti.bounded(self.lower, self.q, self.upper))
        position_component = self.POSITION_TOLERANCE_M / math.sqrt(3)
        rotation_component = 2 * math.sin(self.ROTATION_TOLERANCE_RAD / 2) / 3
        self.opti.subject_to(self.opti.bounded(-1, position / position_component, 1))
        self.opti.subject_to(self.opti.bounded(-1, residual[3:] / rotation_component, 1))
        self.opti.solver('ipopt', {'print_time': False}, {
            'print_level': 0, 'sb': 'yes', 'max_iter': 50, 'tol': 1e-8,
            'constr_viol_tol': 1e-10, 'max_cpu_time': .03,
            'bound_relax_factor': 0,
            'warm_start_init_point': 'yes'})
        self.last = None

    def solve(self, target_tcp, tool, seed, lower, upper, *, rotation_only=False):
        import pinocchio as pin
        from scipy.spatial.transform import Rotation
        started = time.monotonic()
        self.last = None
        seed, lower, upper = (np.asarray(value, dtype=float) for value in (seed, lower, upper))
        if any(value.shape != (6,) or not np.isfinite(value).all() for value in (seed, lower, upper)):
            raise ValueError('invalid regularized IK bounds/seed')
        if np.any(lower >= upper):
            raise ValueError('empty regularized IK local bounds')
        target_tcp, tool = np.asarray(target_tcp, dtype=float), np.asarray(tool, dtype=float)
        for matrix in (target_tcp, tool):
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
                    not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0) or
                    not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-9) or
                    not np.isclose(np.linalg.det(matrix[:3, :3]), 1)):
                raise ValueError('regularized IK requires finite rigid transforms')
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, seed)
        start = data.oMf[self.model.getFrameId('link6')].homogeneous @ tool
        vector = np.asarray(pin.log3(start[:3, :3].T @ target_tcp[:3, :3]))
        angle = float(np.linalg.norm(vector))
        axis = vector / angle if angle > 1e-12 else np.array([1., 0., 0.])
        self.opti.set_value(self.target, target_tcp)
        self.opti.set_value(self.start, start)
        self.opti.set_value(self.axis_angle, np.r_[axis, angle])
        self.opti.set_value(self.pin_position, float(rotation_only))
        self.opti.set_value(self.tool, tool)
        self.opti.set_value(self.reference, seed)
        self.opti.set_value(self.lower, lower)
        self.opti.set_value(self.upper, upper)
        # Linearize only to initialize IPOPT near this tiny target. The guess is
        # never returned as a goal: all constraints and independent checks must
        # still pass the nonlinear optimizer. Starting exactly at the previous
        # command hit the iteration cap for some fixed-TCP rotations.
        frame = self.model.getFrameId('link6')
        jacobian = pin.computeFrameJacobian(self.model, data, seed, frame, pin.LOCAL_WORLD_ALIGNED).copy()
        jacobian[:3] -= pin.skew(data.oMf[frame].rotation @ tool[:3, 3]) @ jacobian[3:]
        error = np.r_[target_tcp[:3, 3] - start[:3, 3],
                      pin.log3(target_tcp[:3, :3] @ start[:3, :3].T)]
        guess = seed + jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6)*1e-8, error)
        self.opti.set_initial(self.q, np.clip(guess, lower, upper))
        self.opti.set_initial(self.progress, 1)
        try:
            solution = self.opti.solve()
            goal = np.asarray(solution.value(self.q)).reshape(6)
            progress = float(solution.value(self.progress))
            stats = solution.stats()
        except RuntimeError as exc:
            # debug.value(var_q) is an unverified iterate, never a fallback goal.
            status = self.opti.stats().get('return_status')
            if status in ('Maximum_CpuTime_Exceeded', 'Maximum_WallTime_Exceeded'):
                raise TimeoutError('regularized teleop IK computation deadline exceeded') from exc
            if status == 'Maximum_Iterations_Exceeded':
                # No iterate is accepted. Feedback may still be catching the
                # previous goal; let the owner wait/recompute under its bounded
                # tracking-stall timer instead of rejecting the held direction.
                raise ValueError('teleop IK iteration budget exhausted') from exc
            if status == 'Infeasible_Problem_Detected':
                from astra_vla.teleop import RateLimitError
                raise RateLimitError('teleop IK local constraints infeasible', .5) from exc
            raise ValueError('regularized teleop IK did not converge: ' + str(status)) from exc
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError('regularized IK invalid progress')
        selected = start.copy()
        selected[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(vector * progress).as_matrix()
        selected[:3, 3] = (target_tcp[:3, 3] if rotation_only else
                          start[:3, 3] + progress * (target_tcp[:3, 3] - start[:3, 3]))
        position = np.asarray(self.position(goal, selected, tool)).reshape(3)
        residual = np.asarray(self.error(goal, selected, tool)).reshape(12)
        if (not stats.get('success') or not np.isfinite(goal).all() or
                np.any(goal < lower - 1e-10) or np.any(goal > upper + 1e-10) or
                np.linalg.norm(position) > self.POSITION_TOLERANCE_M + 1e-8 or
                np.linalg.norm(residual[3:]) > self.ROTATION_TOLERANCE_RAD + 1e-8):
            raise ValueError('regularized teleop IK failed independent accuracy/bounds check')
        self.last = {'solver': 'casadi_ipopt', 'iterations': int(stats.get('iter_count', 0)),
                     'elapsed_ms': (time.monotonic() - started) * 1000,
                     'tcp_position_residual_m': float(np.linalg.norm(position)),
                     'rotation_residual_rad': float(np.linalg.norm(residual[3:])),
                     'progress_fraction': progress, 'accepted_tcp_target': selected.tolist(),
                     'joint_regularization': 'previous_command', 'fallback_used': False}
        return goal

    def prime(self):
        """Load IPOPT before a timed control cycle. Entirely model-only."""
        import pinocchio as pin
        q = np.deg2rad([0., 50., -40., 0., 20., 0.])
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q)
        target = data.oMf[self.model.getFrameId('link6')].homogeneous.copy()
        self.solve(target, np.eye(4), q,
                   q - .01, q + .01)
