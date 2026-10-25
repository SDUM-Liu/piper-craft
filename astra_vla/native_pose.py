"""Controller-limit-aware pose adaptation. Pure calculations; no CAN access.

The MDH convention/parameters match the installed SDK's non-offset Piper
model (S-V1.5-8). SDK measured poses remain authoritative. This model only
solves requested targets; it never compares or substitutes measured poses.
"""
import math
import time

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


class NativePoseUnavailable(ValueError):
    pass


def warm_up():
    """Load optimizer kernels at entry, outside the control-frame deadline."""
    minimize(lambda v: float(v@v), np.ones(3)*.1, jac=lambda v: 2*v,
             method='SLSQP', constraints=[{'type': 'ineq', 'fun': lambda v: 1.-float(v@v)}])
    least_squares(lambda v: v-1., np.zeros(3))
    np.linalg.pinv(np.eye(6)+np.ones((6, 6))*.01)


def controller_bounds(report):
    if not report.get('complete') or report.get('source') != 'controller_query_0x472_reply_0x473':
        raise ValueError('native pose adaptation requires fresh controller joint limits')
    rows = report['limits']
    if len(rows) != 6 or [row['joint'] for row in rows] != list(range(1, 7)):
        raise ValueError('incomplete controller joint limits')
    lower, upper = np.deg2rad([[row['min_deg'] for row in rows], [row['max_deg'] for row in rows]])
    if not np.isfinite([lower, upper]).all() or np.any(lower >= upper):
        raise ValueError('invalid controller joint limits')
    return lower, upper


def placement(q, tcp, with_jacobian=False):
    """SDK MDH target model and its world-aligned TCP Jacobian."""
    alpha = (0., -math.pi/2, 0., math.pi/2, -math.pi/2, math.pi/2)
    lengths = (0., 0., .28503, -.02198, 0., 0.)
    heights = (.123, 0., 0., .25075, 0., .091)
    offsets = np.deg2rad([0., -174.22, -100.78, 0., 0., 0.])
    transform = np.eye(4)
    axes, origins = [], []
    for a, length, height, angle in zip(alpha, lengths, heights, np.asarray(q)+offsets):
        ca, sa, ct, st = math.cos(a), math.sin(a), math.cos(angle), math.sin(angle)
        if with_jacobian:
            origins.append(transform[:3, 3]+transform[:3, :3] @ np.array([length, 0., 0.]))
            axes.append(transform[:3, :3] @ np.array([0., -sa, ca]))
        transform = transform @ np.array([[ct, -st, 0., length],
            [st*ca, ct*ca, -sa, -sa*height], [st*sa, ct*sa, ca, ca*height], [0., 0., 0., 1.]])
    transform = transform @ tcp
    if not with_jacobian:
        return transform, None
    jacobian = np.vstack((np.array([np.cross(axis, transform[:3, 3]-origin)
        for axis, origin in zip(axes, origins)]).T, np.asarray(axes).T))
    return transform, jacobian


def turn(first, second):
    return Rotation.from_matrix(first[:3, :3] @ second[:3, :3].T).as_rotvec()


def adapt_target(state, proposal, tcp, report, policy, allow_orientation, *, budget_s=.025):
    """Keep translation, relax orientation progressively, otherwise hold.

    Returned joints are a model witness only, NEVER an actuator command or a
    claim about the controller's future branch/path. A failed candidate is
    never sent, committed, or replayed.
    """
    if state['controller'].get('firmware') != 'S-V1.5-8':
        raise NativePoseUnavailable('native pose adaptation has no verified model for this firmware')
    lower, upper = controller_bounds(report)
    seed = np.asarray(state['joints_rad'], float)
    desired = np.asarray(proposal['tcp_target'], float)
    start = np.asarray(proposal['start_tcp'], float)
    actual = np.eye(4)
    actual[:3, :3] = Rotation.from_euler('xyz', state['ee_pose']['rpy_deg'], degrees=True).as_matrix()
    actual[:3, 3] = state['ee_pose']['position_m']
    actual = actual @ tcp
    deadline = time.monotonic()+budget_s
    inverse_tcp = np.linalg.inv(tcp)
    def quantized(target):
        ee = target@inverse_tcp
        ee[:3, 3] = np.rint(ee[:3, 3]*1e6)/1e6
        ee[:3, :3] = Rotation.from_euler('xyz', np.rint(
            Rotation.from_matrix(ee[:3, :3]).as_euler('xyz', degrees=True)*1000)/1000,
            degrees=True).as_matrix()
        return ee@tcp
    def check_time():
        if time.monotonic() > deadline:
            raise NativePoseUnavailable('native pose adaptation calculation budget expired')
    def residual(q, target):
        check_time()
        value, _ = placement(q, tcp)
        return np.r_[value[:3, 3]-target[:3, 3], .1*turn(value, target)]
    def inverse(target, initial):
        def jacobian(x, target):
            value, jac = placement(x, tcp, True)
            phi = turn(value, target)
            theta = np.linalg.norm(phi)
            skew = np.array([[0., -phi[2], phi[1]], [phi[2], 0., -phi[0]], [-phi[1], phi[0], 0.]])
            factor = (1./12 if theta < 1e-4 else
                      1/theta**2-(1+math.cos(theta))/(2*theta*math.sin(theta)))
            left_inverse = np.eye(3)-.5*skew+factor*skew@skew
            return np.vstack((jac[:3], .1*left_inverse@jac[3:]))
        return least_squares(residual, initial, args=(target,), jac=jacobian, max_nfev=18,
                             ftol=1e-10, xtol=1e-10, gtol=1e-10).x
    def fits(q, target):
        error = residual(q, target)
        return (np.linalg.norm(error[:3]) <= 2e-6 and np.linalg.norm(error[3:]) <= math.radians(.002)*.1
                and np.all(q >= lower) and np.all(q <= upper))
    rate = math.radians(policy.rotation_deg_s)/policy.control_hz
    lead = math.radians(policy.rotation_lead_deg)
    for progress in (1., .5, .25):
        target = desired.copy()
        target[:3, 3] = start[:3, 3]+progress*(desired[:3, 3]-start[:3, 3])
        target = quantized(target)
        q = inverse(target, seed)
        if fits(q, target):
            return {'tcp_target': target.tolist(), 'adjustment_deg': 0., 'progress': progress,
                    'model_witness_joints_rad': q.tolist()}
        if not allow_orientation:
            raise NativePoseUnavailable('当前朝向的目标超出控制器关节范围；请调整方向或起姿')
        base = target.copy()
        around = np.zeros(3)
        witness = q
        working_lower, working_upper = lower.copy(), upper.copy()
        for iteration in range(3):
            check_time()
            _, jac = placement(witness, tcp, True)
            # Only orientation is free. Translation remains exactly the
            # requested point; numerical IK independently verifies each result.
            effect = np.linalg.pinv(jac, rcond=1e-9)[:, 3:]
            offset = witness-effect@around
            from_start, from_actual = turn(base, start), turn(base, actual)
            def rotations(value):
                check_time()
                return np.array([1.-np.sum((value+from_start)**2)/rate**2,
                                 1.-np.sum((value+from_actual)**2)/lead**2])
            def rotations_jac(value):
                return np.vstack((-2*(value+from_start)/rate**2,
                                  -2*(value+from_actual)/lead**2))
            # Linearization includes the previous quantized target's actual
            # witness, so rounding overrun is corrected without a fixed margin.
            solved = minimize(lambda v: float(v@v), around, jac=lambda v: 2*v,
                method='SLSQP', constraints=[
                    {'type': 'ineq', 'fun': lambda v: np.r_[offset+effect@v-working_lower,
                                                          working_upper-offset-effect@v],
                     'jac': lambda v: np.vstack((effect, -effect))},
                    {'type': 'ineq', 'fun': rotations, 'jac': rotations_jac}],
                options={'maxiter': 12, 'ftol': 1e-12})
            if np.min(rotations(solved.x)) < -1e-6:
                break
            candidate = base.copy()
            candidate[:3, :3] = Rotation.from_rotvec(solved.x).as_matrix()@base[:3, :3]
            # Validate the micrometre/millidegree packet, not just the
            # continuous optimizer's answer.
            candidate = quantized(candidate)
            witness = inverse(candidate, q)
            if (fits(witness, candidate) and np.linalg.norm(turn(candidate, actual)) <= lead and
                    np.linalg.norm(turn(candidate, start)) <= rate+1e-10):
                return {'tcp_target': candidate.tolist(), 'adjustment_deg': math.degrees(
                    np.linalg.norm(turn(candidate, desired))), 'progress': progress,
                    'model_witness_joints_rad': witness.tolist()}
            around = turn(candidate, base)
            working_lower += np.maximum(lower-witness, 0)*2+1e-6
            working_upper -= np.maximum(witness-upper, 0)*2+1e-6
    raise NativePoseUnavailable('自动调姿仍无法满足当前平移目标；松键后调整方向或起姿')
