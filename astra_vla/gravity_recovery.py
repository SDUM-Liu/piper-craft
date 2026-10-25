"""Bounded initial J2/J3 sag recovery; model planning never connects devices."""
from dataclasses import asdict, dataclass
from concurrent.futures import TimeoutError as FutureTimeoutError
import json
import math
import threading
import uuid

import numpy as np

from astra_vla.config import ROOT


class RecoveryPathChanged(RuntimeError):
    """Feedback passed runtime guards, but the remaining target needs a new path."""


@dataclass(frozen=True)
class RecoveryPolicy:
    enabled: bool = True
    selective_wrist_prepare: bool = False
    streamed_recovery: bool = True
    hold_speed_percent: int = 5
    inset_deg: float = .5
    max_start_excess_deg: float = 3.
    speed_percent: int = 1
    timeout_s: float = 10.
    path_tolerance_deg: float = .25
    feedback_speed_deg_s: float = 30.
    feedback_excess_noise_deg: float = .05
    arrival_tolerance_deg: float = .15

    def __post_init__(self):
        if any(not isinstance(value, bool) for value in
               (self.enabled, self.selective_wrist_prepare, self.streamed_recovery)):
            raise ValueError('gravity recovery enabled must be boolean')
        maxima = {'inset_deg': .5, 'max_start_excess_deg': 3.,
                  'speed_percent': 5, 'timeout_s': 15., 'path_tolerance_deg': .25,
                  'hold_speed_percent': 5,
                  'feedback_speed_deg_s': 30., 'feedback_excess_noise_deg': .05, 'arrival_tolerance_deg': .15}
        for name, maximum in maxima.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError('gravity recovery policy outside permitted range: ' + name)
        if int(self.speed_percent) != self.speed_percent or int(self.hold_speed_percent) != self.hold_speed_percent:
            raise ValueError('gravity recovery speed must be an integer')

    @classmethod
    def load(cls):
        path = ROOT / 'config/gravity_recovery.json'
        return cls(**json.loads(path.read_text())) if path.exists() else cls()


def recovery_joint_bounds(model):
    """Recovery bounds, with the J5 upper angle delegated to the controller.

    Keep a finite bound for JSON evidence and integer quantization: this is
    only the signed 32-bit millidegree CAN encoding range, not a physical
    joint-angle limit. Never modify the shared URDF model.
    """
    lower = model.model.lowerPositionLimit[:6].copy()
    upper = model.model.upperPositionLimit[:6].copy()
    upper[4] = math.radians(np.iinfo(np.int32).max / 1000)
    return lower, upper


def recovery_limit_excess(model, q):
    lower, upper = recovery_joint_bounds(model)
    return np.maximum(np.maximum(lower-q[:6], q[:6]-upper), 0)


def recovery_limit_violations(model, state):
    lower, upper = recovery_joint_bounds(model)
    return [item for item in model.joint_limit_violations(state)
            if item['measured_deg'] is None or
            not lower[item['joint']-1] <= state['joints_rad'][item['joint']-1] <= upper[item['joint']-1]]


def plan_recovery(model, state, other_state, geometry, policy):
    """Plan bounded J2/J3 startup correction without local collision queries.

    The caller may request a retreat after the ordinary entry hold detects a
    folded pose. Recovery itself uses measured angles, never mesh contacts.
    The ordinary hold is checked again before keyboard control starts.
    """
    if geometry.get('scope') != 'tcp_jog':
        raise ValueError('PIPER CRAFT recovery requires robot-local geometry')
    current = model.configuration(state)
    if not np.isfinite(current).all():
        raise ValueError('invalid gravity recovery feedback')
    violations = recovery_limit_violations(model, state)
    folded = (not violations and 0 <= math.degrees(current[1]) <= 3.5 and
              -3.5 <= math.degrees(current[2]) <= 0)
    if not violations and not folded:
        return {'required': False}
    if not policy.enabled:
        raise ValueError('automatic gravity recovery is disabled')
    if policy.selective_wrist_prepare:
        raise ValueError('selective wrist preparation remains disabled after failed hardware trial')
    control = state['controller']
    enabled_start = (control['ctrl_mode'] == 1 and control.get('move_mode') == 1 and
                     len(control['motors_enabled']) == 6 and all(control['motors_enabled']))
    if not enabled_start and (control['ctrl_mode'] != 0 or len(control['motors_enabled']) != 6 or
            any(control['motors_enabled']) or control.get('move_mode') in (4, 5)):
        raise ValueError('automatic gravity recovery requires fully disabled standby')
    if control.get('err_code', 0) or control.get('arm_status', 0) or state['gripper'].get('faults'):
        raise ValueError('automatic gravity recovery requires fault-free feedback')
    report = model.start_limit_report(state, operator_recovery=enabled_start)
    eligible = {item['joint'] for item in violations}
    report = {**report, 'blocking': [item for item in report['blocking'] if item['joint'] in eligible],
              'tolerated': [item for item in report['tolerated'] if item['joint'] in eligible]}
    if (report['blocking'] or any(x['joint'] not in (2, 3) or
            x['excess_deg'] > policy.max_start_excess_deg + 1e-10 for x in report['tolerated'])):
        detail = '; '.join(f"J{x['joint']}={x['measured_deg']}deg" for x in report['blocking']+report['tolerated'])
        raise ValueError('automatic gravity recovery only accepts J2/J3 sag up to 3 degrees: '+detail)
    model.feedback_pose(state)  # SDK readback stays authoritative; no FK comparison.
    lower, upper = recovery_joint_bounds(model)
    origin = np.asarray(geometry.get('recovery_origin_joints_rad', current[:6]), dtype=float)
    if origin.shape != (6,) or not np.isfinite(origin).all():
        raise ValueError('invalid recovery origin')
    initial_sdk = np.rint(np.rad2deg(current[:6])*1000).astype(np.int64)
    legal_sdk = np.clip(initial_sdk, np.ceil(np.rad2deg(lower)*1000).astype(np.int64),
                        np.floor(np.rad2deg(upper)*1000).astype(np.int64))
    goal = legal_sdk.copy()
    for item in report['tolerated']:
        index = item['joint']-1
        goal[index] = (math.ceil((math.degrees(lower[index])+policy.inset_deg)*1000)
                       if current[index] < lower[index] else
                       math.floor((math.degrees(upper[index])-policy.inset_deg)*1000))
    if folded:
        goal[1] = max(goal[1], math.ceil((math.degrees(lower[1])+policy.inset_deg)*1000))
        goal[2] = min(goal[2], math.floor((math.degrees(upper[2])-policy.inset_deg)*1000))
    bounded_hold = bool(policy.streamed_recovery and state.get('arm') == 'right')
    stages = []
    startup = None
    if bounded_hold and current[1] < lower[1]:
        # Use an in-range all-arm enable target, never the unrealized negative-J2 hold.
        startup = legal_sdk.tolist()
        stages.append(startup)
    shoulder = goal.copy()
    shoulder[2] = legal_sdk[2]
    for row in (shoulder.tolist(), goal.tolist()):
        if not stages or row != stages[-1]:
            stages.append(row)
    segments, previous = [], current.copy()
    excess = recovery_limit_excess(model, current)
    for row in stages:
        endpoint = current.copy()
        endpoint[:6] = np.deg2rad(np.asarray(row)/1000)
        check_recovery_target(model, dict(state, joints_rad=previous[:6].tolist()), row, origin)
        count = max(20, math.ceil(np.max(np.abs(endpoint[:6]-previous[:6]))/math.radians(.01)))
        for q in np.linspace(previous, endpoint, count+1)[int(bool(segments)):]:
            now = recovery_limit_excess(model, q)
            if np.any(now > excess+1e-10):
                raise ValueError('recovery increases joint-limit excess')
            excess = now
            segments.append(q[:6].tolist())
        previous = endpoint
    result = {'required': True, 'candidate': 'nearest_inset', 'joint_sdk': goal.tolist(),
              'local_collision_check': False, 'stage_joint_sdk': stages,
              'streamed_recovery': bounded_hold, 'wrist_preparation': [],
              'enabled_start': enabled_start, 'start_joints_rad': current[:6].tolist(),
              'goal_joints_rad': np.deg2rad(goal/1000).tolist(),
              'lower_rad': lower.tolist(), 'upper_rad': upper.tolist(),
              'initial_excess_rad': recovery_limit_excess(model, current).tolist(),
              'path_joints_rad': segments, 'samples': len(segments),
              'tolerated_start_joint_limits': report['tolerated']}
    if startup is not None:
        result['legal_startup_joint_sdk'] = startup
    validated_recovery_plan(result, state, policy, origin)
    return result


def check_recovery_target(model, state, integers, origin):
    """Validate ordinary MOVE_J packet encoding, feedback and recovery angles."""
    if (not isinstance(integers, (list, tuple)) or len(integers) != 6 or
            any(isinstance(v, bool) or not isinstance(v, int) for v in integers)):
        raise ValueError('invalid streamed recovery joint packet')
    current = np.asarray(state['joints_rad'], dtype=float)
    origin = np.asarray(origin, dtype=float)
    if (current.shape != (6,) or origin.shape != (6,) or
            not np.isfinite(current).all() or not np.isfinite(origin).all()):
        raise ValueError('invalid recovery origin/feedback')
    target = np.deg2rad(np.asarray(integers)/1000)
    lower, upper = recovery_joint_bounds(model)
    if np.any(target < lower) or np.any(target > upper):
        raise ValueError('streamed recovery target exceeds recovery joint bounds')
    return True


def rebind_recovery_path(model, state, other_state, geometry, plan):
    """Revalidate the complete remaining path after the motor enable handshake."""
    current = model.configuration(state)
    model.feedback_pose(state)
    waypoints = plan.get('stage_joint_sdk', [plan['joint_sdk']])
    segments, previous = [], current.copy()
    for row in waypoints:
        endpoint = current.copy()
        endpoint[:6] = np.deg2rad(np.asarray(row)/1000)
        count = max(20, math.ceil(np.max(np.abs(endpoint[:6]-previous[:6]))/math.radians(.01)))
        segments.extend(np.linspace(previous, endpoint, count+1)[int(bool(segments)):])
        previous = endpoint
    excess = recovery_limit_excess(model, current)
    path = []
    for q in segments:
        now = recovery_limit_excess(model, q)
        if np.any(now > excess+1e-10):
            raise ValueError('enabled recovery path increases joint-limit excess')
        excess = now
        path.append(q[:6].tolist())
    if np.any(excess):
        raise ValueError('enabled recovery endpoint is outside limits')
    return dict(plan, start_joints_rad=current[:6].tolist(), path_joints_rad=path,
                initial_excess_rad=recovery_limit_excess(model, current).tolist())


def validated_recovery_plan(plan, state, policy, origin=None):
    """Validate replanned integer waypoints against fresh feedback and angles."""
    goal = np.asarray(plan['goal_joints_rad'], dtype=float)
    integers = plan['joint_sdk']
    path = np.asarray(plan['path_joints_rad'], dtype=float)
    if (not isinstance(integers, list) or len(integers) != 6 or
            any(isinstance(x, bool) or not isinstance(x, int) for x in integers) or
            goal.shape != (6,) or not np.isfinite(goal).all() or
            not np.allclose(goal, np.deg2rad(np.asarray(integers)/1000), rtol=0, atol=1e-12) or
            path.ndim != 2 or path.shape[1] != 6 or len(path) < 2 or not np.isfinite(path).all() or
            not np.allclose(path[0], state['joints_rad'], rtol=0, atol=1e-12) or
            not np.allclose(path[-1], goal, rtol=0, atol=1e-12) or
            np.any(goal < plan['lower_rad']) or np.any(goal > plan['upper_rad'])):
        raise ValueError('invalid quantized gravity recovery path')
    origin = np.asarray(state['joints_rad'] if origin is None else origin, dtype=float)
    if origin.shape != (6,) or not np.isfinite(origin).all():
        raise ValueError('invalid recovery origin')
    stages = plan.get('stage_joint_sdk', [integers])
    bounded_hold = bool(plan.get('streamed_recovery'))
    if bounded_hold and (not policy.streamed_recovery or state.get('arm') != 'right'):
        raise ValueError('streamed recovery is limited to right-arm recovery')
    if (not isinstance(stages, list) or not 1 <= len(stages) <= (8 if bounded_hold else 3) or stages[-1] != integers or
            any(not isinstance(row, list) or len(row) != 6 or
                any(isinstance(x, bool) or not isinstance(x, int) for x in row) for row in stages)):
        raise ValueError('invalid gravity recovery stage targets')
    for row in stages:
        waypoint = np.deg2rad(np.asarray(row)/1000)
        outside = np.maximum(np.maximum(np.asarray(plan['lower_rad'])-waypoint,
                                         waypoint-np.asarray(plan['upper_rad'])), 0)
        permitted = np.zeros(6)
        if (np.any(outside > permitted+1e-10) or
                not any(np.allclose(sample, waypoint, atol=1e-12, rtol=0) for sample in path)):
            raise ValueError('recovery stage is outside validated path/bounds')
    preparation = plan.get('wrist_preparation', [])
    if preparation:
        if (not policy.selective_wrist_prepare or state.get('arm') != 'right' or
                any(state['controller']['motors_enabled']) or len(preparation) != 4):
            raise ValueError('selective wrist preparation requires a disabled supported right arm')
        previous = np.asarray(state['joints_rad'])
        for item in preparation:
            command = item['joint_sdk']
            expected = np.asarray(item['expected_joints_rad'])
            axis = item['axis']-1
            if (axis not in (3, 4) or expected.shape != (6,) or not np.isfinite(expected).all() or
                    not isinstance(command, list) or len(command) != 6 or
                    any(isinstance(value, bool) or not isinstance(value, int) for value in command)):
                raise ValueError('invalid wrist preparation target')
            target = np.deg2rad(np.asarray(command)/1000)
            passive = [index for index in range(6) if index != axis]
            if (np.any(target < plan['lower_rad']) or np.any(target > plan['upper_rad']) or
                    not np.allclose(expected[passive], previous[passive], rtol=0, atol=1e-12) or
                    not np.allclose(target[3:5], expected[3:5], rtol=0, atol=1e-12) or
                    not any(np.allclose(sample, expected, atol=1e-12, rtol=0) for sample in path)):
                raise ValueError('wrist preparation is outside validated original path/bounds')
            previous = expected
    return goal, integers, path, stages


def recover(controller, arm, state, geometry_plan, *, recovery_origin=None, path_attempt=0):
    """Called under the owner's action lock; no reset, disable or jaw commands."""
    robot, clock = controller.robot, controller.clock
    if controller.policy.follow_mode != 'position_velocity':
        raise ValueError('automatic gravity recovery requires ordinary MOVE_J')
    policy = RecoveryPolicy.load()
    recovery_origin = (list(state['joints_rad']) if recovery_origin is None else list(recovery_origin))
    planning_geometry = dict(controller.geometry, recovery_origin_joints_rad=recovery_origin)
    other = 'right' if arm == 'left' else 'left'
    before = {arm: state, other: robot.healthy(other, False)}
    epoch = getattr(robot, 'stop_epoch', 0)
    cameras = {'frames': controller.camera_read()['cameras'], 'error': None}
    controller.fresh_cameras(cameras['frames'])
    capture_done = threading.Event()
    enabled_seen = False
    preparing_wrist, allowed_motors, live_motors = False, set(), set()
    def capture():
        while not capture_done.wait(.1):
            try:
                frames = controller.camera_read()['cameras']
                controller.fresh_cameras(frames)
                cameras.update(frames=frames, error=None)
            except Exception as exc:
                cameras['error'] = str(exc)
    controller.phase, controller.arm = 'gravity_recovery_check', arm
    controller.recovery = {'phase': 'checking', 'arm': arm}
    def guard():
        nonlocal enabled_seen
        if robot.stopped.is_set() or robot.fault or getattr(robot, 'stop_epoch', 0) != epoch:
            raise RuntimeError('stopped during gravity recovery')
        if controller.token_value() != controller.token:
            raise RuntimeError('configuration changed during gravity recovery')
        if cameras['error']:
            raise RuntimeError(cameras['error'])
        controller.fresh_cameras(cameras['frames'])
        states = {a: robot.healthy(a, False) for a in before}
        for s in states.values():
            robot.check_motion_state(s)
        controller.check_gripper_quarantine(arm, states[arm])
        robot.check_motion_unchanged(other, before[other], states[other], 'during gravity recovery')
        # Manual TCP mode has no other-arm geometry dependency. Use its normal
        # 0.5-degree feedback-change guard for the parked arm; workspace tasks
        # retain the tighter geometry-binding tolerance.
        other_tolerance = .5
        if (np.max(np.abs(np.asarray(states[other]['joints_rad'])-before[other]['joints_rad'])) > math.radians(other_tolerance) or
                abs(states[other]['gripper']['width_m']-before[other]['gripper']['width_m']) > .00025):
            raise RuntimeError('other arm changed during gravity recovery')
        if abs(states[arm]['gripper']['width_m']-state['gripper']['width_m']) > .00025:
            raise RuntimeError('gripper changed during gravity recovery')
        mode = states[arm]['controller']
        enabled = all(mode['motors_enabled'])
        if preparing_wrist:
            actual_motors = {index for index, on in enumerate(mode['motors_enabled']) if on}
            if actual_motors-allowed_motors or live_motors-actual_motors:
                raise RuntimeError('unexpected motor enable state during wrist preparation')
            passive = [index for index in range(6) if index not in allowed_motors]
            if np.max(np.abs(np.asarray(states[arm]['joints_rad'])[passive]-
                             np.asarray(recovery_origin)[passive])) > math.radians(.15):
                raise RuntimeError('supported passive joints moved during wrist preparation')
        if (mode['ctrl_mode'] not in (0, 1) or mode['move_mode'] in (4, 5) or
                (mode['ctrl_mode'] == 1 and mode['move_mode'] != 1) or
                (enabled_seen and (not enabled or mode['ctrl_mode'] != 1))):
            raise RuntimeError('mode/enable changed during gravity recovery')
        enabled_seen |= enabled and mode['ctrl_mode'] == 1
        return states
    for planning_attempt in range(5):
        job = robot.geometry_pool.submit(geometry_plan, state, before[other], planning_geometry, asdict(policy))
        plan = job.result(timeout=15)
        if not plan.get('required'):
            latest = guard()[arm]
            if np.max(np.abs(np.asarray(latest['joints_rad'])-state['joints_rad'])) > math.radians(.05):
                if planning_attempt == 4:
                    raise RuntimeError('gravity recovery start moved during planning')
                state, before[arm] = latest, latest
                continue
            controller.recovery.update(phase='completed')
            return latest
        goal, integers, path, stages = validated_recovery_plan(plan, state, policy, recovery_origin)
        controller.recovery = {'phase': 'prepared', 'arm': arm, 'candidate': plan['candidate'],
                               'target_joints_deg': np.rad2deg(plan['goal_joints_rad']).tolist(),
                               'stage': 1, 'stages': len(stages)}
        cameras['frames'] = controller.camera_read()['cameras']
        latest = guard()[arm]
        robot.check_motion_unchanged(arm, state, latest, 'during gravity recovery planning')
        if np.max(np.abs(np.asarray(latest['joints_rad'])-state['joints_rad'])) <= math.radians(.05):
            break
        if planning_attempt == 4:
            raise RuntimeError('gravity recovery start moved during planning')
        state = latest
        before[arm] = latest
    control = latest['controller']
    enabled_start = plan.get('enabled_start', False)
    expected_mode = (control['ctrl_mode'] == 1 and control['move_mode'] == 1 and all(control['motors_enabled'])
                     if enabled_start else control['ctrl_mode'] == 0 and not any(control['motors_enabled']))
    if not expected_mode:
        raise RuntimeError('gravity recovery start is no longer disabled standby')
    trace = controller.trace_factory(uuid.uuid4().hex, arm, controller.policy)
    trace.context = {'tool': controller.tool, 'owner': 'gravity_recovery', 'recovery_plan': plan,
                     'recovery_policy': asdict(policy), 'configuration_token': controller.token,
                     'planning_attempts': planning_attempt+1, 'path_attempt': path_attempt,
                     'recovery_origin_joints_rad': recovery_origin}
    trace.record('recovery_start', state=before)
    sent, success, failure = False, False, None
    stage_index = 0
    capture_thread = threading.Thread(target=capture, daemon=True, name='gravity-recovery-camera')
    capture_thread.start()
    excess = np.asarray(plan['initial_excess_rad'])
    initial_excess = excess.copy()
    origin = np.asarray(recovery_origin).copy()
    tracking = enabled_start
    previous = latest
    def progress(states, require_path=True):
        nonlocal previous, excess
        now = states[arm]
        q = np.asarray(now['joints_rad'])
        deviation = np.min(np.max(np.abs(path-q), axis=1))
        outside = np.maximum(np.maximum(np.asarray(plan['lower_rad'])-q, q-np.asarray(plan['upper_rad'])), 0)
        # The fresh-feedback start guard elsewhere uses 0.05 degrees. Real
        # ordinary MOVE_J feedback rebounded by 0.021 degrees during an overall
        # retreat; a 0.02-degree monotonicity threshold stopped a valid path.
        # Keep the best-ever excess as the reference (no cumulative drift).
        # Targets and the planned path obey the remaining recovery bounds.
        permitted = excess+math.radians(policy.feedback_excess_noise_deg) if tracking else initial_excess.copy()
        if not tracking:
            # Energizing a previously disabled gravity-loaded joint is not an
            # executed retreat yet. Bind this small transition to the ORIGINAL
            # start; never ratchet the allowance upward with each feedback.
            permitted[1:3] = np.minimum(math.radians(policy.max_start_excess_deg),
                                        permitted[1:3]+math.radians(policy.path_tolerance_deg))
        trace.record('feedback', state=states, path_residual_deg=math.degrees(deviation),
                     stage='tracking' if tracking else 'enabling')
        # Enabling has no deviation threshold against the pre-enable path.
        # Revalidate from powered feedback before trajectory tracking starts.
        if np.any(outside > permitted+1e-12):
            raise RuntimeError('gravity recovery feedback increased limit excess')
        excess = np.minimum(excess, outside)
        dt = now['feedback_stamps']['joint']-previous['feedback_stamps']['joint']
        if dt <= 0 and np.max(np.abs(q-previous['joints_rad'])) > math.radians(.002):
            raise RuntimeError('gravity recovery joint feedback is not advancing')
        # During torque take-up the 200 Hz position feedback can step between
        # encoder samples (recorded: 0.153 deg in 4.89 ms). Judge this
        # startup transition on the 50 ms command period, retaining sag and
        # speed guards. Actual
        # trajectory tracking keeps its timestamp-based speed check.
        rate_interval = max(dt, .05) if not tracking else dt
        if dt > 0 and np.max(np.abs(q-previous['joints_rad'])) > math.radians(policy.feedback_speed_deg_s)*rate_interval+math.radians(.002):
            raise RuntimeError('gravity recovery feedback speed exceeds bound')
        previous = now
        if tracking and require_path and deviation > math.radians(policy.path_tolerance_deg)+1e-12:
            raise RecoveryPathChanged('recovery feedback requires remaining-path revalidation')
        return outside
    def streamed_hold():
        nonlocal sent, plan, path, stages, excess, initial_excess, previous, tracking
        p = robot.arms[arm]
        tracking = False
        controller.phase = 'gravity_recovering'
        controller.recovery.update(phase='enabling_hold')
        def packet(method, arguments):
            nonlocal sent
            progress(guard(), require_path=False)
            robot.actuator_commands_sent, sent = True, True
            trace.record('command', method=method, arguments=list(arguments))
            robot.command(p, method, *arguments)
        def target(arguments):
            states = guard()
            progress(states, require_path=False)
            robot.check_gravity_recovery_target(states[arm], tuple(map(int, arguments)), origin)
            newest = guard()
            progress(newest, require_path=False)
            if np.max(np.abs(np.asarray(newest[arm]['joints_rad'])-
                             states[arm]['joints_rad'])) > math.radians(.05):
                raise RecoveryPathChanged('feedback changed during streamed recovery target validation')
            packet('JointCtrl', tuple(map(int, arguments)))
        def quantized_hold(measured):
            result = np.rint(np.rad2deg(measured['joints_rad'])*1000).astype(np.int64)
            return np.clip(result, np.ceil(np.rad2deg(plan['lower_rad'])*1000).astype(np.int64),
                           np.floor(np.rad2deg(plan['upper_rad'])*1000).astype(np.int64))
        startup = plan.get('legal_startup_joint_sdk')
        hold = np.array(startup, dtype=np.int64) if startup else quantized_hold(guard()[arm])
        def initial_target():
            if startup:
                # Target/corridor angles were validated
                # before any write. Do not send negative encoder
                # sag as a hold: this controller did not execute that target.
                waypoint = np.deg2rad(hold/1000)
                if (not np.array_equal(hold, startup) or np.any(waypoint < plan['lower_rad']) or
                        np.any(waypoint > plan['upper_rad']) or
                        not any(np.allclose(row, waypoint, rtol=0, atol=1e-12) for row in path)):
                    raise ValueError('invalid legal startup target')
                packet('JointCtrl', tuple(map(int, hold)))
            else:
                target(hold)
        for args in ((0, 1, 0), (0, 4, 0)):
            packet('MotionCtrl_1', args)
        packet('MotionCtrl_2', (1, 1, int(policy.hold_speed_percent), 0))
        initial_target()
        packet('EnableArm', (7,))
        packet('MotionCtrl_1', (0, 2, 0))
        enable_deadline, stable, last_enable = clock.monotonic()+5, None, clock.monotonic()
        while clock.monotonic() < enable_deadline:
            if clock.monotonic()-last_enable >= .1:
                packet('EnableArm', (7,))
                last_enable = clock.monotonic()
            packet('MotionCtrl_2', (1, 1, int(policy.hold_speed_percent), 0))
            initial_target()
            states = guard()
            mode = states[arm]['controller']
            ready = (all(mode['motors_enabled']) and mode['ctrl_mode'] == 1 and mode['move_mode'] == 1 and
                     np.max(np.abs(np.asarray(states[arm]['joints_rad'])-np.deg2rad(hold/1000))) <= math.radians(.1))
            stable = (clock.monotonic() if stable is None else stable) if ready else None
            if stable is not None and clock.monotonic()-stable >= .5:
                break
            clock.sleep(.05)
        else:
            raise RuntimeError('legal startup/enable did not settle')
        # Replan against powered encoders while holding a legal settled target.
        from astra_vla.workers.teleop import geometry_recovery_replan
        for attempt in range(5):
            states = guard()
            progress(states, require_path=False)
            hold = quantized_hold(states[arm])
            job = robot.geometry_pool.submit(geometry_recovery_replan, states[arm], states[other],
                controller.geometry, asdict(policy), plan, origin.tolist())
            deadline = clock.monotonic()+15
            while True:
                packet('MotionCtrl_2', (1, 1, int(policy.hold_speed_percent), 0))
                target(hold)
                try:
                    rebound = job.result(timeout=.025)
                    break
                except FutureTimeoutError:
                    if clock.monotonic() >= deadline:
                        raise RuntimeError('streamed recovery planning timed out')
            latest_states = guard()
            progress(latest_states, require_path=False)
            if np.max(np.abs(np.asarray(latest_states[arm]['joints_rad'])-
                             states[arm]['joints_rad'])) <= math.radians(.05):
                break
            if attempt == 4:
                raise RuntimeError('streamed recovery hold moved during planning')
        _, _, path, stages = validated_recovery_plan(rebound, states[arm], policy, origin)
        plan, path = rebound, np.asarray(path)
        excess = np.asarray(plan['initial_excess_rad'])
        initial_excess = np.minimum(initial_excess, excess)
        previous, tracking = latest_states[arm], True
        trace.record('streamed_recovery_revalidated', state=latest_states, plan=plan)
        deadline = clock.monotonic()+15
        for index, goal_sdk in enumerate(stages):
            controller.recovery.update(phase='moving', stage=index+1, stages=len(stages))
            trace.record('recovery_stage', stage=index+1, target=goal_sdk)
            # Bind each complete stage at rest. Tiny replacement MOVE_J goals
            # stalled in the controller planner; use its native trajectory for
            # each independently checked axis, at the bounded startup speed.
            for args in ((0, 1, 0), (0, 4, 0)):
                packet('MotionCtrl_1', args)
            confirmation = robot.stationary_feedback(require_fault_free=True, joint_tolerance_deg=.15,
                timeout_s=5., feedback_check=lambda: progress(guard()))
            if not confirmation['confirmed']:
                raise RuntimeError('recovery stage handoff stop is unconfirmed')
            packet('MotionCtrl_2', (1, 1, int(policy.hold_speed_percent), 0))
            target(goal_sdk)
            packet('MotionCtrl_1', (0, 2, 0))
            quiet = None
            while clock.monotonic() < deadline:
                states = guard()
                progress(states)
                actual = np.asarray(states[arm]['joints_rad'])
                if np.max(np.abs(actual-np.deg2rad(np.asarray(goal_sdk)/1000))) <= math.radians(.1):
                    quiet = clock.monotonic() if quiet is None else quiet
                    if clock.monotonic()-quiet >= .25:
                        break
                else:
                    quiet = None
                packet('MotionCtrl_2', (1, 1, int(policy.hold_speed_percent), 0))
                packet('JointCtrl', tuple(goal_sdk))
                clock.sleep(.05)
            else:
                raise RuntimeError('bounded recovery stages did not converge within 15 seconds')
        for args in ((0, 1, 0), (0, 4, 0)):
            packet('MotionCtrl_1', args)
        confirmation = robot.stationary_feedback(require_fault_free=True, joint_tolerance_deg=.15,
            timeout_s=5., feedback_check=lambda: progress(guard()))
        trace.record('streamed_recovery_stop', confirmation=confirmation)
        if not confirmation['confirmed']:
            raise RuntimeError('streamed recovery stop is unconfirmed')
        final = guard()
        if np.any(progress(final) > 0):
            raise RuntimeError('streamed recovery final pose is outside hard limits')
        return final[arm]
    try:
        trace.activate()
        p = robot.arms[arm]
        if plan.get('streamed_recovery'):
            result = streamed_hold()
            success = True
            controller.recovery.update(phase='completed', trace_path=str(trace.path) if trace.path else None)
            return result
        if plan.get('wrist_preparation'):
            if arm != 'right':
                raise ValueError('selective wrist preparation is limited to the supported right arm')
            preparing_wrist = True
            controller.recovery.update(phase='wrist_preparation')
            for stage, item in enumerate(plan['wrist_preparation'], 1):
                controller.recovery.update(stage=stage, stages=len(plan['wrist_preparation']))
                axis, command = item['axis'], tuple(item['joint_sdk'])
                allowed_motors.add(axis-1)
                trace.record('wrist_preparation_stage', stage=stage, target=item,
                             allowed_motors=sorted(index+1 for index in allowed_motors))
                for method, args in [('MotionCtrl_1', (0, 1, 0)), ('MotionCtrl_1', (0, 4, 0)),
                                     ('MotionCtrl_2', (1, 1, int(policy.speed_percent), 0)),
                                     ('JointCtrl', command), ('EnableArm', (axis,))]:
                    progress(guard())
                    robot.actuator_commands_sent, sent = True, True
                    trace.record('command', method=method, arguments=list(args))
                    robot.command(p, method, *args)
                enable_deadline = clock.monotonic()+3
                while True:
                    states = guard()
                    progress(states)
                    mask = {i for i, on in enumerate(states[arm]['controller']['motors_enabled']) if on}
                    if mask == allowed_motors:
                        live_motors = mask
                        break
                    if clock.monotonic() >= enable_deadline:
                        raise RuntimeError('wrist motor enable unconfirmed')
                    clock.sleep(.05)
                for args in ((0, 4, 0),):
                    progress(guard())
                    trace.record('command', method='MotionCtrl_1', arguments=list(args))
                    robot.command(p, 'MotionCtrl_1', *args)
                for method, args in [('JointCtrl', command), ('MotionCtrl_1', (0, 2, 0))]:
                    progress(guard())
                    trace.record('command', method=method, arguments=list(args))
                    robot.command(p, method, *args)
                deadline, quiet = clock.monotonic()+5, None
                while clock.monotonic() < deadline:
                    states = guard()
                    progress(states)
                    active_axes = sorted(live_motors)
                    arrived = np.max(np.abs(np.asarray(states[arm]['joints_rad'])[active_axes]-
                                            np.asarray(item['expected_joints_rad'])[active_axes])) <= math.radians(.05)
                    quiet = (clock.monotonic() if quiet is None else quiet) if arrived else None
                    if quiet is not None and clock.monotonic()-quiet >= .25:
                        break
                    for method, args in [('MotionCtrl_2', (1, 1, int(policy.speed_percent), 0)),
                                         ('JointCtrl', command)]:
                        progress(guard())
                        trace.record('command', method=method, arguments=list(args))
                        robot.command(p, method, *args)
                    clock.sleep(.05)
                else:
                    raise RuntimeError('wrist preparation timed out')
            for args in ((0, 1, 0), (0, 4, 0)):
                progress(guard())
                trace.record('command', method='MotionCtrl_1', arguments=list(args))
                robot.command(p, 'MotionCtrl_1', *args)
            confirmation = robot.stationary_feedback(require_fault_free=True, joint_tolerance_deg=.15,
                timeout_s=5., feedback_check=lambda: progress(guard()))
            trace.record('wrist_preparation_confirmation', confirmation=confirmation)
            if not confirmation['confirmed']:
                raise RuntimeError('wrist preparation pause is unconfirmed')
            from astra_vla.workers.teleop import geometry_recovery_rebind
            prepared = guard()
            remaining = dict(plan, wrist_preparation=[])
            job = robot.geometry_pool.submit(geometry_recovery_rebind, prepared[arm], prepared[other],
                                              controller.geometry, remaining)
            validation_deadline = clock.monotonic()+15
            while True:
                progress(guard())
                try:
                    rebound = job.result(timeout=.05)
                    break
                except FutureTimeoutError:
                    if clock.monotonic() >= validation_deadline:
                        raise RuntimeError('prepared recovery validation timed out')
            goal, integers, path, stages = validated_recovery_plan(rebound, prepared[arm], policy, recovery_origin)
            latest_states = guard()
            progress(latest_states)
            if np.max(np.abs(np.asarray(latest_states[arm]['joints_rad'])-
                             prepared[arm]['joints_rad'])) > math.radians(.05):
                raise RuntimeError('prepared recovery start moved during validation')
            plan, path = rebound, np.asarray(path)
            latest, previous = latest_states[arm], latest_states[arm]
            excess = np.asarray(plan['initial_excess_rad'])
            initial_excess = excess.copy()
            preparing_wrist = False
            trace.record('wrist_preparation_completed', state=latest_states, plan=plan)
            controller.recovery.update(phase='moving', stage=1, stages=len(stages))
        commands = [('MotionCtrl_1', (0, 1, 0)), ('MotionCtrl_1', (0, 4, 0)),
                    ('JointCtrl', tuple(stages[0])),
                    ('MotionCtrl_2', (1, 1, int(policy.speed_percent), 0)),
                    ('JointCtrl', tuple(stages[0]))]
        if not enabled_start:
            commands.append(('EnableArm', (7,)))
        controller.phase = 'gravity_recovering'
        controller.recovery['phase'] = 'moving'
        for method, arguments in commands:
            progress(guard())
            robot.actuator_commands_sent, sent = True, True
            trace.record('command', method=method, arguments=list(arguments))
            robot.command(p, method, *arguments)
        if not enabled_start:
            enable_deadline = clock.monotonic()+3
            enable_since = None
            last_enable = clock.monotonic()
            while True:
                states = guard()
                progress(states)
                mode = states[arm]['controller']
                if all(mode['motors_enabled']) and mode['ctrl_mode'] == 1 and mode['move_mode'] == 1:
                    enable_since = clock.monotonic() if enable_since is None else enable_since
                    if clock.monotonic()-enable_since >= .5:
                        break
                elif clock.monotonic()-last_enable >= .1:
                    trace.record('command', method='EnableArm', arguments=[7])
                    robot.command(p, 'EnableArm', 7)
                    last_enable = clock.monotonic()
                if clock.monotonic() >= enable_deadline:
                    raise RuntimeError('gravity recovery motor enable not confirmed')
                clock.sleep(.05)
            from astra_vla.workers.teleop import geometry_recovery_replan
            for attempt in range(5):
                states = guard()
                progress(states)
                job = robot.geometry_pool.submit(geometry_recovery_replan, states[arm], states[other],
                    controller.geometry, asdict(policy), plan,
                    origin.tolist())
                validation_deadline = clock.monotonic()+15
                while True:
                    progress(guard())
                    try:
                        rebound = job.result(timeout=.05)
                        break
                    except FutureTimeoutError:
                        if clock.monotonic() >= validation_deadline:
                            raise RuntimeError('enabled recovery validation timed out')
                latest_states = guard()
                progress(latest_states)
                if np.max(np.abs(np.asarray(states[arm]['joints_rad'])-latest_states[arm]['joints_rad'])) <= math.radians(.05):
                    break
                if attempt == 4:
                    raise RuntimeError('enabled recovery start moved during validation')
            if not rebound.get('required'):
                for args in ((0, 1, 0), (0, 4, 0)):
                    progress(guard())
                    trace.record('command', method='MotionCtrl_1', arguments=list(args))
                    robot.command(p, 'MotionCtrl_1', *args)
                confirmation = robot.stationary_feedback(require_fault_free=True, joint_tolerance_deg=.15)
                if not confirmation['confirmed']:
                    raise RuntimeError('gravity recovery stopping is unconfirmed')
                success = True
                controller.recovery.update(phase='completed', trace_path=str(trace.path) if trace.path else None)
                return guard()[arm]
            goal, integers, rebound_path, stages = validated_recovery_plan(rebound, states[arm], policy, origin)
            plan = rebound
            path = np.asarray(plan['path_joints_rad'])
            excess = np.asarray(plan['initial_excess_rad'])
            tracking = True
            controller.recovery.update(candidate=plan['candidate'], target_joints_deg=np.rad2deg(goal).tolist(),
                                       stage=1, stages=len(stages))
            trace.record('enabled_path_revalidated', plan=plan, state=latest_states, replanned=True)
            progress(latest_states)
        # Discard the startup target before loading the plan bound to settled
        # encoders. A resume must not execute a queued pre-enable target.
        for method, arguments in [('MotionCtrl_1', (0, 4, 0)),
                                  ('MotionCtrl_2', (1, 1, int(policy.speed_percent), 0)),
                                  ('JointCtrl', tuple(stages[0])), ('MotionCtrl_1', (0, 2, 0))]:
            progress(guard())
            trace.record('command', method=method, arguments=list(arguments))
            robot.command(p, method, *arguments)
        deadline = clock.monotonic()+policy.timeout_s
        stable_since = None
        while clock.monotonic() < deadline:
            states = guard()
            now = states[arm]
            q = np.asarray(now['joints_rad'])
            mode = now['controller']
            enabled = all(mode['motors_enabled'])
            if (mode['ctrl_mode'] not in (0, 1) or (mode['ctrl_mode'] == 1 and mode['move_mode'] != 1) or
                    (enabled_seen and (not enabled or mode['ctrl_mode'] != 1))):
                raise RuntimeError('mode/enable changed during gravity recovery')
            enabled_seen |= enabled and mode['ctrl_mode'] == 1
            outside = progress(states)
            inside = not np.any(outside > 0)
            arrived = np.max(np.abs(q-np.deg2rad(np.asarray(stages[stage_index])/1000))) <= math.radians(policy.arrival_tolerance_deg)
            last_stage = stage_index == len(stages)-1
            if enabled_seen and inside and arrived:
                stable_since = clock.monotonic() if stable_since is None else stable_since
                if clock.monotonic()-stable_since >= .5:
                    if not last_stage:
                        stage_index += 1
                        stable_since = None
                        controller.recovery['stage'] = stage_index+1
                        trace.record('recovery_stage', stage=stage_index+1, target=stages[stage_index])
                        continue
                    robot.command(p, 'MotionCtrl_1', 0, 1, 0)
                    confirmation = robot.stationary_feedback(require_fault_free=True,
                        joint_tolerance_deg=.15 if controller.owner == 'desktop' else .1)
                    if not confirmation['confirmed']:
                        raise RuntimeError('gravity recovery stopping is unconfirmed')
                    final = guard()
                    if np.any(progress(final) > 0):
                        raise RuntimeError('gravity recovery left hard limits during stop confirmation')
                    success = True
                    controller.recovery.update(phase='completed', trace_path=str(trace.path) if trace.path else None)
                    return final[arm]
            else:
                stable_since = None
            # Firmware handoff needs an active ordinary MOVE_J stream after
            # enabling, just like the existing lifecycle controller. Recheck
            # stop/configuration/feedback before EACH packet; never resume or
            # refresh a target from a failed/cancelled preparation.
            for method, arguments in [('MotionCtrl_2', (1, 1, int(policy.speed_percent), 0)),
                                      ('JointCtrl', tuple(stages[stage_index]))]:
                progress(guard())
                trace.record('command', method=method, arguments=list(arguments))
                robot.command(p, method, *arguments)
            clock.sleep(.05)
        raise RuntimeError('automatic gravity recovery timed out')
    except RecoveryPathChanged as exc:
        try:
            if path_attempt >= 2:
                raise RuntimeError('gravity recovery did not converge after bounded path replanning') from exc
            # Changed feedback requires replanning before replaying any target.
            # Pause, confirm actual quiet, then plan from the new encoders.
            progress(guard(), require_path=False)
            for arguments in ((0, 1, 0), (0, 4, 0)):
                progress(guard(), require_path=False)
                trace.record('command', method='MotionCtrl_1', arguments=list(arguments))
                robot.command(robot.arms[arm], 'MotionCtrl_1', *arguments)
            confirmation = robot.stationary_feedback(require_fault_free=True, joint_tolerance_deg=.15,
                timeout_s=5., feedback_check=lambda: progress(guard(), require_path=False))
            trace.record('replanning_pause_confirmation', confirmation=confirmation)
            if not confirmation['confirmed']:
                raise RuntimeError('recovery replanning pause is unconfirmed')
            latest_states = guard()
            progress(latest_states, require_path=False)
            trace.record('remaining_path_replan', state=latest_states, stop_confirmation=confirmation,
                         next_attempt=path_attempt+1)
            controller.recovery.update(phase='replanning')
            capture_done.set()
            capture_thread.join(timeout=.1)
            result = recover(controller, arm, latest_states[arm], geometry_plan,
                             recovery_origin=recovery_origin,
                             path_attempt=path_attempt+1)
            success = True
            controller.recovery.update(phase='completed', trace_path=str(trace.path) if trace.path else None)
            return result
        except Exception as error:
            failure = str(error)
            trace.record('recovery_failed', reason=failure)
            controller.recovery.update(phase='failed', reason=failure)
            if sent:
                robot.stop('automatic gravity recovery failed: '+failure, manual_tcp=controller.owner == 'desktop')
            raise
    except Exception as exc:
        failure = str(exc)
        trace.record('recovery_failed', reason=failure)
        controller.recovery.update(phase='failed', reason=str(exc))
        if sent:
            robot.stop('automatic gravity recovery failed: '+str(exc), manual_tcp=controller.owner == 'desktop')
        raise
    finally:
        capture_done.set()
        capture_thread.join(timeout=.1)
        trace.close({'recovery_completed': success, 'failure_reason': failure, 'hardware_acceptance_complete': False})
