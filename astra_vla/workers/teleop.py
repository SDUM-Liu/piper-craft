"""Exclusive, bounded MOVE J streaming inside the existing CAN owner."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from astra_vla.config import ROOT, RUNTIME, settings, tcp_configuration, atomic_json
from astra_vla.rpc import RPCError, call
from astra_vla.teleop import Integrator, JOINT_KEYS, Policy, SPEED_PROFILES, keyset, pose, rotation_error
from astra_vla.teleop_detour import DetourPolicy
from astra_vla.workers.detour_control import DetourControl
from astra_vla.workers.native_probe import NativeProbe


_geometry_model = None
_configuration_cache = {}
_configuration_lock = threading.Lock()


class CalculationExpired(TimeoutError):
    pass


class SendWindowExpired(RuntimeError):
    pass


def camera_metadata():
    """Read existing collectors only; keep bulk JSON decoding out of CAN owner."""
    snapshot = call(RUNTIME / 'cameras.sock', 'snapshot', {'depth': False}, timeout=.4)
    names = ('available', 'stamp', 'age_s', 'serial', 'sequence', 'width', 'height')
    return {'cameras': {name: {key: frame[key] for key in names if key in frame}
                        for name, frame in snapshot['cameras'].items()}}


def initialize_teleop_geometry(model):
    global _geometry_model
    _geometry_model = model


def geometry_step(*args):
    # Canonical module owns the pool model even when robot runs as __main__.
    if _geometry_model is None:
        raise RuntimeError('teleop geometry worker not initialized')
    return _geometry_model.teleop_step(*args)


def geometry_recheck(*args):
    """Recheck the integer endpoint and full path without repeating IPOPT."""
    return _geometry_model.teleop_step(*args[:-1], validated_goal=args[-1])


def geometry_detour_plan(*args):
    from astra_vla.teleop_detour import plan_detour
    return plan_detour(_geometry_model, *args)


def geometry_detour_step(*args):
    from astra_vla.teleop_detour import detour_joint_step
    return detour_joint_step(_geometry_model, *args)


def geometry_recovery_plan(state, other_state, geometry, policy):
    from astra_vla.gravity_recovery import RecoveryPolicy, plan_recovery
    return plan_recovery(_geometry_model, state, other_state, geometry, RecoveryPolicy(**policy))


def geometry_recovery_rebind(state, other_state, geometry, plan):
    from astra_vla.gravity_recovery import rebind_recovery_path
    return rebind_recovery_path(_geometry_model, state, other_state, geometry, plan)


def geometry_recovery_replan(state, other_state, geometry, policy, previous, origin):
    """Replan settled encoders and retain the original feedback for evidence."""
    from astra_vla.gravity_recovery import (RecoveryPolicy, plan_recovery, rebind_recovery_path,
                                          validated_recovery_plan)
    recovery_policy = RecoveryPolicy(**policy)
    # Revalidate the already selected route first. Repeating the entire cold
    # candidate search left the enabled arm paused for several seconds, even
    # when its original route was still valid from the settled encoders.
    try:
        rebound = rebind_recovery_path(_geometry_model, state, other_state, geometry, previous)
        validated_recovery_plan(rebound, state, recovery_policy, origin)
        return dict(rebound, replan_strategy='existing_route_revalidated')
    except ValueError:
        pass
    bounded_geometry = dict(geometry, recovery_origin_joints_rad=origin)
    plan = plan_recovery(_geometry_model, state, other_state, bounded_geometry, recovery_policy)
    if not plan.get('required'):
        return plan
    return rebind_recovery_path(_geometry_model, state, other_state, geometry, plan)


def geometry_ready():
    if _geometry_model is not None and Policy.load().ik_solver == 'casadi_ipopt':
        _geometry_model.prepare_teleop_ik()
    return _geometry_model is not None and _geometry_model.frame >= 0


def geometry_probe_ready():
    from astra_vla.native_pose import warm_up
    warm_up()
    return _geometry_model is not None and _geometry_model.frame >= 0


def geometry_native_step(state, geometry, proposal, limits, policy, allow_orientation):
    from astra_vla.native_pose import adapt_target
    native_feedback(state, geometry)
    selected = Policy(**policy)
    return adapt_target(state, proposal, np.asarray(geometry['tcp']), limits, selected,
                        allow_orientation, budget_s=selected.geometry_timeout_s*.7)


def native_feedback(state, geometry):
    """Validate SDK feedback without querying URDF collision geometry."""
    if geometry.get('scope') != 'tcp_jog':
        raise ValueError('PIPER CRAFT native control requires robot-local TCP')
    rows = (state['joints_rad'], state['ee_pose']['position_m'], state['ee_pose']['rpy_deg'])
    if (any(np.asarray(row).shape != (size,) for row, size in zip(rows, (6, 3, 3))) or
            not all(np.isfinite(row).all() for row in rows) or
            not np.isfinite(state['gripper']['width_m'])):
        raise ValueError('invalid native joint/pose/gripper feedback')
    return True


def geometry_hold(state, geometry=None, other_state=None):
    """Check a local joint hold; no camera/workspace transform is involved."""
    _geometry_model.check_hold(state)
    rounded = dict(state, joints_rad=np.deg2rad(
        np.rint(np.rad2deg(state['joints_rad']) * 1000) / 1000).tolist())
    _geometry_model.check_hold(rounded)
    if geometry is not None and geometry.get('scope') == 'tcp_jog':
        for measured in (state, rounded):
            if _geometry_model.bounded_gripper_contacts(_geometry_model.configuration(measured),
                                                       geometry.get('unverified_gripper_max_m')):
                raise ValueError('unverified gripper envelope has URDF self collision')
    elif geometry is not None:
        raise ValueError('PIPER CRAFT hold requires robot-local geometry')
    return True


def configuration_token():
    def identity(path):
        try:
            stat = path.stat()
            return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        except FileNotFoundError:
            return (str(path), None)
    # Reuse parsed YAML while its bytes match. Always hash the actual file bytes:
    # rapid same-size edits can share filesystem timestamps and must still stop
    # a session. YAML parsing, rather than hashing, is the expensive operation.
    with _configuration_lock:
        devices = ROOT / 'config/devices.yaml'
        devices_bytes = devices.read_bytes()
        devices_key = (str(devices), devices_bytes)
        if _configuration_cache.get('devices') != devices_key:
            paths = [ROOT / 'config' / n for n in ('devices.yaml', 'tcp.json', 'teleop.json',
                                                 'teleop_speeds.json', 'gravity_recovery.json',
                                                 'teleop_detour.json')]
            paths.append(Path(settings()['urdf']))
        else:
            paths = _configuration_cache['paths']
        identities = tuple(identity(path) for path in paths)
        digest = hashlib.sha256()
        for path in paths:
            digest.update(str(path).encode())
            digest.update(devices_bytes if path == devices else
                          path.read_bytes() if path.is_file() else b'<missing>')
        if (identities != tuple(identity(path) for path in paths) or
                devices.read_bytes() != devices_bytes):
            raise RuntimeError('configuration changed while reading')
        token = digest.hexdigest()
        _configuration_cache.update(devices=devices_key, paths=paths, identities=identities, token=token)
        return token


def camera_fingerprint(cameras):
    value = {name: {key: frame.get(key) for key in ('serial','width','height')}
             for name, frame in cameras.items()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Trace:
    """Command evidence, explicitly separate from video and physical acceptance."""
    def __init__(self, session, arm, policy):
        self.session, self.arm, self.policy = session, arm, policy
        self.buffer = deque(maxlen=512)
        self.path = None
        self.lock = threading.Lock()
        self.pending = deque()
        self.changed = threading.Event()
        self.writer = None
        self.activated = self.closing = False
        self.error = None

    def check(self):
        if self.error:
            raise RuntimeError('teleop evidence writer: ' + self.error)

    def activate(self):
        with self.lock:
            self.check()
            if self.activated:
                return
            self.path = ROOT / 'runs' / ('teleop_' + time.strftime('%Y%m%d_%H%M%S') + '_' + self.session[:8])
            self.activated = True
            self.pending.extend(self.buffer)
            self.buffer.clear()
            self.writer = threading.Thread(target=self.write, name='teleop-evidence', daemon=True)
            self.writer.start()

    def write(self):
        try:
            self.path.mkdir(parents=True)
            atomic_json(self.path / 'manifest.json', {'kind': 'teleop_command_trace', 'session': self.session,
                'arm': self.arm, 'policy': asdict(self.policy), 'hardware_acceptance_complete': False,
                'physical_accuracy_verified': False, 'video_recorded': False,
                **getattr(self, 'context', {})})
            with (self.path / 'events.jsonl').open('a') as stream:
                while True:
                    self.changed.clear()
                    with self.lock:
                        rows = list(self.pending)
                        self.pending.clear()
                        closing = self.closing
                    for row in rows:
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    if closing:
                        return
                    self.changed.wait(.05)
        except Exception as exc:
            self.error = str(exc)

    def record(self, kind, **data):
        row = {'type': kind, 'stamp': time.time(), 'monotonic': time.monotonic(), **data}
        with self.lock:
            self.check()
            if self.activated:
                if self.closing or len(self.pending) >= 4096:
                    raise RuntimeError('teleop evidence queue closed/full')
                self.pending.append(row)
                self.changed.set()
            else:
                self.buffer.append(row)

    def close(self, result):
        with self.lock:
            if not self.activated:
                return
            self.closing = True
            self.changed.set()
        self.writer.join(timeout=5)
        if self.writer.is_alive():
            raise RuntimeError('teleop evidence writer did not finish')
        self.check()
        atomic_json(self.path / 'result.json', result)
        from astra_vla.operation_memory import record_session
        try:
            record_session(self.path, result, directory=ROOT/'docs/memory')
        except Exception as exc:
            atomic_json(self.path/'memory_error.json', {'error': str(exc)})


class TeleopController(NativeProbe, DetourControl):
    def __init__(self, robot, *, clock=time, camera_read=None, tool_read=tcp_configuration,
                 token_read=configuration_token, trace_factory=Trace, auto_start=True):
        self.robot, self.clock = robot, clock
        self.camera_pool = None
        self.camera_pool_lock = threading.Lock()
        self.camera_read = camera_read or self.read_camera_metadata
        self.tool_read, self.token_read = tool_read, token_read
        self.trace_factory, self.auto_start = trace_factory, auto_start
        self.mutex = threading.RLock()
        self.open_lock = threading.RLock()
        self.active = False
        self.unverified_grippers = {}
        self.owns_lock = False
        self.session = None
        self.reason = None
        self.phase = 'observing'
        self.done = threading.Event()
        self.input_changed = threading.Event()
        self.thread = None
        self.guard_thread = None
        self.camera_thread = None
        self.cameras = {}
        self.camera_error = None
        self.last_confirmation = None
        self.cycles = deque(maxlen=1000)
        self.trace = None
        self.arm = None
        self.gripper_commanded = False
        self.owner = 'desktop'
        self.native_pose_probe = False
        self.native_control = False
        self.execution_backend = 'regularized_move_j'
        self.recovery = None
        self.detour_policy = DetourPolicy.load()
        self.reset_detour()

    def token_value(self):
        token = self.token_read()
        pass
        return token


    def read_camera_metadata(self):
        with self.camera_pool_lock:
            if self.camera_pool is None:
                self.camera_pool = ProcessPoolExecutor(max_workers=1,
                    mp_context=multiprocessing.get_context('spawn'))
            job = self.camera_pool.submit(camera_metadata)
        try:
            return job.result(timeout=3)
        except (TimeoutError, FutureTimeoutError) as exc:
            raise RuntimeError('teleop camera metadata RPC timeout') from exc

    def shutdown_camera(self):
        with self.camera_pool_lock:
            if self.camera_pool is not None:
                self.camera_pool.shutdown(wait=False, cancel_futures=True)
                self.camera_pool = None

    def intent_state(self, state):
        feedback = getattr(self.robot, 'teleop_feedback', None)
        return feedback(state) if feedback else state

    def check_preparation(self, arm):
        prepare = getattr(self.robot, 'check_teleop_prepare', None)
        if prepare and self.owner == 'desktop':
            return prepare(arm, self.policy.follow_mode, arm_only=True)
        return prepare(arm, self.policy.follow_mode) if prepare else self.robot.check_prepare(arm)

    def check_gripper_quarantine(self, arm, state):
        reference = self.unverified_grippers.get(arm)
        if reference is None:
            return False
        if (state['gripper']['enabled'] or not np.isfinite(state['gripper']['width_m']) or
                abs(state['gripper']['width_m'] - reference) > .00025):
            raise RuntimeError(arm + ' unverified gripper changed; re-enter after inspection')
        return True


    def scene_acknowledged(self):
        """Retire an old entry rejection without creating a session or input."""
        with self.mutex:
            if not self.active and not self.owns_lock and self.phase == 'entry_blocked':
                self.phase, self.reason = 'observing', None

    def view(self):
        with self.mutex:
            costs = list(self.cycles)
            now = self.clock.monotonic()
            target_hz = sum(0 <= now-stamp <= 1 for stamp in getattr(self, 'target_stamps', ()))
            joint_jog = next(iter(getattr(self, 'keys', frozenset()) & JOINT_KEYS), None)
            return {'active': self.active, 'owner': self.owner if self.owns_lock else None,
                    'session': self.session, 'arm': self.arm, 'phase': self.phase,
                    'native_pose_probe': self.native_pose_probe,
                    'automatic_orientation': bool(self.native_control and not self.native_pose_probe),
                    'orientation_adjustment_deg': getattr(self, 'native_adjustment_deg', 0.),
                    'execution_backend': 'controller_move_p' if self.native_control else 'regularized_move_j',
                    'control_frame': 'joint' if joint_jog else 'tcp',
                    'joint_jog': joint_jog,
                    'vertical_frame': self.policy.vertical_frame if hasattr(self, 'policy') else None,
                    'ik_solver': ('joint_jog' if joint_jog else 'controller' if self.native_control else
                                  self.policy.ik_solver if hasattr(self, 'policy') else None),
                    'path_precheck': 'none' if self.native_control else 'full_joint_path',
                    'follow_mode': self.policy.follow_mode if hasattr(self, 'policy') else None,
                    'sdk_speed_percent_effective': True if hasattr(self, 'policy') else None,
                    'mode_feedback_verified': False,
                    'collision_scope': 'none' if self.native_control else 'robot_only',
                    'joint_position_limits': 'controller',
                    'environment_guard': False, 'other_arm_guard': False,
                    'blocker': self.reason, 'termination_reason': getattr(self, 'termination_reason', None),
                    'sequence': getattr(self, 'sequence', -1),
                    'input_keys': sorted(getattr(self, 'keys', ())),
                    'motion_speed_fraction': (self.integrator.speed_fraction
                                              if self.active and hasattr(self, 'integrator') else 0.),
                    'stop_confirmed': bool(self.last_confirmation and self.last_confirmation['confirmed']),
                    'control_hz': getattr(self, 'measured_hz', 0),
                    'target_hz': target_hz,
                    'cycle_p95_ms': float(np.percentile(costs, 95)) if costs else None,
                    'cycle_stages_ms': dict(getattr(self, 'cycle_stages', {})),
                    'cycle_overruns': getattr(self, 'cycle_overruns', 0),
                    'deadline_misses': getattr(self, 'deadline_misses', 0),
                    'policy': asdict(self.policy) if hasattr(self, 'policy') else None,
                    'gravity_recovery': self.recovery,
                    'detour': self.detour_status(),
                    'speed_profile': getattr(self, 'requested_speed_profile', 'normal'),
                    'speed_profile_active': self.policy.speed_profile if hasattr(self, 'policy') else 'normal',
                    'speed_profile_pending': getattr(self, 'pending_speed_profile', None),
                    'actual_tcp_speed_mm_s': getattr(self, 'actual_tcp_speed_m_s', 0.)*1000,
                    'release_extra_mm': getattr(self, 'release_extra_observed_m', 0.)*1000,
                    'gripper_target_mm': (self.gripper_target_m*1000 if
                                          getattr(self, 'gripper_target_m', None) is not None else None),
                    'gripper_effort_limit': self.policy.gripper_effort if hasattr(self, 'policy') else None,
                    'gripper_control_available': self.arm not in self.unverified_grippers,
                    'gripper_notice': ('夹爪零位未确认，Q/E已禁用；手臂可控' if
                                       self.arm in self.unverified_grippers else None),
                    'trace_path': str(self.trace.path) if self.trace and self.trace.path else None}

    def fresh_cameras(self, cameras):
        now = self.clock.time()
        if set(cameras) != {'top', 'left_wrist', 'right_wrist'}:
            raise RuntimeError('three real cameras required')
        for name, frame in cameras.items():
            stamp = frame.get('stamp', 0)
            age = frame.get('age_s', float('inf'))
            if (not frame.get('available') or not np.isfinite([stamp, age]).all() or
                    not 0 <= now - stamp <= .6 or not 0 <= age <= .6):
                raise RuntimeError(name + ' RGB missing/stale')
        # RGB views are independent displays, not inputs to TCP geometry.
        # Each must remain fresh, but neither camera-camera nor camera-robot
        # timestamp alignment is needed for robot-local keyboard control.
        if camera_fingerprint(cameras) != self.fingerprint:
            raise RuntimeError('camera configuration changed')

    def prepare_local(self, arm):
        state = self.check_preparation(arm)
        self.robot.check_motion_state(state)
        hold_args = (((state, self.geometry)))
        try:
            validated = self.robot.geometry_pool.submit(geometry_hold, *hold_args).result(timeout=15)
        except ValueError as exc:
            folded_contact = (self.owner == 'desktop' and str(exc) in (
                'measured pose has URDF self collision', 'unverified gripper envelope has URDF self collision'))
            if not folded_contact:
                raise
            if self.owner == 'desktop' and not self.preparation_cancel_supported:
                raise RuntimeError('automatic gravity recovery requires updated desktop cancellation support')
            from astra_vla.gravity_recovery import recover
            state = recover(self, arm, state, geometry_recovery_plan)
            hold_args = (((state, self.geometry)))
            validated = self.robot.geometry_pool.submit(geometry_hold, *hold_args).result(timeout=15)
        if validated is not True:
            raise RuntimeError('local joint hold validation failed')
        latest = self.robot.healthy(arm, False)
        self.robot.check_motion_unchanged(arm, state, latest, 'during teleop hold preparation')
        pass
        if (np.max(np.abs(np.asarray(latest['joints_rad']) - state['joints_rad'])) > np.deg2rad(.05) or
                abs(latest['gripper']['width_m'] - state['gripper']['width_m']) > .00025):
            raise RuntimeError('teleop hold start moved during validation')
        if self.token != self.token_value():
            raise RuntimeError('TCP/device configuration changed during preparation')
        self.check_gripper_quarantine(arm, latest)
        if (latest['controller']['ctrl_mode'] == 1 and all(latest['controller']['motors_enabled'])):
            return latest  # An enabled CAN hold needs no actuator write or IK mode switch.
        hold = tuple(round(np.rad2deg(j) * 1000) for j in latest['joints_rad'])
        p = self.robot.arms[arm]
        self.robot.actuator_commands_sent = True
        commands = [('MotionCtrl_1', (0, 4, 0)), ('JointCtrl', hold),
                             ('MotionCtrl_2', (1, 1, min(5, int(self.policy.speed_percent)), 0)),
                             ('JointCtrl', hold), ('MotionCtrl_1', (0, 2, 0)), ('EnableArm', (7,))]
        if arm not in self.unverified_grippers:
            commands.append(('GripperCtrl', (max(0, round(latest['gripper']['width_m'] * 1e6)),
                                             round(self.policy.gripper_effort * 1000), 1, 0)))
        for method, args in commands:
            self.check_gripper_quarantine(arm, self.robot.healthy(arm, False))
            self.robot.command(p, method, *args)
        deadline = self.clock.monotonic() + 3
        while self.clock.monotonic() < deadline:
            current = self.robot.healthy(arm, False)
            self.robot.check_motion_state(current)
            if (np.max(np.abs(np.asarray(current['joints_rad']) - latest['joints_rad'])) > np.deg2rad(.1) or
                    np.linalg.norm(pose(current)[:3, 3] - pose(latest)[:3, 3]) > .0005 or
                    rotation_error(pose(current), pose(latest)) > np.deg2rad(.2)):
                raise RuntimeError('teleop measured hold moved during enable')
            control = current['controller']
            if control['ctrl_mode'] == 1 and control['move_mode'] == 1 and all(control['motors_enabled']):
                return current
            self.clock.sleep(.05)
        raise RuntimeError('teleop CAN hold enable not confirmed')

    def open(self, args):
        return self._open(args)

    def _open(self, args):
        arm = args.get('arm', 'left')
        if arm not in ('left', 'right'):
            raise ValueError('select left/right arm')
        with self.open_lock:
            if self.owns_lock or not self.robot.lock.acquire(False):
                raise RuntimeError('another arm action/control session owns the robot')
            self.robot.actuator_commands_sent = False
            try:
                self.configure_native_probe(args, arm)
                open_epoch = getattr(self.robot, 'stop_epoch', 0)
                self.recovery = None
                self.termination_reason = None
                self.unverified_grippers = {}
                self.owner = 'desktop'
                self.preparation_cancel_supported = args.get('preparation_cancel_supported') is True
                self.base_policy = Policy.load()
                if self.base_policy.speed_profile != 'normal':
                    raise ValueError('base configuration must use normal speed profile')
                profile = (args.get('speed_profile', 'normal'))
                self.policy = self.base_policy.for_speed(profile)
                if self.native_control:
                    self.native_probe_policy()
                self.requested_speed_profile, self.pending_speed_profile = profile, None
                pass
                token = self.token_value()
                self.tool = self.tool_read()
                self.token = token
                other = 'right' if arm == 'left' else 'left'
                tcp = np.asarray(self.tool['arms'][arm]['T_ee_tcp'], dtype=float)
                if (tcp.shape != (4, 4) or not np.isfinite(tcp).all() or
                    not np.allclose(tcp[3], [0, 0, 0, 1]) or
                    not np.allclose(tcp[:3, :3].T @ tcp[:3, :3], np.eye(3), atol=1e-6) or
                    not np.isclose(np.linalg.det(tcp[:3, :3]), 1)):
                    raise ValueError('configured TCP must be a finite rigid transform')
                g = ({'scope': 'tcp_jog', 'tcp': tcp.tolist()})
                if not np.allclose(g['tcp'], tcp, atol=1e-12, rtol=0):
                    raise ValueError('TCP does not match current configured TCP')
                self.geometry = g
                pass
                cameras = self.camera_read()['cameras']
                self.fingerprint = camera_fingerprint(cameras)
                self.fresh_cameras(cameras)
                self.check_preparation(arm)
                for name in (arm, other):
                    measured = self.robot.healthy(name, False)
                    width = measured['gripper']['width_m']
                    maximum = self.robot.config['limits']['maintenance_gripper_max_m'][name]
                    if not -.002 <= width <= maximum:
                        if not np.isfinite(width) or not -maximum <= width < -.002 or measured['gripper']['enabled']:
                            raise RuntimeError(name + ' gripper encoder outside configured bounds')
                        self.unverified_grippers[name] = width
                if arm in self.unverified_grippers:
                    g['unverified_gripper_max_m'] = self.robot.config['limits']['maintenance_gripper_max_m'][arm]
                # Native control warms target feasibility without constructing local IK.
                ready = geometry_probe_ready if self.native_control else geometry_ready
                if self.robot.geometry_pool.submit(ready).result(timeout=15) is not True:
                    raise RuntimeError('teleop geometry worker not ready')
                confirmation = self.robot.stationary_feedback(require_fault_free=True,
                    joint_tolerance_deg=(.15))
                if not confirmation['confirmed']:
                    raise RuntimeError('fresh stationary feedback required')
                if self.native_control:
                    state = self.robot.healthy(arm, False)
                    self.robot.check_motion_state(state)
                    control = state['controller']
                    if (control['move_mode'] not in (0, 1) or control['ctrl_mode'] != 1 or
                            not all(control['motors_enabled'])):
                        raise RuntimeError('原生方案需要已使能的静止位置控制；请先用本地方案准备后退出控制')
                    if not self.native_pose_probe:
                        from astra_vla.native_pose import controller_bounds
                        if state['controller'].get('firmware') != 'S-V1.5-8':
                            raise RuntimeError('当前固件尚无已验证的原生自动调姿模型')
                        self.native_limits = self.robot.read_controller_limits(arm)
                        controller_bounds(self.native_limits)
                        self.native_adjustment_deg = 0.
                    if self.robot.geometry_pool.submit(native_feedback, state, g).result(timeout=15) is not True:
                        raise RuntimeError('native feedback validation rejected')
                else:
                    state = self.prepare_local(arm)
                if token != self.token_value():
                    raise RuntimeError('configuration changed during preparation')
                self.cameras = self.camera_read()['cameras']
                self.fresh_cameras(self.cameras)
                state = self.robot.healthy(arm)
                self.robot.check_motion_state(state)
                pass
                with self.mutex:
                    if self.robot.stopped.is_set() or getattr(self.robot, 'stop_epoch', 0) != open_epoch:
                        raise RuntimeError('stopped during control preparation')
                    self.arm, self.other = arm, other
                    self.token = token
                    self.integrator = Integrator(self.intent_state(state), g['tcp'], policy=self.policy)
                    if self.native_control:
                        self.begin_native_probe(state)
                    self.maximum_width = self.robot.config['limits']['maintenance_gripper_max_m'][arm]
                    self.session = uuid.uuid4().hex
                    self.keys, self.accepted_keys = frozenset(), frozenset()
                    self.detour_policy = DetourPolicy.load()
                    self.reset_detour()
                    self.sequence, self.generation = -1, 0
                    self.blocked_generation = None
                    self.waiting_since = None
                    self.last_input = self.clock.monotonic()
                    self.velocity = [0.] * 6
                    self.last_goal_joints = None
                    self.paused, self.gripper_commanded = True, False
                    self.gripper_target_m = None
                    self.gripper_enable_request_at = None
                    self.release_origin, self.release_extra_observed_m = None, 0.
                    self.speed_samples, self.actual_tcp_speed_m_s = deque(maxlen=32), 0.
                    self.target_stamps = deque(maxlen=100)
                    self.pause_reference, self.quiet_since = state, self.clock.monotonic()
                    self.pause_stamps = None
                    self.phase, self.reason, self.camera_error = 'ready', None, None
                    self.last_confirmation = confirmation
                    self.measured_hz, self.cycles = 0., deque(maxlen=1000)
                    self.cycle_stages, self.cycle_overruns = {}, 0
                    self.deadline_misses = 0
                    self.trace = self.trace_factory(self.session, arm, self.policy)
                    self.trace.context = {'tool': self.tool, 'control_frame': ('tcp'),
                        'owner': self.owner, 'execution_backend': 'controller_move_p' if self.native_control else 'regularized_move_j',
                        'native_pose_probe': self.native_pose_probe,
                        'automatic_orientation': bool(self.native_control and not self.native_pose_probe),
                        'controller_limits': ({key: self.native_limits[key] for key in ('source', 'complete', 'limits')}
                                              if self.native_control and not self.native_pose_probe else None),
                        'path_precheck': 'none' if self.native_control else 'full_joint_path',
                        'collision_scope': 'none' if self.native_control else 'robot_only',
                        'joint_position_limits': 'controller',
                        'environment_guard': False, 'other_arm_guard': False,
                        'configuration_token': self.token}
                    pass
                    self.done = threading.Event()
                    self.active = True
                    self.owns_lock = True
                if self.auto_start:
                    self.thread = ((threading.Thread(target=self.run, args=(self.done,),
                                                   name='teleop-control', daemon=True)))
                    self.guard_thread = threading.Thread(target=self.guard, args=(self.done,),
                                                         name='teleop-watchdog', daemon=True)
                    self.camera_thread = threading.Thread(target=self.capture, args=(self.done,),
                                                          name='teleop-camera-guard', daemon=True)
                    for thread in (self.thread, self.guard_thread, self.camera_thread):
                        if thread is not threading.current_thread():
                            thread.start()
                return self.view()
            except Exception as exc:
                # prepare can have enabled holding. Never disable/reset on an error.
                if self.recovery and self.recovery['phase'] not in ('failed', 'completed'):
                    self.recovery.update(phase='blocked', reason=str(exc))
                if getattr(self.robot, 'actuator_commands_sent', False):
                    try:
                        stopped = self.robot.stop('teleop preparation failed', manual_tcp=True)
                        if not stopped.get('stop_confirmed'):
                            raise RuntimeError('preparation failed and stopping is unconfirmed')
                    except Exception:
                        self.owns_lock = True
                        self.phase = 'stop_unconfirmed'
                        self.reason = 'preparation failed; explicit stop confirmation required'
                        raise
                if self.robot.lock.locked():
                    self.robot.lock.release()
                self.owns_lock = False
                with self.mutex:
                    self.phase, self.reason = 'entry_blocked', str(exc)
                    self.arm = arm
                raise

    def input(self, args):
        pass
        keys = keyset(args.get('keys'))
        if self.arm in self.unverified_grippers:
            keys = keys - {'q', 'e'}
        sequence = args.get('sequence')
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError('input sequence must be a nonnegative integer')
        ui_age = args.get('ui_age_s', 0)
        if not isinstance(ui_age, (int, float)) or not np.isfinite(ui_age) or not 0 <= ui_age <= .1:
            raise ValueError('desktop event pump is stale')
        if args.get('focused', True) is not True:
            self.fail('desktop focus lost')
            raise RuntimeError('desktop focus lost; control session closed')
        with self.mutex:
            if not self.active or args.get('session') != self.session:
                matching = args.get('session') == self.session
                reason = (getattr(self, 'termination_reason', None) or self.reason) if matching else None
                raise RPCError('teleop session is not active' + (': ' + reason if reason else ''),
                               stage='teleop_session', diagnostics={
                                   'session': args.get('session'), 'reason': reason,
                                   'phase': self.phase if matching else 'expired',
                                   'stop_confirmed': bool(matching and self.last_confirmation and
                                                          self.last_confirmation['confirmed'])})
            if sequence <= self.sequence:
                return {'accepted': False, 'sequence': self.sequence}
            if args.get('execution_backend', self.execution_backend) != self.execution_backend:
                raise ValueError('exit control and confirm stop before changing execution backend')
            if self.native_control and keys & JOINT_KEYS:
                raise ValueError('关节点动请先切换到本地方案')
            profile = args.get('speed_profile', self.requested_speed_profile)
            if self.native_pose_probe and (profile != 'slow' or keys & {'q', 'e'}):
                raise ValueError('native probe permits slow arm-only motion')
            if profile not in SPEED_PROFILES:
                raise ValueError('unknown teleop speed profile')
            # Validate the complete effective policy before accepting a switch.
            changing_profile = profile != self.requested_speed_profile
            if changing_profile:
                self.base_policy.for_speed(profile)
                self.requested_speed_profile = self.pending_speed_profile = profile
                keys = frozenset()  # No held key can accelerate on a gear change.
            previous = self.keys
            route_changed = bool(self.detour_context is not None and (keys != previous or changing_profile))
            self.sequence, self.last_input = sequence, self.clock.monotonic()
            if keys != previous or changing_profile:
                self.generation += 1
                self.waiting_since = None
                self.reset_detour()
            self.keys = keys
            generation = self.generation
            self.trace.record('input', sequence=sequence, keys=sorted(keys), generation=generation,
                              speed_profile=self.requested_speed_profile)
        pairs = (('w', 's'), ('a', 'd'), ('space', 'lshift'), ('i', 'k'),
                 ('j', 'l'), ('o', 'p'), ('q', 'e'))
        cancelled = any(((positive in previous) != (negative in previous)) and
                        ((positive in keys) == (negative in keys)) for positive, negative in pairs)
        if previous - keys or cancelled or changing_profile or route_changed:
            self.pause(generation)
        if keys and not previous:
            self.input_changed.set()
        return {'accepted': True, 'sequence': sequence}

    def pause(self, generation):
        with self.robot.tx_lock, self.mutex:
            if not self.active or generation != self.generation or self.robot.stopped.is_set():
                return
            self.reset_detour()
            p = self.robot.arms[self.arm]
            p.MotionCtrl_1(0, 1, 0)
            self.freeze_gripper()
            state = self.robot.state(self.arm)
            self.paused = True
            self.velocity = [0.] * 6
            self.last_goal_joints = None
            angular = any((a in self.keys) != (b in self.keys) for a, b in
                          (('i', 'k'), ('j', 'l'), ('o', 'p')))
            translating = any((a in self.keys) != (b in self.keys) for a, b in
                              (('w', 's'), ('a', 'd'), ('space', 'lshift')))
            anchor = (self.integrator.target[:3, 3].copy()
                      if self.integrator.rotating and angular and not translating else None)
            # Cancellation must not depend on FK agreement while the arm is
            # still decelerating. SDK pose/joints arrive as separate CAN frames.
            # This reset only discards intent; revalidate FK after quiet feedback
            # and before any new target is calculated.
            self.integrator.reset(state)
            if anchor is not None:
                self.integrator.target[:3, 3] = anchor
                self.integrator.rotating = True
            self.pause_reference, self.quiet_since = state, self.clock.monotonic()
            self.release_origin = (pose(state) @ self.integrator.tcp)[:3, 3].copy()
            self.release_extra_observed_m = 0.
            self.pause_stamps = None
            self.phase, self.last_confirmation = 'pausing', None
            self.trace.record('pause', generation=generation, state=state)

    def freeze_gripper(self):
        """Called under TX lock. Arm planner pause does not stop the finger motor."""
        if not self.gripper_commanded:
            return
        state = self.robot.state(self.arm)
        width = state['gripper']['width_m']
        if (state['age_s'] > .3 or not np.isfinite(width) or width < -.002 or
                width > self.maximum_width or state['gripper']['faults']):
            raise RuntimeError('cannot confirm fresh gripper hold on stop')
        self.robot.arms[self.arm].GripperCtrl(max(0, round(width * 1e6)),
                                            round(self.policy.gripper_effort * 1000), 1, 0)
        self.gripper_target_m = max(0., round(width*1e6)/1e6)
        try:
            self.trace.record('gripper_hold', width_m=self.gripper_target_m, measured=state['gripper'])
        except Exception:
            pass  # Holding the fingers must not depend on diagnostic writes.
        self.gripper_commanded = False

    def invalidate(self, reason):
        with self.mutex:
            self.reset_detour()
            first = getattr(self, 'termination_reason', None) is None
            if first:
                self.termination_reason = reason
            self.active = False
            self.generation = getattr(self, 'generation', 0) + 1
            self.keys = frozenset()
            self.phase, self.reason = 'stopping', self.termination_reason
            self.done.set()
            self.input_changed.set()
            if first and self.trace:
                try:
                    self.trace.record('session_ended', reason=self.termination_reason)
                except Exception:
                    pass  # Evidence failure must never prevent stopping.

    def fail(self, reason):
        with self.mutex:
            if not self.active:
                return
            self.invalidate(reason)
        self.robot.stop(reason)

    def complete_stop(self, confirmation):
        with self.mutex:
            self.last_confirmation = confirmation
            self.phase = 'stopped' if confirmation['confirmed'] else 'stop_unconfirmed'
        thread = self.thread
        for child in (thread, self.guard_thread, self.camera_thread):
            if child and child is not threading.current_thread():
                child.join(timeout=.5)
        if not thread or not thread.is_alive() or thread is threading.current_thread():
            if confirmation['confirmed']:
                with self.open_lock:
                    if self.owns_lock:
                        self.robot.lock.release()
                        self.owns_lock = False
            if self.trace:
                self.trace.close({'stop_confirmed': bool(confirmation['confirmed']), 'reason': self.reason,
                                  'hardware_acceptance_complete': False})

    def close(self, args):
        if not self.owns_lock:
            return self.view()
        if args.get('session') != self.session:
            raise RuntimeError('wrong teleop session')
        self.robot.stop(getattr(self, 'termination_reason', None) or args.get('reason', 'desktop control closed'),
                        **({'manual_tcp': True} if args.get('manual_tcp') is True else {}))
        return self.view()

    def check_watchdog(self):
        if not self.active:
            return
        if self.trace and hasattr(self.trace, 'check'):
            self.trace.check()
        if self.clock.monotonic() - self.last_input > self.policy.input_timeout_s:
            raise RuntimeError(self.owner + ' heartbeat missing/stale')
        if self.camera_error:
            raise RuntimeError(self.camera_error)
        self.fresh_cameras(self.cameras)
        for arm in (self.arm, self.other):
            measured = self.robot.healthy(arm, arm == self.arm)
            self.robot.check_motion_state(measured)
            if self.native_control and arm == self.arm:
                self.native_probe_bounds(measured)
            maximum = self.robot.config['limits']['maintenance_gripper_max_m'][arm]
            quarantined = self.check_gripper_quarantine(arm, measured)
            if not quarantined and not -.002 <= measured['gripper']['width_m'] <= maximum:
                raise RuntimeError(arm + ' gripper encoder outside configured bounds')

    def capture(self, done):
        while not done.is_set():
            try:
                cameras = self.camera_read()['cameras']
                self.fresh_cameras(cameras)
                token = self.token_value()
                if token != self.token:
                    raise RuntimeError('workspace/TCP/configuration changed')
                with self.mutex:
                    if not done.is_set():
                        self.cameras, self.camera_error = cameras, None
            except Exception as exc:
                with self.mutex:
                    if not done.is_set():
                        self.camera_error = str(exc)
            done.wait(.1)

    def guard(self, done):
        while not done.wait(.01):
            try:
                self.check_watchdog()
            except Exception as exc:
                self.fail('teleop guard: ' + str(exc))
                return

    def quiet_feedback(self, state):
        if self.release_origin is not None:
            point = (pose(state) @ self.integrator.tcp)[:3, 3]
            self.release_extra_observed_m = max(self.release_extra_observed_m,
                                               float(np.linalg.norm(point-self.release_origin)))
        reference = self.pause_reference
        stamps = state.get('feedback_stamps', {})
        pair = (stamps.get('pose', 0), stamps.get('joint', 0))
        advancing = self.pause_stamps is None or all(a > b for a, b in zip(pair, self.pause_stamps))
        self.pause_stamps = pair
        quiet = (advancing and np.max(np.abs(np.asarray(state['joints_rad']) - reference['joints_rad'])) <
                 np.deg2rad(.1) and np.linalg.norm(pose(state)[:3, 3] - pose(reference)[:3, 3]) < .0005)
        if not quiet:
            self.pause_reference, self.quiet_since = state, self.clock.monotonic()
        grip_quiet = abs(state['gripper']['width_m'] - reference['gripper']['width_m']) < .0005
        if not grip_quiet:
            self.pause_reference, self.quiet_since = state, self.clock.monotonic()
        window = (.5)
        return quiet and grip_quiet and self.clock.monotonic() - self.quiet_since >= window

    def step(self):
        with self.mutex:
            generation = getattr(self, 'generation', 0)
        try:
            self.calculate_step()
        except (CalculationExpired, SendWindowExpired) as exc:
            with self.mutex:
                if not self.active or generation != self.generation or self.robot.stopped.is_set():
                    return
            if isinstance(exc, CalculationExpired) or not getattr(self, 'cycle_writes', 0):
                # No SDK write was reached. Drop this numerical result and keep
                # only the last validated, bounded target. Recompute from fresh
                # feedback; elapsed time never enlarges the next increment.
                with self.mutex:
                    self.deadline_misses += 1
                    self.trace.record('deadline_drop', generation=generation, reason=str(exc),
                                      consecutive=self.deadline_misses)
                    if self.deadline_misses >= 3:
                        raise RuntimeError('three consecutive control computation deadline misses') from exc
                    self.reason = '本帧计算逾时，已取消候选，正在从新反馈重算'
                return
            # An expired frame never becomes a deferred target. Replace it with
            # a fresh measured hold, confirm quiet feedback, then recompute from
            # the held pose. Persistent misses become a latched fault.
            self.pause(generation)
            with self.mutex:
                if not self.active or generation != self.generation or self.robot.stopped.is_set():
                    return
                self.deadline_misses += 1
                self.trace.record('deadline_hold', generation=generation, reason=str(exc),
                                  consecutive=self.deadline_misses)
                if self.deadline_misses >= 3:
                    raise RuntimeError('three consecutive control deadline misses') from exc
                self.reason = '计算逾时，已保位，等待新验证目标'

    def accept_optimized_target(self, proposal, result, *, update_gain=True):
        if self.policy.ik_solver != 'casadi_ipopt':
            return
        details = result.get('ik', {})
        expected_solver = 'joint_jog' if proposal.get('joint_jog') else 'casadi_ipopt'
        if details.get('solver') != expected_solver:
            raise RuntimeError('requested regularized IK backend was not used')
        target = np.asarray(result.get('tcp_target'), dtype=float)
        progress = details.get('progress_fraction')
        if (target.shape != (4, 4) or not np.isfinite(target).all() or
                not isinstance(progress, (float, int)) or not 0 <= progress <= 1):
            raise RuntimeError('invalid regularized IK selected target')
        proposal['tcp_target'] = target
        proposal['ee_target'] = target @ self.integrator.inverse_tcp
        if update_gain:
            proposal['speed_fraction'] *= progress

    def calculate_step(self):
        started = self.clock.monotonic()
        self.cycle_writes = 0
        checkpoint = started
        self.cycle_stages = {}
        def mark(name):
            nonlocal checkpoint
            now = self.clock.monotonic()
            self.cycle_stages[name] = (now - checkpoint) * 1000
            checkpoint = now
        self.check_watchdog()
        mark('watchdog')
        with self.mutex:
            if not self.active:
                return
            keys, generation = self.keys, self.generation
        before = {a: self.robot.healthy(a, a == self.arm) for a in (self.arm, self.other)}
        state = before[self.arm]
        for measured in before.values():
            self.robot.check_motion_state(measured)
        self.trace.record('feedback', state=before)
        stamp = state.get('stamp', self.clock.time())
        point = (pose(state) @ self.integrator.tcp)[:3, 3]
        if not self.speed_samples or stamp > self.speed_samples[-1][0]:
            self.speed_samples.append((stamp, point))
            previous = next((s for s in reversed(self.speed_samples) if stamp-s[0] >= .2),
                            self.speed_samples[0])
            dt_speed = stamp-previous[0]
            if dt_speed > 0:
                self.actual_tcp_speed_m_s = float(np.linalg.norm(point-previous[1])/dt_speed)
        mark('feedback')
        if self.paused and self.phase == 'pausing':
            if not self.quiet_feedback(state):
                return
            with self.mutex:
                if not self.active or generation != self.generation:
                    return
                anchor = self.integrator.target[:3, 3].copy() if self.integrator.rotating else None
                self.integrator.reset(self.intent_state(state))
                if self.pending_speed_profile is not None:
                    self.policy = self.base_policy.for_speed(self.pending_speed_profile)
                    if self.native_control:
                        self.native_probe_policy()
                    self.pending_speed_profile = None
                    self.integrator = Integrator(self.intent_state(state), self.geometry['tcp'], policy=self.policy)
                    self.trace.record('speed_profile', policy=asdict(self.policy))
                    anchor = None
                if anchor is not None:
                    self.integrator.target[:3, 3] = anchor
                    self.integrator.rotating = True
                self.phase = 'ready'
                self.last_confirmation = {'confirmed': True}
        if not keys:
            return
        if self.blocked_generation == generation:
            return
        self.reason = None
        if self.token_value() != self.token:
            raise RuntimeError('workspace/TCP/configuration changed')
        mark('configuration')
        expected_mode = 0 if self.native_control else 1
        if not self.paused and state['controller']['move_mode'] != expected_mode:
            raise RuntimeError('teleop execution control mode changed')
        if self.detour_context is not None:
            context = self.detour_context
            handled = self.service_detour(keys, generation, before, started)
            if handled is not False or context.get('state') == 'executing':
                return
        intent_state = self.intent_state(state)
        proposal = self.integrator.propose(intent_state, keys, 1 / self.policy.control_hz,
                                           self.maximum_width, self.accepted_keys)
        mark('intent')
        if proposal is None:
            self.trace.record('waiting_feedback', state=before)
            if self.waiting_since is None:
                self.waiting_since = self.clock.monotonic()
            elif self.clock.monotonic() - self.waiting_since > 1:
                self.pause(generation)
                self.reason = 'target tracking stalled; release keys before retry'
                self.blocked_generation = generation
            return
        if not proposal['arm_motion'] and not proposal['gripper_motion']:
            if not self.paused:
                self.pause(generation)
            return
        if self.native_control:
            self.send_native_probe(proposal, before, keys, generation, started, mark)
            return
        job = None
        step_geometry = dict(self.geometry, rotation_only=proposal['rotating'])
        try:
            # Reduce this new increment if acceleration or IK feasibility
            # constraints reject it, under one shared deadline. Joint speed
            # has no local cap; all remaining bounds still apply.
            deadline = self.clock.monotonic() + self.policy.geometry_timeout_s
            motion_scale = 1.
            for attempt in range(5):
                if motion_scale != 1.:
                    proposal = self.integrator.propose(intent_state, keys, 1 / self.policy.control_hz,
                        self.maximum_width, self.accepted_keys, motion_scale=motion_scale)
                if proposal.get('joint_jog'):
                    _, axis, sign = proposal['joint_jog'].split('_')
                    direction = [0.] * 6
                    direction[int(axis)-1] = 1. if sign == 'plus' else -1.
                    step_geometry.update(joint_jog_direction=direction,
                                         joint_jog_gain=proposal['speed_fraction'])
                width_sdk = round(proposal['width_m'] * 1e6)
                proposal['width_m'] = width_sdk / 1e6
                remaining = deadline - self.clock.monotonic()
                if remaining <= 0:
                    raise CalculationExpired('teleop geometry deadline exceeded')
                job = self.robot.geometry_pool.submit(geometry_step, state, before[self.other],
                    proposal['ee_target'].tolist(), step_geometry, proposal['width_m'],
                    asdict(self.policy), self.velocity, self.last_goal_joints)
                try:
                    result = job.result(timeout=remaining)
                    break
                except (TimeoutError, FutureTimeoutError) as exc:
                    raise CalculationExpired('teleop geometry deadline exceeded') from exc
                except ValueError as exc:
                    if str(exc) not in ('teleop IK local constraints infeasible',
                                        'teleop joint acceleration exceeds bound') or attempt == 4:
                        raise
                    self.trace.record('rate_limited', generation=generation,
                                      motion_scale=motion_scale, reason=str(exc))
                    hint = getattr(exc, 'suggested_scale', None)
                    motion_scale *= hint if hint is not None and 0 < hint < 1 else .5
        except ValueError as exc:
            if job is not None:
                job.cancel()
            with self.mutex:
                if not self.active or generation != self.generation:
                    return
            if (str(exc) in ('teleop IK local constraints infeasible', 'teleop IK iteration budget exhausted')
                    and self.consider_detour(keys, generation,
                                             {'detour_recommended': True, 'ik': {'progress_fraction': 0.}})):
                return
            if self.detour_context is not None and str(exc) in (
                    'teleop IK local constraints infeasible', 'teleop joint acceleration exceeds bound',
                    'teleop joint tracking envelope exhausted', 'teleop IK iteration budget exhausted',
                    'teleop TCP target exceeds tracking lead'):
                self.phase = 'detour_planning'
                self.reason = '后台规划绕行中，保持最后验证目标；松键取消'
                return
            if (not self.paused and str(exc) in ('teleop IK local constraints infeasible',
                                                'teleop joint acceleration exceeds bound',
                                                'teleop joint tracking envelope exhausted',
                                                'teleop IK iteration budget exhausted',
                                                'teleop TCP target exceeds tracking lead')):
                # A slow actuator can exhaust the per-joint lead budget even
                # though adjacent command increments are bounded. Keep only the
                # previous validated goal and wait for feedback; a rejected
                # increment is never committed, sent or queued for later replay.
                self.trace.record('waiting_feedback', state=before, reason=str(exc))
                self.phase = 'waiting_feedback'
                if self.waiting_since is None:
                    self.waiting_since = self.clock.monotonic()
                elif self.clock.monotonic() - self.waiting_since > 1:
                    self.pause(generation)
                    self.reason = 'target tracking stalled; release keys before retry'
                    self.blocked_generation = generation
                return
            self.pause(generation)
            with self.mutex:
                if generation == self.generation:
                    self.reason = str(exc)
                    self.blocked_generation = generation
            self.trace.record('rejected', generation=generation, reason=str(exc), state=before)
            return
        except Exception:
            if job is not None:
                job.cancel()
            raise
        if not result.get('reachable'):
            raise RuntimeError('teleop geometry did not validate target')
        if self.consider_detour(keys, generation, result):
            return
        if (self.policy.ik_solver == 'casadi_ipopt' and proposal['arm_motion'] and
                not proposal['gripper_motion'] and result.get('ik', {}).get('progress_fraction', 1) <= 1e-4):
            if self.detour_context is not None:
                self.phase = 'detour_planning'
                self.reason = '后台规划绕行中，保持最后验证目标；松键取消'
                return
            self.trace.record('waiting_feedback', reason='regularized IK has no admissible progress',
                              state=before, ik=result.get('ik'))
            self.phase = 'waiting_feedback'
            if self.waiting_since is None:
                self.waiting_since = self.clock.monotonic()
            elif self.clock.monotonic() - self.waiting_since > 1:
                self.pause(generation)
                self.reason = '当前姿态无法沿该方向连续移动；松键后调整方向或起姿'
                self.blocked_generation = generation
            return
        mark('geometry')
        target = result['joint_sdk']
        if len(target) != 6 or any(isinstance(v, bool) or not isinstance(v, int) for v in target):
            raise RuntimeError('invalid quantized joint target')
        # Freeze the accepted increment before fresh-state revalidation. That
        # revalidation must not go back to a larger, unaccepted requested goal.
        self.accept_optimized_target(proposal, result)
        with self.mutex:
            if not self.active or generation != self.generation:
                return
        # Keep the accepted endpoint while rechecking its path from fresh
        # feedback. Manual readback movement does not cancel the target; the
        # original deadline and every path/gripper check still apply.
        inline_recheck = (getattr(self.robot, 'recheck_teleop_target', None)
                          if self.geometry.get('scope') == 'tcp_jog' and self.policy.ik_solver == 'casadi_ipopt'
                          else None)
        maximum_rechecks = 3 if inline_recheck else 1
        manual_stream = (self.owner == 'desktop' and self.geometry.get('scope') == 'tcp_jog' and
                         self.policy.follow_mode == 'position_velocity' and self.policy.ik_solver == 'casadi_ipopt')
        attempt = 0
        while True:
            with self.mutex:
                if not self.active or generation != self.generation:
                    return
            refreshed = {a: self.robot.healthy(a, a == self.arm) for a in before}
            moved = False
            for arm, old in before.items():
                latest = refreshed[arm]
                joint_change = np.max(np.abs(np.asarray(latest['joints_rad']) - old['joints_rad']))
                # Movement beyond the sampling tolerance requires a fresh path,
                # independent of speed; it is not an overspeed rejection.
                change_limit = np.deg2rad(.5)
                pose_changed = (np.linalg.norm(pose(old)[:3, 3]-pose(latest)[:3, 3]) > .003 or
                                rotation_error(pose(old), pose(latest)) > np.deg2rad(3))
                width_changed = abs(old['gripper']['width_m']-latest['gripper']['width_m']) > .002
                # Manual motion readbacks are validation inputs, not an
                # immutable start pose. Keep every mode/enable/fault check;
                # changed geometry requires a fresh path to this endpoint.
                if manual_stream:
                    reference = dict(old, joints_rad=latest['joints_rad'], ee_pose=latest['ee_pose'],
                                     gripper=dict(old['gripper'], width_m=latest['gripper']['width_m']))
                    requested = getattr(self, 'gripper_enable_request_at', None)
                    if (arm == self.arm and not old['gripper']['enabled'] and latest['gripper']['enabled'] and
                            requested is not None and 0 <= self.clock.monotonic()-requested <= .3):
                        # Accept only the acknowledgement of our own recent
                        # enable packet, never an unsolicited enable or disable.
                        reference['gripper']['enabled'] = True
                        self.trace.record('gripper_enable_confirmed', request_at=requested)
                else:
                    reference = (dict(old, joints_rad=latest['joints_rad'])
                                 if arm == self.arm and joint_change <= change_limit else old)
                try:
                    self.robot.check_motion_unchanged(arm, reference, latest, 'during teleop calculation')
                except Exception:
                    self.trace.record('feedback_changed', arm=arm, previous=old, latest=latest,
                                      joint_change_deg=float(np.rad2deg(joint_change)),
                                      change_limit_deg=float(np.rad2deg(change_limit)))
                    raise
                if manual_stream and (pose_changed or width_changed or
                                      joint_change > (np.deg2rad(.05) if arm != self.arm or self.paused
                                                      else change_limit)):
                    moved = True
                    self.trace.record('feedback_advanced', arm=arm, previous=old, latest=latest,
                                      joint_change_deg=float(np.rad2deg(joint_change)),
                                      pose_changed=bool(pose_changed), width_changed=bool(width_changed))
                    continue
                if joint_change > np.deg2rad(.05):
                    if arm != self.arm or self.paused or joint_change > change_limit:
                        raise RuntimeError('teleop calculation start moved')
                    path = result.get('path_joints_rad')
                    stroke = result.get('stroke_bounds_m')
                    if path is not None and stroke is not None:
                        path = np.asarray(path, dtype=float)
                        if (path.ndim != 2 or path.shape[1] != 6 or len(path) < 2 or
                                not np.isfinite(path).all() or
                                not np.allclose(path[0], old['joints_rad'], atol=1e-12, rtol=0) or
                                not np.allclose(path[-1], np.deg2rad(np.asarray(target)/1000),
                                                atol=1e-12, rtol=0)):
                            raise RuntimeError('invalid teleop path feedback proof')
                        residual = np.min(np.max(np.abs(path - latest['joints_rad']), axis=1))
                        if (residual <= np.deg2rad(.05) and
                                stroke[0] - 5e-7 <= latest['gripper']['width_m'] <= stroke[1] + 5e-7):
                            # Same 0.05 degree start tolerance, now compared
                            # against an already checked point on this exact
                            # path. Expected progress is not a stale start.
                            self.trace.record('validated_progress', generation=generation,
                                              residual_deg=float(np.rad2deg(residual)), latest=latest)
                            continue
                    moved = True
            if not moved:
                break
            with self.mutex:
                if not self.active or generation != self.generation:
                    return
            remaining = min(deadline, started + 1 / self.policy.control_hz) - self.clock.monotonic()
            if remaining <= 0 and manual_stream:
                raise CalculationExpired('teleop fresh path deadline exceeded')
            if (not manual_stream and attempt >= maximum_rechecks) or remaining <= 0:
                self.trace.record('discarded', generation=generation,
                                  reason='selected arm advanced during calculation', latest=refreshed[self.arm])
                return
            attempt += 1
            self.trace.record('revalidating', generation=generation, state=refreshed)
            before, state = refreshed, refreshed[self.arm]
            args = (state, before[self.other], proposal['ee_target'].tolist(), step_geometry,
                    proposal['width_m'], asdict(self.policy), self.velocity, self.last_goal_joints)
            job = None
            try:
                if inline_recheck:
                    result = inline_recheck(*args, result)
                    if self.clock.monotonic() > min(deadline, started + 1 / self.policy.control_hz):
                        raise CalculationExpired('teleop inline recheck deadline exceeded')
                else:
                    job = (self.robot.geometry_pool.submit(geometry_recheck, *args, result)
                           if self.policy.ik_solver == 'casadi_ipopt' else
                           self.robot.geometry_pool.submit(geometry_step, *args))
                    result = job.result(timeout=remaining)
            except (TimeoutError, FutureTimeoutError) as exc:
                if job is not None:
                    job.cancel()
                raise CalculationExpired('teleop fresh geometry deadline exceeded') from exc
            except ValueError as exc:
                if job is not None:
                    job.cancel()
                if str(exc) == 'teleop TCP target exceeds tracking lead':
                    # Fresh feedback can move the admissible lead boundary.
                    # Discard this calculation; never send it or stop a healthy
                    # session merely because its one candidate is now too far.
                    self.trace.record('discarded', generation=generation,
                                      reason='fresh feedback outside TCP tracking lead', state=refreshed)
                    return
                raise
            except Exception:
                if job is not None:
                    job.cancel()
                raise
            if not result.get('reachable'):
                raise RuntimeError('teleop fresh geometry did not validate target')
            target = result['joint_sdk']
            if len(target) != 6 or any(isinstance(v, bool) or not isinstance(v, int) for v in target):
                raise RuntimeError('invalid quantized joint target')
            self.accept_optimized_target(proposal, result, update_gain=False)
            mark('geometry_recheck')
        mark('feedback_recheck')
        with self.robot.tx_lock, self.mutex:
            if (not self.active or generation != self.generation or
                    self.clock.monotonic() - self.last_input > self.policy.input_timeout_s):
                return
            if self.clock.monotonic() - started > 1 / self.policy.control_hz:
                raise SendWindowExpired('teleop control deadline exceeded before write')
            if self.camera_error:
                raise RuntimeError(self.camera_error)
            self.fresh_cameras(self.cameras)
            if self.token_value() != self.token:
                raise RuntimeError('configuration changed before teleop write')
            if self.clock.monotonic() - started > 1 / self.policy.control_hz:
                raise SendWindowExpired('teleop control deadline exceeded before write')
            self.trace.activate()
            mark('configuration_and_evidence')
            p = self.robot.arms[self.arm]
            commands = []
            if self.paused:
                commands += [('MotionCtrl_1', (0, 4, 0)),
                             ('JointCtrl', tuple(round(np.rad2deg(v) * 1000) for v in state['joints_rad']))]
            commands += [('MotionCtrl_2', (1, 1, int(self.policy.speed_percent), 0))]
            commands += [('JointCtrl', tuple(target))]
            if self.paused:
                commands += [('MotionCtrl_1', (0, 2, 0))]
            if proposal['gripper_motion']:
                commands += [('GripperCtrl', (width_sdk, round(self.policy.gripper_effort * 1000), 1, 0))]
            for method, arguments in commands:
                if self.robot.stopped.is_set() or self.robot.fault:
                    raise RuntimeError('execution stopped before teleop write')
                self.trace.record('command', method=method, arguments=list(arguments), generation=generation)
                if self.clock.monotonic() - started > 1 / self.policy.control_hz:
                    raise SendWindowExpired('teleop control deadline exceeded before write')
                self.cycle_writes += 1  # An SDK attempt may reach hardware even if its return is late.
                self.robot.actuator_commands_sent = True
                if method == 'GripperCtrl' and arguments[2] == 1 and not state['gripper']['enabled']:
                    self.gripper_enable_request_at = self.clock.monotonic()
                getattr(p, method)(*arguments)
                if method == 'GripperCtrl':
                    self.gripper_commanded = True
                    self.gripper_target_m = arguments[0]/1e6
            mark('SDK_send')
            if self.clock.monotonic() - started > 1 / self.policy.control_hz:
                raise SendWindowExpired('teleop SDK send deadline exceeded')
            self.integrator.commit(proposal)
            self.deadline_misses = 0
            self.waiting_since = None
            self.accepted_keys = keys
            self.velocity = result['velocity']
            self.last_goal_joints = np.deg2rad(np.asarray(target, dtype=float) / 1000).tolist()
            self.paused, self.phase, self.last_confirmation = False, 'moving', None
            if self.detour_context is not None:
                self.phase = 'detour_planning'
                self.reason = '后台规划绕行中，当前局部控制继续；松键取消'
            self.target_stamps.append(self.clock.monotonic())
            self.trace.record('target', generation=generation, keys=sorted(keys), joints_sdk=target,
                              width_m=proposal['width_m'], tcp_target=proposal['tcp_target'].tolist(),
                              state=before, geometry_samples=result['samples'], ik=result.get('ik'),
                              joint_limit_status=result.get('joint_limit_status'),
                              gripper_envelope_mode=result.get('gripper_envelope_mode'),
                              gripper_envelope_max_m=result.get('gripper_envelope_max_m'),
                              speed_profile=self.policy.speed_profile)
            mark('publish')

    def run(self, done):
        last = None
        while not done.is_set():
            self.input_changed.clear()
            tick = self.clock.monotonic()
            try:
                self.step()
            except Exception as exc:
                elapsed = self.clock.monotonic() - tick
                self.cycles.append(elapsed * 1000)
                try:
                    self.trace.record('cycle', elapsed_ms=elapsed * 1000,
                                      stages_ms=dict(self.cycle_stages), error=str(exc),
                                      error_type=type(exc).__name__)
                except Exception:
                    pass  # Evidence failure must not prevent the actuator pause.
                self.fail('teleop control: ' + (str(exc) or type(exc).__name__))
                return
            elapsed = self.clock.monotonic() - tick
            self.cycles.append(elapsed * 1000)
            self.measured_hz = 1 / max(1e-6, tick - last) if last is not None else 0.
            last = tick
            if elapsed > 1 / self.policy.control_hz:
                # Sending has its own strict deadline inside step. A scheduling
                # or bookkeeping overrun after a valid send is a timing sample,
                # not permission to integrate a larger step or queue old inputs.
                self.cycle_overruns += 1
            if self.trace and not done.is_set():
                try:
                    self.trace.record('cycle', elapsed_ms=elapsed * 1000,
                                      stages_ms=dict(self.cycle_stages))
                except Exception as exc:
                    self.fail('teleop evidence: ' + str(exc))
                    return
            self.input_changed.wait(max(0, 1 / self.policy.control_hz - elapsed))
