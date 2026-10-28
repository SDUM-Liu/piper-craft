"""Isolated continuous input, transforms, watchdog and write-order regressions."""
import copy
from dataclasses import asdict
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from astra_vla.teleop import Integrator, Policy, keyset, pose, rotation_error
from astra_vla.workers.teleop import TeleopController, camera_fingerprint


def state():
    return {'ee_pose': {'position_m': [.2, 0, .3], 'rpy_deg': [0, 0, 0]},
            'joints_rad': [0.] * 6, 'gripper': {'width_m': .03, 'enabled': True, 'faults': []},
            'controller': {'ctrl_mode': 1, 'move_mode': 0, 'arm_status': 0, 'err_code': 0,
                           'motors_enabled': [True] * 6, 'firmware': 'S-V1.5-8'}, 'stamp': 100., 'age_s': .01,
            'rates_hz': {'pose': 200, 'joint': 200},
            'feedback_stamps': {'pose': 100., 'joint': 100.}}


def transforms():
    tcp, camera = np.eye(4), np.eye(4)
    tcp[2, 3] = .145
    camera[:3, :3] = Rotation.from_euler('x', -90, degrees=True).as_matrix()
    return tcp, camera


@pytest.mark.parametrize('keys,expected', [(['w'], [1, 0, 0]), (['s'], [-1, 0, 0]),
    (['a'], [0, 1, 0]), (['d'], [0, -1, 0]), (['space'], [0, 0, 1]), (['lshift'], [0, 0, -1])])
def test_translation_directions_and_continuous_hold(keys, expected):
    s = state()
    tcp, camera = transforms()
    integrator = Integrator(s, tcp, policy=Policy())
    start = integrator.target[:3, 3].copy()
    for _ in range(200):  # ten seconds without repeated keydown events
        proposal = integrator.propose(s, frozenset(keys), .05, .065)
        assert proposal is not None
        integrator.commit(proposal)
        s['ee_pose']['position_m'] = proposal['ee_target'][:3, 3].tolist()
    np.testing.assert_allclose(integrator.target[:3, 3] - start, np.array(expected) * .0495, atol=1e-10)


def test_diagonal_normalized_and_opposites_cancel():
    s = state()
    tcp, camera = transforms()
    integrator = Integrator(s, tcp, policy=Policy())
    p = integrator.propose(s, frozenset(('w', 'd', 'space')), .05, .065)
    assert np.linalg.norm(p['tcp_target'][:3, 3] - integrator.target[:3, 3]) == pytest.approx(.00005)
    p = integrator.propose(s, keyset(['w', 's', 'q', 'e', 'i', 'k']), .05, .065)
    np.testing.assert_allclose(p['tcp_target'], integrator.target)
    assert not p['arm_motion'] and not p['gripper_motion']


@pytest.mark.parametrize('keys', [('i',), ('k',), ('j',), ('l',), ('o',), ('p',), ('i', 'j', 'o')])
def test_fixed_tcp_rotations_compensate_tool_offset(keys):
    s = state()
    tcp, camera = transforms()
    integrator = Integrator(s, tcp, policy=Policy())
    tcp[0, 3] = .03
    integrator = Integrator(s, tcp, policy=Policy())
    anchor = integrator.target[:3, 3].copy()
    for _ in range(100):
        proposal = integrator.propose(s, frozenset(keys), .05, .065)
        np.testing.assert_allclose(proposal['tcp_target'][:3, 3], anchor, atol=1e-12)
        np.testing.assert_allclose((proposal['ee_target'] @ tcp)[:3, 3], anchor, atol=1e-12)
        integrator.commit(proposal)
        s['ee_pose'] = {'position_m': proposal['ee_target'][:3, 3].tolist(),
                       'rpy_deg': Rotation.from_matrix(proposal['ee_target'][:3, :3]).as_euler(
                           'xyz', degrees=True).tolist()}
    assert np.linalg.norm(pose(s)[:3, 3] - [.2, 0, .3]) > .0001


def test_rotation_anchor_survives_key_changes_and_feedback_noise():
    s = state()
    tcp, camera = transforms()
    integrator = Integrator(s, tcp, policy=Policy())
    anchor = integrator.target[:3, 3].copy()
    p = integrator.propose(s, keyset(['i']), .05, .065)
    integrator.commit(p)
    s['ee_pose']['position_m'][0] += .0005
    p = integrator.propose(s, keyset(['j']), .05, .065, keyset(['i']))
    np.testing.assert_allclose(p['tcp_target'][:3, 3], anchor)
    s['ee_pose']['position_m'][0] += .003
    with pytest.raises(RuntimeError, match='drift'):
        integrator.propose(s, keyset(['i']), .05, .065)


def test_no_time_debt_and_lead_bound():
    s = state()
    tcp, camera = transforms()
    integrator = Integrator(s, tcp, policy=Policy())
    p = integrator.propose(s, keyset(['w']), 20, .065)
    assert np.linalg.norm(p['tcp_target'][:3, 3] - integrator.target[:3, 3]) == pytest.approx(.00005)
    for _ in range(10):
        p = integrator.propose(s, keyset(['w']), .05, .065)
        integrator.commit(p)
    proposal=integrator.propose(s, keyset(['w']), .05, .065)
    assert proposal is None or np.linalg.norm(proposal['tcp_target'][:3,3]-(pose(s)@tcp)[:3,3])<=integrator.policy.position_lead_m


def test_gripper_combination_clamping_and_changed_translation():
    s = state()
    tcp, camera = transforms()
    integrator = Integrator(s, tcp, policy=Policy())
    p = integrator.propose(s, keyset(['w', 'e']), .05, .03)
    assert p['width_m'] == .03 and p['arm_motion'] and p['gripper_motion']
    integrator.commit(p)
    p = integrator.propose(s, keyset(['d']), .05, .065, keyset(['w', 'e']))
    assert p['tcp_target'][0, 3] == pytest.approx(.2)


@pytest.mark.parametrize('key,local_direction', [('w', [1, 0, 0]), ('s', [-1, 0, 0]),
    ('a', [0, 1, 0]), ('d', [0, -1, 0]), ('space', [0, 0, 1]), ('lshift', [0, 0, -1])])
def test_tcp_local_ten_second_hold_needs_only_feedback_and_tool_offset(key, local_direction):
    measured = state()
    measured['ee_pose']['rpy_deg'] = [23., -31., 67.]
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler('xyz', [12, 28, -15], degrees=True).as_matrix()
    tcp[:3, 3] = [.02, -.03, .145]
    integrator = Integrator(measured, tcp, policy=Policy(coordinate_mode='motion'))
    initial = integrator.target.copy()
    for _ in range(200):
        proposal = integrator.propose(measured, frozenset((key,)), .05, .065)
        integrator.commit(proposal)
        measured['ee_pose'] = {'position_m': proposal['ee_target'][:3, 3].tolist(),
            'rpy_deg': Rotation.from_matrix(proposal['ee_target'][:3, :3]).as_euler(
                'xyz', degrees=True).tolist()}
    expected = initial[:3, :3] @ np.asarray(local_direction) * .0495
    np.testing.assert_allclose(integrator.target[:3, 3] - initial[:3, 3], expected, atol=1e-10)


@pytest.mark.parametrize('key,rotation_axis', [('i', [0, -1, 0]), ('k', [0, 1, 0]),
    ('j', [0, 0, 1]), ('l', [0, 0, -1]), ('o', [-1, 0, 0]), ('p', [1, 0, 0])])
def test_tcp_local_rotation_axes_pin_offset_tool_position(key, rotation_axis):
    measured = state()
    measured['ee_pose']['rpy_deg'] = [25, 30, -17]
    tcp = np.eye(4)
    tcp[:3, 3] = [.02, -.03, .145]
    integrator = Integrator(measured, tcp, policy=Policy(coordinate_mode='motion'))
    initial = integrator.target.copy()
    for _ in range(200):
        proposal = integrator.propose(measured, frozenset((key,)), .05, .065)
        integrator.commit(proposal)
        measured['ee_pose'] = {'position_m': proposal['ee_target'][:3, 3].tolist(),
            'rpy_deg': Rotation.from_matrix(proposal['ee_target'][:3, :3]).as_euler(
                'xyz', degrees=True).tolist()}
        np.testing.assert_allclose((pose(measured) @ tcp)[:3, 3], initial[:3, 3], atol=1e-12)
    change = Rotation.from_matrix(initial[:3, :3].T @ integrator.target[:3, :3]).as_rotvec()
    np.testing.assert_allclose(change, np.asarray(rotation_axis) * np.deg2rad(19.8), atol=1e-10)


@pytest.mark.parametrize('value', [['mouse'], ['w', 2], 'w', None])
def test_invalid_keys(value):
    with pytest.raises(ValueError):
        keyset(value)


@pytest.mark.parametrize('values', [{'input_timeout_s': 1}, {'speed_percent': 101},
    {'translation_m_s': float('nan')}, {'control_hz': True}, {'speed_percent': 1.5}])
def test_invalid_policy(values):
    with pytest.raises(ValueError):
        Policy(**values)


@pytest.mark.parametrize('profile', ['slow', 'normal', 'fast'])
def test_commanded_speed_has_no_application_ceiling(profile):
    from dataclasses import replace
    policy = replace(Policy().for_speed(profile), translation_m_s=.3,
                     rotation_deg_s=90., gripper_m_s=.05, speed_percent=100)
    assert policy.translation_m_s == .3 and policy.speed_percent == 100


class MemoryTrace:
    def __init__(self, *_):
        self.path, self.rows, self.activated = None, [], False

    def record(self, kind, **data):
        self.rows.append({'type': kind, **data})

    def activate(self):
        self.activated = True

    def close(self, result):
        self.result = result


@pytest.fixture
def controller(tmp_path, monkeypatch):
    clock = [100.]
    timer = SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0])
    tcp, camera = transforms()
    verified = {'verified': True, 'arms': {a: {'T_ee_tcp': tcp.tolist(),
        'table': {'normal': [0, 0, 1], 'point_m': [0, 0, 0]}} for a in ('left', 'right')}}
    cameras = {name: {'available': True, 'depth_available': True, 'depth_aligned_to_color': True,
                     'stamp': 100., 'depth_stamp': 100., 'age_s': .01, 'serial': name, 'intrinsics': {}}
               for name in ('top', 'left_wrist', 'right_wrist')}
    verified['fingerprint'] = camera_fingerprint(cameras)
    commands, states = [], {a: state() for a in ('left', 'right')}
    token = ['same']
    robot = SimpleNamespace(lock=threading.Lock(), tx_lock=threading.Lock(), stopped=threading.Event(),
        fault=None, config={'calibration_dir': str(tmp_path), 'limits': {'synchronization_s': .15,
            'collision_margin_m': .015, 'table_clearance_m': .003,
            'maintenance_gripper_max_m': {'left': .065, 'right': .0855}}})
    def feedback(a, *_):
        s = copy.deepcopy(states[a])
        s['stamp'] = clock[0]
        s['feedback_stamps'] = {'pose': clock[0], 'joint': clock[0]}
        return s
    robot.read_controller_limits = lambda arm: {'source': 'controller_query_0x472_reply_0x473',
        'complete': True, 'limits': [dict(joint=i+1, min_deg=lo, max_deg=hi)
        for i, (lo, hi) in enumerate(zip([-150,0,-170,-100,-70,-180],[150,180,0,100,70,180]))]}
    robot.state = feedback
    robot.healthy = feedback
    robot.check_motion_state = lambda s: None
    robot.check_motion_unchanged = lambda *_: None
    robot.check_prepare = lambda a: feedback(a)
    robot.stationary_feedback = lambda **_: {'confirmed': True}
    robot.prepare = lambda a, g: feedback(a)
    robot.motion_geometry = lambda g: g
    def send(a, method):
        def invoke(*values):
            commands.append((a, method, values))
            if method == 'MotionCtrl_2':
                states[a]['controller']['ctrl_mode'] = values[0]
                states[a]['controller']['move_mode'] = values[1]
            elif method == 'EnableArm':
                states[a]['controller']['motors_enabled'] = [True] * 6
            elif method == 'GripperCtrl':
                states[a]['gripper']['enabled'] = True
        return invoke
    robot.arms = {a: SimpleNamespace(**{m: send(a, m) for m in
        ('MotionCtrl_1', 'MotionCtrl_2', 'JointCtrl', 'GripperCtrl', 'EnableArm')}) for a in ('left', 'right')}
    def command(p, method, *args):
        if robot.stopped.is_set():
            raise RuntimeError('stopped before hold write')
        getattr(p, method)(*args)
    robot.command = command
    completion = [None]
    geometry_error = [None]
    def submit(_function, *arguments):
        if _function.__name__ == 'geometry_native_step':
            def selected(**_):
                if completion[0]:
                    completion[0]()
                if geometry_error[0]:
                    raise geometry_error[0]
                return {'tcp_target': arguments[2]['tcp_target'].tolist(),
                        'adjustment_deg': 0., 'progress': 1.}
            return SimpleNamespace(result=selected, cancel=lambda: None)
        if not arguments or _function.__name__ in ('geometry_hold', 'native_feedback'):
            return SimpleNamespace(result=lambda **_: True)
        def result(timeout):
            if completion[0]:
                completion[0]()
            if geometry_error[0]:
                raise geometry_error[0]
            return {'reachable': True, 'joint_sdk': [0] * 6, 'velocity': [0.] * 6, 'samples': 22,
                    'ik': {'solver': 'casadi_ipopt', 'progress_fraction': 1.},
                    'tcp_target': (np.asarray(arguments[2]) @ np.asarray(arguments[3]['tcp'])).tolist()}
        return SimpleNamespace(result=result, cancel=lambda: None)
    robot.geometry_pool = SimpleNamespace(submit=submit)
    monkeypatch.setattr(Policy, 'load', lambda: Policy())
    manager = TeleopController(robot, clock=timer, camera_read=lambda: {'cameras': cameras},
        tool_read=lambda: verified, token_read=lambda: token[0], trace_factory=MemoryTrace, auto_start=False)
    robot.teleop = manager
    def stop(reason, **_kwargs):
        robot.stopped.set()
        robot.fault = reason
        manager.invalidate(reason)
        with robot.tx_lock:
            for a in robot.arms:
                robot.arms[a].MotionCtrl_1(0, 1, 0)
                if hasattr(robot, 'hold_joint_follow'):
                    robot.hold_joint_follow(a)
            manager.freeze_gripper()
        confirmation = robot.stationary_feedback()
        manager.complete_stop(confirmation)
        return {'stop_confirmed': confirmation['confirmed']}
    robot.stop = stop
    return SimpleNamespace(manager=manager, robot=robot, states=states, commands=commands,
        clock=clock, cameras=cameras, verified=verified, token=token,
        completion=completion, geometry_error=geometry_error)


def enter(f, keys=('w',), sequence=0):
    view = f.manager.open({'arm': 'left'})
    f.manager.input({'session': view['session'], 'sequence': sequence, 'keys': list(keys)})
    return view['session']


def test_open_does_not_require_verified_workspace_or_depth(controller):
    f = controller
    f.verified['verified'] = False
    for frame in f.cameras.values():
        frame['depth_available'] = False
    f.manager.open({'arm': 'left'})
    assert not f.commands and f.robot.lock.locked() and f.manager.active
    assert f.manager.geometry['scope'] == 'tcp_jog'
    assert f.manager.view()['environment_guard'] is False


def test_failed_entry_retains_reason_and_arm_without_owning_control(controller):
    f = controller
    def rejected(*_):
        raise ValueError('measured pose has URDF self collision')
    f.manager.prepare_local = rejected
    with pytest.raises(ValueError, match='self collision'):
        f.manager.open({'arm': 'right'})
    view = f.manager.view()
    assert view['phase'] == 'entry_blocked' and view['arm'] == 'right'
    assert view['blocker'] == 'measured pose has URDF self collision'
    assert not view['active'] and view['owner'] is None
    assert not f.robot.lock.locked() and not f.commands


def test_release_pauses_before_fk_check_then_revalidates_after_quiet(controller):
    f = controller
    session = enter(f)
    f.manager.step()
    writes = len(f.commands)
    def incomplete_feedback(_state):
        raise ValueError('teleop SDK pose disagrees with FK')
    f.robot.teleop_feedback = incomplete_feedback
    # A transient mixed pose/joint sample must not turn release into an input
    # failure. Pause/cancel happens now, with no replacement motion target.
    f.manager.input({'session': session, 'sequence': 1, 'keys': []})
    assert f.manager.phase == 'pausing' and f.manager.active
    assert all(method != 'JointCtrl' for _, method, _ in f.commands[writes:])
    f.clock[0] += .1
    f.manager.last_input = f.clock[0]
    f.manager.step()
    # Persistent disagreement still blocks recovery before another target.
    f.clock[0] += .51
    f.manager.last_input = f.clock[0]
    for camera in f.cameras.values():
        camera['stamp'] = f.clock[0]
    with pytest.raises(ValueError, match='disagrees with FK'):
        f.manager.step()
    assert all(method != 'JointCtrl' for _, method, _ in f.commands[writes:])
    f.robot.teleop_feedback = lambda s: s
    f.clock[0] += .01
    f.manager.last_input = f.clock[0]
    f.manager.step()
    assert f.manager.phase == 'ready'


def test_exclusive_session_sends_move_j_and_simultaneous_gripper(controller):
    f = controller
    enter(f, ('w', 'q'))
    assert f.robot.lock.locked()
    with pytest.raises(RuntimeError, match='owns'):
        f.manager.open({'arm': 'right'})
    f.manager.step()
    assert ('left', 'MotionCtrl_2', (1, 1, 5, 0)) in f.commands
    assert f.commands[-1][1] == 'GripperCtrl'
    assert all(c[0] == 'left' for c in f.commands)
    assert f.manager.trace.activated


@pytest.mark.parametrize('ages', [(.2, .2, .2), (.05, .3, .5)])
def test_tcp_control_accepts_independently_fresh_rgb_without_timestamp_alignment(controller, ages):
    f = controller
    for frame, age in zip(f.cameras.values(), ages):
        frame.update(stamp=f.clock[0] - age, age_s=age)
    enter(f, ('w', 'q'))
    f.manager.step()
    assert any(c[1] == 'JointCtrl' for c in f.commands)
    assert f.commands[-1][1] == 'GripperCtrl'
    assert f.manager.phase == 'moving'


@pytest.mark.parametrize('camera', ['top', 'left_wrist', 'right_wrist'])
def test_one_stale_rgb_still_rejects_before_any_target(controller, camera):
    f = controller
    enter(f)
    f.cameras[camera].update(stamp=f.clock[0] - .601, age_s=.601)
    with pytest.raises(RuntimeError, match='RGB missing/stale'):
        f.manager.step()
    assert not f.commands
    assert not f.manager.trace.activated


def test_local_prepare_disabled_uses_quantized_measured_joint_hold(controller):
    f = controller
    f.states['left']['controller'].update(ctrl_mode=0, motors_enabled=[False] * 6)
    f.manager.open({'arm': 'left'})
    methods = [c[1] for c in f.commands]
    assert methods == ['MotionCtrl_1', 'JointCtrl', 'MotionCtrl_2', 'JointCtrl',
                       'MotionCtrl_1', 'EnableArm', 'GripperCtrl']
    assert f.commands[1][2] == (0,) * 6
    assert f.commands[-1][2] == (30000, 500, 1, 0)


def test_configuration_token_does_not_read_verified_or_handeye(tmp_path, monkeypatch):
    from astra_vla.workers import teleop
    folder = tmp_path / 'config'
    folder.mkdir()
    model = tmp_path / 'model.urdf'
    model.write_text('offline-model')
    for name in ('devices.yaml', 'tcp.json', 'teleop.json'):
        (folder / name).write_text('offline-config')
    monkeypatch.setattr(teleop, 'ROOT', tmp_path)
    monkeypatch.setattr(teleop, 'settings', lambda: {'urdf': str(model)})
    initial = teleop.configuration_token()
    (folder / 'verified.json').write_text('{"verified":false}')
    (tmp_path / 'left_handeye.json').write_text('not used')
    assert teleop.configuration_token() == initial
    (folder / 'tcp.json').write_text('new-offset')
    assert teleop.configuration_token() != initial


def test_configuration_cache_skips_yaml_parsing_and_detects_rapid_same_size_edits(tmp_path, monkeypatch):
    from astra_vla.workers import teleop
    folder = tmp_path / 'config'
    folder.mkdir()
    model = tmp_path / 'model.urdf'
    paths = [folder / name for name in ('devices.yaml', 'tcp.json', 'teleop.json')] + [model]
    for path in paths:
        path.write_text('initial')
    parses = []
    monkeypatch.setattr(teleop, 'ROOT', tmp_path)
    monkeypatch.setattr(teleop, 'settings', lambda: parses.append(True) or {'urdf': str(model)})
    initial = teleop.configuration_token()
    assert teleop.configuration_token() == initial
    assert len(parses) == 1
    for path in reversed(paths):
        path.write_text('changed')
        new = teleop.configuration_token()
        assert new != initial
        initial = new
    assert len(parses) == 2  # YAML is parsed again only when devices.yaml changes.


def test_trace_disk_writer_does_not_block_first_command_and_drains_in_order(tmp_path, monkeypatch):
    from astra_vla.workers import teleop
    monkeypatch.setattr(teleop, 'ROOT', tmp_path)
    (tmp_path / 'runs').mkdir()
    started, release = threading.Event(), threading.Event()
    write_json = teleop.atomic_json
    def slow_disk(path, value):
        if path.name == 'manifest.json':
            started.set()
            assert release.wait(2)
        write_json(path, value)
    monkeypatch.setattr(teleop, 'atomic_json', slow_disk)
    trace = teleop.Trace('offline_test', 'left', Policy())
    trace.record('input', keys=['w'])
    trace.activate()
    assert started.wait(1)
    try:
        trace.record('command', method='JointCtrl')
        trace.record('target', joints_sdk=[0] * 6)
        assert trace.writer.is_alive()  # Control-side calls returned during blocked disk I/O.
    finally:
        release.set()
        trace.close({'stop_confirmed': True})
    rows = [json.loads(line) for line in (trace.path / 'events.jsonl').read_text().splitlines()]
    assert [row['type'] for row in rows] == ['input', 'command', 'target']
    assert json.loads((trace.path / 'result.json').read_text())['stop_confirmed']


def test_trace_writer_failure_is_reported_to_control_guard(tmp_path, monkeypatch, controller):
    from astra_vla.workers import teleop
    monkeypatch.setattr(teleop, 'ROOT', tmp_path)
    (tmp_path / 'runs').mkdir()
    def failed_write(*_):
        raise OSError('disk full')
    monkeypatch.setattr(teleop, 'atomic_json', failed_write)
    trace = teleop.Trace('offline_failure', 'left', Policy())
    trace.activate()
    trace.writer.join(timeout=1)
    f = controller
    enter(f)
    f.manager.trace = trace
    with pytest.raises(RuntimeError, match='disk full'):
        f.manager.step()
    assert not f.commands


def test_final_configuration_read_cannot_allow_an_expired_target(controller):
    f = controller
    enter(f)
    reads = []
    def slow_final_read():
        reads.append(True)
        if len(reads) == 2:
            f.clock[0] += .06
        return f.token[0]
    f.manager.token_read = slow_final_read
    f.manager.step()
    assert f.manager.deadline_misses == 1 and f.manager.active
    assert any(row['type'] == 'deadline_drop' for row in f.manager.trace.rows)
    assert not f.commands and not f.manager.trace.activated


def test_sdk_send_overrun_pauses_and_never_queues_expired_target(controller):
    f = controller
    enter(f)
    f.manager.paused = False
    f.states['left']['controller']['move_mode'] = 1
    send = f.robot.arms['left'].JointCtrl
    def slow_send(*args):
        send(*args)
        f.clock[0] += .06
    f.robot.arms['left'].JointCtrl = slow_send
    f.manager.step()
    assert f.manager.paused and f.manager.phase == 'pausing'
    assert any(c[1:] == ('MotionCtrl_1', (0, 1, 0)) for c in f.commands)
    assert len([c for c in f.commands if c[1] == 'JointCtrl']) == 1
    assert f.manager.last_goal_joints is None
    assert any(row['type'] == 'deadline_hold' for row in f.manager.trace.rows)
    assert not f.robot.stopped.is_set()


def test_bookkeeping_overrun_after_valid_send_is_counted_without_extra_motion(controller):
    f = controller
    enter(f)
    step = f.manager.step
    def expensive_bookkeeping():
        step()
        f.clock[0] += .06
    f.manager.step = expensive_bookkeeping
    done = threading.Event()
    class OneCycle(threading.Event):
        def wait(self, timeout=None):
            done.set()
            return True
    f.manager.input_changed = OneCycle()
    f.manager.run(done)
    assert f.manager.cycle_overruns == 1 and not f.robot.stopped.is_set()
    assert len([c for c in f.commands if c[1] == 'JointCtrl']) == 2  # measured hold + one target


def test_release_during_geometry_cannot_send_old_target(controller):
    f = controller
    session = enter(f)
    f.completion[0] = lambda: f.manager.input({'session': session, 'sequence': 1, 'keys': []})
    f.manager.step()
    assert f.commands == [('left', 'MotionCtrl_1', (0, 1, 0))]
    assert not f.manager.trace.activated


def test_stop_during_geometry_cannot_write_after_pause(controller):
    f = controller
    enter(f)
    f.completion[0] = lambda: f.robot.stop('operator')
    f.manager.step()
    assert all(c[1] == 'MotionCtrl_1' and c[2] == (0, 1, 0) for c in f.commands)
    assert not f.robot.lock.locked()


def test_heartbeat_updates_do_not_invalidate_computation(controller):
    f = controller
    session = enter(f)
    f.completion[0] = lambda: f.manager.input({'session': session, 'sequence': 1, 'keys': ['w']})
    f.manager.step()
    assert any(c[1] == 'JointCtrl' for c in f.commands)
    accepted = f.manager.input({'session': session, 'sequence': 0, 'keys': ['s']})
    assert not accepted['accepted'] and f.manager.keys == frozenset(['w'])


@pytest.mark.parametrize('reason', ['heartbeat', 'rgb', 'identity', 'config'])
def test_failed_guards_prevent_writes(controller, reason):
    f = controller
    enter(f)
    if reason == 'heartbeat':
        f.clock[0] += .251
    elif reason == 'rgb':
        f.cameras['top']['stamp'] -= .7
    elif reason == 'identity':
        f.cameras['left_wrist']['serial'] = 'changed-device'
    else:
        f.token[0] = 'changed'
    with pytest.raises(RuntimeError):
        f.manager.step()
    assert not f.commands


def test_focus_loss_closes_session(controller):
    f = controller
    session = enter(f)
    with pytest.raises(RuntimeError, match='focus'):
        f.manager.input({'session': session, 'sequence': 1, 'keys': ['w'], 'focused': False})
    assert not f.manager.active and not f.robot.lock.locked()


def test_geometric_rejection_blocks_direction_without_stopping_latch(controller):
    f = controller
    enter(f)
    f.geometry_error[0] = ValueError('self collision')
    f.manager.step()
    assert f.manager.reason == 'self collision' and not f.robot.stopped.is_set()
    assert not any(c[1] == 'JointCtrl' for c in f.commands)
    f.manager.step()
    assert f.manager.active


@pytest.mark.parametrize('reason', ['teleop joint acceleration exceeds bound',
                                   'teleop IK local constraints infeasible'])
def test_rate_limited_start_reduces_new_increment_without_relaxing_limits(controller, reason):
    f = controller
    enter(f, ('w', 'q'))
    full = f.manager.integrator.propose(f.states['left'], keyset(['w', 'q']), .05, .065)
    attempts = []
    submit = f.robot.geometry_pool.submit
    def constrained(function, *arguments):
        attempts.append(arguments)
        if len(attempts) == 1:
            def reject(**_):
                raise ValueError(reason)
            return SimpleNamespace(result=reject, cancel=lambda: None)
        return submit(function, *arguments)
    f.robot.geometry_pool.submit = constrained
    f.manager.step()
    assert len(attempts) == 2
    initial = pose(f.states['left']) @ f.manager.integrator.tcp
    np.testing.assert_allclose(f.manager.integrator.target[:3, 3] - initial[:3, 3],
                              (full['tcp_target'][:3, 3] - initial[:3, 3]) / 2)
    assert f.manager.integrator.width == pytest.approx(full['width_m'])
    assert attempts[0][-3] == attempts[1][-3] == asdict(Policy())
    assert f.manager.phase == 'moving'
    assert any(c[1] == 'JointCtrl' for c in f.commands)


def test_repeated_rate_rejection_is_bounded_and_never_writes_target(controller):
    f = controller
    enter(f)
    attempts = []
    f.geometry_error[0] = ValueError('teleop joint acceleration exceeds bound')
    f.completion[0] = lambda: attempts.append(True)
    f.manager.step()
    assert len(attempts) == 5
    assert not any(c[1] == 'JointCtrl' for c in f.commands)
    assert not f.manager.trace.activated
    assert f.manager.reason == 'teleop joint acceleration exceeds bound'


def test_rate_backoffs_share_one_geometry_deadline(controller):
    f = controller
    enter(f)
    f.geometry_error[0] = ValueError('teleop IK local constraints infeasible')
    f.completion[0] = lambda: f.clock.__setitem__(0, f.clock[0] + .024)
    f.manager.step()
    assert f.manager.deadline_misses == 1
    assert not any(c[1] == 'JointCtrl' for c in f.commands)


def test_tracking_bound_waits_for_feedback_without_committing_or_replaying_rejected_target(controller):
    f = controller
    enter(f)
    f.manager.step()
    accepted = f.manager.integrator.target.copy()
    commands = list(f.commands)
    f.geometry_error[0] = ValueError('teleop joint acceleration exceeds bound')
    f.manager.step()
    assert f.manager.phase == 'waiting_feedback' and not f.manager.paused
    assert f.commands == commands and f.manager.reason is None
    np.testing.assert_array_equal(f.manager.integrator.target, accepted)
    f.geometry_error[0] = None
    f.manager.step()
    assert f.manager.phase == 'moving' and f.manager.waiting_since is None
    assert len(f.commands) == len(commands) + 2


def test_unconfirmed_stop_retains_exclusive_lease_until_retry(controller):
    f = controller
    session = enter(f)
    f.robot.stationary_feedback = lambda **_: {'confirmed': False}
    f.manager.close({'session': session})
    assert f.robot.lock.locked() and f.manager.phase == 'stop_unconfirmed'
    f.robot.stationary_feedback = lambda **_: {'confirmed': True}
    f.manager.close({'session': session})
    assert not f.robot.lock.locked()


def test_gripper_release_installs_measured_hold(controller):
    f = controller
    session = enter(f, ('w', 'q'))
    f.manager.step()
    f.manager.input({'session': session, 'sequence': 1, 'keys': ['w']})
    assert f.commands[-1] == ('left', 'GripperCtrl', (30000, 500, 1, 0))
    assert f.manager.paused
    assert f.manager.last_goal_joints is None


def test_policy_serializable_for_hardware_pool():
    assert json.loads(json.dumps(asdict(Policy())))['control_hz'] == 20


def test_camera_reader_returns_metadata_and_uses_lazy_spawn_pool(controller, monkeypatch):
    from astra_vla.workers import teleop
    reads, pools = [], []
    def rpc(path, method, args, **_):
        reads.append((path.name, method, args))
        return {'cameras': {'left_wrist': {'stamp': 10., 'serial': 'fake', 'available': True,
                'intrinsics': {'K': [1]*9}, 'jpeg': 'huge image', 'depth_npy': 'huge depth'}}}
    monkeypatch.setattr(teleop, 'call', rpc)
    def pool(**kwargs):
        pools.append(kwargs)
        return SimpleNamespace(submit=lambda function: SimpleNamespace(result=lambda **_: function()),
                               shutdown=lambda **_: None)
    monkeypatch.setattr(teleop, 'ProcessPoolExecutor', pool)
    f = controller
    assert f.manager.camera_pool is None and not reads
    for _ in range(2):
        result = f.manager.read_camera_metadata()
        assert set(result['cameras']['left_wrist']) == {'stamp', 'serial', 'available'}
    assert len(pools) == 1 and pools[0]['mp_context'].get_start_method() == 'spawn'
    assert reads == [('cameras.sock', 'snapshot', {'depth': False})]*2
    f.manager.shutdown_camera()
    assert f.manager.camera_pool is None


def test_rate_hint_survives_process_boundary_and_is_revalidated(controller):
    import pickle
    from astra_vla.teleop import rate_rejection
    error = rate_rejection('teleop joint acceleration exceeds bound', [20]*6, [10]*6, 6)
    error = pickle.loads(pickle.dumps(error))
    assert error.suggested_scale == pytest.approx(.784)
    f = controller
    enter(f)
    submissions = []
    submit = f.robot.geometry_pool.submit
    def constrained(function, *arguments):
        submissions.append(arguments)
        if len(submissions) == 1:
            def reject(**_):
                raise error
            return SimpleNamespace(result=reject, cancel=lambda: None)
        return submit(function, *arguments)
    f.robot.geometry_pool.submit = constrained
    f.manager.step()
    assert len(submissions) == 2
    assert submissions[0][-3] == submissions[1][-3] == asdict(Policy())
    assert f.manager.integrator.speed_fraction == pytest.approx(.2*.784)


def test_only_fresh_press_wakes_control_and_release_still_pauses(controller):
    f = controller
    session = enter(f, ())
    f.manager.input_changed.clear()
    f.manager.input({'session': session, 'sequence': 1, 'keys': ['w']})
    assert f.manager.input_changed.is_set()
    f.manager.input_changed.clear()
    f.manager.input({'session': session, 'sequence': 2, 'keys': ['w']})
    assert not f.manager.input_changed.is_set()
    f.manager.input({'session': session, 'sequence': 3, 'keys': []})
    assert f.commands[-1][1] == 'MotionCtrl_1' and f.manager.paused


def test_late_computation_cannot_write_even_if_future_returns(controller):
    f = controller
    enter(f)
    f.completion[0] = lambda: f.clock.__setitem__(0, f.clock[0] + .08)
    f.manager.step()
    assert f.manager.deadline_misses == 1
    assert not any(c[1] in ('JointCtrl', 'GripperCtrl') for c in f.commands)


def test_wrong_session_and_sequence_dont_refresh_heartbeat(controller):
    f = controller
    session = enter(f)
    stamp = f.manager.last_input
    f.clock[0] += .1
    with pytest.raises(RuntimeError, match='session'):
        f.manager.input({'session': 'wrong', 'sequence': 2, 'keys': ['w']})
    assert f.manager.last_input == stamp
    f.manager.input({'session': session, 'sequence': 0, 'keys': ['w']})
    assert f.manager.last_input == stamp


def test_other_arm_change_rechecks_same_target_before_write(controller):
    f = controller
    enter(f)
    f.completion[0] = lambda: f.states['right']['joints_rad'].__setitem__(0, .02)
    f.manager.step()
    assert any(c[1] == 'JointCtrl' for c in f.commands)
    assert any(row['type'] == 'revalidating' for row in f.manager.trace.rows)


def test_advancing_selected_arm_revalidates_same_target_from_fresh_feedback(controller):
    f = controller
    enter(f)
    f.manager.paused = False
    f.states['left']['controller']['move_mode'] = 1
    calls = []
    submit = f.robot.geometry_pool.submit
    def captured(function, *arguments):
        calls.append(copy.deepcopy(arguments))
        return submit(function, *arguments)
    f.robot.geometry_pool.submit = captured
    def advance_once():
        f.states['left']['joints_rad'][0] = np.deg2rad(.06)
        f.completion[0] = None
    f.completion[0] = advance_once
    f.manager.step()
    assert len(calls) == 2 and calls[0][0]['joints_rad'][0] == 0
    assert calls[1][0]['joints_rad'][0] == pytest.approx(np.deg2rad(.06))
    np.testing.assert_array_equal(calls[0][2], calls[1][2])
    assert [command[1] for command in f.commands] == ['MotionCtrl_2', 'JointCtrl']
    assert any(r['type'] == 'revalidating' for r in f.manager.trace.rows)


@pytest.mark.parametrize('off_path', [False, True])
def test_following_validated_path_streams_but_off_path_requires_new_geometry(controller, off_path):
    f = controller
    enter(f)
    f.manager.paused = False
    f.states['left']['controller']['move_mode'] = 1
    calls = []
    def advance_once():
        f.states['left']['joints_rad'][0] = np.deg2rad(.1)
        if off_path:
            f.states['left']['joints_rad'][1] = np.deg2rad(.11)
        f.completion[0] = None
    f.completion[0] = advance_once
    def submit(_function, *arguments):
        initial = copy.deepcopy(arguments[0]['joints_rad'])
        calls.append(initial)
        def result(**_):
            if f.completion[0]:
                f.completion[0]()
            goal = np.deg2rad([.4, 0, 0, 0, 0, 0])
            return {'reachable': True, 'joint_sdk': [400, 0, 0, 0, 0, 0], 'velocity': [0.]*6,
                    'samples': 11, 'path_joints_rad': np.linspace(initial, goal, 11).tolist(),
                    'stroke_bounds_m': [.03, .03],
                    'tcp_target': (np.asarray(arguments[2]) @ np.asarray(arguments[3]['tcp'])).tolist(),
                    'ik': {'solver': 'casadi_ipopt', 'progress_fraction': 1.}}
        return SimpleNamespace(result=result, cancel=lambda: None)
    f.robot.geometry_pool.submit = submit
    f.manager.step()
    assert len(calls) == (2 if off_path else 1)
    assert f.commands == [('left', 'MotionCtrl_2', (1, 1, 5, 0)),
                          ('left', 'JointCtrl', (400, 0, 0, 0, 0, 0))]
    assert any(r['type'] == ('revalidating' if off_path else 'validated_progress')
               for r in f.manager.trace.rows)


@pytest.mark.parametrize('interruption', ['release', 'continued_movement'])
def test_revalidation_is_bounded_and_cannot_send_after_release(controller, interruption):
    f = controller
    session = enter(f)
    f.manager.paused = False
    f.states['left']['controller']['move_mode'] = 1
    calls = [0]
    def advance():
        calls[0] += 1
        f.clock[0] += .02
        f.states['left']['joints_rad'][0] += np.deg2rad(.06)
        if interruption == 'release' and calls[0] == 2:
            f.manager.input({'session': session, 'sequence': 1, 'keys': []})
    f.completion[0] = advance
    original = f.manager.integrator.target.copy()
    f.manager.step()
    assert calls[0] >= 2 and not any(c[1] == 'JointCtrl' for c in f.commands)
    if interruption == 'release':
        assert f.manager.paused
        return
    np.testing.assert_array_equal(f.manager.integrator.target, original)
    assert any(r['type'] == 'deadline_drop' for r in f.manager.trace.rows)


def test_timeout_does_not_send_targets(controller):
    f = controller
    enter(f)
    f.geometry_error[0] = TimeoutError('geometry deadline')
    f.manager.step()
    f.manager.step()
    with pytest.raises(RuntimeError, match='three consecutive'):
        f.manager.step()
    assert not any(c[1] == 'JointCtrl' for c in f.commands)


@pytest.mark.parametrize('key,direction', [('w',[0,0,1]),('s',[0,0,-1]),
    ('a',[0,1,0]),('d',[0,-1,0]),('space',[-1,0,0]),('lshift',[1,0,0])])
def test_piper_input_axes_follow_tool_extension_not_generic_x(key,direction):
    measured=state()
    measured['ee_pose']['rpy_deg']=[29.,-37.,81.]
    tcp=np.eye(4)
    tcp[:3,3]=[.023,-.017,.145]
    tcp[:3,:3]=Rotation.from_euler('xyz',[17,23,-11],degrees=True).as_matrix()
    policy=Policy(axis_convention='piper_tool',coordinate_mode='motion')
    integrator=Integrator(measured,tcp,policy=policy)
    initial=integrator.target.copy()
    for _ in range(200):
        proposal=integrator.propose(measured,keyset([key]),.05,.065)
        integrator.commit(proposal)
        measured['ee_pose']={'position_m':proposal['ee_target'][:3,3].tolist(),
            'rpy_deg':Rotation.from_matrix(proposal['ee_target'][:3,:3]).as_euler('xyz',degrees=True).tolist()}
    np.testing.assert_allclose(integrator.target[:3,3]-initial[:3,3],
                               initial[:3,:3]@np.array(direction)*.0495,atol=1e-10)
    np.testing.assert_allclose(integrator.target[:3,:3],initial[:3,:3],atol=1e-10)


@pytest.mark.parametrize('key,axis', [('i',[0,-1,0]),('k',[0,1,0]),
    ('j',[-1,0,0]),('l',[1,0,0]),('o',[0,0,-1]),('p',[0,0,1])])
def test_piper_rotation_axes_pin_nonzero_offset_tcp(key,axis):
    measured=state()
    measured['ee_pose']['rpy_deg']=[-19.,47.,12.]
    tcp=np.eye(4)
    tcp[:3,3]=[.025,-.015,.145]
    integrator=Integrator(measured,tcp,policy=Policy(axis_convention='piper_tool',coordinate_mode='motion'))
    initial=integrator.target.copy()
    proposal=integrator.propose(measured,keyset([key]),.05,.065)
    np.testing.assert_allclose(proposal['tcp_target'][:3,3],initial[:3,3],atol=1e-12)
    np.testing.assert_allclose((proposal['ee_target']@tcp)[:3,3],initial[:3,3],atol=1e-12)
    expected=Rotation.from_rotvec(initial[:3,:3]@np.array(axis)*np.deg2rad(.02)).as_matrix()@initial[:3,:3]
    np.testing.assert_allclose(proposal['tcp_target'][:3,:3],expected,atol=1e-12)


@pytest.mark.parametrize('key,sign', [('space',1),('lshift',-1)])
@pytest.mark.parametrize('rpy', [[0,0,0],[90,0,0],[23,71,139]])
def test_base_vertical_input_never_rotates_with_wrist(key,sign,rpy):
    measured=state()
    measured['ee_pose']['rpy_deg']=rpy
    tcp=np.eye(4)
    tcp[:3,3]=[.025,-.017,.145]
    integrator=Integrator(measured,tcp,policy=Policy(axis_convention='piper_tool',vertical_frame='base'))
    start=integrator.target.copy()
    for _ in range(200):
        proposal=integrator.propose(measured,keyset([key]),.05,.065)
        integrator.commit(proposal)
        measured['ee_pose']={'position_m':proposal['ee_target'][:3,3].tolist(),
            'rpy_deg':Rotation.from_matrix(proposal['ee_target'][:3,:3]).as_euler('xyz',degrees=True).tolist()}
    np.testing.assert_allclose(integrator.target[:3,3]-start[:3,3],[0,0,sign*.0495],atol=1e-10)


def test_regularized_backend_commits_only_selected_progress(controller,monkeypatch):
    f=controller
    monkeypatch.setattr(Policy,'load',lambda:Policy(ik_solver='casadi_ipopt',vertical_frame='base'))
    session=enter(f,('space',))
    desired=f.manager.integrator.propose(f.states['left'],keyset(['space']),.05,.065)
    selected=desired['tcp_target'].copy()
    initial=f.manager.integrator.target.copy()
    selected[:3,3]=initial[:3,3]+.5*(selected[:3,3]-initial[:3,3])
    def submit(function,*args):
        assert function.__name__=='geometry_step'
        return SimpleNamespace(result=lambda **_:dict(reachable=True,joint_sdk=[0]*6,velocity=[0.]*6,
            samples=11,tcp_target=selected.tolist(),ik={'solver':'casadi_ipopt','progress_fraction':.5}),
            cancel=lambda:None)
    f.robot.geometry_pool.submit=submit
    f.manager.step()
    np.testing.assert_allclose(f.manager.integrator.target,selected)
    assert f.manager.integrator.speed_fraction==pytest.approx(.1)
    f.manager.input({'session':session,'sequence':1,'keys':[]})
    writes=len(f.commands)
    f.clock[0]+=.1
    f.manager.last_input=f.clock[0]
    f.manager.step()
    assert not any(method=='JointCtrl' for _,method,_ in f.commands[writes:])


def test_missing_requested_ipopt_result_never_sends_newton_goal(controller,monkeypatch):
    f=controller
    monkeypatch.setattr(Policy,'load',lambda:Policy(ik_solver='casadi_ipopt'))
    enter(f)
    submit=f.robot.geometry_pool.submit
    def invalid_backend(function,*args):
        future=submit(function,*args)
        def bad(**kwargs):
            result=future.result(**kwargs)
            result.pop('ik',None)
            return result
        return SimpleNamespace(result=bad,cancel=lambda:None)
    f.robot.geometry_pool.submit=invalid_backend
    with pytest.raises(RuntimeError,match='backend was not used'):
        f.manager.step()
    assert not any(method=='JointCtrl' for _,method,_ in f.commands)


def test_regularized_zero_progress_waits_then_pauses_without_joint_write(controller,monkeypatch):
    f=controller
    monkeypatch.setattr(Policy,'load',lambda:Policy(ik_solver='casadi_ipopt'))
    session=enter(f)
    def submit(_function,*args):
        return SimpleNamespace(result=lambda **_:dict(reachable=True,joint_sdk=[0]*6,velocity=[0.]*6,
            samples=11,ik={'solver':'casadi_ipopt','progress_fraction':0}),cancel=lambda:None)
    f.robot.geometry_pool.submit=submit
    f.manager.step()
    assert f.manager.phase=='waiting_feedback' and not f.commands
    f.clock[0]+=1.01
    for camera in f.cameras.values():
        camera['stamp']=f.clock[0]
    f.manager.input({'session':session,'sequence':1,'keys':['w']})
    f.manager.step()
    assert f.manager.blocked_generation==f.manager.generation
    assert not any(method=='JointCtrl' for _,method,_ in f.commands)


def test_prepare_uses_five_percent_before_new_twenty_percent_motion(controller,monkeypatch):
    f=controller
    monkeypatch.setattr(Policy,'load',lambda:Policy(ik_solver='casadi_ipopt',speed_percent=20))
    f.states['left']['controller'].update(ctrl_mode=0,motors_enabled=[False]*6)
    f.manager.open({'arm':'left'})
    modes=[args for arm,method,args in f.commands if method=='MotionCtrl_2']
    assert modes==[(1,1,5,0)]


def test_input_tracking_uses_controller_pose_and_ignores_alternate_fk_basis():
    measured=state()
    measured['reported_ee_pose']={'position_m':[9.,8.,7.],'rpy_deg':[90.,90.,90.]}
    integrator=Integrator(measured,np.eye(4))
    integrator.target[0,3]=.20196
    proposal=integrator.propose(measured,keyset(['w']),.05,.065)
    assert proposal is not None
    assert np.linalg.norm(proposal['tcp_target'][:3,3]-pose(measured)[:3,3])<=integrator.policy.position_lead_m
    assert measured['ee_pose']['position_m']==[.2,0,.3]


def test_fresh_tracking_lead_rejection_discards_without_joint_write(controller):
    f=controller
    enter(f)
    f.manager.step()
    sent=len(f.commands)
    computations=[0]
    def advance():
        computations[0]+=1
        if computations[0]==1:
            f.states['left']['joints_rad'][0]+=.001
        else:
            f.geometry_error[0]=ValueError('teleop TCP target exceeds tracking lead')
    f.completion[0]=advance
    f.manager.step()
    assert computations[0]==2 and f.manager.active
    assert not any(method=='JointCtrl' for _,method,_ in f.commands[sent:])
    assert f.manager.trace.rows[-1]['reason']=='fresh feedback outside TCP tracking lead'


def test_fresh_revalidation_cannot_expand_optimizer_selected_increment(controller,monkeypatch):
    f=controller
    monkeypatch.setattr(Policy,'load',lambda:Policy(ik_solver='casadi_ipopt',vertical_frame='base'))
    enter(f,('space',))
    calls=[]
    change=[False]
    selected=[None]
    def submit(_function,*args):
        calls.append(args)
        if len(calls)%2 or not change[0]:
            point=np.asarray(args[2])@f.manager.integrator.tcp
            point[:3,3]=(point[:3,3]+f.manager.integrator.target[:3,3])/2
            selected[0]=point
            progress=.5
        else:
            point=np.asarray(args[2])@f.manager.integrator.tcp
            np.testing.assert_allclose(point,selected[0])
            progress=1.
        def result(**_):
            if change[0] and len(calls)%2:
                f.states['left']['joints_rad'][0]+=.001
            return dict(reachable=True,joint_sdk=[0]*6,velocity=[0.]*6,samples=11,
                        tcp_target=point.tolist(),ik={'solver':'casadi_ipopt','progress_fraction':progress})
        return SimpleNamespace(result=result,cancel=lambda:None)
    f.robot.geometry_pool.submit=submit
    f.manager.step()
    calls.clear()
    change[0]=True
    f.manager.step()
    assert len(calls)==2
    np.testing.assert_allclose(f.manager.integrator.target,selected[0])
    assert f.manager.integrator.speed_fraction==pytest.approx(.15)


def test_regularized_computation_timeout_drops_without_write_then_recovers(controller,monkeypatch):
    from astra_vla.workers.teleop import CalculationExpired
    f=controller
    monkeypatch.setattr(Policy,'load',lambda:Policy(ik_solver='casadi_ipopt'))
    enter(f)
    f.geometry_error[0]=CalculationExpired('synthetic numerical deadline')
    f.manager.step()
    assert f.manager.deadline_misses==1 and f.manager.active and not f.commands
    assert any(x['type']=='deadline_drop' for x in f.manager.trace.rows)
    f.manager.step()
    assert f.manager.deadline_misses==2 and not f.commands
    with pytest.raises(RuntimeError,match='three consecutive'):
        f.manager.step()
    assert not f.commands


def test_regularized_prewrite_deadline_drops_without_pause_or_sdk(controller,monkeypatch):
    from astra_vla.workers.teleop import SendWindowExpired
    f=controller
    monkeypatch.setattr(Policy,'load',lambda:Policy(ik_solver='casadi_ipopt'))
    enter(f)
    def expired():
        f.manager.cycle_writes=0
        raise SendWindowExpired('before SDK write')
    f.manager.calculate_step=expired
    f.manager.step()
    assert not f.commands and f.manager.active
    assert f.manager.trace.rows[-1]['type']=='deadline_drop'


def test_regularized_late_sdk_attempt_requests_pause(controller,monkeypatch):
    from astra_vla.workers.teleop import SendWindowExpired
    f=controller
    monkeypatch.setattr(Policy,'load',lambda:Policy(ik_solver='casadi_ipopt'))
    enter(f)
    def expired():
        f.manager.cycle_writes=1
        raise SendWindowExpired('SDK returned late')
    f.manager.calculate_step=expired
    f.manager.step()
    assert any(method=='MotionCtrl_1' and args==(0,1,0) for _,method,args in f.commands)
    assert not any(method=='JointCtrl' for _,method,_ in f.commands)
    assert f.manager.phase=='pausing' and f.manager.active


def test_regularized_increment_uses_remaining_tracking_budget_without_debt():
    measured=state()
    policy=Policy(ik_solver='casadi_ipopt',translation_m_s=.02)
    integ=Integrator(measured,np.eye(4),policy=policy)
    integ.speed_fraction=1.
    integ.target[0,3]+=.0016
    proposal=integ.propose(measured,keyset(['w']),.05,.065)
    assert proposal is not None
    step=proposal['tcp_target'][0,3]-integ.target[0,3]
    assert 0<step<.0004
    assert np.linalg.norm(proposal['tcp_target'][:3,3]-pose(measured)[:3,3])<.002
    original=integ.target.copy()
    # Proposed/abandoned increments do not move the controller target.
    np.testing.assert_array_equal(integ.target,original)
    integ.target[0,3]+=.001
    assert integ.propose(measured,keyset(['w']),.05,.065) is None


def test_regularized_rotation_tracking_clip_retains_tcp_anchor():
    measured=state()
    tcp=np.eye(4)
    tcp[:3,3]=[.025,0,.145]
    policy=Policy(ik_solver='casadi_ipopt',rotation_deg_s=5)
    integ=Integrator(measured,tcp,policy=policy)
    integ.rotating=True
    integ.speed_fraction=1.
    anchor=integ.target[:3,3].copy()
    integ.target[:3,:3]=Rotation.from_euler('z',.4,degrees=True).as_matrix()@integ.target[:3,:3]
    p=integ.propose(measured,keyset(['j']),.05,.065)
    assert p is not None
    np.testing.assert_array_equal(p['tcp_target'][:3,3],anchor)
    assert np.deg2rad(.4)<rotation_error(pose(measured)@tcp,p['tcp_target'])<np.deg2rad(.5)


@pytest.mark.parametrize('reason', ['teleop joint tracking envelope exhausted',
                                  'teleop IK iteration budget exhausted'])
def test_joint_tracking_saturation_waits_without_rejecting_or_writing(controller,reason):
    f=controller
    enter(f)
    f.manager.step()
    sent=len(f.commands)
    f.geometry_error[0]=ValueError(reason)
    f.manager.step()
    assert f.manager.active and f.manager.phase=='waiting_feedback'
    assert f.manager.blocked_generation is None
    assert not any(method=='JointCtrl' for _,method,_ in f.commands[sent:])


@pytest.mark.parametrize('kind', ['bounded_progress','excess_speed','mode_change'])
def test_feedback_time_bound_requires_fresh_path_and_preserves_fault_guards(controller,monkeypatch,kind):
    f=controller
    monkeypatch.setattr(Policy,'load',lambda:Policy(joint_lead_deg=2))
    enter(f)
    f.manager.step()
    f.commands.clear()
    calls=[]
    def ordinary_guard(arm,previous,latest,stage):
        if max(abs(np.asarray(previous['joints_rad'])-latest['joints_rad']))>np.deg2rad(.5):
            raise RuntimeError('start moved')
        if previous['controller']['move_mode']!=latest['controller']['move_mode']:
            raise RuntimeError('mode changed')
    f.robot.check_motion_unchanged=ordinary_guard
    def calculation_done():
        calls.append(copy.deepcopy(f.states['left']))
        if len(calls)==1:
            f.clock[0]+=.02
            f.states['left']['joints_rad'][0]+=np.deg2rad(1. if kind=='excess_speed' else .536)
            if kind=='mode_change':
                f.states['left']['controller']['move_mode']=2
        assert not f.commands, 'joint command sent before fresh validation completed'
    f.completion[0]=calculation_done
    if kind!='mode_change':
        f.manager.step()
        assert len(calls)==2
        expected=1. if kind=='excess_speed' else .536
        assert calls[1]['joints_rad'][0]==pytest.approx(np.deg2rad(expected))
        assert any(method=='JointCtrl' for _,method,_ in f.commands)
    else:
        with pytest.raises(RuntimeError):
            f.manager.step()
        assert not f.commands


def test_opening_keeps_deadband_allowance_while_closing_can_continue_to_zero():
    measured=state()
    policy=Policy(gripper_opening_lead_m=.003)
    integ=Integrator(measured,np.eye(4),policy=policy)
    start=measured['gripper']['width_m']
    for _ in range(40):
        p=integ.propose(measured,keyset(['e']),.05,.065)
        assert 0<=p['width_m']-measured['gripper']['width_m']<=.003+1e-12
        integ.commit(p)
        # Measured opening did not move with the old <=1mm target budget.
        if p['width_m']-measured['gripper']['width_m']>.0012:
            measured['gripper']['width_m']=p['width_m']-.0012
    assert measured['gripper']['width_m']-start>.005
    integ.reset(measured)
    for _ in range(200):
        p=integ.propose(measured,keyset(['q']),.05,.065)
        integ.commit(p)
    assert p['width_m'] == 0.


@pytest.mark.parametrize('values', [{'gripper_opening_lead_m':.0031},{'gripper_effort':.8001}])
def test_gripper_opening_allowance_and_effort_are_still_bounded(values):
    with pytest.raises(ValueError):
        Policy(**values)


@pytest.mark.parametrize('profile', ['normal', 'fast'])
def test_held_q_stalled_feedback_keeps_closing_to_zero_at_same_speed_and_effort(profile):
    policy = Policy(ik_solver='casadi_ipopt').for_speed(profile)
    measured = state()
    measured['gripper']['width_m'] = .05166
    integrator = Integrator(measured, np.eye(4), policy=policy)
    previous = measured['gripper']['width_m']
    for _ in range(int(12*policy.control_hz)):
        proposal = integrator.propose(measured, keyset(['q']), 1/policy.control_hz, .065)
        assert 0 <= proposal['width_m'] <= previous
        assert previous-proposal['width_m'] <= policy.gripper_m_s/policy.control_hz+1e-12
        integrator.commit(proposal)
        previous = proposal['width_m']
    assert previous == 0. and policy.gripper_effort == .5
    # Release/reversal do not replay the large pending closing error.
    released = integrator.propose(measured, keyset([]), 1/policy.control_hz, .065, keyset(['q']))
    opened = integrator.propose(measured, keyset(['e']), 1/policy.control_hz, .065, keyset(['q']))
    assert released['width_m'] == measured['gripper']['width_m']
    assert opened['width_m'] >= measured['gripper']['width_m']


@pytest.fixture
def regularized_controller(controller, monkeypatch):
    f = controller
    f.robot.config['limits'].update(state_age_s=.3, action_timeout_s=3.)
    monkeypatch.setattr(Policy, 'load', lambda: Policy(ik_solver='casadi_ipopt'))
    monkeypatch.setattr(f.manager, 'token_value', lambda: f.token[0])
    def sleep(seconds):
        f.clock[0] += max(seconds, .001)
        for frame in f.cameras.values():
            frame['stamp'] = f.clock[0]
    f.manager.clock.sleep = sleep
    f.jobs, f.pending = [], {}
    def submit(function, *args):
        f.jobs.append((function.__name__, copy.deepcopy(args)))
        def result(timeout):
            if f.completion[0]:
                f.completion[0]()
            if f.geometry_error[0]:
                raise f.geometry_error[0]
            if function.__name__ in ('geometry_hold', 'geometry_ready'):
                return True
            target, geometry = np.array(args[2]), args[3]
            n = len(f.pending) + 1
            f.pending[n] = target
            return {'reachable': True, 'joint_sdk': [n] + [0] * 5, 'velocity': [0.] * 6,
                    'samples': 11, 'tcp_target': (target @ np.array(geometry['tcp'])).tolist(),
                    'ik': {'solver': 'casadi_ipopt', 'progress_fraction': 1., 'fallback_used': False}}
        return SimpleNamespace(result=result, cancel=lambda: None)
    f.robot.geometry_pool = SimpleNamespace(submit=submit)
    for name in ('left', 'right'):
        previous = f.robot.arms[name].JointCtrl
        def joint(*values, name=name, previous=previous):
            previous(*values)
            if values[0] in f.pending:
                target = f.pending[values[0]]
                f.states[name]['ee_pose'] = {'position_m': target[:3, 3].tolist(),
                    'rpy_deg': Rotation.from_matrix(target[:3, :3]).as_euler('xyz', degrees=True).tolist()}
                f.states[name]['joints_rad'] = np.deg2rad(np.array(values) / 1000).tolist()
        f.robot.arms[name].JointCtrl = joint
    return f


@pytest.mark.parametrize('event', ['progress', 'release', 'moving', 'timeout', 'collision'])
def test_local_inline_recheck_keeps_full_validation_and_bounded_send(regularized_controller, event):
    f = regularized_controller
    session = enter(f)
    f.manager.step()
    f.commands.clear()
    calls = []
    def advance():
        f.states['left']['joints_rad'][0] += np.deg2rad(.06)
        f.manager.clock.sleep(.002)
    f.completion[0] = advance
    def recheck(*args):
        assert args[3]['scope'] == 'tcp_jog'
        calls.append(copy.deepcopy(args))
        if event == 'release':
            f.manager.input({'session': session, 'sequence': 1, 'keys': []})
        elif event == 'moving':
            advance()
        elif event == 'timeout':
            f.manager.clock.sleep(.06)
        elif event == 'collision':
            raise ValueError('fresh self collision')
        return copy.deepcopy(args[-1])
    f.robot.recheck_teleop_target = recheck
    submitted = len([1 for name, _ in f.jobs if name == 'geometry_step'])
    if event == 'collision':
        with pytest.raises(ValueError, match='fresh self collision'):
            f.manager.step()
    else:
        f.manager.step()
    assert len([1 for name, _ in f.jobs if name == 'geometry_step']) == submitted + 1
    assert calls
    if event == 'progress':
        assert len(calls) == 1 and any(method == 'JointCtrl' for _, method, _ in f.commands)
    else:
        assert not any(method == 'JointCtrl' for _, method, _ in f.commands)
    if event == 'moving':
        assert len(calls) > 3  # Moving feedback uses the original time budget, not an attempt cap.
        assert f.manager.deadline_misses == 1
    if event == 'timeout':
        assert f.manager.deadline_misses == 1


@pytest.fixture
def gravity_controller(controller):
    f = controller
    f.states['left']['joints_rad'] = np.deg2rad([0, -2., -.6, 0, 20, 0]).tolist()
    f.states['left']['controller'].update(ctrl_mode=0, motors_enabled=[False]*6)
    f.traces, f.plans, f.target, f.replans = [], [], None, []
    def trace_factory(*args):
        t = MemoryTrace(*args)
        f.traces.append(t)
        return t
    f.manager.trace_factory = trace_factory
    previous_joint = f.robot.arms['left'].JointCtrl
    def joint(*values):
        previous_joint(*values)
        f.target = np.deg2rad(np.array(values)/1000)
    f.robot.arms['left'].JointCtrl = joint
    def submit(function, *args):
        def result(timeout):
            if function.__name__ == 'geometry_hold':
                if args[0]['joints_rad'][1] < 0:
                    raise ValueError('measured pose has URDF self collision')
                return True
            if function.__name__ in ('geometry_recovery_rebind', 'geometry_recovery_replan'):
                plan = copy.deepcopy(args[4] if function.__name__ == 'geometry_recovery_replan' else args[-1])
                start = np.array(args[0]['joints_rad'])
                if function.__name__ == 'geometry_recovery_replan':
                    f.replans.append(copy.deepcopy(args))
                    plan['goal_joints_rad'][3] = float(start[3])
                    plan['joint_sdk'][3] = round(np.rad2deg(start[3])*1000)
                    plan['goal_joints_rad'][3] = float(np.deg2rad(plan['joint_sdk'][3]/1000))
                plan['path_joints_rad'] = np.linspace(start, plan['goal_joints_rad'], 351).tolist()
                plan['initial_excess_rad'] = np.maximum(np.array(plan['lower_rad'])-start, 0).tolist()
                return plan
            if function.__name__ != 'geometry_recovery_plan':
                return True
            before = np.array(args[0]['joints_rad'])
            goal = before.copy()
            goal[1:3] = np.deg2rad([.5, -3.])
            lower = np.deg2rad([-150, 0, -170, -100, -70, -120])
            upper = np.deg2rad([150, 180, 0, 100, 70, 120])
            plan = {'required': True, 'candidate': 'folded_elbow_clearance',
                    'enabled_start': args[0]['controller']['ctrl_mode'] == 1 and
                                     all(args[0]['controller']['motors_enabled']),
                    'joint_sdk': np.rint(np.rad2deg(goal)*1000).astype(int).tolist(),
                    'goal_joints_rad': goal.tolist(), 'start_joints_rad': before.tolist(),
                    'lower_rad': lower.tolist(), 'upper_rad': upper.tolist(),
                    'initial_excess_rad': np.maximum(lower-before, 0).tolist(),
                    'path_joints_rad': np.linspace(before, goal, 351).tolist(),
                    'local_collision_check': False, 'samples': 1053}
            f.plans.append(plan)
            if f.completion[0]:
                f.completion[0]()
            return plan
        return SimpleNamespace(result=result)
    f.robot.geometry_pool = SimpleNamespace(submit=submit)
    f.after_tick = None
    def sleep(seconds):
        f.clock[0] += seconds
        for camera in f.cameras.values():
            camera['stamp'] = f.clock[0]
        if f.target is not None and all(f.states['left']['controller']['motors_enabled']):
            current = np.array(f.states['left']['joints_rad'])
            delta = f.target-current
            scale = min(1., np.deg2rad(.5)/max(np.max(np.abs(delta)), 1e-12))
            f.states['left']['joints_rad'] = (current+delta*scale).tolist()
        if f.after_tick:
            f.after_tick()
    f.manager.clock.sleep = sleep
    def no_collision(*args):
        raise AssertionError('recovery queried local collision feedback')
    f.robot.check_gravity_recovery_feedback = no_collision
    def valid_feedback(s):
        if s['age_s'] > .3:
            raise RuntimeError('stale feedback')
    f.robot.check_motion_state = valid_feedback
    return f


def test_gravity_entry_recovers_selected_arm_then_rechecks_strict_hold(gravity_controller):
    f = gravity_controller
    view = f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
    assert view['active'] and view['gravity_recovery']['phase'] == 'completed'
    assert len(f.plans) == 1
    assert any(row['type'] == 'feedback' for row in f.traces[0].rows)
    assert all(arm == 'left' for arm, _, _ in f.commands)
    targets = [args for _, method, args in f.commands if method == 'JointCtrl']
    assert len(targets) >= 3 and all(t[1:3] == (500, -3000) for t in targets)
    assert ('left', 'MotionCtrl_2', (1, 1, 1, 0)) in f.commands
    assert not any(m in ('GripperCtrl', 'ResetPiper', 'DisableArm') for _, m, _ in f.commands)
    assert f.traces[0].context['owner'] == 'gravity_recovery'
    assert f.traces[0].result['recovery_completed']
    assert f.manager.integrator.target is not None


def test_legacy_client_cannot_start_gravity_recovery_without_cancellation(gravity_controller):
    f = gravity_controller
    with pytest.raises(RuntimeError, match='updated desktop cancellation'):
        f.manager.open({'arm': 'left'})
    assert not f.plans and not f.commands and not f.robot.lock.locked()


def test_recovery_completion_still_requires_ordinary_collision_hold(gravity_controller):
    f = gravity_controller
    submit = f.robot.geometry_pool.submit
    holds = []
    def reject_hold(function, *args):
        if function.__name__ == 'geometry_hold':
            holds.append(copy.deepcopy(args[0]))
            def result(timeout):
                raise ValueError('measured pose has URDF self collision')
            return SimpleNamespace(result=result)
        return submit(function, *args)
    f.robot.geometry_pool.submit = reject_hold
    with pytest.raises(ValueError, match='URDF self collision'):
        f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
    assert len(holds) == 2 and holds[0]['joints_rad'][1] < 0 <= holds[1]['joints_rad'][1]
    assert f.traces[0].result['recovery_completed']
    assert not f.manager.active and not f.robot.lock.locked()


@pytest.mark.parametrize('cause', ['stop', 'start_moved', 'configuration', 'mode', 'stop_epoch', 'bad_plan'])
def test_gravity_recovery_invalidated_before_write_sends_nothing(gravity_controller, cause):
    f = gravity_controller
    def change():
        if cause == 'stop':
            f.robot.stopped.set()
        elif cause == 'start_moved':
            f.states['left']['joints_rad'][1] += np.deg2rad(.06)
        elif cause == 'configuration':
            f.token[0] = 'changed'
        elif cause == 'stop_epoch':
            f.robot.stop_epoch = 1
        elif cause == 'bad_plan':
            f.plans[-1]['joint_sdk'][1] = -100
        else:
            f.states['left']['controller']['ctrl_mode'] = 2
    f.completion[0] = change
    with pytest.raises((RuntimeError, ValueError)):
        f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
    assert not f.commands and not f.robot.lock.locked()


@pytest.mark.parametrize('cause', ['stop', 'camera', 'other_arm', 'excess', 'mode', 'timeout',
                                 'stale_feedback', 'gripper'])
def test_gravity_recovery_failure_pauses_without_reset_or_old_target(gravity_controller, cause):
    f = gravity_controller
    def change():
        if cause == 'stop':
            f.robot.stopped.set()
        elif cause == 'camera':
            f.cameras['left_wrist']['stamp'] -= 1
        elif cause == 'other_arm':
            f.states['right']['joints_rad'][0] += np.deg2rad(.6)
        elif cause == 'excess':
            f.states['left']['joints_rad'][1] = np.deg2rad(-2.1)
        elif cause == 'mode':
            f.states['left']['controller']['ctrl_mode'] = 2
        elif cause == 'timeout':
            f.states['left']['joints_rad'][1:3] = np.deg2rad([-2., -.6]).tolist()
        elif cause == 'stale_feedback':
            f.states['left']['age_s'] = .4
        elif cause == 'gripper':
            f.states['left']['gripper']['width_m'] += .001
    f.after_tick = change
    with pytest.raises((RuntimeError, ValueError)):
        f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
    assert f.robot.stopped.is_set() and not f.robot.lock.locked()
    assert f.manager.recovery['phase'] == 'failed'
    assert len([1 for _, m, _ in f.commands if m == 'JointCtrl']) >= 2
    assert not any(m in ('GripperCtrl', 'ResetPiper', 'DisableArm') for _, m, _ in f.commands)
    assert f.commands[-1][1] == 'MotionCtrl_1'


def test_unverified_jaw_does_not_block_arm_entry_or_receive_any_gripper_command(controller):
    f = controller
    f.states['left']['gripper'].update(width_m=-.0021, enabled=False)
    f.states['left']['controller'].update(ctrl_mode=0, motors_enabled=[False]*6)
    seen = []
    def check(a, mode, *, arm_only=False):
        seen.append(arm_only)
        return f.robot.healthy(a, False)
    f.robot.check_teleop_prepare = check
    opened = f.manager.open({'arm': 'left'})
    assert opened['active'] and opened['phase'] == 'ready'
    assert not opened['gripper_control_available'] and 'Q/E' in opened['gripper_notice']
    assert all(seen)
    assert f.manager.geometry['unverified_gripper_max_m'] == .065
    assert any(method == 'EnableArm' for _, method, _ in f.commands)
    assert not any(method == 'GripperCtrl' for _, method, _ in f.commands)
    f.manager.input({'session': opened['session'], 'sequence': 0, 'keys': ['space', 'q', 'e']})
    assert f.manager.keys == frozenset(['space'])
    f.manager.step()
    assert any(method == 'JointCtrl' for _, method, _ in f.commands)
    assert not any(method == 'GripperCtrl' for _, method, _ in f.commands)
    assert f.states['left']['gripper']['width_m'] == -.0021


@pytest.mark.parametrize('change', ['enabled', 'width'])
def test_quarantined_gripper_change_stops_control_before_new_target(controller, change):
    f = controller
    f.states['left']['gripper'].update(width_m=-.0021, enabled=False)
    opened = f.manager.open({'arm': 'left'})
    f.manager.input({'session': opened['session'], 'sequence': 0, 'keys': ['space']})
    f.commands.clear()
    if change == 'enabled':
        f.states['left']['gripper']['enabled'] = True
    else:
        f.states['left']['gripper']['width_m'] += .0003
    with pytest.raises(RuntimeError, match='unverified gripper changed'):
        f.manager.step()
    assert not any(method in ('JointCtrl', 'GripperCtrl') for _, method, _ in f.commands)


def test_negative_other_disabled_gripper_does_not_block_selected_arm_watchdog(controller):
    f = controller
    f.states['right']['gripper'].update(width_m=-.0021, enabled=False)
    opened = f.manager.open({'arm': 'left'})
    assert opened['gripper_control_available']
    f.manager.check_watchdog()
    assert f.manager.active


@pytest.mark.parametrize('sag,success', [(.1, True), (.3, False)])
def test_gravity_waits_for_all_motors_and_revalidates_after_bounded_enable_sag(gravity_controller, sag, success):
    f = gravity_controller
    pending, ticks, resumed = [None], [0], []
    joint = f.robot.arms['left'].JointCtrl
    motion = f.robot.arms['left'].MotionCtrl_1
    def install(*values):
        joint(*values)
        pending[0] = np.deg2rad(np.array(values)/1000)
        f.target = pending[0] if resumed else None
    def enable(*values):
        f.commands.append(('left', 'EnableArm', values))
    def resume(*values):
        if values == (0, 2, 0):
            assert all(f.states['left']['controller']['motors_enabled'])
            assert ticks[0] >= 2
            resumed.append(True)
            f.target = pending[0]
        motion(*values)
    def advance():
        ticks[0] += 1
        if ticks[0] == 1:
            f.states['left']['joints_rad'][1] -= np.deg2rad(sag)
        if ticks[0] == 2:
            f.states['left']['controller']['motors_enabled'] = [True]*6
    f.robot.arms['left'].JointCtrl = install
    f.robot.arms['left'].EnableArm = enable
    f.robot.arms['left'].MotionCtrl_1 = resume
    f.after_tick = advance
    if success:
        assert f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})['active']
        assert resumed
    else:
        with pytest.raises(RuntimeError, match='increased limit excess'):
            f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
        assert not resumed


@pytest.mark.parametrize('wrist_shift,success', [(-.518, True), (-.751, True), (-1.4, True), (-1.6, False)])
def test_enable_wrist_shift_replans_fresh_feedback_without_settling_angle_limit(gravity_controller, wrist_shift, success):
    f = gravity_controller
    pending, resumed, changed = [None], [], [False]
    joint, motion = f.robot.arms['left'].JointCtrl, f.robot.arms['left'].MotionCtrl_1
    def install(*values):
        joint(*values)
        pending[0] = np.deg2rad(np.array(values)/1000)
        f.target = pending[0] if resumed else None
    def resume(*values):
        if values == (0, 2, 0):
            resumed.append(True)
            assert f.replans
            assert f.replans[-1][0]['joints_rad'][3] == pytest.approx(np.deg2rad(wrist_shift))
            assert pending[0][3] == pytest.approx(np.deg2rad(wrist_shift))
            clear = max(i for i, (_, method, args) in enumerate(f.commands)
                        if method == 'MotionCtrl_1' and args == (0, 4, 0))
            enable = max(i for i, (_, method, _) in enumerate(f.commands) if method == 'EnableArm')
            assert clear > enable
            assert all(args[3] == round(wrist_shift*1000)
                       for _, method, args in f.commands[clear+1:] if method == 'JointCtrl')
            f.target = pending[0]
        motion(*values)
    def settle():
        if not changed[0]:
            f.states['left']['joints_rad'][3] += np.deg2rad(wrist_shift)
            changed[0] = True
    f.robot.arms['left'].JointCtrl, f.robot.arms['left'].MotionCtrl_1 = install, resume
    f.after_tick = settle
    if success:
        assert f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})['active']
        assert resumed and f.replans[-1][-1][3] == 0  # Original feedback is retained for records.
    else:
        with pytest.raises(RuntimeError, match='feedback speed exceeds bound'):
            f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
        assert not resumed and not f.replans


def test_stop_during_enable_replanning_sends_no_replanned_target(gravity_controller):
    f = gravity_controller
    submit = f.robot.geometry_pool.submit
    at_stop = []
    def interrupted(function, *args):
        future = submit(function, *args)
        if function.__name__ != 'geometry_recovery_replan':
            return future
        def result(timeout):
            plan = future.result(timeout)
            at_stop.append(len([1 for _, method, _ in f.commands if method == 'JointCtrl']))
            f.robot.stopped.set()
            return plan
        return SimpleNamespace(result=result)
    f.robot.geometry_pool.submit = interrupted
    with pytest.raises(RuntimeError, match='stopped during gravity recovery'):
        f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
    assert at_stop and len([1 for _, method, _ in f.commands if method == 'JointCtrl']) == at_stop[0]
    assert not any(method == 'MotionCtrl_1' and args == (0, 2, 0) for _, method, args in f.commands)


def test_enabled_validation_wait_checks_stop_between_future_polls(gravity_controller):
    f = gravity_controller
    submit = f.robot.geometry_pool.submit
    targets_at_stop = []
    def waiting(function, *args):
        future = submit(function, *args)
        if function.__name__ != 'geometry_recovery_replan':
            return future
        def result(timeout):
            assert timeout <= .05
            targets_at_stop.append(len([1 for _, method, _ in f.commands if method == 'JointCtrl']))
            f.robot.stopped.set()
            raise TimeoutError()
        return SimpleNamespace(result=result)
    f.robot.geometry_pool.submit = waiting
    with pytest.raises(RuntimeError, match='stopped during gravity recovery'):
        f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
    assert len(targets_at_stop) == 1
    assert len([1 for _, method, _ in f.commands if method == 'JointCtrl']) == targets_at_stop[0]
    assert not any(method == 'MotionCtrl_1' and args == (0, 2, 0) for _, method, args in f.commands)


@pytest.mark.parametrize('event', ['settles', 'settled_at_pause', 'stop_in_replan',
                                  'stop_after_clear', 'never_settles',
                                  'pause_unconfirmed', 'stop_during_pause'])
def test_safe_following_error_replans_and_retains_original_feedback(gravity_controller, event):
    f = gravity_controller
    moving, changed = [False], [False]
    motion = f.robot.arms['left'].MotionCtrl_1
    def mode(*args):
        if args == (0, 2, 0):
            moving[0] = True
        elif args == (0, 1, 0):
            moving[0] = False
        motion(*args)
    def drift():
        if moving[0] and (event == 'never_settles' or not changed[0]):
            f.states['left']['joints_rad'][4] += np.deg2rad(.592)
            changed[0] = True
    f.robot.arms['left'].MotionCtrl_1 = mode
    f.after_tick = drift
    stopped_at = []
    if event in ('pause_unconfirmed', 'stop_during_pause'):
        def pause_confirmation(**kwargs):
            check = kwargs.get('feedback_check')
            if check is None:
                return {'confirmed': True}
            assert kwargs['timeout_s'] == 5. and kwargs['joint_tolerance_deg'] == .15
            assert f.commands[-1][1:] == ('MotionCtrl_1', (0, 4, 0))
            stopped_at.append(len([1 for _, name, _ in f.commands if name == 'JointCtrl']))
            if event == 'stop_during_pause':
                f.robot.stopped.set()
            check()
            return {'confirmed': False, 'reason': 'joint feedback still moving'}
        f.robot.stationary_feedback = pause_confirmation
    if event in ('settled_at_pause', 'stop_after_clear'):
        submit = f.robot.geometry_pool.submit
        def clear_after_pause(function, *args):
            future = submit(function, *args)
            if function.__name__ != 'geometry_recovery_plan':
                return future
            def result(timeout):
                plan = future.result(timeout)
                if len(f.plans) > 1:
                    if event == 'stop_after_clear':
                        stopped_at.append(len([1 for _, name, _ in f.commands if name == 'JointCtrl']))
                        f.robot.stopped.set()
                    return {'required': False}
                return plan
            return SimpleNamespace(result=result)
        f.robot.geometry_pool.submit = clear_after_pause
    if event == 'stop_in_replan':
        def stop():
            if len(f.plans) == 2:
                stopped_at.append(len([1 for _, name, _ in f.commands if name == 'JointCtrl']))
                f.robot.stopped.set()
        f.completion[0] = stop
    if event in ('settles', 'settled_at_pause'):
        view = f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
        assert view['active'] and view['gravity_recovery']['phase'] == 'completed'
        traces = [trace for trace in f.traces if trace.context.get('owner') == 'gravity_recovery']
        assert len(traces) == (2 if event == 'settles' else 1)
        assert all(trace.result['recovery_completed'] for trace in traces)
        origins = [trace.context['recovery_origin_joints_rad'] for trace in traces]
        assert all(origin == origins[0] for origin in origins)
        if event == 'settles':
            assert traces[1].context['path_attempt'] == 1
        assert f.plans[-1]['start_joints_rad'][4] > origins[0][4]
    else:
        with pytest.raises((RuntimeError, ValueError)):
            f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
        if event in ('stop_in_replan', 'stop_after_clear', 'pause_unconfirmed', 'stop_during_pause'):
            assert stopped_at and len([1 for _, name, _ in f.commands if name == 'JointCtrl']) == stopped_at[0]
        if event == 'never_settles':
            assert len(f.traces) == 3 and 'did not converge' in f.traces[0].result['failure_reason']
    assert not any(name in ('ResetPiper', 'DisableArm', 'GripperCtrl') for _, name, _ in f.commands)


def test_gravity_stop_between_refresh_mode_and_target_cannot_send_old_goal(gravity_controller):
    f = gravity_controller
    mode = f.robot.arms['left'].MotionCtrl_2
    count = [0]
    targets_at_stop = []
    def refresh(*args):
        mode(*args)
        count[0] += 1
        if count[0] == 3:
            targets_at_stop.append(len([1 for _, name, _ in f.commands if name == 'JointCtrl']))
            f.robot.stopped.set()
    f.robot.arms['left'].MotionCtrl_2 = refresh
    with pytest.raises(RuntimeError, match='stopped during gravity recovery'):
        f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
    assert targets_at_stop
    assert len([1 for _, name, _ in f.commands if name == 'JointCtrl']) == targets_at_stop[0]
    assert f.manager.recovery['phase'] == 'failed'


def test_manual_recovery_uses_normal_other_arm_guard_for_parked_servo_jitter(gravity_controller):
    f = gravity_controller
    changed = [False]
    def jitter():
        if not changed[0]:
            f.states['right']['joints_rad'][3] += np.deg2rad(.1)
            changed[0] = True
    f.after_tick = jitter
    view = f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
    assert view['active'] and view['gravity_recovery']['phase'] == 'completed'


def test_ordinary_enabled_manual_sag_recovery_revalidates_without_reset_or_reenable(gravity_controller):
    f = gravity_controller
    f.states['left']['controller'].update(ctrl_mode=1, move_mode=1, motors_enabled=[True]*6)
    view = f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
    assert view['active'] and view['gravity_recovery']['phase'] == 'completed'
    assert not any(name in ('EnableArm', 'DisableArm', 'ResetPiper') for _, name, _ in f.commands)
    assert f.plans[0]['enabled_start']


def test_ordinary_mode_and_speed_are_refreshed_before_each_continuous_goal(controller):
    f = controller
    enter(f, ('w',))
    f.manager.step()
    assert not f.manager.paused
    f.commands.clear()
    f.clock[0] += .05
    f.manager.step()
    assert [name for _, name, _ in f.commands] == ['MotionCtrl_2', 'JointCtrl']
    assert f.commands[0] == ('left', 'MotionCtrl_2', (1, 1, 5, 0))


def test_in_limit_folded_collision_invokes_bounded_recovery_before_hold(gravity_controller):
    f = gravity_controller
    f.states['left']['joints_rad'][1] = np.deg2rad(.443)
    f.states['left']['controller'].update(ctrl_mode=1, move_mode=1, motors_enabled=[True]*6)
    previous_submit = f.robot.geometry_pool.submit
    calls = []
    def submit(function, *args):
        calls.append(function.__name__)
        if function.__name__ == 'geometry_hold' and calls.count('geometry_hold') == 1:
            def result(timeout):
                raise ValueError('measured pose has URDF self collision')
            return SimpleNamespace(result=result)
        return previous_submit(function, *args)
    f.robot.geometry_pool.submit = submit
    view = f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
    assert view['active'] and view['gravity_recovery']['phase'] == 'completed'
    assert calls.count('geometry_recovery_plan') == 1 and calls.count('geometry_hold') == 2
    assert not any(method in ('EnableArm', 'DisableArm', 'ResetPiper') for _, method, _ in f.commands)


def test_stop_during_ordinary_mode_refresh_prevents_following_target(controller):
    f = controller
    enter(f, ('w',))
    f.manager.step()
    f.commands.clear()
    previous = f.robot.arms['left'].MotionCtrl_2
    def refresh(*args):
        previous(*args)
        f.robot.stopped.set()
    f.robot.arms['left'].MotionCtrl_2 = refresh
    with pytest.raises(RuntimeError, match='execution stopped before teleop write'):
        f.manager.step()
    assert [name for _, name, _ in f.commands] == ['MotionCtrl_2']


@pytest.mark.parametrize('profile,hz,speed,lead', [('slow', 20, .005, .002),
    ('normal', 20, .02, .002), ('fast', 50, .1, .008)])
def test_manual_profile_limits_preserve_joint_motion_guards(profile, hz, speed, lead):
    base = Policy(ik_solver='casadi_ipopt', translation_m_s=.02,
                  joint_acceleration_deg_s2=120, joint_lead_deg=2)
    p = base.for_speed(profile)
    assert (p.control_hz, p.translation_m_s, p.position_lead_m) == (hz, speed, lead)
    assert p.joint_acceleration_deg_s2 == 120
    assert p.tcp_drift_m == .002 and p.rotation_lead_deg == .5
    assert p.geometry_timeout_s < 1/p.control_hz


@pytest.mark.parametrize('values', [{'position_lead_m': .008},
    {'control_hz': 50, 'geometry_timeout_s': .018}, {'joint_lead_deg': 6}])
def test_normal_profile_cannot_borrow_fast_following_budget(values):
    with pytest.raises(ValueError):
        Policy(**values)


def test_switching_speed_drops_held_keys_and_waits_for_quiet_before_changing_limits(controller, monkeypatch):
    f = controller
    monkeypatch.setattr(Policy, 'load', lambda: Policy(ik_solver='casadi_ipopt'))
    opened = f.manager.open({'arm': 'left'})
    f.manager.input({'session': opened['session'], 'sequence': 0, 'keys': ['w']})
    f.manager.input({'session': opened['session'], 'sequence': 1, 'keys': ['w'], 'speed_profile': 'fast'})
    assert not f.manager.keys
    assert f.manager.policy.speed_profile == 'normal'
    assert f.manager.pending_speed_profile == 'fast' and f.manager.phase == 'pausing'
    f.manager.step()
    assert f.manager.policy.speed_profile == 'normal'
    f.clock[0] += .51
    for frame in f.cameras.values():
        frame['stamp'] = f.clock[0]
    f.manager.last_input = f.clock[0]
    f.manager.step()
    assert f.manager.policy.speed_profile == 'fast'
    assert f.manager.policy.control_hz == 50
    assert not f.manager.keys and f.manager.phase == 'ready'
    assert not any(method == 'JointCtrl' for _, method, _ in f.commands)


@pytest.mark.parametrize('profile', ['slow', 'normal', 'fast'])
@pytest.mark.parametrize('drift', [.00218, .0101, .05])
def test_release_displacement_is_recorded_without_distance_limit(controller, monkeypatch, profile, drift):
    f = controller
    monkeypatch.setattr(Policy, 'load', lambda: Policy(ik_solver='casadi_ipopt'))
    opened = f.manager.open({'arm': 'left', 'speed_profile': profile})
    f.manager.input({'session': opened['session'], 'sequence': 0, 'keys': ['w']})
    f.manager.input({'session': opened['session'], 'sequence': 1, 'keys': []})
    f.states['left']['ee_pose']['position_m'][0] += drift
    f.manager.step()
    assert f.manager.active and not f.robot.stopped.is_set()
    assert f.manager.view()['release_extra_mm'] == pytest.approx(drift*1000)
    assert 'release_extra_m' not in f.manager.view()['policy']
    assert not any(method == 'JointCtrl' for _, method, _ in f.commands)


@pytest.mark.parametrize('field,value', [('input_timeout_s', .101), ('geometry_timeout_s', .019),
    ('joint_lead_deg', 6.01), ('position_lead_m', .0081),
    ('joint_acceleration_deg_s2', 120.1)])
def test_fast_profile_cannot_exceed_declared_motion_and_stop_caps(field, value):
    from dataclasses import replace
    fast = Policy(ik_solver='casadi_ipopt').for_speed('fast')
    with pytest.raises(ValueError):
        replace(fast, **{field: value})


def test_speed_config_change_invalidates_session_token(tmp_path, monkeypatch):
    from astra_vla.workers import teleop
    import astra_vla.teleop as policy_module
    folder = tmp_path/'config'
    folder.mkdir()
    (folder/'devices.yaml').write_text('urdf: model.urdf\n')
    (tmp_path/'model.urdf').write_text('model')
    monkeypatch.setattr(teleop, 'ROOT', tmp_path)
    monkeypatch.setattr(policy_module, 'ROOT', tmp_path)
    monkeypatch.setattr(teleop, 'settings', lambda: {'urdf': str(tmp_path/'model.urdf')})
    token = teleop.configuration_token()
    path = folder/'teleop_speeds.json'
    path.write_text(json.dumps({'slow': {}, 'normal': {}, 'fast': {'translation_m_s': .08}}))
    assert token != teleop.configuration_token()
    assert Policy(ik_solver='casadi_ipopt').for_speed('fast').translation_m_s == .08
    path.write_text(json.dumps({'slow': {}, 'normal': {}, 'fast': {'release_extra_m': .02}}))
    with pytest.raises(ValueError):
        Policy(ik_solver='casadi_ipopt').for_speed('fast')


def test_fast_worker_sends_ordinary_mode_with_validated_goal_under_twenty_ms_period(controller, monkeypatch):
    f = controller
    monkeypatch.setattr(Policy, 'load', lambda: Policy(ik_solver='casadi_ipopt', speed_percent=20))
    opened = f.manager.open({'arm': 'left', 'speed_profile': 'fast'})
    submitted = []
    def submit(function, *args):
        submitted.append(args)
        def result(**kwargs):
            tcp_target = np.asarray(args[2]) @ np.asarray(args[3]['tcp'])
            return {'reachable': True, 'joint_sdk': [0]*6, 'velocity': [0.]*6, 'samples': 11,
                    'tcp_target': tcp_target.tolist(), 'ik': {'solver': 'casadi_ipopt', 'progress_fraction': 1.}}
        return SimpleNamespace(result=result, cancel=lambda: None)
    f.robot.geometry_pool.submit = submit
    for seq in range(2):
        f.manager.input({'session': opened['session'], 'sequence': seq, 'keys': ['w'], 'speed_profile': 'fast'})
        f.manager.step()
        f.clock[0] += .02
    assert all(args[-3]['control_hz'] == 50 and args[-3]['geometry_timeout_s'] == .018 for args in submitted)
    assert len([1 for _, name, _ in f.commands if name == 'JointCtrl']) == 3
    assert [args for _, name, args in f.commands if name == 'MotionCtrl_2'] == [(1, 1, 20, 0)]*2
    assert f.manager.view()['target_hz'] == 2
    f.clock[0] += 1.01
    assert f.manager.view()['target_hz'] == 0


def test_fast_heartbeat_expires_at_one_hundred_ms_before_any_new_target(controller, monkeypatch):
    f = controller
    monkeypatch.setattr(Policy, 'load', lambda: Policy(ik_solver='casadi_ipopt'))
    f.manager.open({'arm': 'left', 'speed_profile': 'fast'})
    f.clock[0] += .101
    with pytest.raises(RuntimeError, match='heartbeat'):
        f.manager.step()
    assert not f.commands


def test_new_input_during_quiet_check_cannot_reset_trajectory_or_apply_stale_profile(controller):
    f = controller
    enter(f)
    f.manager.pause(f.manager.generation)
    f.commands.clear()
    def quiet(state):
        f.manager.generation += 1
        return True
    def stale_reset(state):
        raise AssertionError('stale control frame reset the new input trajectory')
    f.manager.quiet_feedback = quiet
    f.manager.integrator.reset = stale_reset
    f.manager.step()
    assert f.manager.phase == 'pausing' and not f.commands


@pytest.mark.parametrize('cancel_second_stage', [False, True])
def test_recovery_executes_validated_stages_in_order_and_stop_cancels_later_targets(gravity_controller, cancel_second_stage):
    f = gravity_controller
    stages = [[0, 500, -700, 0, 20000, 0], [0, 500, -700, 0, 18500, 0], [0, 500, -3000, 0, 18500, 0]]
    submit = f.robot.geometry_pool.submit
    def staged(function, *args):
        future = submit(function, *args)
        if function.__name__ not in ('geometry_recovery_plan', 'geometry_recovery_rebind', 'geometry_recovery_replan'):
            return future
        def result(timeout):
            plan = future.result(timeout)
            plan['joint_sdk'] = stages[-1]
            plan['goal_joints_rad'] = np.deg2rad(np.array(stages[-1])/1000).tolist()
            plan['stage_joint_sdk'] = copy.deepcopy(stages)
            previous = np.array(args[0]['joints_rad'])
            path = []
            for row in stages:
                goal = np.deg2rad(np.array(row)/1000)
                path.extend(np.linspace(previous, goal, 351)[bool(path):].tolist())
                previous = goal
            plan['path_joints_rad'] = path
            return plan
        return SimpleNamespace(result=result)
    f.robot.geometry_pool.submit = staged
    joint = f.robot.arms['left'].JointCtrl
    seen = []
    def send(*values):
        joint(*values)
        if not seen or list(values) != seen[-1]:
            seen.append(list(values))
        if cancel_second_stage and list(values) == stages[1]:
            f.robot.stopped.set()
    f.robot.arms['left'].JointCtrl = send
    if cancel_second_stage:
        with pytest.raises(RuntimeError, match='stopped during gravity recovery'):
            f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
        assert seen == stages[:2]
        assert not f.traces[0].result['recovery_completed']
        assert f.traces[0].result['failure_reason']
    else:
        view = f.manager.open({'arm': 'left', 'preparation_cancel_supported': True})
        assert view['active'] and seen == stages
        assert view['gravity_recovery']['stage'] == 3
        assert f.traces[0].result['recovery_completed']


@pytest.mark.parametrize('event', ['complete', 'passive_moved', 'unexpected_enable',
                                  'lost_wrist', 'stop_between_stages', 'pause_unconfirmed'])
def test_selective_wrist_preparation_never_energizes_shoulder_before_revalidation(
        gravity_controller, monkeypatch, event):
    from astra_vla.gravity_recovery import RecoveryPolicy
    f = gravity_controller
    cold, parked = copy.deepcopy(f.states['left']), copy.deepcopy(f.states['right'])
    cold['arm'] = 'right'
    f.states.update(right=cold, left=parked)
    monkeypatch.setattr(RecoveryPolicy, 'load', classmethod(lambda _: RecoveryPolicy(selective_wrist_prepare=True)))
    origin = np.array(cold['joints_rad'])
    goal = origin.copy()
    goal[[1, 2, 3, 4]] = np.deg2rad([.5, -3., -3.25, 18.])
    first = goal.copy()
    first[2] = origin[2]
    body_stages = [np.rint(np.rad2deg(q)*1000).astype(int).tolist() for q in (first, goal)]
    preparation = []
    for axis, j4, j5 in ((4, .5, 20.), (5, .5, 19.5), (4, -3.25, 19.5), (5, -3.25, 18.)):
        expected = origin.copy()
        expected[3:5] = np.deg2rad([j4, j5])
        command = np.array(body_stages[0])
        command[3:5] = np.rint([j4*1000, j5*1000]).astype(int)
        preparation.append({'axis': axis, 'joint_sdk': command.tolist(), 'expected_joints_rad': expected.tolist()})
    def path(start, rows):
        result = [start]
        for row in rows:
            result.extend(np.linspace(result[-1], row, 101)[1:])
        return np.asarray(result).tolist()
    template = {'required': True, 'candidate': 'folded_coupled_wrist_fixture', 'enabled_start': False,
        'joint_sdk': body_stages[-1], 'stage_joint_sdk': body_stages,
        'goal_joints_rad': goal.tolist(), 'start_joints_rad': origin.tolist(),
        'lower_rad': np.deg2rad([-150, 0, -170, -100, -70, -120]).tolist(),
        'upper_rad': np.deg2rad([150, 180, 0, 100, 70, 120]).tolist(),
        'initial_excess_rad': np.maximum(np.deg2rad([0, 0, -170, -100, -70, -120])-origin, 0).tolist(),
        'wrist_preparation': preparation,
        'path_joints_rad': path(origin, [p['expected_joints_rad'] for p in preparation]+[first, goal])}
    rebound_calls, target, phase, stopped_at = [], [None], [0], []
    def submit(function, *args):
        def result(timeout):
            if function.__name__ == 'geometry_hold':
                if args[0]['joints_rad'][1] < 0:
                    raise ValueError('measured pose has URDF self collision')
                return True
            if function.__name__ == 'geometry_recovery_plan':
                return copy.deepcopy(template)
            if function.__name__ in ('geometry_recovery_rebind', 'geometry_recovery_replan'):
                rebound_calls.append(copy.deepcopy(args[0]))
                plan = copy.deepcopy(args[-1] if function.__name__ == 'geometry_recovery_rebind' else args[4])
                start = np.array(args[0]['joints_rad'])
                plan['start_joints_rad'] = start.tolist()
                plan['path_joints_rad'] = path(start, [np.deg2rad(np.array(row)/1000) for row in body_stages])
                plan['initial_excess_rad'] = np.maximum(np.array(plan['lower_rad'])-start, 0).tolist()
                return plan
            return True
        return SimpleNamespace(result=result)
    f.robot.geometry_pool.submit = submit
    def enable(axis):
        f.commands.append(('right', 'EnableArm', (axis,)))
        if axis == 7:
            assert rebound_calls and all(x['controller']['motors_enabled'][3:5] == [True, True] for x in rebound_calls)
            assert all(not x['controller']['motors_enabled'][1] for x in rebound_calls)
            f.states['right']['controller']['motors_enabled'] = [True]*6
        else:
            assert axis in (4, 5)
            f.states['right']['controller']['motors_enabled'][axis-1] = True
            phase[0] += 1
            if event == 'unexpected_enable':
                f.states['right']['controller']['motors_enabled'][1] = True
            if event == 'lost_wrist' and phase[0] == 2:
                f.states['right']['controller']['motors_enabled'][3] = False
            if event == 'stop_between_stages' and phase[0] == 2:
                stopped_at.append(len(f.commands))
                f.robot.stopped.set()
    def joint(*args):
        f.commands.append(('right', 'JointCtrl', args))
        target[0] = np.deg2rad(np.array(args)/1000)
    f.robot.arms['right'].EnableArm, f.robot.arms['right'].JointCtrl = enable, joint
    def sleep(seconds):
        f.clock[0] += seconds
        for camera in f.cameras.values():
            camera['stamp'] = f.clock[0]
        if target[0] is not None:
            enabled = f.states['right']['controller']['motors_enabled']
            q = np.array(f.states['right']['joints_rad'])
            change = np.clip(target[0]-q, -np.deg2rad(.5), np.deg2rad(.5))
            f.states['right']['joints_rad'] = (q+np.array(enabled)*change).tolist()
        if event == 'passive_moved' and phase[0]:
            f.states['right']['joints_rad'][1] += np.deg2rad(.16)
    f.manager.clock.sleep = sleep
    if event == 'pause_unconfirmed':
        f.robot.stationary_feedback = lambda **kw: {'confirmed': kw.get('feedback_check') is None}
    if event == 'complete':
        view = f.manager.open({'arm': 'right', 'preparation_cancel_supported': True})
        assert view['active'] and view['gravity_recovery']['phase'] == 'completed'
        assert any(name == 'EnableArm' and args == (7,) for _, name, args in f.commands)
    else:
        with pytest.raises(RuntimeError):
            f.manager.open({'arm': 'right', 'preparation_cancel_supported': True})
        assert not any(name == 'EnableArm' and args == (7,) for _, name, args in f.commands)
        if stopped_at:
            assert not any(name == 'JointCtrl' for _, name, _ in f.commands[stopped_at[0]:])
    assert not any(name in ('ResetPiper', 'DisableArm', 'GripperCtrl') for _, name, _ in f.commands)


@pytest.mark.parametrize('event', ['complete', 'startup_burst', 'enable_wrist_shift', 'stop_at_send',
                                 'planning_timeout', 'target_rejected', 'camera_stale'])
def test_all_enabled_recovery_uses_legal_startup_and_bounded_packets(gravity_controller, event):
    f = gravity_controller
    cold, parked = copy.deepcopy(f.states['left']), copy.deepcopy(f.states['right'])
    cold['arm'] = 'right'
    f.states.update(right=cold, left=parked)
    origin = np.array(cold['joints_rad'])
    rows = []
    for j4, j5, j2, j3 in ((0, 20, 0, -.6), (0, 18, 0, -.6),
                           (0, 18, .5, -.6), (0, 18, .5, -3)):
        row = origin.copy()
        row[[1, 2, 3, 4]] = np.deg2rad([j2, j3, j4, j5])
        rows.append(np.rint(np.rad2deg(row)*1000).astype(int).tolist())
    def make_path(start):
        path = [start]
        for row in rows:
            path.extend(np.linspace(path[-1], np.deg2rad(np.array(row)/1000), 101)[1:])
        return np.asarray(path).tolist()
    template = {'required': True, 'candidate': 'bounded_hold_fixture', 'enabled_start': False,
        'streamed_recovery': True, 'legal_startup_joint_sdk': rows[0], 'joint_sdk': rows[-1], 'stage_joint_sdk': rows,
        'goal_joints_rad': np.deg2rad(np.array(rows[-1])/1000).tolist(),
        'start_joints_rad': origin.tolist(), 'path_joints_rad': make_path(origin),
        'lower_rad': np.deg2rad([-150, 0, -170, -100, -70, -120]).tolist(),
        'upper_rad': np.deg2rad([150, 180, 0, 100, 70, 120]).tolist(),
        'initial_excess_rad': [0, np.deg2rad(2), 0, 0, 0, 0]}
    target, checked, stopped_at, burst = [None], [], [], [None]
    feedback = f.robot.healthy
    def measured(arm, *args):
        result = feedback(arm, *args)
        if arm == 'right' and burst[0] == f.clock[0]:
            result['feedback_stamps']['joint'] = f.clock[0]-.0451102734
        return result
    f.robot.healthy = measured
    def planned():
        return any(row['type'] == 'streamed_recovery_revalidated' for trace in f.traces for row in trace.rows)
    def submit(function, *args):
        def result(timeout):
            if function.__name__ == 'geometry_hold':
                if args[0]['joints_rad'][1] < 0:
                    raise ValueError('measured pose has URDF self collision')
                return True
            if function.__name__ == 'geometry_recovery_plan':
                return copy.deepcopy(template)
            if function.__name__ == 'geometry_recovery_replan':
                assert f.states['right']['controller']['motors_enabled'] == [True]*6
                assert abs(f.states['right']['joints_rad'][1]) <= np.deg2rad(.1)
                if event == 'planning_timeout':
                    f.manager.clock.sleep(.025)
                    raise TimeoutError()
                plan = copy.deepcopy(template)
                plan['start_joints_rad'] = args[0]['joints_rad']
                plan['path_joints_rad'] = make_path(np.array(args[0]['joints_rad']))
                return plan
            return True
        return SimpleNamespace(result=result)
    f.robot.geometry_pool.submit = submit
    def joint(*values):
        f.commands.append(('right', 'JointCtrl', values))
        target[0] = np.deg2rad(np.array(values)/1000)
    f.robot.arms['right'].JointCtrl = joint
    mode = f.robot.arms['right'].MotionCtrl_2
    def mode_stop(*values):
        mode(*values)
        if event == 'stop_at_send' and planned():
            stopped_at.append(len([1 for _, name, _ in f.commands if name == 'JointCtrl']))
            f.robot.stopped.set()
    f.robot.arms['right'].MotionCtrl_2 = mode_stop
    def sleep(seconds):
        f.clock[0] += seconds
        for frame in f.cameras.values():
            frame['stamp'] = f.clock[0]
        if target[0] is not None and all(f.states['right']['controller']['motors_enabled']):
            q = np.array(f.states['right']['joints_rad'])
            f.states['right']['joints_rad'] = (q+np.clip(target[0]-q, -np.deg2rad(.075), np.deg2rad(.075))).tolist()
        if event == 'startup_burst' and burst[0] is None:
            f.states['right']['joints_rad'][1] += np.deg2rad(.153)
            burst[0] = f.clock[0]
        if event == 'enable_wrist_shift' and burst[0] is None:
            f.states['right']['joints_rad'][3] -= np.deg2rad(.9)
            burst[0] = f.clock[0]
        if event == 'camera_stale' and planned():
            f.cameras['top']['stamp'] -= 1
    f.manager.clock.sleep = sleep
    def check(measured, integers, actual_origin):
        assert all(f.states['right']['controller']['motors_enabled']) or not planned()
        q = np.deg2rad(np.array(integers)/1000)
        assert q[1] >= 0
        assert np.all(q>=template['lower_rad']) and np.all(q<=template['upper_rad'])
        if event == 'target_rejected' and planned():
            raise ValueError('target outside recovery joint bounds')
        checked.append(list(integers))
    f.robot.check_gravity_recovery_target = check
    if event in ('complete', 'startup_burst', 'enable_wrist_shift'):
        view = f.manager.open({'arm': 'right', 'preparation_cancel_supported': True})
        assert view['active'] and view['gravity_recovery']['phase'] == 'completed'
        assert checked[0][1] >= 0
        assert any(row[4] == 18000 for row in checked)
        assert all(args[1]>=0 for _, name, args in f.commands if name=='JointCtrl')
        assert checked[-1] == rows[-1]
    else:
        with pytest.raises((RuntimeError, ValueError)):
            f.manager.open({'arm': 'right', 'preparation_cancel_supported': True})
        if stopped_at:
            assert stopped_at == [len([1 for _, name, _ in f.commands if name == 'JointCtrl'])]
    assert all(args == (7,) for _, name, args in f.commands if name == 'EnableArm')
    assert not any(name in ('DisableArm', 'ResetPiper', 'GripperCtrl') for _, name, _ in f.commands)
