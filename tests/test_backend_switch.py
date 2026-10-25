"""Real backend selection and asynchronous desktop handoff; no device access."""
import copy
import threading

import numpy as np
import pytest

import test_teleop as fixtures
from astra_vla.teleop_desktop import Client

controller = fixtures.controller


def native_session(f, arm='left', profile='normal'):
    for name in f.robot.arms:
        f.robot.arms[name].EndPoseCtrl = lambda *v, name=name: f.commands.append((name, 'EndPoseCtrl', v))
    return f.manager.open({'arm': arm, 'speed_profile': profile, 'execution_backend': 'controller_move_p'})['session']


@pytest.mark.parametrize('probe,keys', [(False, ['w']), (False, ['k']), (False, ['q']),
                                      (False, ['e']), (False, ['w', 'q']), (True, ['space'])])
def test_native_entry_and_writes_never_query_local_collision_model(controller, monkeypatch, probe, keys):
    from types import SimpleNamespace
    from astra_vla.workers import teleop
    from astra_vla import native_pose
    f = controller
    def forbidden(*args, **kwargs):
        raise AssertionError('native control queried local collision geometry')
    monkeypatch.setattr(teleop, '_geometry_model', SimpleNamespace(
        check_hold=forbidden, bounded_gripper_contacts=forbidden))
    monkeypatch.setattr(native_pose, 'adapt_target', lambda state, proposal, *args, **kwargs:
                        {'tcp_target': proposal['tcp_target'].tolist(), 'progress': 1., 'adjustment_deg': 0.})
    original = f.robot.geometry_pool.submit
    def submit(function, *args):
        if function in (teleop.native_feedback, teleop.geometry_native_step):
            return SimpleNamespace(result=lambda **kwargs: function(*args), cancel=lambda: None)
        if function is teleop.geometry_hold:
            forbidden()
        return original(function, *args)
    f.robot.geometry_pool.submit = submit
    if probe:
        from test_native_probe import open_probe
        session = open_probe(f)
    else:
        session = native_session(f, 'right')
    assert not f.commands  # Entry remains observation-only.
    f.manager.input({'session': session, 'sequence': 0, 'keys': keys})
    f.manager.step()
    assert any(method == 'EndPoseCtrl' for _, method, _ in f.commands)
    if 'q' in keys or 'e' in keys:
        assert any(method == 'GripperCtrl' for _, method, _ in f.commands)
    assert f.manager.view()['collision_scope'] == 'none'
    assert any(row['type'] == 'target' and row['geometry_scope'] == 'none'
               for row in f.manager.trace.rows)


@pytest.mark.parametrize('field,value', [('joints_rad', [float('nan')]*6),
                                       ('joints_rad', [0.]*5),
                                       ('position_m', [float('inf'), 0., 0.]),
                                       ('rpy_deg', [0., 0.]), ('width_m', float('nan'))])
def test_native_feedback_still_rejects_malformed_measurements(controller, field, value):
    from astra_vla.workers.teleop import native_feedback
    state = copy.deepcopy(controller.states['right'])
    if field == 'joints_rad':
        state[field] = value
    elif field == 'width_m':
        state['gripper'][field] = value
    else:
        state['ee_pose'][field] = value
    with pytest.raises(ValueError, match='invalid native'):
        native_feedback(state, {'scope': 'tcp_jog'})


def test_native_unreachable_pose_pauses_input_without_ending_session_or_sending_bad_target(controller):
    from astra_vla.native_pose import NativePoseUnavailable
    f = controller
    session = native_session(f)
    f.manager.input({'session': session, 'sequence': 0, 'keys': ['w']})
    f.geometry_error[0] = NativePoseUnavailable('no feasible adapted pose')
    f.manager.step()
    assert f.manager.active and f.manager.session == session
    assert f.manager.reason == 'no feasible adapted pose'
    assert all(method == 'MotionCtrl_1' for _, method, _ in f.commands)
    f.geometry_error[0] = None
    f.manager.input({'session': session, 'sequence': 1, 'keys': []})
    for delay in (.1, .6, .1):
        f.clock[0] += delay
        f.manager.last_input = f.clock[0]
        for frame in f.cameras.values():
            frame['stamp'] = f.clock[0]
        f.manager.step()
    f.manager.input({'session': session, 'sequence': 2, 'keys': ['s']})
    f.manager.step()
    assert any(method == 'EndPoseCtrl' for _, method, _ in f.commands)
    assert not any(method in ('JointCtrl', 'EnableArm') for _, method, _ in f.commands)


@pytest.mark.parametrize('cancel', ['release', 'stop', 'stale_input'])
def test_native_adaptation_cannot_send_after_cancellation(controller, cancel):
    f = controller
    session = native_session(f)
    f.manager.input({'session': session, 'sequence': 0, 'keys': ['w']})
    def complete():
        if cancel == 'release':
            f.manager.input({'session': session, 'sequence': 1, 'keys': []})
        elif cancel == 'stop':
            f.robot.stop('operator stop')
        else:
            f.clock[0] += .3
    f.completion[0] = complete
    f.manager.step()
    assert not any(method == 'EndPoseCtrl' for _, method, _ in f.commands)


def test_native_entry_does_not_fall_back_to_urdf_when_controller_limits_missing(controller):
    f = controller
    f.robot.read_controller_limits = lambda arm: {'complete': False}
    with pytest.raises(ValueError, match='fresh controller'):
        native_session(f)
    assert not f.manager.active and not f.commands


@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('profile', ['slow', 'normal', 'fast'])
def test_native_backend_supports_both_arms_profiles_and_gripper_without_local_joint_commands(controller, arm, profile):
    f = controller
    session = native_session(f, arm, profile)
    submit = f.robot.geometry_pool.submit
    def geometry_only(fn, *args):
        assert fn.__name__ in ('native_feedback', 'geometry_probe_ready', 'geometry_native_step')
        return submit(fn, *args)
    f.robot.geometry_pool.submit = geometry_only
    f.manager.input({'session': session, 'sequence': 0, 'keys': ['space', 'q'],
                     'execution_backend': 'controller_move_p'})
    f.manager.step()
    view = f.manager.view()
    assert not view['native_pose_probe'] and view['execution_backend'] == 'controller_move_p'
    assert view['policy']['ik_solver'] == view['ik_solver'] == 'controller'
    assert view['path_precheck'] == view['collision_scope'] == 'none'
    assert f.manager.trace.context['path_precheck'] == f.manager.trace.context['collision_scope'] == 'none'
    assert view['speed_profile_active'] == profile and not view['detour']['enabled']
    assert set(a for a, _, _ in f.commands) == {arm}
    assert any(m == 'EndPoseCtrl' for _, m, _ in f.commands)
    assert any(m == 'GripperCtrl' for _, m, _ in f.commands)
    assert not any(m in ('JointCtrl', 'EnableArm') for _, m, _ in f.commands)
    assert f.manager.gripper_commanded and f.manager.gripper_target_m is not None
    f.manager.input({'session': session, 'sequence': 1, 'keys': []})
    assert f.manager.phase == 'pausing'
    assert f.commands[-1][1] == 'GripperCtrl'  # Cancellation holds actual jaw feedback.


def test_formal_native_does_not_inherit_temporary_probe_envelope(controller):
    f = controller
    native_session(f)
    state = copy.deepcopy(f.states['left'])
    state['ee_pose']['position_m'][0] += .02
    state['joints_rad'][0] += np.deg2rad(10)
    f.manager.native_started -= 21
    f.manager.native_probe_bounds(state)
    assert not f.manager.native_pose_probe and f.manager.native_control


def test_backend_is_immutable_during_session_and_invalid_open_keeps_live_owner(controller):
    f = controller
    session = native_session(f)
    with pytest.raises(ValueError, match='exit control'):
        f.manager.input({'session': session, 'sequence': 0, 'keys': ['space'],
                         'execution_backend': 'regularized_move_j'})
    with pytest.raises(RuntimeError, match='owns'):
        f.manager.open({'arm': 'right', 'execution_backend': 'regularized_move_j'})
    assert f.manager.native_control and f.manager.execution_backend == 'controller_move_p'
    assert not f.commands


def test_native_refuses_disabled_entry_without_sending_enable(controller):
    f = controller
    f.states['left']['controller']['motors_enabled'][0] = False
    with pytest.raises(RuntimeError, match='已使能'):
        native_session(f)
    assert not f.commands


def test_native_speed_switch_rebuilds_controller_policy_after_existing_quiet_window(controller):
    f = controller
    session = native_session(f)
    f.manager.input({'session': session, 'sequence': 0, 'keys': ['space']})
    f.manager.step()
    f.manager.input({'session': session, 'sequence': 1, 'keys': ['space'], 'speed_profile': 'slow'})
    assert f.manager.keys == frozenset() and f.manager.paused
    f.clock[0] += .6
    f.manager.last_input = f.clock[0]
    for frame in f.cameras.values():
        frame['stamp'] = f.clock[0]
    f.manager.step()
    assert f.manager.policy.ik_solver == 'controller' and f.manager.policy.speed_percent == 5
    assert f.manager.policy.speed_profile == 'slow'


@pytest.mark.parametrize('confirmed', [True, False])
def test_ui_backend_commits_only_after_positive_stop_confirmation(confirmed):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def rpc(_path, method, args=None, **kwargs):
        calls.append(method)
        entered.set()
        assert release.wait(2)
        return {'stop_confirmed': confirmed, 'owner': None}
    c = Client(rpc)
    c.session, c.control_active, c.keys = 'old', True, ['w']
    c.request_backend('controller_move_p')
    assert entered.wait(1)
    assert c.execution_backend == 'regularized_move_j' and c.pending_backend == 'controller_move_p'
    assert not c.keys and not c.control_active
    c.request('open', 'left')  # Cannot queue a new control entry while stopping.
    assert c.commands.empty()
    release.set()
    for t in c.threads:
        t.join(2)
    assert calls == ['stop']
    assert c.execution_backend == ('controller_move_p' if confirmed else 'regularized_move_j')
    assert not c.control_active and c.pending_backend is None
    if not confirmed:
        assert '尚未确认静止' in c.error


def test_late_backend_switch_does_not_override_new_stop():
    entered, release = threading.Event(), threading.Event()
    def rpc(*a, **kw):
        entered.set()
        release.wait(2)
        return {'stop_confirmed': True}
    c = Client(rpc)
    c.request_backend('controller_move_p')
    assert entered.wait(1)
    c.request('stop')
    release.set()
    for t in c.threads:
        t.join(2)
    assert c.execution_backend == 'regularized_move_j' and not c.control_active


def test_backend_change_cancels_previously_queued_open():
    calls = []
    c = Client(lambda _p, m, *a, **kw: (calls.append(m), {'stop_confirmed': True})[1])
    c.request('open', 'left')
    action, arm = c.commands.get_nowait()
    c.request_backend('controller_move_p')
    for t in c.threads:
        t.join(2)
    c.execute(action, arm)
    assert calls == ['stop'] and not c.opening and not c.session


def test_native_legacy_worker_cannot_silently_open_local_backend():
    calls = []
    c = Client(lambda _p, m, *a, **kw: calls.append(m))
    c.execution_backend = 'controller_move_p'
    with pytest.raises(RuntimeError, match='尚未加载'):
        c.execute('open', 'left')
    assert calls == []


def test_native_open_checks_worker_reply_and_drops_mismatched_owner():
    calls = []
    def rpc(_p, method, args=None, **kwargs):
        calls.append((method, args))
        return {'session': 'wrong', 'execution_backend': 'regularized_move_j'}
    c = Client(rpc)
    c.execution_backend = 'controller_move_p'
    c.robot = {'teleop': {}, 'motion_control': {'execution_backends': ['controller_move_p']}}
    with pytest.raises(RuntimeError, match='所选控制方案'):
        c.execute('open', 'right')
    assert [m for m, _ in calls] == ['teleop_open', 'teleop_close']
    assert not c.session and not c.control_active
