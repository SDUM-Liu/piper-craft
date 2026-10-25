"""Session failures and advancing feedback, using isolated SDK spies only."""
import copy
import time

import numpy as np
import pytest

from astra_vla.rpc import RPCError, error_response
from astra_vla.teleop_desktop import Client, control_message
import test_robot_maintenance as maintenance_fixtures
import test_teleop as teleop_fixtures

worker = maintenance_fixtures.worker
controller = teleop_fixtures.controller
regularized_controller = teleop_fixtures.regularized_controller
enter = teleop_fixtures.enter


def test_scene_acknowledgement_clears_old_entry_blocker_without_opening_control(regularized_controller):
    f = regularized_controller
    f.manager.phase, f.manager.reason = 'entry_blocked', 'teach/MIT/master-slave state requires manual mode handoff'
    f.manager.scene_acknowledged()
    assert f.manager.phase == 'observing' and f.manager.reason is None
    assert not f.manager.active and not f.manager.owns_lock and f.manager.session is None
    assert not f.commands


def test_scene_acknowledgement_does_not_alter_an_active_session(regularized_controller):
    f = regularized_controller
    session = enter(f)
    phase = f.manager.phase
    f.manager.scene_acknowledged()
    assert f.manager.active and f.manager.session == session and f.manager.phase == phase


@pytest.mark.parametrize('confirmed', [False, True])
def test_expired_input_keeps_first_stop_cause_across_cleanup(controller, confirmed):
    f = controller
    session = enter(f)
    f.robot.stationary_feedback = lambda **_: {'confirmed': confirmed}
    f.manager.fail('teleop guard: camera RGB stale')
    f.manager.close({'session': session, 'reason': 'desktop close'})
    with pytest.raises(RPCError) as error:
        f.manager.input({'session': session, 'sequence': 2, 'keys': ['w']})
    packet = error_response(error.value)
    assert packet['stage'] == 'teleop_session'
    assert packet['diagnostics']['reason'] == 'teleop guard: camera RGB stale'
    assert packet['diagnostics']['session'] == session
    assert f.manager.view()['termination_reason'] == 'teleop guard: camera RGB stale'
    if confirmed:
        assert f.manager.trace.result['reason'] == 'teleop guard: camera RGB stale'
    else:
        assert f.manager.owns_lock
    assert len([r for r in f.manager.trace.rows if r['type'] == 'session_ended']) == 1


@pytest.mark.parametrize('change', ['joint', 'pose', 'orientation', 'width', 'other_joint'])
def test_manual_changed_feedback_rechecks_same_goal_and_sends_without_ending_session(regularized_controller, worker, change):
    f = regularized_controller
    real_robot, _ = worker
    f.robot.check_motion_unchanged = real_robot.check_motion_unchanged
    enter(f)
    f.manager.step()
    f.commands.clear()
    rechecks = []
    def recheck(*args):
        rechecks.append(copy.deepcopy(args))
        return copy.deepcopy(args[-1])
    f.robot.recheck_teleop_target = recheck
    def advance():
        if change == 'joint':
            f.states['left']['joints_rad'][3] += np.deg2rad(2.)
        elif change == 'other_joint':
            f.states['right']['joints_rad'][3] += np.deg2rad(2.)
        elif change == 'pose':
            f.states['left']['ee_pose']['position_m'][1] += .004
        elif change == 'orientation':
            f.states['left']['ee_pose']['rpy_deg'][1] += 4.
        else:
            f.states['left']['gripper']['width_m'] += .003
    f.completion[0] = advance
    f.manager.step()
    assert f.manager.active and not f.robot.stopped.is_set()
    assert any(method == 'JointCtrl' for _, method, _ in f.commands)
    assert len(rechecks) == 1
    sent = next(args for _, method, args in f.commands if method == 'JointCtrl')
    assert list(sent) == rechecks[0][-1]['joint_sdk']
    if change == 'joint':
        assert rechecks[0][0]['joints_rad'][3] == pytest.approx(np.deg2rad(2.))
    assert not any(r['type'] == 'discarded' for r in f.manager.trace.rows)
    assert any(r['type'] == 'feedback_advanced' for r in f.manager.trace.rows)


@pytest.mark.parametrize('change', ['mode', 'enabled', 'error', 'gripper_enabled', 'stale'])
def test_feedback_recheck_does_not_hide_controller_or_enable_changes(regularized_controller, worker, change):
    f = regularized_controller
    real_robot, _ = worker
    f.robot.check_motion_unchanged = real_robot.check_motion_unchanged
    enter(f)
    f.manager.step()
    f.commands.clear()
    def advance():
        s = f.states['left']
        s['joints_rad'][3] += np.deg2rad(2.)
        if change == 'mode':
            s['controller']['move_mode'] = 2
        elif change == 'enabled':
            s['controller']['motors_enabled'][0] = False
        elif change == 'error':
            s['controller']['err_code'] = 1
        elif change == 'stale':
            s['age_s'] = .301
        else:
            s['gripper']['enabled'] = not s['gripper']['enabled']
    f.completion[0] = advance
    with pytest.raises(RuntimeError, match='arm/gripper changed|feedback missing/stale'):
        f.manager.step()
    assert not f.commands


def test_feedback_recheck_collision_still_prevents_send(regularized_controller):
    f = regularized_controller
    enter(f)
    f.manager.step()
    f.commands.clear()
    f.completion[0] = lambda: f.states['left']['joints_rad'].__setitem__(3, np.deg2rad(2))
    def recheck(*_args):
        raise ValueError('fresh self collision')
    f.robot.recheck_teleop_target = recheck
    with pytest.raises(ValueError, match='fresh self collision'):
        f.manager.step()
    assert not any(method == 'JointCtrl' for _, method, _ in f.commands)


@pytest.mark.parametrize('delayed_seconds,accepted', [(0., True), (.31, False)])
def test_first_gripper_enable_only_accepts_own_recent_acknowledgement(regularized_controller, worker, delayed_seconds, accepted):
    f = regularized_controller
    real_robot, _ = worker
    f.robot.check_motion_unchanged = real_robot.check_motion_unchanged
    f.states['left']['gripper']['enabled'] = False
    session = enter(f, ('q',))
    f.manager.step()
    assert f.manager.gripper_enable_request_at is not None
    f.commands.clear()
    f.states['left']['gripper']['enabled'] = False  # The cached enable bit is delayed.
    f.manager.clock.sleep(delayed_seconds)
    f.manager.input({'session': session, 'sequence': 1, 'keys': ['q']})
    f.completion[0] = lambda: f.states['left']['gripper'].__setitem__('enabled', True)
    if accepted:
        f.manager.step()
        assert f.manager.active and any(method == 'GripperCtrl' for _, method, _ in f.commands)
        assert any(r['type'] == 'gripper_enable_confirmed' for r in f.manager.trace.rows)
    else:
        with pytest.raises(RuntimeError, match='arm/gripper changed'):
            f.manager.step()
        assert not f.commands


def test_feedback_rechecks_keep_target_past_three_updates_within_original_deadline(regularized_controller):
    f = regularized_controller
    enter(f)
    f.manager.step()
    f.commands.clear()
    f.completion[0] = lambda: f.states['left']['joints_rad'].__setitem__(3, np.deg2rad(.8))
    goals = []
    def recheck(*args):
        goals.append(copy.deepcopy(args[-1]))
        if len(goals) <= 4:
            f.states['left']['joints_rad'][3] += np.deg2rad(.06)
        f.manager.clock.sleep(.002)
        return copy.deepcopy(args[-1])
    f.robot.recheck_teleop_target = recheck
    started = f.clock[0]
    f.manager.step()
    assert len(goals) == 5 and f.clock[0]-started < f.manager.policy.geometry_timeout_s
    assert all(g['joint_sdk'] == goals[0]['joint_sdk'] for g in goals)
    assert [args for _, method, args in f.commands if method == 'JointCtrl'] == [tuple(goals[0]['joint_sdk'])]
    assert not any(r['type'] == 'discarded' for r in f.manager.trace.rows)


@pytest.mark.parametrize('cancel', ['release', 'stop', 'timeout'])
def test_feedback_recheck_still_cancels_on_release_stop_or_real_timeout(regularized_controller, cancel):
    f = regularized_controller
    session = enter(f)
    f.manager.step()
    f.commands.clear()
    f.completion[0] = lambda: f.states['left']['joints_rad'].__setitem__(3, np.deg2rad(.8))
    def recheck(*args):
        if cancel == 'release':
            f.manager.input({'session': session, 'sequence': 1, 'keys': []})
        elif cancel == 'stop':
            f.robot.stopped.set()
        else:
            f.manager.clock.sleep(.06)
        return copy.deepcopy(args[-1])
    f.robot.recheck_teleop_target = recheck
    if cancel == 'stop':
        with pytest.raises(RuntimeError, match='execution stopped'):
            f.manager.step()
    else:
        f.manager.step()
    assert not any(method == 'JointCtrl' for _, method, _ in f.commands)


def client_session(rpc=lambda *_args, **_kwargs: {}):
    c = Client(rpc)
    c.session, c.control_active, c.keys = 'current', True, ['w']
    return c


def test_feedback_session_identity_is_checked_and_never_adopts_another_owner():
    c = client_session()
    c.accept_feedback({'teleop': {'session': 'another', 'active': True, 'owner': 'desktop'}}, time.monotonic())
    assert c.session is None and not c.control_active and not c.keys


def test_old_status_and_cancelled_input_cannot_reactivate_session():
    c = client_session()
    c.session_started = 100.
    c.accept_feedback({'teleop': {'session': 'old', 'active': False}}, 99.)
    assert c.session == 'current' and c.control_active
    c.control_active = False
    c.accept_feedback({'teleop': {'session': 'current', 'active': True, 'owner': 'desktop'}}, 101.)
    assert not c.control_active and not c.keys


def test_terminated_worker_reason_replaces_generic_expired_input_without_duplicate_close():
    calls = []
    def rpc(_path, method, *_args, **_kwargs):
        calls.append(method)
        return {'teleop': {'session': 'current', 'active': False, 'owner': 'desktop',
                          'termination_reason': 'teleop guard: RGB stale'}}
    c = client_session(rpc)
    c.control_error(RuntimeError('teleop session is not active'))
    assert c.error == 'teleop guard: RGB stale'
    assert calls == ['status'] and not c.control_active and not c.keys
    assert c.session == 'current'  # Keep stop retry available until confirmed.


def test_diagnostics_survive_failed_status_and_cleanup():
    calls = []
    def rpc(_path, method, *_args, **_kwargs):
        calls.append(method)
        raise RuntimeError('read unavailable' if method == 'status' else 'close unavailable')
    c = client_session(rpc)
    c.control_error(RPCError('teleop session is not active', stage='teleop_session',
                            diagnostics={'session': 'current', 'reason': 'original fault'}))
    assert c.error.startswith('original fault') and '停止确认' in c.error
    assert calls == ['status', 'teleop_close'] and not c.control_active


def test_input_failure_closes_matching_active_session_and_does_not_reopen():
    calls = []
    def rpc(_path, method, *_args, **_kwargs):
        calls.append(method)
        if method == 'status':
            return {'teleop': {'session': 'current', 'active': True, 'owner': 'desktop'}}
        return {'owner': 'desktop', 'stop_confirmed': False}
    c = client_session(rpc)
    c.control_error(RuntimeError('input connection failed'))
    assert c.session == 'current' and not c.control_active
    assert c.error.startswith('input connection failed') and '尚未确认静止' in c.error
    assert calls == ['status', 'teleop_close']


def test_expired_input_cleanup_never_stops_newer_session():
    calls = []
    def rpc(_path, method, *_args, **_kwargs):
        calls.append(method)
        return {'teleop': {'session': 'newer', 'active': True, 'owner': 'desktop'}}
    c = client_session(rpc)
    c.control_error(RuntimeError('teleop session is not active'))
    assert calls == ['status'] and c.session is None and not c.control_active


def test_inflight_input_error_keeps_cause_when_status_poll_already_cleared_session():
    result = {'teleop': {'session': 'current', 'active': False, 'owner': None,
                        'termination_reason': 'original camera fault'}}
    c = client_session(lambda *_args, **_kwargs: result)
    c.accept_feedback(result, time.monotonic())
    assert c.session is None
    c.control_error(RuntimeError('teleop session is not active'), 'current')
    assert c.error == 'original camera fault' and c.session is None


def test_late_old_input_error_does_not_cancel_current_session():
    c = client_session(lambda *_args, **_kwargs: pytest.fail('late old error touched the new worker'))
    c.control_error(RuntimeError('teleop session is not active'), 'old')
    assert c.session == 'current' and c.control_active and not c.error


def test_unexpected_stop_poll_shows_primary_cause_but_intentional_close_is_not_an_error():
    result = {'teleop': {'session': 'current', 'active': False, 'owner': None,
                        'termination_reason': 'desktop close'}}
    c = client_session()
    c.control_active = False
    c.accept_feedback(result, time.monotonic())
    assert not c.error and c.session is None
    c = client_session()
    result['teleop']['termination_reason'] = 'camera RGB stale'
    c.accept_feedback(result, time.monotonic())
    assert c.error == 'camera RGB stale' and not c.control_active
    assert 'camera RGB stale' in control_message('teleop session is not active: camera RGB stale')
