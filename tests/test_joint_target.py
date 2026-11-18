"""Absolute degree control with fake SDKs and SDL; never opens live workers."""
from concurrent.futures import Future
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from astra_vla.joint_target import joint_target
from astra_vla.teleop_desktop import Client
from astra_vla.teleop_preferences import DesktopPreferences
import test_teleop as fixtures
import test_teleop_desktop as desktop
from test_sleep_return import tick

controller = fixtures.controller
window = desktop.window
pg = desktop.pg


@pytest.mark.parametrize('values', [None, [], [0]*5, [0]*7, ['0']*6,
                                    [True]*6, [float('nan')]*6, [float('inf')]*6, [3e6]*6])
def test_invalid_targets_never_reach_robot(values):
    with pytest.raises(ValueError):
        joint_target(values)


def test_encoding_uses_degrees_and_allows_angles_outside_urdf():
    target = joint_target([180.001, -3.125, 20, -110, 85, 270])
    assert target['joint_sdk'] == [180001, -3125, 20000, -110000, 85000, 270000]
    np.testing.assert_allclose(np.rad2deg(target['goal_joints_rad']), [180.001, -3.125, 20, -110, 85, 270])


def prepare(f, arm='left'):
    f.pending = Future()
    f.robot.detour_pool = SimpleNamespace(submit=lambda *_: f.pending)
    opened = f.manager.open({'arm': arm})
    f.manager.input({'session': opened['session'], 'sequence': 0, 'keys': []})
    f.commands.clear()
    args = {'session': opened['session'], 'arm': arm, 'joints_deg': [5., 0, 0, 0, 0, 0],
            'input_sequence': 0}
    return args


def start_plan(f, args):
    f.manager.set_joint_target(args)
    tick(f, 12)
    assert f.manager.joint_target['state'] == 'planning'
    plan = {'start_joints_rad': [0.]*6, 'goal_joints_rad': f.manager.joint_target['goal_joints_rad'],
            'samples': 0, 'envelope_max_m': .03}
    f.pending.set_result(plan)
    tick(f)
    return plan


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_joint_target_sends_only_selected_arm_move_j_and_waits_for_stationary(controller, arm):
    f = controller
    args = prepare(f, arm)
    plan = start_plan(f, args)
    assert f.manager.view()['control_frame'] == 'joint'
    assert f.manager.view()['ik_solver'] == 'joint_target'
    assert f.manager.joint_target['state'] == 'executing'
    f.commands.clear()
    def submit(fn, *values):
        assert fn.__name__ == 'geometry_detour_step'
        assert values[2]['speed_percent'] == f.manager.policy.speed_percent
        assert values[2]['joint_lead_deg'] == f.manager.policy.joint_lead_deg
        return SimpleNamespace(result=lambda **_: {'joint_sdk': [100, 0, 0, 0, 0, 0],
                               'velocity': [0.]*6, 'samples': 0})
    f.robot.geometry_pool = SimpleNamespace(submit=submit)
    tick(f)
    assert (arm, 'JointCtrl', (100, 0, 0, 0, 0, 0)) in f.commands
    assert (arm, 'MotionCtrl_2', (1, 1, f.manager.policy.speed_percent, 0)) in f.commands
    assert all(a == arm and method not in ('GripperCtrl', 'EnableArm', 'EndPoseCtrl')
               for a, method, _ in f.commands)
    f.states[arm]['joints_rad'] = plan['goal_joints_rad']
    f.states[arm]['controller']['motion_status'] = 0
    tick(f)
    assert f.manager.joint_target['state'] == 'settling'
    assert not any(r['type'] == 'joint_target_completed' for r in f.manager.trace.rows)
    tick(f, 12)
    assert f.manager.joint_target['state'] == 'completed'
    assert f.manager.active and f.manager.last_confirmation['confirmed']
    assert any(r['type'] == 'joint_target_completed' for r in f.manager.trace.rows)


@pytest.mark.parametrize('cause', ['menu', 'keys', 'speed', 'focus', 'stop', 'stale', 'camera', 'config'])
def test_cancelled_joint_target_never_uses_late_plan(controller, cause):
    f = controller
    args = prepare(f)
    f.manager.set_joint_target(args)
    tick(f, 12)
    f.pending.set_running_or_notify_cancel()
    packet = {'session': args['session'], 'sequence': f.manager.sequence+1, 'keys': []}
    if cause == 'menu':
        f.manager.input(dict(packet, cancel_detour=True))
    elif cause == 'keys':
        f.manager.input(dict(packet, keys=['w']))
    elif cause == 'speed':
        f.manager.input(dict(packet, speed_profile='slow'))
    elif cause == 'focus':
        with pytest.raises(RuntimeError):
            f.manager.input(dict(packet, focused=False))
    elif cause == 'stop':
        f.manager.fail('operator stop')
    elif cause == 'stale':
        f.clock[0] += .3
    elif cause == 'camera':
        f.cameras['right_wrist']['available'] = False
    else:
        f.token[0] = 'changed'
    f.pending.set_result({'invalid': 'late'})
    f.commands.clear()
    if cause in ('stale', 'camera', 'config'):
        with pytest.raises(RuntimeError):
            f.manager.step()
    else:
        f.manager.step()
    assert not any(method == 'JointCtrl' for _, method, _ in f.commands)
    assert f.manager.joint_target['state'] != 'completed'


@pytest.mark.parametrize('change', [dict(arm='right'), dict(session='old'), dict(input_sequence=-1),
                                     dict(joints_deg=[float('nan')]*6)])
def test_wrong_arm_session_sequence_and_invalid_target_send_nothing(controller, change):
    f = controller
    args = prepare(f)
    with pytest.raises((ValueError, RuntimeError)):
        f.manager.set_joint_target(dict(args, **change))
    assert not f.commands


def test_changed_joint_plan_is_blocked(controller):
    f = controller
    args = prepare(f)
    f.manager.set_joint_target(args)
    tick(f, 12)
    f.pending.set_result({'start_joints_rad': [0.]*6, 'goal_joints_rad': [1.]*6})
    f.commands.clear()
    tick(f)
    assert f.manager.joint_target['state'] == 'blocked'
    assert not any(method == 'JointCtrl' for _, method, _ in f.commands)


def test_old_fast_startup_preference_migrates_to_normal(tmp_path):
    path = tmp_path/'desktop.json'
    data = vars(DesktopPreferences())
    data['startup_speed'] = 'fast'
    path.write_text(json.dumps(data))
    assert DesktopPreferences.load(path).startup_speed == 'normal'


@pytest.mark.parametrize('size', [(1100, 700), (1280, 800), (1920, 1080)])
def test_modal_layout_typing_and_submission(window, size):
    window.screen = pg.display.set_mode(size)
    window.client.request('open', 'left')
    window.menu_open = False
    window.draw()
    assert window.buttons['joint_angles'].collidelist(list(window.key_rects.values())) == -1
    window.activate('joint_angles')
    window.draw()
    assert window.screen.get_rect().contains(window.joint_editor_rect)
    assert all(window.joint_editor_rect.contains(r) for r in window.buttons.values())
    for index, a in enumerate(window.joint_field_rects):
        assert all(not a.colliderect(b) for b in window.joint_field_rects[index+1:])
    calls = []
    window.client.request_joint_target = lambda arm, values: calls.append((arm, values))
    # Motion keys and arm shortcuts belong to the editor, never the robot.
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_w, repeat=False))
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_2, repeat=False))
    assert not window.controls.keys and window.controls.arm == 'left'
    window.handle(pg.event.Event(pg.TEXTINPUT, text='-12.345'))
    assert window.joint_editor['values'][0] == '-12.345'
    window.activate('joint_execute')
    assert calls == [('left', [-12.345, 0., 0., 0., 0., 0.])]
    assert window.joint_editor is None and not window.menu_open


@pytest.mark.parametrize('cancel', ['escape', 'focus', 'stop'])
def test_editor_cancellation_never_submits(window, cancel):
    window.client.request('open', 'left')
    window.activate('joint_angles')
    calls = []
    window.client.request_joint_target = lambda *args: calls.append(args)
    if cancel == 'focus':
        window.handle(pg.event.Event(pg.WINDOWFOCUSLOST))
    else:
        key = pg.K_ESCAPE if cancel == 'escape' else pg.K_F9
        window.handle(pg.event.Event(pg.KEYDOWN, key=key, repeat=False))
    assert window.joint_editor is None and not calls and not window.controls.keys


def test_editor_invalid_input_does_not_close_or_submit(window):
    window.client.request('open', 'left')
    window.activate('joint_angles')
    window.joint_editor['values'][2] = 'nan'
    calls = []
    window.client.request_joint_target = lambda *args: calls.append(args)
    window.activate('joint_execute')
    assert window.joint_editor is not None and not calls


@pytest.mark.parametrize('cause', ['stop', 'session', 'menu', 'expired'])
def test_client_drops_delayed_queued_request(cause):
    calls = []
    client = Client(lambda *args, **kwargs: calls.append(args))
    client.session, client.control_active = 's', True
    client.ui_tick = time.monotonic()
    client.request_joint_target('left', [0.]*6)
    action, payload = client.commands.get_nowait()
    if cause == 'stop':
        client.cancel_epoch += 1
    elif cause == 'session':
        client.session = 'new'
    elif cause == 'menu':
        client.cancel_detour = True
    else:
        payload['requested_at'] -= 1
    with pytest.raises(RuntimeError):
        client.execute(action, payload)
    assert not calls


def test_client_sends_empty_heartbeat_then_bound_degree_request():
    calls = []
    client = Client(lambda path, method, args=None, **kwargs: calls.append((method, args)))
    client.session, client.control_active = 's', True
    client.ui_tick = time.monotonic()
    client.request_joint_target('right', [1.25, 2, 3, 4, 5, -6.125])
    action, payload = client.commands.get_nowait()
    client.execute(action, payload)
    assert [method for method, _ in calls] == ['teleop_input', 'teleop_joint_target']
    assert calls[0][1]['keys'] == [] and calls[0][1]['cancel_detour'] is False
    assert calls[1][1] == {'session': 's', 'arm': 'right', 'joints_deg': [1.25, 2., 3., 4., 5., -6.125],
                           'input_sequence': calls[0][1]['sequence']}


def test_duplicate_degree_request_is_rejected_even_after_cancellation(controller):
    f = controller
    args = prepare(f)
    f.manager.set_joint_target(args)
    f.manager.pause(f.manager.generation)
    f.commands.clear()
    with pytest.raises(RuntimeError, match='过期'):
        f.manager.set_joint_target(args)
    assert not f.commands
