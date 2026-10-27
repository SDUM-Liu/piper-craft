"""Sleep return checks with isolated SDK spies; no physical motion."""
from concurrent.futures import Future
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest

from astra_vla.config import ROOT, settings
from astra_vla.sleep_return import load_sleep_target, plan_sleep_return
from astra_vla.teleop_desktop import Client
import test_teleop as fixtures

controller = fixtures.controller


@pytest.fixture
def sleep_controller(controller, monkeypatch, tmp_path):
    import astra_vla.sleep_return as sleep
    path = tmp_path / 'sleep_pose.json'
    path.write_text(json.dumps({'version': 1, 'arms': {
        arm: {'joints_raw': [5000, 0, 0, 0, 0, 0]} for arm in ('left', 'right')}}))
    monkeypatch.setattr(sleep, 'sleep_pose_path', lambda: path)
    controller.sleep_path = path
    controller.pending = Future()
    controller.robot.detour_pool = SimpleNamespace(submit=lambda *_: controller.pending)
    return controller


def request(f, arm='left'):
    result = f.manager.open({'arm': arm})
    f.manager.input({'session': result['session'], 'sequence': 0, 'keys': []})
    f.commands.clear()
    return f.manager.return_sleep({'session': result['session'], 'arm': arm})


def tick(f, count=1):
    for _ in range(count):
        f.clock[0] += .05
        for frame in f.cameras.values():
            frame['stamp'] = f.clock[0]
        f.manager.input({'session': f.manager.session, 'sequence': f.manager.sequence + 1, 'keys': []})
        f.manager.step()


def executing(f):
    request(f)
    tick(f, 12)
    assert f.manager.detour_context['state'] == 'planning'
    plan = dict(start_joints_rad=[0.] * 6,
                goal_joints_rad=f.manager.sleep_return['goal_joints_rad'], envelope_max_m=.03, samples=100)
    f.pending.set_result(plan)
    tick(f)
    assert f.manager.sleep_return['state'] == 'executing'
    return plan


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_request_pauses_selected_arm_and_waits_before_planning(sleep_controller, arm):
    f = sleep_controller
    view = request(f, arm)
    assert view['sleep_return']['state'] == 'waiting'
    assert view['path_precheck'] == 'none' and not view['path_collision_check']
    assert view['collision_scope'] == 'entry_hold_only'
    assert f.manager.trace.context['path_precheck'] == 'none'
    assert f.manager.trace.context['collision_scope'] == 'entry_hold_only'
    assert f.commands == [(arm, 'MotionCtrl_1', (0, 1, 0))]
    tick(f, 3)
    assert f.manager.detour_context['state'] == 'starting'
    tick(f, 9)
    assert f.manager.sleep_return['state'] == 'planning'
    assert not any(method in ('JointCtrl', 'GripperCtrl', 'EnableArm') for _, method, _ in f.commands)


def test_sleep_uses_current_speed_and_follow_budget_with_fresh_geometry(sleep_controller):
    f = sleep_controller
    executing(f)
    calls = []
    def submit(fn, *args):
        calls.append((fn.__name__, copy.deepcopy(args)))
        return SimpleNamespace(result=lambda **_: {'joint_sdk': [100, 0, 0, 0, 0, 0],
                                                  'velocity': [0.] * 6, 'samples': 80})
    f.robot.geometry_pool = SimpleNamespace(submit=submit)
    f.commands.clear()
    tick(f)
    assert calls[0][0] == 'geometry_detour_step'
    policy = calls[0][1][2]
    assert policy['speed_percent'] == f.manager.policy.speed_percent
    assert policy['joint_lead_deg'] == f.manager.policy.joint_lead_deg
    assert ('left', 'MotionCtrl_2', (1, 1, f.manager.policy.speed_percent, 0)) in f.commands
    assert any(method == 'JointCtrl' for _, method, _ in f.commands)
    assert all(arm == 'left' and method not in ('GripperCtrl', 'EnableArm') for arm, method, _ in f.commands)


@pytest.mark.parametrize('interruption', ['menu', 'keys', 'speed', 'focus', 'stop', 'stale', 'camera', 'config'])
def test_sleep_cancellation_discards_late_plan(sleep_controller, interruption):
    f = sleep_controller
    request(f)
    tick(f, 12)
    # A running process future can complete after cancellation.
    f.pending.set_running_or_notify_cancel()
    packet = {'session': f.manager.session, 'sequence': f.manager.sequence + 1, 'keys': []}
    if interruption == 'menu':
        f.manager.input(dict(packet, cancel_detour=True))
    elif interruption == 'keys':
        f.manager.input(dict(packet, keys=['j']))
    elif interruption == 'speed':
        f.manager.input(dict(packet, speed_profile='slow'))
    elif interruption == 'focus':
        with pytest.raises(RuntimeError, match='focus lost'):
            f.manager.input(dict(packet, focused=False))
    elif interruption == 'stop':
        f.manager.fail('operator stop')
    elif interruption == 'stale':
        f.clock[0] += .3
    elif interruption == 'camera':
        f.cameras['left_wrist']['available'] = False
    else:
        f.token[0] = 'changed'
    f.pending.set_result({'invalid': 'late result'})
    f.commands.clear()
    if interruption in ('stale', 'camera', 'config'):
        with pytest.raises(RuntimeError):
            f.manager.step()
    else:
        f.manager.step()
    assert not any(method == 'JointCtrl' for _, method, _ in f.commands)
    assert f.manager.sleep_return['state'] != 'completed'


@pytest.mark.parametrize('failure', ['collision', 'goal_changed', 'start_changed'])
def test_sleep_plan_failure_pauses_without_joint_commands(sleep_controller, failure):
    f = sleep_controller
    request(f)
    tick(f, 12)
    if failure == 'collision':
        f.pending.set_exception(ValueError('URDF self collision'))
    else:
        goal = f.manager.sleep_return['goal_joints_rad']
        f.pending.set_result({'start_joints_rad': [0.] * 6, 'goal_joints_rad':
                             [1.] * 6 if failure == 'goal_changed' else goal})
        if failure == 'start_changed':
            f.states['left']['joints_rad'][0] = .2
    f.commands.clear()
    tick(f)
    assert f.manager.sleep_return['state'] == 'blocked'
    assert not any(method == 'JointCtrl' for _, method, _ in f.commands)


def test_joint_arrival_needs_stationary_confirmation_and_does_not_compare_fk(sleep_controller):
    f = sleep_controller
    plan = executing(f)
    f.states['left']['joints_rad'] = plan['goal_joints_rad']
    f.states['left']['controller']['motion_status'] = 0
    # SDK end pose stays unchanged; sleep completion is explicitly joint-based.
    f.commands.clear()
    tick(f)
    assert f.manager.sleep_return['state'] == 'settling'
    tick(f, 12)
    assert f.manager.sleep_return['state'] == 'completed'
    assert f.manager.last_confirmation['confirmed']
    assert f.manager.active and not f.manager.keys
    assert not any(method in ('JointCtrl', 'GripperCtrl', 'EnableArm') for _, method, _ in f.commands)
    assert any(row['type'] == 'sleep_completed' for row in f.manager.trace.rows)


def test_sleep_rejects_no_session_wrong_arm_native_and_menu(sleep_controller):
    f = sleep_controller
    with pytest.raises(RuntimeError):
        f.manager.return_sleep({'session': None, 'arm': 'left'})
    request(f)
    with pytest.raises(RuntimeError):
        f.manager.return_sleep({'session': 'old-session', 'arm': 'left'})
    with pytest.raises(RuntimeError):
        f.manager.return_sleep({'session': f.manager.session, 'arm': 'right'})
    f.manager.input({'session': f.manager.session, 'sequence': 1, 'keys': [], 'cancel_detour': True})
    with pytest.raises(RuntimeError, match='关闭菜单'):
        f.manager.return_sleep({'session': f.manager.session, 'arm': 'left'})
    f.manager.native_control = True
    with pytest.raises(ValueError, match='本地'):
        f.manager.return_sleep({'session': f.manager.session, 'arm': 'left'})


@pytest.mark.parametrize('row', [None, [0]*5, [True]*6, [1.5]*6, ['0']*6])
def test_unconfigured_or_invalid_sleep_targets_are_rejected(sleep_controller, row):
    sleep_controller.sleep_path.write_text(json.dumps({'version': 1, 'arms': {'left': {'joints_raw': row}}}))
    with pytest.raises(ValueError, match='joints_raw'):
        request(sleep_controller)
    assert not sleep_controller.commands


def test_legacy_sleep_pose_and_exact_sdk_units(sleep_controller):
    f = sleep_controller
    f.sleep_path.write_text(json.dumps({'version': 1, 'arms': {'left_piper': {
        'joints_raw': [778, 0, -2430, -3757, 25811, 1498]}}}))
    target = load_sleep_target('left')
    np.testing.assert_allclose(np.rad2deg(target['goal_joints_rad']), [.778, 0, -2.430, -3.757, 25.811, 1.498])


def test_sleep_does_not_query_path_or_finger_collision():
    def forbidden(*a, **kw):
        raise AssertionError('sleep queried local collision geometry')
    model = SimpleNamespace(configuration=forbidden, bounded_gripper_collision=forbidden)
    state = {'joints_rad': [0.] * 6, 'gripper': {'width_m': .03}}
    plan = plan_sleep_return(model, state, {'goal_joints_rad': [1., 0, 0, 0, 0, 0]}, {},
                             {'planning_timeout_s': 2})
    assert plan['samples'] == 0 and plan['scope'] == plan['path_precheck'] == 'none'
    assert not plan['path_collision_check']
    for bad in ([float('nan')] * 6, [0.] * 5):
        with pytest.raises(ValueError, match='invalid sleep joint plan'):
            plan_sleep_return(model, dict(state, joints_rad=bad), {'goal_joints_rad': [0.] * 6}, {},
                              {'planning_timeout_s': 2})


def test_desktop_sleep_requires_active_focused_local_session():
    calls = []
    client = Client(rpc=lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(RuntimeError):
        client.execute('sleep', 'left')
    assert not calls
    import time
    client.session, client.control_active = 'session', True
    client.focused, client.ui_tick = True, time.monotonic()
    client.execute('sleep', 'left')
    assert calls[0][0][1:] == ('teleop_sleep', {'session': 'session', 'arm': 'left'})
    client.cancel_detour = True
    with pytest.raises(RuntimeError):
        client.execute('sleep', 'left')
    assert len(calls) == 1


def test_real_model_sleep_route_without_path_collision_check():
    cfg = settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('offline model environment unavailable')
    script = r"""
import numpy as np
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.sleep_return import plan_sleep_return
k=Kinematics(settings())
s={'joints_rad':np.deg2rad([-35.179,45.298,-21.786,97.077,14.347,-108.083]).tolist(),
   'gripper':{'width_m':.03}}
goal=np.array(s['joints_rad']);goal[0]+=np.deg2rad(1)
p=plan_sleep_return(k,s,{'goal_joints_rad':goal.tolist()},{},{'planning_timeout_s':8})
assert p['samples']==0 and p['scope']=='none' and 'target_tcp' not in p
rng=np.random.default_rng(1234)
for _ in range(200):
 q=rng.uniform(k.model.lowerPositionLimit[:6],k.model.upperPositionLimit[:6])
 bad=dict(s,joints_rad=q.tolist())
 if k.bounded_gripper_collision(k.configuration(bad),.03):break
else:raise AssertionError('no collision fixture found')
def forbidden(*a,**kw):raise AssertionError('sleep queried path collision')
k.bounded_gripper_collision=forbidden
p=plan_sleep_return(k,bad,{'goal_joints_rad':q.tolist()},{},{'planning_timeout_s':8})
assert not p['path_collision_check']
print('offline real-model sleep checks passed',p['samples'])
"""
    result = subprocess.run([cfg['robot_python'], '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('interruption', ['menu', 'stop', 'timeout', 'config'])
def test_sleep_late_geometry_cannot_send_after_cancel(sleep_controller, interruption):
    f = sleep_controller
    executing(f)
    def submit(fn, *args):
        def result(**_):
            if interruption == 'menu':
                f.manager.input({'session': f.manager.session, 'sequence': f.manager.sequence + 1,
                                 'keys': [], 'cancel_detour': True})
            elif interruption == 'stop':
                f.manager.fail('operator stop')
            elif interruption == 'timeout':
                f.clock[0] += .3
            else:
                f.token[0] = 'changed'
            return {'joint_sdk': [100, 0, 0, 0, 0, 0], 'velocity': [0.] * 6, 'samples': 80}
        return SimpleNamespace(result=result)
    f.robot.geometry_pool = SimpleNamespace(submit=submit)
    f.commands.clear()
    if interruption == 'config':
        with pytest.raises(RuntimeError, match='configuration changed'):
            f.manager.step()
    else:
        f.manager.step()
    assert not any(method == 'JointCtrl' for _, method, _ in f.commands)


@pytest.mark.parametrize('change', ['drift', 'menu'])
def test_settling_cannot_claim_completion_after_drift_or_cancellation(sleep_controller, change):
    f = sleep_controller
    plan = executing(f)
    f.states['left']['joints_rad'] = list(plan['goal_joints_rad'])
    f.states['left']['controller']['motion_status'] = 0
    tick(f)
    if change == 'drift':
        f.states['left']['joints_rad'][0] += np.deg2rad(.2)
    else:
        f.manager.input({'session': f.manager.session, 'sequence': f.manager.sequence + 1,
                         'keys': [], 'cancel_detour': True})
    tick(f, 15)
    assert f.manager.sleep_return['state'] == ('blocked' if change == 'drift' else 'cancelled')
    assert not any(row['type'] == 'sleep_completed' for row in f.manager.trace.rows)
