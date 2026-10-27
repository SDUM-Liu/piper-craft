"""Live detour lifecycle with isolated SDK spies, plus real offline geometry."""
from concurrent.futures import Future
import copy
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest

from astra_vla.config import ROOT, settings
from astra_vla.teleop import Policy, pose
from astra_vla.teleop_detour import DetourPolicy, endpoint_intent
import test_teleop as fixtures

controller = fixtures.controller
enter = fixtures.enter


def ready_route(f):
    session = enter(f, ('l',))
    f.manager.step()
    pending = Future()
    f.robot.detour_pool = SimpleNamespace(submit=lambda *_args: pending)
    f.manager.detour_policy = DetourPolicy()
    f.commands.clear()
    velocity = list(f.manager.velocity)
    assert f.manager.consider_detour(f.manager.keys, f.manager.generation,
                                   {'detour_recommended': True})
    assert not f.commands
    assert f.manager.velocity == velocity
    assert not f.manager.paused
    f.manager.step()  # Submits the long search while ordinary streaming continues.
    assert f.manager.detour_context['state'] == 'planning'
    assert any(c[1] == 'JointCtrl' for c in f.commands)
    assert not any(c[1] == 'MotionCtrl_1' for c in f.commands)
    plan = {'start_joints_rad': [0.]*6, 'goal_joints_rad': np.deg2rad([5, 0, 0, 0, 0, 0]).tolist(),
            'target_tcp': f.manager.detour_context['target'], 'envelope_max_m': .065, 'samples': 100}
    return session, pending, plan


def test_detour_background_search_has_no_pause_and_preserves_velocity(controller):
    f = controller
    _, pending, plan = ready_route(f)
    old = list(f.manager.velocity)
    pending.set_result(plan)
    f.commands.clear()
    f.manager.step()
    assert f.manager.detour_context['state'] == 'executing'
    assert f.manager.velocity == old
    assert not f.commands


@pytest.mark.parametrize('release_phase', ['starting', 'planning', 'executing'])
def test_detour_release_keeps_fixed_goal_and_uses_independent_fast_policy(controller, release_phase):
    f = controller
    session, pending, plan = ready_route(f)
    if release_phase == 'starting':
        f.manager.reset_detour()
        pending = Future()
        f.robot.detour_pool = SimpleNamespace(submit=lambda *_args: pending)
        f.manager.consider_detour(f.manager.keys, f.manager.generation, {'detour_recommended': True})
    if release_phase == 'executing':
        pending.set_result(plan)
        f.manager.step()
    generation = f.manager.generation
    f.commands.clear()
    f.manager.input({'session': session, 'sequence': 1, 'keys': []})
    assert f.manager.generation == generation
    assert f.manager.detour_context is not None and not f.manager.paused
    assert not f.commands
    if release_phase == 'starting':
        f.manager.step()
        plan['target_tcp'] = f.manager.detour_context['target']
    if not pending.done():
        pending.set_result(plan)
        f.manager.step()
    calls = []
    def submit(function, *args):
        calls.append(args)
        return SimpleNamespace(result=lambda **_: {
            'joint_sdk': [100, 0, 0, 0, 0, 0], 'velocity': [0.]*6, 'samples': 70})
    f.robot.geometry_pool = SimpleNamespace(submit=submit)
    f.manager.step()
    assert calls[0][2]['joint_lead_deg'] == 6.
    assert calls[0][2]['joint_acceleration_deg_s2'] == f.manager.policy.joint_acceleration_deg_s2
    assert f.manager.keys == frozenset()
    assert ('left', 'MotionCtrl_2', (1, 1, 100, 0)) in f.commands
    assert ('left', 'JointCtrl', (100, 0, 0, 0, 0, 0)) in f.commands


def test_released_detour_completion_pauses_and_does_not_resume_old_keys(controller):
    f = controller
    session, pending, plan = ready_route(f)
    f.manager.input({'session': session, 'sequence': 1, 'keys': []})
    pending.set_result(plan)
    f.manager.step()
    f.states['left']['joints_rad'] = plan['goal_joints_rad']
    f.states['left']['controller']['motion_status'] = 0
    from scipy.spatial.transform import Rotation
    target = np.array(plan['target_tcp']) @ f.manager.integrator.inverse_tcp
    f.states['left']['ee_pose'] = {'position_m': target[:3, 3].tolist(),
        'rpy_deg': Rotation.from_matrix(target[:3, :3]).as_euler('xyz', degrees=True).tolist()}
    f.commands.clear()
    f.manager.step()
    assert f.manager.detour_context is None and f.manager.paused and not f.manager.keys
    assert not any(c[1] == 'JointCtrl' for c in f.commands)
    f.commands.clear()
    f.manager.step()
    assert not f.commands


def test_released_detour_still_requires_live_input_heartbeat(controller):
    f = controller
    session, pending, plan = ready_route(f)
    f.manager.input({'session': session, 'sequence': 1, 'keys': []})
    pending.set_result(plan)
    f.manager.step()
    f.clock[0] += f.manager.policy.input_timeout_s + .01
    f.commands.clear()
    with pytest.raises(RuntimeError, match='heartbeat missing/stale'):
        f.manager.step()
    assert not any(c[1] == 'JointCtrl' for c in f.commands)


@pytest.mark.parametrize('keys', [['j'], ['l', 'q']])
def test_reverse_or_gripper_change_discards_search(controller, keys):
    f = controller
    session, pending, plan = ready_route(f)
    f.manager.input({'session': session, 'sequence': 1, 'keys': keys})
    assert f.manager.detour_context is None
    assert pending.cancelled()
    f.commands.clear()
    f.manager.step()
    assert not any(r['type'] == 'detour_target' for r in f.manager.trace.rows)
    assert not any(c[1] == 'JointCtrl' for c in f.commands)


@pytest.mark.parametrize('interrupt', ['menu', 'stop', 'timeout', 'config'])
def test_detour_inflight_revalidation_cannot_send_after_cancel(controller, interrupt):
    f = controller
    session, pending, plan = ready_route(f)
    pending.set_result(plan)
    f.manager.step()
    original = f.robot.geometry_pool.submit
    def submit(function, *args):
        if function.__name__ != 'geometry_detour_step':
            return original(function, *args)
        def result(**_kwargs):
            if interrupt == 'menu':
                f.manager.input({'session': session, 'sequence': 1, 'keys': [], 'cancel_detour': True})
            elif interrupt == 'stop':
                f.robot.stopped.set()
            elif interrupt == 'timeout':
                f.clock[0] += .3
            else:
                f.token[0] = 'changed'
            return {'joint_sdk': [100, 0, 0, 0, 0, 0], 'velocity': [0.]*6, 'samples': 70}
        return SimpleNamespace(result=result, cancel=lambda: None)
    f.robot.geometry_pool.submit = submit
    f.commands.clear()
    if interrupt == 'config':
        with pytest.raises(RuntimeError, match='configuration changed'):
            f.manager.step()
    else:
        f.manager.step()
    assert not any(c[1] == 'JointCtrl' for c in f.commands)


def test_detour_execution_emits_fast_move_j_and_menu_cancels(controller):
    f = controller
    session, pending, plan = ready_route(f)
    pending.set_result(plan)
    f.manager.step()
    f.robot.geometry_pool = SimpleNamespace(submit=lambda *_: SimpleNamespace(
        result=lambda **_: {'joint_sdk': [100, 0, 0, 0, 0, 0], 'velocity': [0.]*6, 'samples': 70}))
    f.commands.clear()
    f.manager.step()
    assert f.commands == [('left', 'MotionCtrl_2', (1, 1, f.manager.detour_policy.speed_percent, 0)),
                          ('left', 'JointCtrl', (100, 0, 0, 0, 0, 0))]
    assert f.manager.phase == 'detour_executing'
    f.manager.input({'session': session, 'sequence': 1, 'keys': [], 'cancel_detour': True})
    assert f.manager.detour_context is None and f.manager.paused


def test_detour_late_search_never_reaches_a_new_session(controller):
    f = controller
    session, pending, plan = ready_route(f)
    pending.set_running_or_notify_cancel()
    f.manager.input({'session': session, 'sequence': 1, 'keys': [], 'cancel_detour': True})
    pending.set_result(plan)
    f.commands.clear()
    f.manager.step()
    assert f.manager.detour_context is None
    assert not any(c[1] == 'JointCtrl' for c in f.commands)


def test_detour_completion_returns_to_local_control_without_pause(controller):
    f = controller
    _, pending, plan = ready_route(f)
    pending.set_result(plan)
    f.manager.step()
    f.states['left']['joints_rad'] = plan['goal_joints_rad']
    f.states['left']['controller']['motion_status'] = 0
    from scipy.spatial.transform import Rotation
    target = np.array(plan['target_tcp']) @ f.manager.integrator.inverse_tcp
    f.states['left']['ee_pose'] = {'position_m': target[:3, 3].tolist(),
                                  'rpy_deg': Rotation.from_matrix(target[:3, :3]).as_euler('xyz', degrees=True).tolist()}
    f.commands.clear()
    f.manager.step()
    assert f.manager.detour_context is None
    assert not f.manager.paused
    assert not f.commands
    assert f.manager.keys == frozenset(['l'])


@pytest.mark.parametrize('geometry_s,feedback_s,send_expected', [
    (.042, .002, True),   # 46ms: geometry valid, full frame still has TX budget.
    (.043, .0025, True),  # 48ms: reproduces the failed hardware frame boundary.
    (.042, .006, False),  # 54ms: full frame expired, no SDK write is permitted.
    (.046, 0., False),    # Geometry budget itself is still 45ms.
])
def test_detour_geometry_and_full_frame_deadlines_are_separate(controller, geometry_s, feedback_s, send_expected):
    f = controller
    _, pending, plan = ready_route(f)
    pending.set_result(plan)
    f.manager.step()
    old_feedback = f.robot.healthy
    in_geometry = [False]
    def feedback(*args):
        if in_geometry[0]:
            f.clock[0] += feedback_s
        return old_feedback(*args)
    def result(timeout):
        in_geometry[0] = True
        f.clock[0] += min(timeout, geometry_s)
        if geometry_s > timeout:
            raise TimeoutError('synthetic geometry deadline')
        return {'joint_sdk': [100, 0, 0, 0, 0, 0], 'velocity': [0.]*6, 'samples': 70}
    f.robot.healthy = feedback
    f.robot.geometry_pool = SimpleNamespace(submit=lambda *_: SimpleNamespace(result=result, cancel=lambda: None))
    f.commands.clear()
    f.manager.step()
    assert any(c[1] == 'JointCtrl' for c in f.commands) is send_expected
    if send_expected:
        assert f.manager.deadline_misses == 0
        assert f.manager.cycle_stages['detour_geometry'] == pytest.approx(geometry_s*1000)
        assert f.manager.cycle_stages['detour_feedback_recheck'] == pytest.approx(feedback_s*2000)
    else:
        assert f.manager.deadline_misses == 1


def test_detour_completed_future_cannot_bypass_geometry_deadline(controller):
    f = controller
    _, pending, plan = ready_route(f)
    pending.set_result(plan)
    f.manager.step()
    def result(**_kwargs):
        # A completed Future can return after the caller's timeout has elapsed.
        f.clock[0] += .046
        return {'joint_sdk': [100, 0, 0, 0, 0, 0], 'velocity': [0.]*6, 'samples': 70}
    f.robot.geometry_pool = SimpleNamespace(submit=lambda *_: SimpleNamespace(result=result, cancel=lambda: None))
    f.commands.clear()
    f.manager.step()
    assert not f.commands
    assert f.manager.deadline_misses == 1


def test_detour_fresh_recheck_timeout_drops_candidate_and_preserves_three_miss_stop(controller):
    f = controller
    _, pending, plan = ready_route(f)
    pending.set_result(plan)
    f.manager.step()
    def result(**_kwargs):
        f.states['left']['joints_rad'][0] += np.deg2rad(.06)
        return {'joint_sdk': [100, 0, 0, 0, 0, 0], 'velocity': [0.]*6, 'samples': 70}
    def recheck(*_args):
        raise TimeoutError('detour geometry deadline exceeded')
    f.robot.geometry_pool = SimpleNamespace(submit=lambda *_: SimpleNamespace(result=result, cancel=lambda: None))
    f.robot.recheck_detour_target = recheck
    f.commands.clear()
    old_velocity = list(f.manager.velocity)
    old_goal = copy.deepcopy(f.manager.last_goal_joints)
    for misses in (1, 2):
        f.manager.step()
        assert f.manager.deadline_misses == misses
        assert f.manager.velocity == old_velocity
        assert f.manager.last_goal_joints == old_goal
        assert not f.commands
    with pytest.raises(RuntimeError, match='three consecutive'):
        f.manager.step()
    assert not f.commands


def test_endpoint_intent_uses_sdk_goal_and_no_integrated_detour_debt():
    state = fixtures.state()
    original = copy.deepcopy(state)
    policy = Policy.load()
    target = endpoint_intent(state, np.eye(4), policy, frozenset(['l']), .065, 1.)
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_matrix(target[:3, :3] @ pose(state)[:3, :3].T).as_rotvec()
    assert np.rad2deg(np.linalg.norm(rotation)) == pytest.approx(policy.rotation_deg_s)
    np.testing.assert_allclose(target[:3, 3], state['ee_pose']['position_m'])
    assert state == original
    with pytest.raises(ValueError, match='arm-only'):
        endpoint_intent(state, np.eye(4), policy, frozenset(['l', 'q']), .065, 1.)


def test_real_model_swept_collision_boolean_matches_contact_query():
    cfg = settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('offline model environment unavailable')
    script = r'''
import numpy as np
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
k=Kinematics(settings());rng=np.random.default_rng(20261020)
placements=[obj.placement.homogeneous.copy() for obj in k.geometry.geometryObjects]
rows=[np.deg2rad([-35.179,45.298,-21.786,97.077,14.347,-108.083])]
rows += [rng.uniform(k.model.lowerPositionLimit[:6],k.model.upperPositionLimit[:6]) for _ in range(300)]
colliding=0;clear=0
for envelope in (.001,.064,.1):
 for joints in rows:
  for opening in (0.,.06314,.1):
   q=k.configuration({'joints_rad':joints,'gripper':{'width_m':opening}})
   expected=bool(k.bounded_gripper_contacts(q,envelope))
   assert k.bounded_gripper_collision(q,envelope)==expected,(envelope,opening,joints)
   colliding+=expected;clear+=not expected
assert colliding>0 and clear>0
for obj,original in zip(k.geometry.geometryObjects,placements):
 np.testing.assert_array_equal(obj.placement.homogeneous,original)
'''
    result = subprocess.run([cfg['robot_python'], '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr


def test_real_model_endpoint_route_and_acceleration_handoff():
    cfg = settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('offline model environment unavailable')
    script = r'''
import numpy as np
from dataclasses import asdict
from scipy.spatial.transform import Rotation
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.teleop import Policy
from astra_vla.teleop_detour import plan_detour,DetourPolicy,detour_joint_step
k=Kinematics(settings());p=Policy.load()
transition=dict(asdict(p),joint_lead_deg=6.,speed_percent=100)
s={'joints_rad':np.deg2rad([-35.179,45.298,-21.786,97.077,14.347,-108.083]).tolist(),
   'gripper':{'width_m':.06314}}
# Synthetic model-only fixture; never compare or replace a real SDK reading.
target=k.placement(k.configuration(s)).homogeneous.copy()
target[:3,:3]=Rotation.from_rotvec(target[:3,0]*np.deg2rad(5)).as_matrix()@target[:3,:3]
plan=plan_detour(k,s,target,{'tcp':np.eye(4).tolist()},asdict(DetourPolicy(search_seeds=5)))
assert plan['samples']==0 and plan['scope']=='none'
def forbidden(*a,**kw):raise AssertionError('transition queried path collision')
k.bounded_gripper_collision=forbidden
goal=np.array(plan['goal_joints_rad'])
assert np.all(goal>=k.model.lowerPositionLimit[:6]) and np.all(goal<=k.model.upperPositionLimit[:6])
previous=np.array(s['joints_rad']);velocity=np.deg2rad([1,0,0,2,-1,-1])
consecutive=0;dropped=0;accepted=0
for tick in range(600):
 old_velocity=velocity.copy()
 try:
  r=detour_joint_step(k,s,plan,transition,previous,velocity)
  assert r is not None, tick
  checked=detour_joint_step(k,s,plan,transition,previous,velocity,validated=r)
 except TimeoutError:
  # Match the controller: a timed-out candidate commits neither target nor
  # velocity. Three consecutive missed geometry deadlines must still fail.
  consecutive+=1;dropped+=1
  assert consecutive<3, ('persistent geometry deadline misses',tick)
  continue
 consecutive=0;accepted+=1
 assert checked['joint_sdk']==r['joint_sdk']
 next_q=np.deg2rad(np.array(r['joint_sdk'])/1000)
 velocity=np.array(r['velocity'])
 assert max(abs(velocity-old_velocity))<=np.deg2rad(p.joint_acceleration_deg_s2)/p.control_hz+1e-10
 assert max(abs(next_q-np.array(s['joints_rad'])))<=np.deg2rad(transition['joint_lead_deg'])+1e-10
 previous=next_q;s=dict(s,joints_rad=next_q.tolist())
 if max(abs(previous-goal))<np.deg2rad(.01):break
else:raise AssertionError('route never converged')
print('real model route ticks',tick+1,'accepted',accepted,'deadline drops',dropped)
'''
    result = subprocess.run([cfg['robot_python'], '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
