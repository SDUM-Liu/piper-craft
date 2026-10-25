"""Joint button cancellation, guarded MOVE_J routing and offline geometry; no live RPCs."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from astra_vla.teleop import JOINT_KEYS, Integrator, keyset
import test_teleop as teleop_fixtures
import test_teleop_desktop as desktop_fixtures
from test_teleop_kinematics import run_model

controller = teleop_fixtures.controller
window = desktop_fixtures.window
pg = desktop_fixtures.pg
state = teleop_fixtures.state
transforms = teleop_fixtures.transforms


@pytest.mark.parametrize('keys', [['joint_0_plus'], ['joint_7_minus'], ['joint_1_plus', 'w'],
                                 ['joint_1_plus', 'q'], ['joint_1_plus', 'joint_2_minus']])
def test_joint_input_rejects_mixed_or_unknown_axes(keys):
    with pytest.raises(ValueError):
        keyset(keys)


@pytest.mark.parametrize('key', sorted(JOINT_KEYS))
def test_joint_intent_has_one_axis_and_no_gripper_or_tcp_command(key):
    s = state()
    integrator = Integrator(s, transforms()[0])
    before = copy.deepcopy(s)
    result = integrator.propose(s, keyset([key]), 5., .065)
    assert result['joint_jog'] == key and result['arm_motion'] and not result['gripper_motion']
    assert result['speed_fraction'] == pytest.approx(.2)  # one tick, never catch-up motion
    np.testing.assert_allclose(result['ee_target'][:3, 3], s['ee_pose']['position_m'])
    assert s == before


@pytest.mark.parametrize('cause', ['release', 'drag', 'lost_release', 'focus', 'menu', 'hide', 'backend', 'gear'])
def test_mouse_joint_hold_cancels_without_replay(window, monkeypatch, cause):
    window.client.request('open', 'left')
    window.menu_open = False
    window.draw()
    key = 'joint_4_plus'
    pos = window.joint_rects[key].center
    pressed = [True]
    monkeypatch.setattr(pg.mouse, 'get_pressed', lambda: (pressed[0], False, False))
    monkeypatch.setattr(pg.mouse, 'get_pos', lambda: pos)
    window.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=pos))
    assert window.controls.keys == {key}
    window.publish_intent()
    assert window.controls.keys == {key}
    if cause == 'release':
        window.handle(pg.event.Event(pg.MOUSEBUTTONUP, button=1, pos=(0, 0)))
    elif cause == 'drag':
        window.handle(pg.event.Event(pg.MOUSEMOTION, pos=(0, 0)))
    elif cause == 'lost_release':
        pressed[0] = False
    elif cause == 'focus':
        window.handle(pg.event.Event(pg.WINDOWFOCUSLOST))
    elif cause == 'menu':
        window.activate('menu')
    elif cause == 'hide':
        window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F1, repeat=False))
    elif cause == 'backend':
        window.activate('backend')
    else:
        window.activate('speed')
    window.publish_intent()
    assert not window.controls.keys and window.mouse_joint is None
    pressed[0] = True
    window.publish_intent()
    assert not window.controls.keys


@pytest.mark.parametrize('active,native', [(False, False), (True, True)])
def test_joint_buttons_never_enter_control_or_switch_backend(window, monkeypatch, active, native):
    if active:
        window.client.request('open', 'left')
    if native:
        window.client.execution_backend = 'controller_move_p'
    window.menu_open = False
    window.draw()
    key = 'joint_1_minus'
    pos = window.joint_rects[key].center
    monkeypatch.setattr(pg.mouse, 'get_pressed', lambda: (True, False, False))
    monkeypatch.setattr(pg.mouse, 'get_pos', lambda: pos)
    before = window.client.session, window.client.execution_backend
    window.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=pos))
    assert not window.controls.keys
    assert (window.client.session, window.client.execution_backend) == before


@pytest.mark.parametrize('size,debug,native', [((1280,800),False,False), ((1100,700),False,False),
                                             ((1100,700),False,True), ((1280,800),True,False)])
def test_all_twelve_joint_buttons_fit_and_stay_separate_from_controls(window, size, debug, native):
    window.screen = pg.display.set_mode(size)
    window.menu_open, window.show_debug = False, debug
    if native:
        window.client.execution_backend = 'controller_move_p'
        window.client.robot['teleop']['speed_profile_pending'] = 'fast'
    window.draw()
    assert set(window.joint_rects) == JOINT_KEYS
    for key, rect in window.joint_rects.items():
        assert window.screen.get_rect().contains(rect)
        assert all(not rect.colliderect(other) for name, other in window.buttons.items() if key != name)
        assert all(not rect.colliderect(other) for other in window.key_rects.values())


def test_joint_geometry_all_axes_both_signs_recheck_and_collision_guard():
    run_model(r'''
import copy,numpy as np
from dataclasses import asdict
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.teleop import Policy
k=Kinematics(settings())
def forbidden(*a,**kw):raise AssertionError('joint jog called TCP IK')
k.prepare_teleop_ik=forbidden
s={'joints_rad':np.deg2rad([0,50,-40,0,20,0]).tolist(),'gripper':{'width_m':.03},
   'ee_pose':{'position_m':[.4,.2,.3],'rpy_deg':[10,20,45]}}
original=copy.deepcopy(s)
for profile in ('slow','normal','fast'):
 p=asdict(Policy.load().for_speed(profile))
 for axis in range(6):
  for sign in (-1,1):
   direction=np.zeros(6);direction[axis]=sign
   g={'scope':'tcp_jog','tcp':np.eye(4).tolist(),'joint_jog_direction':direction.tolist(),'joint_jog_gain':.2}
   r=k.teleop_step(s,s,np.eye(4),g,.03,p,[0.]*6)
   delta=np.deg2rad(np.array(r['joint_sdk'])/1000)-s['joints_rad']
   assert delta[axis]*sign>0 and np.max(np.abs(np.delete(delta,axis)))<1e-12
   assert r['ik']['solver']=='joint_jog' and not r['detour_recommended']
   assert r['samples']>=11 and r['path_joints_rad'][0]==s['joints_rad']
   moved=dict(s,joints_rad=(np.array(s['joints_rad'])+delta*.4).tolist())
   check=k.teleop_step(moved,s,np.eye(4),g,.03,p,[0.]*6,validated_goal=r)
   assert check['joint_sdk']==r['joint_sdk'] and check['ik']['fresh_path_rechecked']
assert s==original  # No replacement/comparison of SDK readback with FK.
# Acceleration and following checks remain independent of the TCP pose budget.
p=asdict(Policy.load())
g={'scope':'tcp_jog','tcp':np.eye(4).tolist(),'joint_jog_direction':[1,0,0,0,0,0],'joint_jog_gain':1.}
try:k.teleop_step(s,s,np.eye(4),g,.03,p,[0,1.,0,0,0,0])
except ValueError as e:assert 'acceleration' in str(e)
else:raise AssertionError('acceleration check bypassed')
far=np.array(s['joints_rad']);far[0]+=np.deg2rad(p['joint_lead_deg']+.1)
try:k.teleop_step(s,s,np.eye(4),g,.03,p,[0.]*6,previous_goal=far)
except ValueError as e:assert 'tracking envelope' in str(e)
else:raise AssertionError('following check bypassed')
# A stalled physical axis consumes finite lead and brakes, never accumulates debt.
previous=None;velocity=[0.]*6
for _ in range(50):
 r=k.teleop_step(s,s,np.eye(4),g,.03,p,velocity,previous_goal=previous)
 previous=np.deg2rad(np.array(r['joint_sdk'])/1000);velocity=r['velocity']
 assert np.max(np.abs(previous-s['joints_rad']))<=np.deg2rad(p['joint_lead_deg'])+1e-10
 if r['ik']['progress_fraction']==0:break
else:raise AssertionError('joint jog failed to stop advancing into a stalled axis')
k.bounded_gripper_contacts=lambda *a,**kw:{(1,5):.001}
try:k.teleop_step(s,s,np.eye(4),g,.03,p,[0.]*6)
except ValueError as e:assert 'path has URDF self collision' in str(e)
else:raise AssertionError('joint jog bypassed collision guard')
''')


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_joint_worker_sends_move_j_then_release_pauses_and_drops_target(controller, arm):
    f = controller
    opened = f.manager.open({'arm': arm})
    f.manager.input({'session': opened['session'], 'sequence': 0, 'keys': ['joint_3_minus']})
    f.commands.clear()
    submit = f.robot.geometry_pool.submit
    seen = []
    def joint_step(function, *args):
        if function.__name__ != 'geometry_step':
            return submit(function, *args)
        seen.append(args)
        assert args[3]['joint_jog_direction'] == [0., 0., -1., 0., 0., 0.]
        def result(timeout):
            return {'reachable': True, 'joint_sdk': [0,0,-20,0,0,0], 'velocity':[0,0,-.007,0,0,0],
                    'tcp_target': np.asarray(args[2]).tolist(), 'samples':11,
                    'ik':{'solver':'joint_jog','progress_fraction':1.}}
        return SimpleNamespace(result=result, cancel=lambda:None)
    f.robot.geometry_pool.submit = joint_step
    f.manager.step()
    assert seen and (arm, 'JointCtrl', (0,0,-20,0,0,0)) in f.commands
    assert f.manager.view()['control_frame'] == 'joint'
    assert f.manager.view()['ik_solver'] == 'joint_jog'
    assert all(a == arm for a,_,_ in f.commands)
    assert not any(m in ('GripperCtrl','EndPoseCtrl','EnableArm') for _,m,_ in f.commands)
    f.manager.input({'session': opened['session'], 'sequence': 1, 'keys': []})
    assert f.manager.paused and f.manager.last_goal_joints is None
    count = len([1 for _,m,_ in f.commands if m == 'JointCtrl'])
    f.clock[0] += .05
    for frame in f.cameras.values():
        frame['stamp'] = f.clock[0]
    f.manager.step()
    assert len([1 for _,m,_ in f.commands if m == 'JointCtrl']) == count


@pytest.mark.parametrize('cause', ['collision', 'stop', 'changed_input', 'stale'])
def test_joint_candidate_never_sends_after_guard_or_generation_change(controller, cause):
    f = controller
    session = teleop_fixtures.enter(f, keys=('joint_1_plus',))
    f.commands.clear()
    submit = f.robot.geometry_pool.submit
    def wrapped(function, *args):
        future = submit(function, *args)
        def result(timeout):
            selected = future.result(timeout=timeout)
            selected['ik']['solver'] = 'joint_jog'
            return selected
        return SimpleNamespace(result=result, cancel=lambda: None)
    f.robot.geometry_pool.submit = wrapped
    if cause == 'collision':
        f.geometry_error[0] = ValueError('teleop sampled path has URDF self collision')
    elif cause == 'stop':
        f.completion[0] = lambda: f.robot.stop('during joint calculation')
    elif cause == 'changed_input':
        f.completion[0] = lambda: f.manager.input({'session':session, 'sequence':1, 'keys':[]})
    else:
        def stale(*_):
            raise RuntimeError('Piper feedback missing/stale')
        f.robot.healthy = stale
    if cause == 'stale':
        with pytest.raises(RuntimeError, match='feedback missing/stale'):
            f.manager.calculate_step()
    else:
        f.manager.step()
    assert not any(method == 'JointCtrl' for _,method,_ in f.commands)


def test_native_session_rejects_direct_joint_input_without_mode_switch(controller):
    f = controller
    session = teleop_fixtures.enter(f, keys=())
    f.manager.native_control = True
    f.commands.clear()
    with pytest.raises(ValueError, match='关节点动请先切换到本地方案'):
        f.manager.input({'session':session, 'sequence':1, 'keys':['joint_1_plus']})
    assert not f.commands and not f.manager.keys
