"""Gravity recovery policy and real-model planning. Never connects CAN."""
from pathlib import Path
import subprocess

import pytest

from astra_vla.config import ROOT, settings
from astra_vla.gravity_recovery import RecoveryPolicy


@pytest.mark.parametrize('values', [{'enabled': 1}, {'max_start_excess_deg': 3.01},
    {'speed_percent': 6}, {'timeout_s': 16},
    {'path_tolerance_deg': .26}, {'feedback_speed_deg_s': 31}, {'inset_deg': float('nan')},
    {'feedback_excess_noise_deg': .051}, {'selective_wrist_prepare': 1},
    {'streamed_recovery': 1}])
def test_recovery_policy_cannot_increase_bounds(values):
    with pytest.raises(ValueError):
        RecoveryPolicy(**values)


def run_model_script(script):
    cfg = settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('configured model environment unavailable')
    result = subprocess.run([cfg['robot_python'], '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_recovery_uses_joint_bounds_without_collision_queries_or_correction_budget():
    run_model_script(r'''
import copy,json,math,numpy as np
from dataclasses import asdict
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.gravity_recovery import (RecoveryPolicy,plan_recovery,validated_recovery_plan,
    rebind_recovery_path,check_recovery_target,recovery_joint_bounds,recovery_limit_excess)
from astra_vla.workers import teleop
k=Kinematics(settings());policy=RecoveryPolicy();g={'scope':'tcp_jog'}
original_upper=k.model.upperPositionLimit.copy()
def forbidden(*args,**kwargs):raise AssertionError('recovery called local collision checker')
k.contacts=k.recovery_contacts=k.bounded_gripper_contacts=forbidden
teleop.initialize_teleop_geometry(k)
s={'arm':'right','joints_rad':np.deg2rad([-49.488,-2.178,.474,70.352,73.683,-58.254]).tolist(),
   'gripper':{'width_m':.0643,'faults':[]},
   'ee_pose':{'position_m':[.1,0,.2],'rpy_deg':[0,70,0]},
   'controller':{'ctrl_mode':0,'move_mode':0,'motors_enabled':[False]*6,'arm_status':0,'err_code':0}}
saved=copy.deepcopy(s)
p=teleop.geometry_recovery_plan(s,s,g,asdict(policy))
assert p['required'] and p['local_collision_check'] is False
assert 'contact_envelope' not in p and 'contact_measure' not in p
assert p['joint_sdk'][4]==73683 and p['joint_sdk'][2]==-500
assert p['legal_startup_joint_sdk'][1]==0
assert all(row[1]>=0 for row in p['stage_joint_sdk'])
validated_recovery_plan(p,s,policy)
rebound=rebind_recovery_path(k,s,s,g,p)
assert rebound['joint_sdk'][4]==73683
for target in p['stage_joint_sdk']:
 check_recovery_target(k,s,target,s['joints_rad'])
for row in p['path_joints_rad']:
 assert row[4]>original_upper[4]
 assert not recovery_limit_excess(k,k.configuration(dict(s,joints_rad=row)))[4]
json.dumps(p,allow_nan=False)
assert np.array_equal(k.model.upperPositionLimit,original_upper) and s==saved
# A legal packet can exceed the former travel budget and differ from the origin.
larger=p['joint_sdk'].copy();larger[4]+=20000
assert check_recovery_target(k,s,larger,s['joints_rad'])
origin=np.array(s['joints_rad']);origin[0]+=math.radians(4)
validated_recovery_plan(p,s,policy,origin)
lower,upper=recovery_joint_bounds(k)
assert np.array_equal(lower,k.model.lowerPositionLimit[:6])
assert np.array_equal(upper[[0,1,2,3,5]],original_upper[[0,1,2,3,5]])
bad=copy.deepcopy(s);bad['joints_rad'][4]=math.radians(-73.683)
try:plan_recovery(k,bad,s,g,policy)
except ValueError as exc:assert 'J5=' in str(exc)
else:raise AssertionError('unrelated J5 lower bound removed')
# Ordinary entry/keyboard control continues to invoke the same collision guard.
try:teleop.geometry_hold(s,g)
except AssertionError as exc:assert 'collision checker' in str(exc)
else:raise AssertionError('ordinary hold bypassed local collisions')
''')


def test_recovery_rejects_ineligible_states_and_keeps_packet_validation():
    run_model_script(r'''
import copy,numpy as np
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.gravity_recovery import RecoveryPolicy,plan_recovery,check_recovery_target
k=Kinematics(settings());g={'scope':'tcp_jog'};policy=RecoveryPolicy()
def forbidden(*a,**kw):raise AssertionError('collision query during recovery')
k.contacts=k.recovery_contacts=k.bounded_gripper_contacts=forbidden
s={'arm':'left','joints_rad':np.deg2rad([0,-2.,.8,0,20,0]).tolist(),
 'gripper':{'width_m':.03,'faults':[]},'ee_pose':{'position_m':[.1,0,.2],'rpy_deg':[0,70,0]},
 'controller':{'ctrl_mode':0,'move_mode':0,'motors_enabled':[False]*6,'err_code':0,'arm_status':0}}
for change in ('large','wrist','enabled','teach','partial','fault','gripper_fault',
               'disabled_policy','selective','nan'):
 bad=copy.deepcopy(s);p=policy
 if change=='large':bad['joints_rad'][1]=np.deg2rad(-3.001)
 if change=='wrist':bad['joints_rad'][5]=np.deg2rad(-120.004)
 if change=='enabled':bad['controller'].update(ctrl_mode=1,motors_enabled=[True]*6)
 if change=='teach':bad['controller']['ctrl_mode']=2
 if change=='partial':bad['controller']['motors_enabled'][0]=True
 if change=='fault':bad['controller']['err_code']=1
 if change=='gripper_fault':bad['gripper']['faults']=['fault']
 if change=='disabled_policy':p=RecoveryPolicy(enabled=False)
 if change=='selective':p=RecoveryPolicy(selective_wrist_prepare=True)
 if change=='nan':bad['joints_rad'][0]=float('nan')
 try:plan_recovery(k,bad,s,g,p)
 except ValueError:pass
 else:raise AssertionError('ineligible '+change+' accepted')
held=copy.deepcopy(s);held['controller'].update(ctrl_mode=1,move_mode=1,motors_enabled=[True]*6)
assert plan_recovery(k,held,s,g,policy)['enabled_start']
ordinary=dict(s,joints_rad=np.deg2rad([0,50,-40,0,20,0]).tolist())
assert not plan_recovery(k,ordinary,s,g,policy)['required']
try:plan_recovery(k,s,s,{},policy)
except ValueError as exc:assert 'robot-local' in str(exc)
else:raise AssertionError('retired workspace recovery accepted')
for packet in ([0]*5,[False]*6,[0.]*6,[0,-1,-500,0,20000,0]):
 try:check_recovery_target(k,s,packet,s['joints_rad'])
 except ValueError:pass
 else:raise AssertionError('invalid/out-of-range packet accepted')
''')
