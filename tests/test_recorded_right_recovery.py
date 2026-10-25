"""Recorded starts with recovery collision checks disabled; never opens SDK/CAN."""
from test_gravity_recovery import run_model_script


def test_recorded_starts_and_powered_replanning_need_no_contact_envelope():
    run_model_script(r'''
import copy,numpy as np
from dataclasses import asdict
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.gravity_recovery import RecoveryPolicy,validated_recovery_plan,check_recovery_target
from astra_vla.workers import teleop
k=Kinematics(settings());policy=RecoveryPolicy();g={'scope':'tcp_jog','unverified_gripper_max_m':.065}
def forbidden(*args,**kwargs):raise AssertionError('local collision queried during recovery')
k.contacts=k.recovery_contacts=k.bounded_gripper_contacts=forbidden
teleop.initialize_teleop_geometry(k)
# Last blocked start plus historical cold, powered, in-range and left starts.
fixtures=[[-.139,-2.429,-.314,-7.941,29.630,3.627],
 [-.139,-2.429,-.314,-8.459,29.630,3.627],[-.682,-2.177,-.663,-4.312,30.781,2.393],
 [-.690,-2.242,-.643,-4.199,30.887,2.393],[-.722,.443,-.986,-4.299,27.185,-8.751],
 [5.83, -2.46, 1.9420000000000002, -2.578, 19.271, 8.23],[3.796,-1.912,.910,-5.106,16.269,9.701]]
for arm in ('left','right'):
 for joints in fixtures:
  for enabled in (False,True):
   s={'arm':arm,'joints_rad':np.deg2rad(joints).tolist(),
    'gripper':{'width_m':.0644,'faults':[]},
    'ee_pose':{'position_m':[.045,0,.16],'rpy_deg':[0,70,0]},
    'controller':{'ctrl_mode':1 if enabled else 0,'move_mode':1 if enabled else 0,
     'motors_enabled':[enabled]*6,'err_code':0,'arm_status':0}}
   saved=copy.deepcopy(s)
   p=teleop.geometry_recovery_plan(s,s,g,asdict(policy))
   assert p['required'] and p['local_collision_check'] is False
   assert not p['wrist_preparation'] and 'contact_envelope' not in p
   validated_recovery_plan(p,s,policy,s['joints_rad'])
   path=np.array(p['path_joints_rad'])
   np.testing.assert_allclose(path[:,[0,3,4,5]],np.tile(np.deg2rad(joints)[[0,3,4,5]],(len(path),1)),atol=1e-12)
   for row in p['stage_joint_sdk']:check_recovery_target(k,s,row,s['joints_rad'])
   assert s==saved
   held=copy.deepcopy(s)
   held['controller'].update(ctrl_mode=1,move_mode=1,motors_enabled=[True]*6)
   # Torque take-up reaches the legal startup target; retain origin as evidence.
   if 'legal_startup_joint_sdk' in p:
    held['joints_rad']=np.deg2rad(np.array(p['legal_startup_joint_sdk'])/1000).tolist()
   rebound=teleop.geometry_recovery_replan(held,s,g,asdict(policy),p,s['joints_rad'])
   assert rebound['replan_strategy']=='existing_route_revalidated'
   validated_recovery_plan(rebound,held,policy,s['joints_rad'])
   distant_origin=np.array(s['joints_rad']);distant_origin[0]+=np.deg2rad(4.)
   validated_recovery_plan(rebound,held,policy,distant_origin)
''')


def test_recorded_enable_deviation_replans_without_a_travel_budget():
    run_model_script(r'''
import copy,numpy as np
from dataclasses import asdict
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.gravity_recovery import RecoveryPolicy,validated_recovery_plan,check_recovery_target
from astra_vla.workers import teleop
k=Kinematics(settings());policy=RecoveryPolicy();g={'scope':'tcp_jog'}
teleop.initialize_teleop_geometry(k)
s={'arm':'right','joints_rad':np.deg2rad([5.830,-2.460,1.942,-2.578,19.309,8.230]).tolist(),
 'gripper':{'width_m':.0643,'faults':[]},
 'ee_pose':{'position_m':[.051182,.003867,.163889],'rpy_deg':[158.274,75.193,162.555]},
 'controller':{'ctrl_mode':0,'move_mode':0,'motors_enabled':[False]*6,'err_code':0,'arm_status':0}}
p=teleop.geometry_recovery_plan(s,s,g,asdict(policy))
held=copy.deepcopy(s)
held['joints_rad']=np.deg2rad([2.997,-2.050,2.611,-12.569,19.640,4.662]).tolist()
held['controller'].update(ctrl_mode=1,move_mode=1,motors_enabled=[True]*6)
rebound=teleop.geometry_recovery_replan(held,s,g,asdict(policy),p,s['joints_rad'])
assert rebound['replan_strategy']=='existing_route_revalidated'
validated_recovery_plan(rebound,held,policy,s['joints_rad'])
assert np.max(np.abs(np.array(rebound['goal_joints_rad'])-held['joints_rad']))>np.deg2rad(9.)
for target in rebound['stage_joint_sdk']:
 assert check_recovery_target(k,held,target,s['joints_rad'])
bad=rebound['joint_sdk'].copy();bad[3]=200000
try:check_recovery_target(k,held,bad,s['joints_rad'])
except ValueError as exc:assert 'joint bounds' in str(exc)
else:raise AssertionError('recovery joint bounds removed')
''')
