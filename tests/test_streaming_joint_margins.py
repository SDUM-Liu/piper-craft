"""Controller-owned angle limits with real local IK/collision checks; no CAN."""
from pathlib import Path
import subprocess

import pytest

from astra_vla.config import ROOT, settings


def test_streaming_and_hold_leave_absolute_joint_angles_to_controller():
    cfg = settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('configured model environment unavailable')
    script = r'''
import copy,math,numpy as np
from dataclasses import asdict
from scipy.spatial.transform import Rotation
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.teleop import Policy
from astra_vla.workers import teleop
k=Kinematics(settings());k.prepare_teleop_ik();policy=Policy.load()
teleop.initialize_teleop_geometry(k)
g={'scope':'tcp_jog','tcp':np.eye(4).tolist()}
def state(q):
 s={'joints_rad':np.deg2rad(q).tolist(),'gripper':{'width_m':.03},
    'controller':{'ctrl_mode':1,'motors_enabled':[True]*6}}
 f=k.placement(k.configuration(s)).homogeneous.copy()
 s['ee_pose']={'position_m':f[:3,3].tolist(),
   'rpy_deg':Rotation.from_matrix(f[:3,:3]).as_euler('xyz',degrees=True).tolist()}
 return s
def forbidden(*args,**kwargs):raise AssertionError('normal control used recovery joint limits')
k.start_limit_report=k.joint_limit_violations=k.joint_limit_excess=forbidden
for index,value in [(3,97.077),(3,100.01),(3,-100.01),(4,70.01),
                    (4,-70.01),(5,120.01),(0,150.01),(2,.01)]:
 q=[0,50,-40,0,20,0];q[index]=value;s=state(q);saved=copy.deepcopy(s)
 assert teleop.geometry_hold(s,g) is True
 current=k.configuration(s);target=current.copy()
 sign=1 if value>0 else -1;target[index]+=math.radians(.02*sign)
 f=k.placement(target).homogeneous.copy()
 checked=k.teleop_step(s,s,f,g,.03,asdict(policy),[0.]*6)
 goal=np.array(checked['goal_joints_rad'])
 assert sign*(goal[index]-current[index])>math.radians(.005),(index,value,goal)
 assert checked['joint_limit_status']==[] and checked['joint_position_limits']=='controller'
 if index!=3 or abs(value)>100:
  assert goal[index]<k.model.lowerPositionLimit[index] or goal[index]>k.model.upperPositionLimit[index]
 assert max(abs(np.rad2deg(checked['velocity'])))<=policy.joint_acceleration_deg_s2/policy.control_hz+1e-8
 solver=k.regularized_teleop_ik;k.regularized_teleop_ik=None
 again=k.teleop_step(s,s,np.array(checked['tcp_target']),g,.03,asdict(policy),
                     [0.]*6,validated_goal=checked)
 assert again['joint_sdk']==checked['joint_sdk'] and k.regularized_teleop_ik is None
 k.regularized_teleop_ik=solver
 assert s==saved
# Crossing the actual URDF J4 endpoint must pass the full sampled path too.
s=state([0,50,-40,99.98,20,0]);q=k.configuration(s);t=q.copy();t[3]+=math.radians(.03)
r=k.teleop_step(s,s,k.placement(t).homogeneous.copy(),g,.03,asdict(policy),[0.]*6)
assert r['goal_joints_rad'][3]>k.model.upperPositionLimit[3]
# Nonfinite joints and actual self collision remain blockers at entry.
bad=copy.deepcopy(s);bad['joints_rad'][0]=float('nan')
try:teleop.geometry_hold(bad,g)
except ValueError as e:assert 'invalid teleop joint feedback' in str(e)
else:raise AssertionError('nonfinite joints accepted')
folded=state([-.198,-2.235,-.640,-4.956,30.453,2.705])
try:teleop.geometry_hold(folded,g)
except ValueError as e:assert 'self collision' in str(e)
else:raise AssertionError('colliding hold accepted')
# Send-time recheck retains endpoint/rate validation without collision queries.
def forbidden(*a,**kw):raise AssertionError('send-time path collision queried')
k.bounded_gripper_contacts=forbidden
checked=k.teleop_step(s,s,np.array(r['tcp_target']),g,.03,asdict(policy),[0.]*6,validated_goal=r)
assert checked['joint_sdk']==r['joint_sdk'] and not checked['path_collision_check']
print('absolute angle bounds/margins removed; finite feedback, acceleration and entry hold checks retained')
'''
    result = subprocess.run([cfg['robot_python'], '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
