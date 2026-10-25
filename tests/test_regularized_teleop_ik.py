"""Real model/IPOPT in the configured environment; no SDK/CAN construction."""
from pathlib import Path
import subprocess

import pytest

from astra_vla.config import ROOT, settings


def test_joint_speed_above_old_cap_passes_solver_and_recheck_with_other_guards():
    cfg = settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('configured offline IK environment unavailable')
    script = r'''
import copy,numpy as np
from dataclasses import asdict
from scipy.spatial.transform import Rotation
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.teleop import Policy
k=Kinematics(settings());k.prepare_teleop_ik()
p=Policy.load().for_speed('fast');policy=asdict(p)
assert 'joint_speed_deg_s' not in policy
s={'joints_rad':np.deg2rad([0,50,-40,0,20,0]).tolist(),'gripper':{'width_m':.03},
   'controller':{'ctrl_mode':1,'motors_enabled':[True]*6}}
def feedback(s):
 f=k.placement(k.configuration(s)).homogeneous.copy()
 s['ee_pose']={'position_m':f[:3,3].tolist(),
   'rpy_deg':Rotation.from_matrix(f[:3,:3]).as_euler('xyz',degrees=True).tolist()}
feedback(s);q=k.configuration(s);goal=q.copy()
# Opposite shoulder/elbow increments keep the TCP turn within its lead.
goal[1:3]+=np.deg2rad([.64,-.64]);target=k.placement(goal).homogeneous.copy()
v=np.deg2rad([0,32,-32,0,0,0]);g={'scope':'tcp_jog','tcp':np.eye(4).tolist()}
r=k.teleop_step(s,s,target,g,.03,policy,v,q[:6])
speed=max(abs(np.rad2deg(r['velocity'])))
assert speed>30, speed
assert max(abs(np.array(r['velocity'])-v))<=np.deg2rad(p.joint_acceleration_deg_s2)/p.control_hz+1e-10
accepted=np.array(r['tcp_target'])
checked=k.teleop_step(s,s,accepted,g,.03,policy,v,q[:6],validated_goal=r)
assert checked['joint_sdk']==r['joint_sdk'] and checked['ik']['fresh_path_rechecked']
try:k.teleop_step(s,s,accepted,g,.03,policy,[0.]*6,q[:6],validated_goal=r)
except ValueError as e:assert str(e)=='teleop joint acceleration exceeds bound', str(e)
else:raise AssertionError('acceleration guard removed')
lagged=copy.deepcopy(s);lagged['joints_rad'][1]-=np.deg2rad(6);feedback(lagged)
try:k.teleop_step(lagged,s,accepted,g,.03,policy,v,q[:6],validated_goal=r)
except ValueError as e:assert str(e)=='teleop joint tracking envelope exhausted', str(e)
else:raise AssertionError('tracking lead guard removed')
print('offline speed above old cap:',speed,'deg/s; acceleration and tracking guards retained')
'''
    result = subprocess.run([cfg['robot_python'], '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ipopt_bounds_fixed_tcp_vertical_and_failure_never_use_debug_iterate():
    cfg=settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('configured offline IK environment unavailable')
    script=r'''
import copy, numpy as np, pinocchio as pin
from dataclasses import asdict
from scipy.spatial.transform import Rotation
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.teleop import Policy,Integrator,keyset
k=Kinematics(settings());solver=k.prepare_teleop_ik()
def forbidden(*a,**kw):raise AssertionError('new solver fell back to Newton')
k.continuous_goal=forbidden
s={'joints_rad':np.deg2rad([0,50,-40,0,20,0]).tolist(),'gripper':{'width_m':.03},
   'controller':{'ctrl_mode':1,'motors_enabled':[True]*6}}
def feedback(measured):
 fk=k.placement(k.configuration(measured)).homogeneous.copy()
 measured['ee_pose']={'position_m':fk[:3,3].tolist(),
  'rpy_deg':Rotation.from_matrix(fk[:3,:3]).as_euler('xyz',degrees=True).tolist()}
feedback(s);saved=copy.deepcopy(s)
tcp=np.eye(4);tcp[:3,3]=[.025,-.015,.145]
policy=Policy(ik_solver='casadi_ipopt',vertical_frame='base',axis_convention='piper_tool',
 joint_acceleration_deg_s2=120,joint_lead_deg=2)
for key in ('w','s','a','d','space','lshift','i','k','j','l','o','p'):
 measured=copy.deepcopy(s);integ=Integrator(measured,tcp,policy=policy)
 initial=integ.target.copy();v=[0.]*6;previous=None
 for tick in range(200 if key=='space' else 20):
  p=integ.propose(k.feedback_pose(measured),keyset([key]),.05,.065)
  assert p is not None
  checked=k.teleop_step(measured,s,p['ee_target'],
   {'scope':'tcp_jog','tcp':tcp.tolist(),'rotation_only':p['rotating']},.03,asdict(policy),v,previous)
  assert checked['ik']['solver']=='casadi_ipopt' and not checked['ik']['fallback_used']
  last=np.array(measured['joints_rad'] if previous is None else previous)
  next_v=np.array(checked['velocity']);a=np.deg2rad(120);dt=.05
  room=np.deg2rad(2)+np.sign(next_v)*(np.array(measured['joints_rad'])-last)
  assert np.all(abs(next_v)*dt+next_v**2/(2*a)<=room+1e-5), str((key,tick,next_v,room,abs(next_v)*dt+next_v**2/(2*a)))
  assert max(abs(np.rad2deg(np.array(checked['velocity'])-v)))<=6+1e-8
  v=checked['velocity'];previous=checked['goal_joints_rad']
  p['tcp_target']=np.array(checked['tcp_target']);p['ee_target']=p['tcp_target']@integ.inverse_tcp
  p['speed_fraction']*=checked['ik']['progress_fraction'];integ.commit(p)
  measured['joints_rad']=previous;feedback(measured)
  actual=k.state_pose_matrix(measured)@tcp
  assert np.linalg.norm(actual[:3,3]-p['tcp_target'][:3,3])<50e-6
  if key in ('i','k','j','l','o','p'):
   assert np.linalg.norm(actual[:3,3]-initial[:3,3])<50e-6
  if key=='space':
   assert abs(actual[0,3]-initial[0,3])<50e-6 and abs(actual[1,3]-initial[1,3])<50e-6
 assert np.linalg.norm(actual[:3,3]-initial[:3,3])>.001 if key in ('w','s','a','d','space','lshift') else True
assert s==saved
# Hard rejection and a solver failure export no fallback/debug iterate.
original=solver.opti
class Fail:
 def __getattr__(self,n):
  if n=='debug':raise AssertionError('unverified debug goal used')
  return getattr(original,n)
 def solve(self):raise RuntimeError('simulated optimizer failure')
 def stats(self):return {'return_status':'Maximum_CpuTime_Exceeded'}
solver.opti=Fail()
try:solver.solve(np.eye(4),tcp,np.array(s['joints_rad']),np.full(6,-2.),np.full(6,2.))
except TimeoutError:pass
else:raise AssertionError('timeout became a target')
assert solver.last is None
solver.opti=original
try:solver.solve(np.eye(4),tcp,np.zeros(6),np.zeros(6),np.zeros(6))
except ValueError as e:assert 'bounds' in str(e)
else:raise AssertionError('invalid bounds accepted')
print('IPOPT used for 12 directions; vertical 10s, fixed TCP, bounds and no fallback passed')
'''
    result=subprocess.run([cfg['robot_python'],'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=35)
    assert result.returncode==0,result.stdout+result.stderr


def test_fresh_integer_recheck_keeps_endpoint_and_checks_path_without_resolving():
    cfg=settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('configured offline IK environment unavailable')
    script=r'''
import copy,numpy as np
from dataclasses import asdict
from scipy.spatial.transform import Rotation
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.teleop import Integrator,Policy,keyset
k=Kinematics(settings());k.prepare_teleop_ik()
s={'joints_rad':np.deg2rad([0,50,-40,0,20,0]).tolist(),'gripper':{'width_m':.03},
 'controller':{'ctrl_mode':1,'motors_enabled':[True]*6}}
def feedback(s):
 fk=k.placement(k.configuration(s)).homogeneous.copy()
 s['ee_pose']={'position_m':fk[:3,3].tolist(),'rpy_deg':Rotation.from_matrix(fk[:3,:3]).as_euler('xyz',degrees=True).tolist()}
feedback(s);tcp=np.eye(4);tcp[2,3]=.145;g={'scope':'tcp_jog','tcp':tcp.tolist()}
p=Policy(ik_solver='casadi_ipopt',translation_m_s=.02,joint_lead_deg=2,
 joint_acceleration_deg_s2=120)
i=Integrator(s,tcp,policy=p);proposal=i.propose(s,keyset(['w']),.05,.065)
r=k.teleop_step(s,s,proposal['ee_target'],g,.031,asdict(p),[0.]*6)
target=np.array(r['tcp_target'])@np.linalg.inv(tcp)
fresh=copy.deepcopy(s);fresh['joints_rad']=(np.array(s['joints_rad'])+.5*(np.array(r['goal_joints_rad'])-s['joints_rad'])).tolist();feedback(fresh)
def forbidden(*args,**kwargs):raise AssertionError('IPOPT was called twice for one command')
k.regularized_teleop_ik.solve=forbidden
v=k.teleop_step(fresh,s,target,g,.031,asdict(p),[0.]*6,validated_goal=r)
assert v['joint_sdk']==r['joint_sdk'] and v['ik']['fresh_path_rechecked']
np.testing.assert_allclose(v['path_joints_rad'][0],fresh['joints_rad'],atol=1e-12)
assert len(v['path_joints_rad'])>=11 and v['samples']>=55
# Recheck still tests all intermediate finger strokes against genuine path.
contacts=k.contacts
def collision(q):return {(0,1)} if abs(q[6]*2-.0305)<1e-8 else contacts(q)
k.contacts=collision
try:k.teleop_step(fresh,s,target,g,.031,asdict(p),[0.]*6,validated_goal=r)
except ValueError as e:assert 'self collision' in str(e)
else:raise AssertionError('fresh recheck skipped gripper envelope')
k.contacts=contacts
bad=copy.deepcopy(r);bad['joint_sdk'][0]+=100000
try:k.teleop_step(fresh,s,target,g,.031,asdict(p),[0.]*6,validated_goal=bad)
except ValueError:pass
else:raise AssertionError('changed integer endpoint escaped accuracy/rate checks')
'''
    result=subprocess.run([cfg['robot_python'],'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
