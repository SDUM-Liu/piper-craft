"""Hardware-environment Python timeout contract with fake geometry and no SDK."""
import subprocess
from pathlib import Path

import pytest

from astra_vla.config import ROOT,settings


def test_future_timeout_is_wrapped_in_actual_robot_python_without_can():
    cfg=settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('robot Python unavailable')
    script=r'''
import copy,threading
from types import SimpleNamespace
from concurrent.futures import Future
from astra_vla.teleop import Policy
from astra_vla.workers.teleop import TeleopController
clock=SimpleNamespace(time=lambda:100.,monotonic=lambda:100.)
s={'joints_rad':[0.]*6,'ee_pose':{'position_m':[.2,0,.3],'rpy_deg':[0,0,0]},
 'gripper':{'width_m':.03,'enabled':True,'faults':[]},'controller':{'ctrl_mode':1,'move_mode':1,
 'motors_enabled':[True]*6,'arm_status':0,'err_code':0},'stamp':100.,'age_s':.01,
 'rates_hz':{'joint':200,'pose':200},'feedback_stamps':{'joint':100.,'pose':100.}}
class Trace:
 path=None
 def __init__(self,*a):self.rows=[]
 def record(self,kind,**kw):self.rows.append((kind,kw))
 def check(self):pass
 def activate(self):raise AssertionError('expired calculation activated evidence/target')
class Pool:
 def submit(self,fn,*args):
  if fn.__name__=='geometry_step':
   f=Future()
   class Pending:
    def result(self,**kwargs):return f.result(timeout=0)
    def cancel(self):return f.cancel()
   return Pending()
  return SimpleNamespace(result=lambda **kw:True)
robot=SimpleNamespace(lock=threading.Lock(),tx_lock=threading.Lock(),stopped=threading.Event(),
 fault=None,geometry_pool=Pool(),config={'limits':{'maintenance_gripper_max_m':{'left':.065,'right':.0855}}})
robot.state=lambda *a:copy.deepcopy(s)
robot.healthy=robot.check_prepare=robot.state
robot.check_motion_state=lambda *a:None
robot.check_motion_unchanged=lambda *a:None
robot.stationary_feedback=lambda **kw:{'confirmed':True}
robot.arms={'left':None,'right':None}
robot.command=lambda *a:(_ for _ in ()).throw(AssertionError('SDK write reached'))
cameras={n:{'available':True,'stamp':100.,'age_s':.01,'serial':n,'intrinsics':{}}
 for n in ['top','left_wrist','right_wrist']}
tool={'arms':{n:{'T_ee_tcp':[[1,0,0,0],[0,1,0,0],[0,0,1,.145],[0,0,0,1]]} for n in ['left','right']}}
Policy.load=classmethod(lambda cls:Policy(ik_solver='casadi_ipopt'))
c=TeleopController(robot,clock=clock,camera_read=lambda:{'cameras':cameras},
 tool_read=lambda:tool,token_read=lambda:'fixed',trace_factory=Trace,auto_start=False)
v=c.open({'arm':'left'});c.input({'session':v['session'],'sequence':0,'keys':['space']})
c.step()
assert c.active and c.deadline_misses==1
assert any(kind=='deadline_drop' and 'geometry deadline exceeded' in kw['reason'] for kind,kw in c.trace.rows)
print('actual robot Python Future timeout correctly wrapped; no SDK imports/writes')
'''
    r=subprocess.run([cfg['robot_python'],'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=10)
    assert r.returncode==0,r.stdout+r.stderr
