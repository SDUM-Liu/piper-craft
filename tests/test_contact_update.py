"""Real mesh parity for allocation-free local self-collision checks; no CAN."""
import subprocess
from pathlib import Path

import pytest

from astra_vla.config import ROOT, settings


def test_contacts_match_original_world_objects_path_for_all_finger_widths():
    cfg=settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('configured offline geometry environment unavailable')
    script=r'''
import numpy as np,pinocchio as pin
from astra_vla.kinematics import Kinematics
from astra_vla.config import settings
fast=Kinematics(settings());original=Kinematics(settings())
poses=[[-.198,-2.235,-.640,-4.956,30.453,2.705],[-.198,.5,-3,-4.956,30.453,2.705],
 [0,50,-40,0,45,0],[13.258,50.396,-39.885,-12.235,61.502,16.712],
 [0,80,-100,0,40,40],[0,10,-10,-30,20,-20]]
def forbidden(*args,**kwargs):raise AssertionError('self check constructed unused world objects')
fast.objects=forbidden
colliding=free=0
for joints in poses:
 for width in np.linspace(0,.0855,19):
  state={'joints_rad':np.deg2rad(joints).tolist(),'gripper':{'width_m':float(width)}}
  q=fast.configuration(state)
  got=fast.contacts(q)
  original.objects(q)
  pin.computeCollisions(original.geometry,original.gdata,False)
  expected={(p.first,p.second):max(float(c.penetration_depth) for c in result.getContacts())
   for p,result in zip(original.geometry.collisionPairs,original.gdata.collisionResults)
   if result.isCollision() and not original.closed_finger_contact(p,result)}
  assert got.keys()==expected.keys()
  for pair in expected:assert abs(got[pair]-expected[pair])<1e-10
  colliding+=bool(got);free+=not got
assert colliding and free,'parity only covered one collision outcome'
'''
    result=subprocess.run([cfg['robot_python'],'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr


def test_unverified_finger_sweep_still_contains_every_opening():
    cfg = settings()
    if not Path(cfg['robot_python']).exists():
        pytest.skip('configured offline geometry environment unavailable')
    script = r'''
import numpy as np
from astra_vla.kinematics import Kinematics
from astra_vla.config import settings
k=Kinematics(settings())
s={'joints_rad':[0.]*6,'gripper':{'width_m':-.0021}}
# Every sampled exact collision belongs to the conservative swept set.
# Include folded and deliberately colliding arm configurations.
for joints in [[0,0,0,0,0,0],[0,10,-10,0,20,0],[0,50,-40,0,45,0],
               [-3,-2.5,2.5,-8.5,19.7,3.5],[3.796,.5,-.5,-5.106,16.269,9.701]]:
 st=dict(s,joints_rad=np.deg2rad(joints).tolist());q=k.configuration(st)
 swept=k.bounded_gripper_contacts(q,.065)
 for w in np.linspace(0,.065,66):
  qq=k.configuration(dict(st,gripper={'width_m':float(w)}))
  assert set(k.contacts(qq))<=set(swept), (joints,w)

'''
    result = subprocess.run([cfg['robot_python'], '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
