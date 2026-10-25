"""Actual URDF checks in the robot environment; tests never construct an SDK/CAN owner."""
from pathlib import Path
import subprocess

import pytest

from astra_vla.config import ROOT, settings


def run_model(script):
    python = settings()['robot_python']
    if not Path(python).is_file():
        pytest.skip('configured offline robot environment unavailable')
    result = subprocess.run([python, '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr


def test_controller_pose_is_preserved_even_when_local_fk_disagrees():
    run_model('''
import copy,numpy as np
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
k=Kinematics(settings())
s={'joints_rad':np.deg2rad([0,50,-40,0,20,0]).tolist(),'gripper':{'width_m':.03},
   'ee_pose':{'position_m':[.4,.2,.3],'rpy_deg':[10.,20.,30.]}}
def forbidden(*a,**kw): raise AssertionError('feedback path evaluated local FK')
k.placement=forbidden
result=k.feedback_pose(s)
assert result['ee_pose']==s['ee_pose'] and 'reported_ee_pose' not in result
assert s['ee_pose']['position_m']==[.4,.2,.3]
bad=copy.deepcopy(s);bad['ee_pose']['position_m'][0]=float('nan')
try:k.feedback_pose(bad)
except ValueError:pass
else:raise AssertionError('invalid feedback accepted')
''')


def test_recorded_yaw_disagreement_does_not_stop_validated_joint_hold():
    run_model('''
import numpy as np
from dataclasses import asdict
from scipy.spatial.transform import Rotation
from astra_vla.config import settings
from astra_vla.kinematics import Kinematics
from astra_vla.teleop import Policy
k=Kinematics(settings())
s={'joints_rad':np.deg2rad([0,50,-40,0,20,0]).tolist(),'gripper':{'width_m':.03},
   'controller':{'ctrl_mode':1,'motors_enabled':[True]*6}}
target=k.placement(k.configuration(s)).homogeneous.copy()
rpy=Rotation.from_matrix(target[:3,:3]).as_euler('xyz',degrees=True);rpy[2]+=.338529
s['ee_pose']={'position_m':(target[:3,3]+[.00008,0,0]).tolist(),'rpy_deg':rpy.tolist()}
validated={'joint_sdk':[0,50000,-40000,0,20000,0],'ik':{'solver':'casadi_ipopt'}}
result=k.teleop_step(s,s,target,{'scope':'tcp_jog','tcp':np.eye(4).tolist()},.03,
                    asdict(Policy.load()),[0.]*6,validated_goal=validated)
assert result['reachable'] and result['samples']>=11
assert result['joint_sdk']==validated['joint_sdk']
''')
