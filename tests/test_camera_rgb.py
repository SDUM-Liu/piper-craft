"""Snapshot and ownership behavior only; no ROS, camera or CAN connection."""
import importlib.util
import sys
import threading
from types import ModuleType

import pytest

from astra_vla.config import ROOT


@pytest.fixture
def camera(monkeypatch):
    for name in ('rospy','cv_bridge','sensor_msgs','sensor_msgs.msg'):
        module=ModuleType(name)
        if name=='cv_bridge':
            module.CvBridge=object
        if name=='sensor_msgs.msg':
            module.Image=object
        monkeypatch.setitem(sys.modules,name,module)
    spec=importlib.util.spec_from_file_location('isolated_rgb_camera',ROOT/'astra_vla/workers/cameras.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    camera=module.Cameras.__new__(module.Cameras)
    camera.config={'cameras':{'top':{'serial':'top'},'left_wrist':{'serial':'left'},'right_wrist':{'serial':'right'}}}
    camera.lock=threading.Lock()
    camera.frames={'top':{'available':True,'serial':'top','stamp':100,'jpeg':'rgb'}}
    camera.errors={}
    monkeypatch.setattr(module.time,'time',lambda:100.2)
    return module,camera


def test_snapshot_only_accepts_matching_native_identity_and_keeps_source_age(camera,monkeypatch):
    module,cam=camera
    def native(*a,**kw):
        return {'cameras':{'left_wrist':{'available':True,'serial':'left','stamp':100.1,'jpeg':'left'},
                           'right_wrist':{'available':True,'serial':'wrong','stamp':100.1,'jpeg':'wrong'}},'errors':{}}
    monkeypatch.setattr(module,'call',native)
    result=cam.snapshot()
    assert result['pipeline']=='rgb_only'
    assert result['cameras']['left_wrist']['jpeg']=='left'
    assert result['cameras']['left_wrist']['age_s']==pytest.approx(.1)
    assert not result['cameras']['right_wrist']['available']
    assert 'depth' not in result['cameras']['left_wrist']


def test_missing_native_owner_preserves_ros_rgb_and_reports_error(camera,monkeypatch):
    module,cam=camera
    def missing(*a,**kw):raise FileNotFoundError('not connected')
    monkeypatch.setattr(module,'call',missing)
    result=cam.snapshot()
    assert result['cameras']['top']['jpeg']=='rgb'
    assert result['cameras']['top']['age_s']==pytest.approx(.2)
    assert 'native_rgb' in result['error']


def test_depth_and_old_recovery_rpc_are_not_supported(camera,monkeypatch):
    module,cam=camera
    with pytest.raises(ValueError):
        cam.dispatch('rgbd_start',{})
    calls=[]
    monkeypatch.setattr(module,'call',lambda *a,**kw:calls.append((a,kw)) or {'started':True})
    assert cam.dispatch('retry',{'camera':'left_wrist'})['started']
    assert calls[0][0][1]=='retry'


def test_camera_fingerprint_requires_no_intrinsics_and_ignores_frame_content():
    from astra_vla.workers.teleop import camera_fingerprint
    frame={'serial':'real-id','width':640,'height':480,'jpeg':'frame1','stamp':1}
    before=camera_fingerprint({'top':frame})
    frame.update(jpeg='frame2',stamp=2)
    assert camera_fingerprint({'top':frame})==before
    frame['width']=1280
    assert camera_fingerprint({'top':frame})!=before
