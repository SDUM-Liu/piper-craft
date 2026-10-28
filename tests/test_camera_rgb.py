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


def test_lossless_snapshot_preserves_source_pixels_dimensions_and_stamp():
    import base64
    import cv2
    import numpy as np
    from astra_vla.rgb_frame import snapshot_frames
    pixels = np.random.default_rng(7).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    frame = {'available': True, 'stamp': 100., 'width': 53, 'height': 37, 'jpeg': 'lossy'}
    result = snapshot_frames({'top': frame}, {'top': (100., pixels)}, 'png')['top']
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(result['png']), dtype=np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, pixels)
    assert result['stamp'] == 100 and result['lossless'] and not result['resized']
    assert 'jpeg' not in result and frame['jpeg'] == 'lossy'
    assert not snapshot_frames({'top': frame}, {'top': (99., pixels)}, 'png')['top']['available']
    with pytest.raises(ValueError):
        snapshot_frames({}, {}, 'unsupported')


def test_observation_export_is_read_only_and_rejects_stale_source(tmp_path, monkeypatch):
    import base64
    import json
    import numpy as np
    import cv2
    from astra_vla import observe
    monkeypatch.setattr(observe, 'settings', lambda: {
        'cameras': {n: {'serial': n} for n in ('top', 'left_wrist', 'right_wrist')},
        'limits': {'camera_age_s': .6}})
    pixels = np.zeros((17, 23, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.png', pixels)
    assert ok
    frames = {n: {'serial': n, 'available': True, 'stamp': 100., 'png': base64.b64encode(encoded).decode()}
              for n in ('top', 'left_wrist', 'right_wrist')}
    calls = []
    def rpc(_path, method, args, **kw):
        calls.append((method, args))
        return {'cameras': frames}
    metadata = observe.capture(tmp_path, rpc=rpc, clock=lambda: 100.1)
    assert calls == [('snapshot', {'format': 'png'})]
    assert set(metadata) == set(frames)
    assert 'png' not in json.loads((tmp_path/'observation.json').read_text())['top']
    frames['top']['stamp'] = 99.
    with pytest.raises(RuntimeError, match='stale'):
        observe.capture(tmp_path/'stale', rpc=rpc, clock=lambda: 100.1)
    assert not (tmp_path/'stale').exists()
