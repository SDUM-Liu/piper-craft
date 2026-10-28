"""Native short-probe dispatch/cancellation with SDK spies; no CAN access."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from astra_vla.workers.native_probe import pose_sdk
import test_teleop as fixtures

controller = fixtures.controller


def open_probe(f):
    for arm in f.robot.arms:
        f.robot.arms[arm].EndPoseCtrl = lambda *v, arm=arm: f.commands.append((arm, "EndPoseCtrl", v))
    return f.manager.open({"arm": "right", "speed_profile": "slow", "native_pose_probe": True})["session"]


def test_pose_units_and_orientation_roundtrip():
    t = np.eye(4)
    t[:3, 3] = [0.047481, -0.004268, 0.170996]
    t[:3, :3] = Rotation.from_euler("xyz", [-173.059, 67.093, -175.566], degrees=True).as_matrix()
    assert pose_sdk(t) == (47481, -4268, 170996, -173059, 67093, -175566)


@pytest.mark.parametrize('args', [
    {'arm': arm, **options}
    for arm in ('left', 'right')
    for options in (
        {'execution_backend': 'controller_move_p'},
        {'native_pose_probe': True, 'speed_profile': 'slow'},
        {'native_pose_probe': True, 'execution_backend': 'controller_move_p'},
    )
])
def test_native_backend_and_legacy_probe_disabled_before_any_preparation(controller, args):
    def forbidden(*_a, **_kw):
        raise AssertionError('disabled backend reached preparation/geometry')
    controller.robot.geometry_pool.submit = forbidden
    controller.manager.check_preparation = forbidden
    with pytest.raises(ValueError, match='已禁用'):
        controller.manager.open(args)
    assert not controller.commands and not controller.manager.active


def test_only_local_backend_is_advertised():
    from astra_vla.teleop import EXECUTION_BACKENDS
    assert EXECUTION_BACKENDS == ('regularized_move_j',)
