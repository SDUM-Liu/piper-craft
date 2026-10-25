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


@pytest.mark.parametrize(
    "args",
    [
        dict(arm="left", speed_profile="slow", native_pose_probe=True),
        dict(arm="right", speed_profile="normal", native_pose_probe=True),
    ],
)
def test_probe_cannot_use_left_arm_or_fast_profile(controller, args):
    with pytest.raises(ValueError, match="right arm"):
        controller.manager.open(args)
    assert not controller.commands and not controller.manager.active


def test_native_sends_pose_only_and_never_calls_local_ik(controller):
    f = controller
    session = open_probe(f)
    submit = f.robot.geometry_pool.submit

    def measured_only(function, *args):
        assert function.__name__ in ("geometry_hold", "geometry_probe_ready"), "local IK/path solver called"
        return submit(function, *args)

    f.robot.geometry_pool.submit = measured_only
    f.manager.input({"session": session, "sequence": 0, "keys": ["space"]})
    f.manager.step()
    assert {a for a, _, _ in f.commands} == {"right"}
    assert not any(m in ("JointCtrl", "EnableArm", "GripperCtrl") for _, m, _ in f.commands)
    assert any(m == "EndPoseCtrl" for _, m, _ in f.commands)
    assert ("right", "MotionCtrl_2", (1, 0, 5, 0)) in f.commands
    assert f.manager.view()["ik_solver"] == "controller"
    assert f.manager.view()["path_precheck"] == "measured_geometry_only"
    f.manager.input({"session": session, "sequence": 1, "keys": []})
    assert f.manager.phase == "pausing"
    assert f.commands[-1] == ("right", "MotionCtrl_1", (0, 1, 0))


@pytest.mark.parametrize("cancel", ["release", "stop", "stale_input", "stale_rgb", "moved"])
def test_native_cancel_and_guards_prevent_late_pose_write(controller, cancel):
    f = controller
    session = open_probe(f)
    f.manager.input({"session": session, "sequence": 0, "keys": ["space"]})

    def complete():
        if cancel == "release":
            f.manager.input({"session": session, "sequence": 1, "keys": []})
        elif cancel == "stop":
            f.robot.stopped.set()
        elif cancel == "stale_input":
            f.clock[0] += 0.3
        elif cancel == "stale_rgb":
            f.cameras["top"]["age_s"] = 2
        else:
            f.states["right"]["ee_pose"]["position_m"][0] += 0.006

    original = f.robot.geometry_pool.submit

    def submit(function, *args):
        job = original(function, *args)

        def result(**kw):
            value = job.result(**kw)
            complete()
            return value

        from types import SimpleNamespace

        return SimpleNamespace(result=result, cancel=lambda: None)

    f.robot.geometry_pool.submit = submit
    if cancel in ("stop", "stale_rgb", "moved"):
        with pytest.raises(RuntimeError):
            f.manager.step()
    else:
        f.manager.step()
    assert not any(m == "EndPoseCtrl" for _, m, _ in f.commands)


@pytest.mark.parametrize("bound", ["position", "rotation", "joint", "time"])
def test_native_test_envelope_remains_bounded(controller, bound):
    f = controller
    session = open_probe(f)
    f.manager.input({"session": session, "sequence": 0, "keys": ["space"]})
    if bound == "position":
        f.states["right"]["ee_pose"]["position_m"][0] += 0.006
    elif bound == "rotation":
        f.states["right"]["ee_pose"]["rpy_deg"][0] += 3
    elif bound == "joint":
        f.states["right"]["joints_rad"][0] += np.deg2rad(9)
    else:
        f.manager.native_started -= 21
    with pytest.raises(RuntimeError, match="native probe"):
        f.manager.step()
    assert not f.commands


def test_native_disallows_gripper_and_profile_switch(controller):
    f = controller
    session = open_probe(f)
    for args in ({"keys": ["q"]}, {"keys": ["space"], "speed_profile": "normal"}):
        with pytest.raises(ValueError):
            f.manager.input({"session": session, "sequence": 0, **args})
    assert not f.commands
