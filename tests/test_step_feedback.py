"""Readback display tests use synthetic SDK samples; no real control."""

import math

import pytest

from astra_vla.step_feedback import KeyStepFeedback, feedback_text
import test_teleop as fixtures

controller = fixtures.controller


def test_each_release_reports_actual_joints_gripper_then_settled_sample(controller):
    f = controller
    session = f.manager.open({"arm": "left"})["session"]
    f.manager.input(dict(session=session, sequence=0, keys=["w", "q"]))
    f.manager.step()
    f.states["left"]["joints_rad"] = [math.radians(i * 10) for i in range(6)]
    f.states["left"]["gripper"].update(width_m=0.0432, effort=-0.2)
    f.clock[0] += 0.1
    f.manager.input(dict(session=session, sequence=1, keys=[]))
    row = f.manager.view()["step_feedback"][-1]
    assert row["joints_deg"] == pytest.approx([i * 10 for i in range(6)])
    assert row["gripper"]["width_m"] == 0.0432 and row["targets_sent"] == 1
    assert not row["settled"] and "J6=+50.00°" in feedback_text(row)
    assert "43.20mm" in feedback_text(row) and "effort=-0.200" in feedback_text(row)
    for delay in (0.1, 0.6):
        f.clock[0] += delay
        f.manager.last_input = f.clock[0]
        for frame in f.cameras.values():
            frame["stamp"] = f.clock[0]
        f.manager.step()
    final = f.manager.view()["step_feedback"][-1]
    assert final["settled"] and final["revision"] > row["revision"]
    assert any(r["type"] == "key_step_feedback" for r in f.manager.trace.rows)


def test_stop_captures_held_step_and_history_survives_new_session(controller):
    f = controller
    session = f.manager.open({"arm": "right"})["session"]
    f.manager.input(dict(session=session, sequence=0, keys=["q"]))
    f.manager.step()
    f.robot.stop("operator stop")
    row = f.manager.view()["step_feedback"][-1]
    assert row["keys"] == ["q"] and row["settled"] and row["reason"] == "operator stop"
    f.robot.stopped.clear()
    f.robot.fault = None
    f.manager.open({"arm": "left"})
    assert f.manager.view()["step_feedback"][0]["session"] == session


def test_stale_or_nonfinite_feedback_never_appears_as_valid_settled_numbers():
    recorder = KeyStepFeedback("one", "right")
    recorder.change(["w"], 1.0, 100.0)
    recorder.change([], 2.0, 101.0)
    sample = fixtures.state()
    sample["stamp"] = 101.0
    assert recorder.capture(sample, settled=False, stamp=101.0)[0]["feedback_status"] == "measured"
    sample["joints_rad"][2] = float("nan")
    row = recorder.capture(sample, settled=True, stamp=101.0)[0]
    assert row["feedback_status"] == "unavailable" and "joints_deg" not in row
    assert "反馈不可用" in feedback_text(row)


def test_input_combination_change_preserves_its_own_readback_when_motion_continues():
    recorder = KeyStepFeedback("one", "left")
    recorder.change(["w"], 1.0, 100.0)
    recorder.change(["w", "q"], 2.0, 101.0)
    sample = fixtures.state()
    sample["stamp"] = 101.0
    recorder.capture(sample, settled=False, stamp=101.0)
    recorder.sent(["w", "q"])
    sample["stamp"] = 102.0
    sample["joints_rad"] = [1.0] * 6
    assert recorder.capture(sample, settled=True, stamp=102.0) == []
    row = recorder.view()[0]
    assert row["joints_deg"] == [0.0] * 6 and not row["settled"] and row["interrupted"]


def test_heartbeat_dedup_and_bounded_history():
    recorder = KeyStepFeedback("one", "right")
    for i in range(40):
        recorder.change(["w"], float(i), 100.0 + i)
        assert not recorder.change(["w"], i + 0.1, 100.0 + i)
        recorder.change([], i + 0.2, 100.0 + i + 0.2)
    assert len(recorder.view()) == 32 and recorder.view()[-1]["id"] == 40
