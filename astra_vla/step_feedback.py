"""Bounded SDK readback after each held input segment; contains no robot writes."""

import copy
import math


class KeyStepFeedback:
    def __init__(self, session="", arm=None, history=()):
        self.session, self.arm = session, arm
        self.current, self.reports, self.sequence = None, copy.deepcopy(list(history)[-32:]), 0

    def change(self, keys, monotonic, stamp, reason="keys changed"):
        keys = sorted(keys)
        if self.current is not None and self.current["keys"] == keys:
            return False
        ended = self.current is not None
        if ended:
            row = dict(
                self.current,
                ended_at=stamp,
                duration_s=max(0.0, monotonic - self.current["start_tick"]),
                reason=reason,
                revision=0,
                settled=False,
                finalized=False,
                feedback_status="pending",
            )
            row.pop("start_tick")
            self.reports = (self.reports + [row])[-32:]
        self.current = None
        if keys:
            self.sequence += 1
            self.current = dict(
                session=self.session,
                id=self.sequence,
                arm=self.arm,
                keys=keys,
                started_at=stamp,
                start_tick=monotonic,
                targets_sent=0,
            )
        return ended

    def sent(self, keys):
        for row in self.reports:
            if not row["finalized"]:
                row.update(finalized=True, revision=row["revision"] + 1, interrupted=True)
        if self.current is not None and self.current["keys"] == sorted(keys):
            self.current["targets_sent"] += 1

    def capture(self, state, *, settled, stamp, reason=""):
        changed = []
        for row in self.reports:
            if row["finalized"] or (not settled and row["revision"]):
                continue
            row.update(
                revision=row["revision"] + 1,
                observed_at=stamp,
                settled=bool(settled),
                finalized=bool(settled),
                feedback_status="unavailable",
                feedback_error=reason or "SDK feedback missing/stale",
            )
            try:
                q = state["joints_rad"]
                grip = state["gripper"]
                source_stamp = state["stamp"]
                if (
                    len(q) != 6
                    or not all(math.isfinite(v) for v in q)
                    or not math.isfinite(grip["width_m"])
                    or not math.isfinite(grip.get("effort", 0.0))
                    or not math.isfinite(source_stamp)
                    or not 0 <= stamp - source_stamp <= 0.3
                    or (settled and source_stamp < row["ended_at"])
                    or not 0 <= state["age_s"] <= 0.3
                ):
                    raise ValueError("SDK feedback missing/stale")
                row.update(
                    feedback_status="measured",
                    feedback_error="",
                    source_stamp=source_stamp,
                    joints_deg=[math.degrees(v) for v in q],
                    gripper=copy.deepcopy(grip),
                    sdk_ee_pose=copy.deepcopy(state.get("ee_pose")),
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                # Never retain old measured numbers under a new "settled" label.
                for key in ("joints_deg", "gripper", "sdk_ee_pose", "source_stamp"):
                    row.pop(key, None)
            changed.append(copy.deepcopy(row))
        return changed

    def view(self):
        return copy.deepcopy(self.reports)


def feedback_text(row):
    arm = "左臂" if row["arm"] == "left" else "右臂"
    keys = "+".join(key.upper().replace("LSHIFT", "Shift") for key in row["keys"])
    title = f"#{row['id']} {arm} {keys} {row['duration_s']:.2f}s结束"
    phase = (
        "静止已确认"
        if row["settled"]
        else "输入已变，未确认静止"
        if row.get("interrupted")
        else "松键实读，待静止"
    )
    if row["feedback_status"] != "measured":
        return title + " · " + phase + " · 关节/夹爪反馈不可用：" + row.get("feedback_error", "")
    joints = " ".join(f"J{i + 1}={q:+.2f}°" for i, q in enumerate(row["joints_deg"]))
    grip = row["gripper"]
    effort = f" · effort={grip['effort']:+.3f}" if "effort" in grip else ""
    enabled = "使能" if grip.get("enabled") else "未使能"
    sent = "" if row["targets_sent"] else " · 未发送动作目标"
    return (
        title
        + " · "
        + phase
        + sent
        + " · "
        + joints
        + f" · 夹爪={grip['width_m'] * 1000:.2f}mm"
        + effort
        + " · "
        + enabled
    )
