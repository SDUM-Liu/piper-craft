"""Durable task reflections for Codex planning reference. Never sends motion."""

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import time

from astra_vla.config import ROOT, atomic_json

MAX_MAPS, MAX_NODES, MAX_CASES = 8, 24, 8
REUSE_CHECKS = (
    "instruction",
    "objects",
    "scene",
    "arm_roles",
    "start_state",
    "backend",
    "observation_views",
    "grasp",
    "grasp_depth",
    "completion",
    "reference_limits",
)
PLANNING_FOCUS = {
    "observation_views": "逐阶段写明顶视、操作腕和另一腕各看清什么、哪里遮挡；"
    "未选中臂的腕相机也须主动查看。抓前、试提/短移后、倾倒/插入前及释放前重新核查。"
    "关键接触/高度看不清先暂停、补观察；调整观察臂仍须新RGB/双臂反馈和普通运动保护。",
    "grasp_depth": "闭爪前用腕部近视与可用辅助侧视交叉检查物体进入双指的程度、"
    "两侧接触区域、夹持高度及指尖到桌面/邻物间隙，避免仅指尖浅夹；"
    "并非越深越好，不盲推到底。试提/短移后复查相对滑移/转动。"
    "开度、effort、SDK末端或已提起不能证明夹深/保持力；未知写未知，"
    "无标定不报毫米深度，不跨瓶身/球头/筒身套用旧抓点。",
}
REASONS = ("reference_failure", "outdated", "invalid", "success_revoked")


def sentence(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 1200:
        raise ValueError(name + " requires nonempty text, at most 1200 characters")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", value):
        raise ValueError("invalid task/node/case identifier")
    return value


class TaskMemory:
    def __init__(self, directory=None, *, root=ROOT, clock=time.time):
        self.root, self.clock = Path(root).resolve(), clock
        self.directory = Path(directory) if directory else self.root / "docs/memory"
        self.path = self.directory / "task_maps.json"

    def artifact(self, value):
        if not isinstance(value, str) or Path(value).is_absolute():
            raise ValueError("evidence paths must be relative to the repository root")
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("missing or external evidence: " + value)
        return path

    def document(self, value):
        path = self.artifact(value)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("source/review JSON is too large")
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("source/review must be a JSON object")
        return data

    def fingerprint(self, value):
        path = self.artifact(value)
        stat = path.stat()
        result = {"size": stat.st_size}
        # Large videos remain references, not files to hash on every context read.
        if stat.st_size <= 4 * 1024 * 1024:
            result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            result["mtime_ns"] = stat.st_mtime_ns
        return result

    def read(self):
        if not self.path.exists():
            return {"version": 2, "maps": [], "assessments": [], "invalidations": []}
        state = json.loads(self.path.read_text())
        if state.get("version") == 1:
            # These graphs were reviewed on admission. Preserve their recorded
            # conclusions even when the original run has already been cleaned.
            # Migration cannot invent a causal explanation or new physical facts.
            for graph in state["maps"]:
                for case in graph["cases"]:
                    case.setdefault("summary", case["review"]["statement"])
                    case.setdefault("reflection", "旧版未独立总结原因；仅沿用已有节点的条件与调整规则，效果与因果仍需复核。")
                    case.setdefault("limitations", list(graph["non_transferable"]))
                for node in graph["nodes"]:
                    node.setdefault("reflection", "旧版未独立解释策略原因；保留既有条件和调整规则，作用机理未验证。")
                self.distill(graph)
                graph["revision"] += 1
            state["version"] = 2
            state["assessments"] = []
        if state.get("version") != 2:
            raise ValueError("unsupported task-memory version")
        return state

    @staticmethod
    def distill(graph):
        """Keep conclusions and inline references, never run-file dependencies."""
        graph.pop("fingerprints", None)
        for case in graph["cases"]:
            case.pop("source", None)
            case["review"].pop("evidence", None)
        for node in graph["nodes"]:
            node.pop("evidence", None)
            if "reference_state" in node:
                node["reference_state"].pop("source", None)

    @contextmanager
    def edit(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".task_maps.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.read()
            yield state
            state["assessments"] = state["assessments"][-24:]
            state["invalidations"] = state["invalidations"][-16:]
            atomic_json(self.path, state)
            self.render(state)

    def render(self, state):
        def brief(value, limit):
            return ' '.join(str(value).split())[:limit]
        lines = [
            "# 成功案例任务地图",
            "",
            "仅供规划参考，不重放坐标、关节或按键。每次复用先评估新指令/现场，执行中动态调整。",
            "案例事实、反思、策略和适用边界自包含于本机 `task_maps.json`，不依赖运行文件；失败仅进入问题记忆。",
            "",
        ]
        for graph in state["maps"]:
            lines += [
                f"- **{graph['id']}** · {brief(graph['name'], 100)} · v{graph['revision']} · "
                f"{len(graph['nodes'])}节点 / {len(graph['cases'])}成功案例",
                "  任务：" + brief(graph['task'], 180),
            ]
        if not state["maps"]:
            lines.append("暂无经过真实成功复核并总结的地图。")
        lines += ["", "## 自动发现的人类示范（无需手动选择）", ""]
        demonstrations = self.demonstrations(limit=13)
        for item in demonstrations[:12]:
            lines.append("- " + brief(item['title'], 90) + " · " + item["outcome"] + " · `" +
                         brief(item['source'], 240) + "`")
        if not demonstrations:
            lines.append("暂无完整真实示范；新录制保存后自动刷新此索引。")
        if len(demonstrations) > 12:
            lines.append('更多示范由 planning-context 自动分页查看，不需要手动选择参考。')
        lines += [
            "",
            "复用评估：" + " / ".join(REUSE_CHECKS) + "。",
            "- 腕部辅助视角：" + PLANNING_FOCUS["observation_views"],
            "- 抓夹深度：" + PLANNING_FOCUS["grasp_depth"],
            "清理运行记录不删除经验。结论被推翻、参考过时/到期或导致失败时修正或移除；不保留失败路线。",
        ]
        (self.directory / "task_maps.md").write_text("\n".join(lines) + "\n")

    def demonstrations(self, *, offset=0, limit=32):
        """Discover real recordings regardless of the legacy focus selector."""
        from astra_vla.demonstrations import available, read_demonstration

        if type(offset) is not int or offset < 0:
            raise ValueError('invalid demonstration catalogue offset')
        result, skipped = [], 0
        for path in available(self.root / "runs"):
            data = read_demonstration(path)
            if data.get("outcome") == "failure":
                continue
            if skipped < offset:
                skipped += 1
                continue
            qualified = (data.get('outcome') == 'success' and
                         data.get('outcome_basis') in ('visual_review', 'user_confirmation') and
                         bool(data.get('outcome_evidence')))
            if qualified:
                try:
                    for proof in data['outcome_evidence']:
                        self.artifact(proof)
                except (ValueError, OSError):
                    qualified = False
            source = str((path / "demonstration.json").relative_to(self.root))
            result.append(
                dict(
                    source=source,
                    title=data.get("title") or data.get("task") or path.name,
                    description=data.get("description", ""),
                    limitations=data.get("limitations", ""),
                    task=data.get("task", ""),
                    tags=data.get("task_tags", []),
                    mode=data["mode"],
                    outcome='success' if qualified else 'unknown',
                    analysis_required=data.get("annotation_status") != "reviewed"
                    or not qualified,
                    success_review_required=not qualified,
                    frame_index=str((path / "reference/index.md").relative_to(self.root)),
                    export_required=not (path / "reference/index.md").is_file(),
                    videos={
                        n: str((path / (n + ".mp4")).relative_to(self.root))
                        for n in ("top", "left_wrist", "right_wrist")
                    },
                    reference_only=True,
                )
            )
            if len(result) == limit:
                break
        return result

    def validate(self, graph):
        if len(json.dumps(graph, ensure_ascii=False, allow_nan=False)) > 200000:
            raise ValueError("task map exceeds bounded storage")
        identifier(graph["id"])
        for field in ("name", "task", "description"):
            sentence(graph[field], field)
        for field in ("scene_assumptions", "success_criteria", "non_transferable"):
            if not isinstance(graph.get(field), list) or not graph[field]:
                raise ValueError(field + " must explicitly describe applicability")
            for item in graph[field]:
                sentence(item, field)
        cases = set()
        if not 1 <= len(graph["cases"]) <= MAX_CASES:
            raise ValueError("a map requires 1–8 reviewed successful cases")
        for case in graph["cases"]:
            key = identifier(case["id"])
            if key in cases:
                raise ValueError("duplicate case")
            cases.add(key)
            for field in ("summary", "reflection"):
                sentence(case.get(field), "case " + field)
            if not isinstance(case.get("limitations"), list) or not case["limitations"]:
                raise ValueError("case limitations must describe the limits of the conclusion")
            for limitation in case["limitations"]:
                sentence(limitation, "case limitation")
            if case["kind"] not in ("robot_task", "human_objects", "human_teleop"):
                raise ValueError("unknown success case kind")
            review = case["review"]
            if review.get("outcome") != "success" or review.get("basis") not in (
                "user_confirmation",
                "visual_review",
            ):
                raise ValueError("success needs explicit physical outcome review")
            sentence(review["statement"], "success statement")
            checks = review.get("result_checks", {})
            if not isinstance(checks, dict) or not checks:
                raise ValueError("success needs explicit physical result checks")
            if self.unsuccessful(checks):
                raise ValueError("failed/incomplete/simulated cases cannot support a map")
            if not set(checks) - {
                "closed",
                "stop_confirmed",
                "final_stop_confirmed",
                "final_stopped",
                "error",
            }:
                raise ValueError("file closure or stopping alone is not task success")
        nodes = set()
        if not 1 <= len(graph["nodes"]) <= MAX_NODES or len(graph["edges"]) > MAX_NODES * 2:
            raise ValueError("task map node/edge budget exceeded")
        for node in graph["nodes"]:
            key = identifier(node["id"])
            if key in nodes:
                raise ValueError("duplicate node")
            nodes.add(key)
            for field in ("operation", "preconditions", "completion", "adaptation", "reflection"):
                sentence(node.get(field), field)
            if not node.get("case_ids") or not set(node["case_ids"]) <= cases:
                raise ValueError("every node requires successful case provenance")
            if "reference_state" in node:
                reference = node["reference_state"]
                if reference.get("reference_only") is not True or reference.get("arm") not in (
                    "left",
                    "right",
                    "both",
                ):
                    raise ValueError("recorded state is reference-only, never an executable waypoint")
                if "joints_deg" in reference and (
                    len(reference["joints_deg"]) != 6
                    or not all(
                        isinstance(v, (float, int)) and math.isfinite(v) for v in reference["joints_deg"]
                    )
                ):
                    raise ValueError("reference joints require six finite SDK angles")
        node_cases = {node["id"]: set(node["case_ids"]) for node in graph["nodes"]}
        for edge in graph["edges"]:
            if edge["from"] not in nodes or edge["to"] not in nodes or edge["from"] == edge["to"]:
                raise ValueError("invalid graph edge")
            sentence(edge["condition"], "edge transition condition")
            if not edge.get("case_ids") or not set(edge["case_ids"]) <= cases:
                raise ValueError("every transition requires successful case provenance")
            if not set(edge["case_ids"]) <= node_cases[edge["from"]] & node_cases[edge["to"]]:
                raise ValueError("transition must be supported by the same successful endpoint cases")
        expiry = graph.get("valid_until")
        if expiry is not None and (
            not isinstance(expiry, (int, float)) or not math.isfinite(expiry) or expiry <= self.clock()
        ):
            raise ValueError("task map expired")

    @staticmethod
    def unsuccessful(result):
        return (
            result.get("simulated")
            or result.get("task_completed") is False
            or result.get("success") is False
            or result.get("outcome") in ("failure", "failed", "incomplete", "unknown", "pending")
            or result.get("error")
        )

    def review_sources(self, graph):
        """Optional run import: cross-check once, then discard all file links.

        Inline reviews are authored by Codex after observing a real outcome.
        Structural validation cannot independently establish physical success.
        """
        paths = set()
        for case in graph["cases"]:
            if "source" not in case:
                if case["review"].get("evidence"):
                    raise ValueError("run evidence import needs a source result")
                continue
            source = self.document(case["source"])
            if self.unsuccessful(source):
                raise ValueError("failed/incomplete/simulated cases cannot support a map")
            review = case["review"]
            if not review.get("evidence"):
                raise ValueError("success needs a physical outcome review before run import")
            paths.update(review["evidence"])
            if any(k not in source or source[k] != v for k, v in review["result_checks"].items()):
                raise ValueError("success criteria do not match source result")
            if case["kind"] in ("human_objects", "human_teleop"):
                if (
                    source.get("kind") != "human_demonstration"
                    or source.get("mode") != case["kind"]
                    or not source.get("closed")
                    or source.get("outcome") != "success"
                    or source.get("outcome_basis") not in ("visual_review", "user_confirmation")
                    or not source.get("outcome_evidence")
                ):
                    raise ValueError("human demonstrations need a reviewed successful outcome")
                paths.update(source["outcome_evidence"])
                for camera in ("top", "left_wrist", "right_wrist"):
                    paths.add(str(Path(case["source"]).parent / (camera + ".mp4")))
        for node in graph["nodes"]:
            paths.update(node.get("evidence", []))
            source = node.get("reference_state", {}).get("source")
            if source is not None:
                paths.add(source)
        for path in paths:
            self.artifact(path)

    def put(self, graph):
        graph = copy.deepcopy(graph)
        self.validate(graph)
        self.review_sources(graph)
        self.distill(graph)
        with self.edit() as state:
            old = next((m for m in state["maps"] if m["id"] == graph["id"]), None)
            if old is None and len(state["maps"]) >= MAX_MAPS:
                raise ValueError("8 maps already stored; remove obsolete references first")
            graph.update(
                revision=old["revision"] + 1 if old else 1,
                updated_at=self.clock(),
                reference_only=True,
            )
            state["maps"] = [m for m in state["maps"] if m["id"] != graph["id"]] + [graph]
            state["assessments"] = [a for a in state["assessments"] if a["map_id"] != graph["id"]]
        return graph

    def stale(self, graph):
        try:
            if graph.get("valid_until") is not None and graph["valid_until"] <= self.clock():
                return "outdated"
            self.validate(graph)
        except (OSError, ValueError, KeyError, TypeError):
            return "invalid"
        return None

    def remove(self, state, key, reason, explanation):
        state["maps"] = [m for m in state["maps"] if m["id"] != key]
        state["assessments"] = [a for a in state["assessments"] if a["map_id"] != key]
        state["invalidations"].append(
            dict(map_id=key, reason=reason, explanation=explanation[:500], stamp=self.clock())
        )

    def invalidate(self, key, reason, explanation):
        if reason not in REASONS:
            raise ValueError("unknown invalidation reason")
        sentence(explanation, "invalidation explanation")
        with self.edit() as state:
            if not any(m["id"] == key for m in state["maps"]):
                raise ValueError("unknown map")
            self.remove(state, key, reason, explanation)

    def compact(self):
        with self.edit() as state:
            # Older reviews did not assess wrist visibility or insertion depth.
            state["assessments"] = [
                a for a in state["assessments"]
                if set(a.get("checks", {})) == set(REUSE_CHECKS) and a["expires_at"] > self.clock()
            ]
            for graph in list(state["maps"]):
                reason = self.stale(graph)
                if reason:
                    self.remove(
                        state,
                        graph["id"],
                        reason,
                        "Stored conclusion invalid or explicitly expired; re-review required.",
                    )

    def assess(self, key, review):
        self.compact()
        graph = next((m for m in self.read()["maps"] if m["id"] == key), None)
        if graph is None:
            raise ValueError("unknown or invalidated task map")
        sentence(review["instruction"], "new instruction")
        sentence(review["scene_id"], "new scene identifier")
        stamp = review["observed_at"]
        if (
            not isinstance(stamp, (int, float))
            or not math.isfinite(stamp)
            or not 0 <= self.clock() - stamp <= 60
        ):
            raise ValueError("reuse assessment requires a scene observed within 60 seconds")
        views = review.get("scene_evidence", {})
        if (
            not isinstance(views, dict)
            or set(views) != {"top", "left_wrist", "right_wrist"}
            or len(set(views.values())) != 3
        ):
            raise ValueError("fresh evidence from all three RGB views is required")
        signatures = {p: self.fingerprint(p) for p in views.values()}
        snapshot = self.document(review["scene_snapshot"])
        signatures[review["scene_snapshot"]] = self.fingerprint(review["scene_snapshot"])
        cameras = snapshot["cameras"].get("cameras", snapshot["cameras"])
        if set(cameras) != {"top", "left_wrist", "right_wrist"} or any(
            not c.get("available") or not 0 <= stamp - c.get("stamp", 0) <= 0.6 for c in cameras.values()
        ):
            raise ValueError("assessment requires three fresh RGB views")
        arms = snapshot["robot"]["arms"]
        if set(arms) != {"left", "right"} or any(
            not 0 <= stamp - a.get("stamp", 0) <= 0.3 for a in arms.values()
        ):
            raise ValueError("assessment requires fresh feedback from both arms")
        for arm in arms.values():
            rows = (arm.get('joints_rad', []), arm.get('ee_pose', {}).get('position_m', []),
                    arm.get('ee_pose', {}).get('rpy_deg', []))
            if (any(not isinstance(row, list) or len(row) != size for row, size in zip(rows, (6, 3, 3))) or
                    any(not isinstance(v, (int, float)) or not math.isfinite(v) for row in rows for v in row) or
                    arm.get('feedback_coherent') is False):
                raise ValueError('assessment requires valid SDK joint/pose feedback')
        checks = review.get("checks", {})
        if set(checks) != set(REUSE_CHECKS):
            raise ValueError("review every applicability dimension: " + ", ".join(REUSE_CHECKS))
        verdicts = []
        for check in checks.values():
            if check["verdict"] not in ("pass", "adapt", "reject"):
                raise ValueError("invalid reuse verdict")
            sentence(check["reason"], "reuse assessment reason")
            verdicts.append(check["verdict"])
        decision = "reject" if "reject" in verdicts else "adapt" if "adapt" in verdicts else "use"
        if decision == "adapt" and not review.get("adjustments"):
            raise ValueError("scene differences require explicit dynamic adjustments")
        for adjustment in review.get("adjustments", []):
            sentence(adjustment, "dynamic adjustment")
        if review.get("node_id") and review["node_id"] not in {n["id"] for n in graph["nodes"]}:
            raise ValueError("unknown current node")
        result = dict(
            copy.deepcopy(review),
            map_id=key,
            revision=graph["revision"],
            decision=decision,
            evidence_fingerprints=signatures,
            created_at=self.clock(),
            expires_at=stamp + 60,
            reference_only=True,
        )
        with self.edit() as state:
            current = next((m for m in state["maps"] if m["id"] == key), None)
            if current is None or current["revision"] != graph["revision"]:
                raise ValueError("map changed during assessment; review again")
            state["assessments"].append(result)
        return result

    def context(self, instruction, scene_id="", demo_offset=0):
        self.compact()
        state, result = self.read(), []
        for graph in state["maps"]:
            assessment = next(
                (
                    a
                    for a in reversed(state["assessments"])
                    if a["map_id"] == graph["id"]
                    and a["revision"] == graph["revision"]
                    and a["instruction"] == instruction
                    and a["scene_id"] == scene_id
                    and a["expires_at"] > self.clock()
                ),
                None,
            )
            if assessment:
                try:
                    if any(
                        self.fingerprint(p) != sig for p, sig in assessment["evidence_fingerprints"].items()
                    ):
                        assessment = None
                except (ValueError, OSError):
                    assessment = None
            result.append(
                dict(
                    id=graph["id"],
                    name=graph["name"],
                    task=graph["task"],
                    description=graph['description'],
                    cases=graph['cases'],
                    success_criteria=graph['success_criteria'],
                    revision=graph["revision"],
                    decision=assessment["decision"] if assessment else "assessment_required",
                    assessment=assessment,
                    nodes=graph["nodes"],
                    edges=graph["edges"],
                    scene_assumptions=graph["scene_assumptions"],
                    non_transferable=graph["non_transferable"],
                )
            )
        demonstrations = self.demonstrations(offset=demo_offset, limit=33)
        return dict(
            instruction=instruction,
            scene_id=scene_id,
            reference_only=True,
            reuse_checks=list(REUSE_CHECKS),
            planning_focus=copy.deepcopy(PLANNING_FOCUS),
            maps=result,
            demonstrations=demonstrations[:32],
            next_demo_offset=demo_offset+32 if len(demonstrations) > 32 else None,
            workflow="Codex automatically reads candidates, examines all views, names/describes and "
            "reviews pending demos; distill successful cases into self-contained facts, reflections, "
            "strategies and limits. Run cleanup does not invalidate learned summaries. Reassess the new "
            "instruction/scene before reuse and after meaningful changes; never replay.",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    put = sub.add_parser("put")
    put.add_argument("file", type=Path)
    assess = sub.add_parser("assess")
    assess.add_argument("map_id")
    assess.add_argument("review", type=Path)
    context = sub.add_parser("context")
    context.add_argument("--instruction", required=True)
    context.add_argument("--scene-id", default="")
    context.add_argument('--demo-offset', type=int, default=0)
    invalidate = sub.add_parser("invalidate")
    invalidate.add_argument("map_id")
    invalidate.add_argument("--reason", choices=REASONS, required=True)
    invalidate.add_argument("--explanation", required=True)
    sub.add_parser("compact")
    args = parser.parse_args()
    library = TaskMemory(args.directory)
    if args.action == "put":
        result = library.put(json.loads(args.file.read_text()))
    elif args.action == "assess":
        result = library.assess(args.map_id, json.loads(args.review.read_text()))
    elif args.action == "context":
        result = library.context(args.instruction, args.scene_id, args.demo_offset)
    elif args.action == "invalidate":
        result = library.invalidate(args.map_id, args.reason, args.explanation)
    else:
        result = library.compact()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
