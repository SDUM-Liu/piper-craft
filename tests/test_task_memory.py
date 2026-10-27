"""Synthetic fixtures: durable reflections and fresh scene reviews, never hardware."""

import copy
import json

import pytest

from astra_vla.task_memory import TaskMemory, REUSE_CHECKS


@pytest.fixture
def library(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"task_completed": True, "stop_confirmed": True}))
    (tmp_path / "image.jpg").write_bytes(b"synthetic image fixture")
    for name in ("top", "left_wrist", "right_wrist"):
        (tmp_path / (name + ".jpg")).write_bytes(b"synthetic image fixture")
    (tmp_path / "scene.json").write_text(
        json.dumps(
            {
                "cameras": {
                    name: {"available": True, "stamp": 100.0} for name in ("top", "left_wrist", "right_wrist")
                },
                "robot": {"arms": {name: {"stamp": 100.0, 'joints_rad':[0.]*6,
                                           'ee_pose':{'position_m':[.2,0.,.3], 'rpy_deg':[0.]*3}}
                                   for name in ("left", "right")}},
            }
        )
    )
    clock = [100.0]
    return TaskMemory(tmp_path / "memory", root=tmp_path, clock=lambda: clock[0]), clock


def graph():
    return dict(
        id="store",
        name="归位",
        task="把物体归位",
        description="观察后靠近、释放",
        scene_assumptions=["桌面支撑可见"],
        success_criteria=["物体独立稳定"],
        non_transferable=["不重放旧关节或开度"],
        cases=[
            dict(
                id="one",
                kind="robot_task",
                summary="合成案例：观察桌面支撑后分步释放，物体独立稳定。",
                reflection="分步释放提供再次观察支撑的机会；此测试不验证实体效果。",
                limitations=["合成测试，不是实机成功经验"],
                source="result.json",
                review=dict(
                    outcome="success",
                    basis="visual_review",
                    statement="合成测试证据",
                    evidence=["image.jpg"],
                    result_checks={"task_completed": True},
                ),
            )
        ],
        nodes=[
            dict(
                id="observe",
                operation="观察",
                preconditions="反馈和RGB有效",
                completion="物体及支撑位置可判读",
                adaptation="遮挡则换视角",
                reflection="图像存在不代表能判断支撑关系；看不清需补观察。",
                case_ids=["one"],
                evidence=["image.jpg"],
            ),
            dict(
                id="place",
                operation="释放",
                preconditions="支撑已确认",
                completion="物体独立稳定",
                adaptation="分步开爪再观察",
                reflection="释放前后分别观察，避免把持物状态误当作独立稳定。",
                case_ids=["one"],
                evidence=["image.jpg"],
            ),
        ],
        edges=[dict(**{"from": "observe", "to": "place"}, condition="当次支撑已确认", case_ids=["one"])],
    )


def review():
    return dict(
        instruction="把物体归位",
        scene_id="scene-a",
        observed_at=100.0,
        scene_evidence={name: name + ".jpg" for name in ("top", "left_wrist", "right_wrist")},
        scene_snapshot="scene.json",
        checks={key: dict(verdict="pass", reason="当次检查已满足，合成测试") for key in REUSE_CHECKS},
    )


def test_success_graph_requires_new_instruction_scene_review_and_expires(library):
    store, clock = library
    saved = store.put(graph())
    assert saved["revision"] == 1 and saved["reference_only"]
    assert store.context("把物体归位")["maps"][0]["decision"] == "assessment_required"
    assert store.assess("store", review())["decision"] == "use"
    assert store.context("把物体归位", "scene-a")["maps"][0]["decision"] == "use"
    for instruction, scene in [("倒水", "scene-a"), ("把物体归位", "scene-b")]:
        assert store.context(instruction, scene)["maps"][0]["decision"] == "assessment_required"
    clock[0] += 61
    assert store.context("把物体归位", "scene-a")["maps"][0]["decision"] == "assessment_required"


def test_planning_focus_is_available_without_any_prior_success_or_demo(library):
    store, _ = library
    context = store.context("抓起新物体")
    assert not context["maps"] and not context["demonstrations"]
    assert {"observation_views", "grasp_depth"} <= set(context["reuse_checks"])
    assert "另一腕" in context["planning_focus"]["observation_views"]
    assert "两侧接触" in context["planning_focus"]["grasp_depth"]
    assert "抓夹深度" in (store.directory / "task_maps.md").read_text()


@pytest.mark.parametrize("dimension", ["observation_views", "grasp_depth"])
def test_reuse_requires_explicit_view_and_depth_reviews(library, dimension):
    store, _ = library
    store.put(graph())
    assessment = review()
    assessment["checks"].pop(dimension)
    with pytest.raises(ValueError, match=dimension):
        store.assess("store", assessment)
    assessment = review()
    assessment["checks"][dimension] = dict(verdict="reject", reason="关键接触被挡住，不能确认夹深")
    assert store.assess("store", assessment)["decision"] == "reject"
    assert store.context(assessment["instruction"], "scene-a")["maps"][0]["decision"] == "reject"
    assessment["checks"][dimension]["verdict"] = "adapt"
    with pytest.raises(ValueError, match="adjustments"):
        store.assess("store", assessment)
    assessment["adjustments"] = ["暂停抓持推进，补腕部侧视后重新评估接触区域和夹深"]
    assert store.assess("store", assessment)["decision"] == "adapt"


def test_legacy_nine_dimension_review_is_dropped_without_removing_success(library):
    store, _ = library
    store.put(graph())
    store.assess("store", review())
    with store.edit() as state:
        for dimension in ("observation_views", "grasp_depth"):
            state["assessments"][0]["checks"].pop(dimension)
    context = store.context("把物体归位", "scene-a")
    assert context["maps"][0]["decision"] == "assessment_required"
    state = store.read()
    assert not state["assessments"] and len(state["maps"]) == 1
    assert not state["invalidations"]


@pytest.mark.parametrize(
    "change",
    [
        "failed",
        "unknown",
        "simulated",
        "closure_only",
        "missing_evidence",
        "missing_summary",
        "missing_reflection",
        "missing_limitations",
        "missing_node_reflection",
        "node_provenance",
        "edge_provenance",
        "dangling_edge",
        "expired",
    ],
)
def test_unsuccessful_or_unproven_routes_never_enter_library(library, change):
    store, _ = library
    data = graph()
    case = data["cases"][0]
    if change == "failed":
        (store.root / "result.json").write_text(json.dumps({"task_completed": False}))
    elif change == "unknown":
        case["review"]["outcome"] = "unknown"
    elif change == "simulated":
        (store.root / "result.json").write_text(json.dumps({"task_completed": True, "simulated": True}))
    elif change == "closure_only":
        case["review"]["result_checks"] = {"stop_confirmed": True}
    elif change == "missing_evidence":
        case["review"]["evidence"] = ["missing.jpg"]
    elif change in ("missing_summary", "missing_reflection", "missing_limitations"):
        case.pop(change.removeprefix("missing_"))
    elif change == "missing_node_reflection":
        data["nodes"][0].pop("reflection")
    elif change == "node_provenance":
        data["nodes"][0]["case_ids"] = ["unknown"]
    elif change == "edge_provenance":
        data["edges"][0]["case_ids"] = []
    elif change == "dangling_edge":
        data["edges"][0]["to"] = "missing"
    else:
        data["valid_until"] = 99.0
    with pytest.raises(ValueError):
        store.put(data)
    assert not store.read()["maps"]


def test_reference_failure_removes_graph_and_assessments_not_failed_nodes(library):
    store, _ = library
    store.put(graph())
    store.assess("store", review())
    store.invalidate("store", "reference_failure", "新现场夹持不适用，删除参考而非保存失败路线")
    state = store.read()
    assert state["maps"] == state["assessments"] == []
    assert state["invalidations"][-1]["reason"] == "reference_failure"
    assert "nodes" not in state["invalidations"][-1]


@pytest.mark.parametrize("change", ["missing", "edited", "new_failure"])
def test_run_cleanup_or_rewrite_does_not_remove_learned_summaries(library, change):
    store, _ = library
    saved = store.put(graph())
    if change == "missing":
        (store.root / "image.jpg").unlink()
        (store.root / "result.json").unlink()
    elif change == "edited":
        (store.root / "image.jpg").write_bytes(b"different")
    else:
        (store.root / "result.json").write_text(json.dumps({"task_completed": False}))
    store.compact()
    context = store.context("把物体归位")["maps"][0]
    assert context["cases"] == saved["cases"]
    assert context["nodes"] == saved["nodes"]
    assert context["decision"] == "assessment_required"
    assert not store.read()["invalidations"]
    assert "image.jpg" not in json.dumps(context)
    assert "result.json" not in json.dumps(context)


def test_explicit_expiry_still_removes_outdated_knowledge(library):
    store, clock = library
    data = graph()
    data["valid_until"] = 101.0
    store.put(data)
    clock[0] = 102.0
    store.compact()
    assert not store.read()["maps"]
    assert "store**" not in (store.directory / "task_maps.md").read_text()


def test_self_contained_reflections_need_no_run_files_and_revise_after_cleanup(library):
    store, _ = library
    data = graph()
    data["nodes"][0]["reference_state"] = dict(
        source="result.json", arm="right", reference_only=True, joints_deg=[0.0] * 6
    )
    # A reviewed summary can be authored directly, without file pointers.
    store.distill(data)
    for path in store.root.glob("*.*"):
        path.unlink()
    saved = store.put(data)
    assert saved["cases"][0]["reflection"] == data["cases"][0]["reflection"]
    assert saved["nodes"][0]["reference_state"]["joints_deg"] == [0.0] * 6
    assert "source" not in saved["nodes"][0]["reference_state"]
    saved["cases"][0]["limitations"].append("同案澄清，不新增实机成功")
    assert store.put(saved)["revision"] == 2
    assert len(store.context("新任务")["maps"]) == 1


def test_legacy_maps_migrate_after_all_run_files_are_deleted(library):
    store, _ = library
    old = graph()
    for case in old["cases"]:
        for field in ("summary", "reflection", "limitations"):
            case.pop(field)
    for node in old["nodes"]:
        node.pop("reflection")
    old.update(revision=3, fingerprints={"image.jpg": {"sha256": "old"}}, reference_only=True)
    store.directory.mkdir()
    store.path.write_text(json.dumps(dict(version=1, maps=[old], assessments=[{}], invalidations=[])))
    for path in store.root.glob("*.*"):
        path.unlink()
    store.compact()
    state = store.read()
    assert state["version"] == 2 and not state["assessments"]
    migrated = state["maps"][0]
    assert migrated["revision"] == 4
    assert migrated["cases"][0]["summary"] == old["cases"][0]["review"]["statement"]
    assert "未独立" in migrated["cases"][0]["reflection"]
    assert migrated["nodes"][0]["adaptation"] == old["nodes"][0]["adaptation"]
    assert not any(key in json.dumps(migrated) for key in ("image.jpg", "result.json", "fingerprints"))
    store.compact()
    assert store.read()["maps"] == state["maps"]  # Migration is idempotent.


def test_success_revocation_is_explicit_and_clears_scene_assessments(library):
    store, _ = library
    store.put(graph())
    store.assess("store", review())
    store.invalidate("store", "success_revoked", "复核发现原案例完成判定错误")
    assert store.read()["maps"] == store.read()["assessments"] == []


def test_cleanup_of_current_scene_requires_reassessment_but_preserves_knowledge(library):
    store, _ = library
    saved = store.put(graph())
    store.assess("store", review())
    assert store.context("把物体归位", "scene-a")["maps"][0]["decision"] == "use"
    for path in store.root.glob("*.*"):
        path.unlink()
    context = store.context("把物体归位", "scene-a")["maps"][0]
    assert context["decision"] == "assessment_required"
    assert context["assessment"] is None
    assert context["cases"] == saved["cases"] and context["nodes"] == saved["nodes"]


@pytest.mark.parametrize("change", ["failed", "unknown", "simulated", "closure_only"])
def test_inline_case_review_cannot_admit_unsuccessful_or_unreviewed_outcomes(library, change):
    store, _ = library
    data = graph()
    store.distill(data)
    result = data["cases"][0]["review"]
    if change == "failed":
        result["result_checks"]["task_completed"] = False
    elif change == "unknown":
        result["outcome"] = "unknown"
    elif change == "simulated":
        result["result_checks"]["simulated"] = True
    else:
        result["result_checks"] = {"closed": True}
    with pytest.raises(ValueError):
        store.put(data)
    assert not store.read()["maps"]


def test_dynamic_adaptation_and_map_revision_require_new_review(library):
    store, _ = library
    store.put(graph())
    assessment = review()
    assessment["checks"]["objects"] = dict(verdict="adapt", reason="物体尺寸改变")
    with pytest.raises(ValueError, match="adjustments"):
        store.assess("store", assessment)
    assessment["adjustments"] = ["按新RGB重新选抓点，不沿用开度"]
    assert store.assess("store", assessment)["decision"] == "adapt"
    updated = graph()
    updated["description"] = "基于同一成功案例澄清适用边界"
    assert store.put(updated)["revision"] == 2
    assert not store.read()["assessments"]
    assessment["checks"]["scene"] = dict(verdict="reject", reason="当前被遮挡")
    assert store.assess("store", assessment)["decision"] == "reject"


def test_review_rejects_old_missing_camera_feedback_and_modified_scene(library):
    store, clock = library
    store.put(graph())
    bad = review()
    bad["checks"].pop("backend")
    with pytest.raises(ValueError):
        store.assess("store", bad)
    clock[0] = 161.0
    with pytest.raises(ValueError, match="60 seconds"):
        store.assess("store", review())
    clock[0] = 100.0
    snapshot = json.loads((store.root / "scene.json").read_text())
    original = copy.deepcopy(snapshot)
    snapshot["cameras"]["top"]["stamp"] = 90.0
    (store.root / "scene.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="RGB"):
        store.assess("store", review())
    (store.root / "scene.json").write_text(json.dumps(original))
    store.assess("store", review())
    (store.root / "scene.json").write_text(json.dumps(snapshot))
    assert store.context("把物体归位", "scene-a")["maps"][0]["decision"] == "assessment_required"


def test_human_teleop_demonstration_is_not_success_just_because_closed(library):
    store, _ = library
    data = graph()
    case = data["cases"][0]
    case["kind"] = "human_teleop"
    source = dict(kind="human_demonstration", mode="human_teleop", closed=True, outcome="unknown")
    (store.root / "result.json").write_text(json.dumps(source))
    case["review"]["result_checks"] = {"outcome": "unknown"}
    with pytest.raises(ValueError):
        store.put(data)
    source.update(outcome="success", outcome_basis="visual_review", outcome_evidence=["image.jpg"])
    for name in ("top", "left_wrist", "right_wrist"):
        (store.root / (name + ".mp4")).write_bytes(b"synthetic video fixture")
    (store.root / "result.json").write_text(json.dumps(source))
    case["review"]["result_checks"] = {"outcome": "success"}
    assert store.put(data)["cases"][0]["kind"] == "human_teleop"


def test_library_and_reference_state_are_bounded_and_not_executable(library):
    store, _ = library
    for index in range(8):
        data = graph()
        data["id"] += str(index)
        store.put(data)
    with pytest.raises(ValueError, match="8 maps"):
        store.put(graph())
    assert len((store.directory / "task_maps.md").read_text().splitlines()) < 40
    data = graph()
    data["id"] = "store0"
    data["nodes"][0]["reference_state"] = {"source": "result.json", "arm": "right", "reference_only": False}
    with pytest.raises(ValueError, match="reference-only"):
        store.put(data)


def test_memory_auto_discovers_all_real_demos_without_focus_selector(library):
    store, _ = library
    runtime = store.root / ".runtime"
    runtime.mkdir()
    (runtime / "demonstration_reference.json").write_text('{"enabled": false}')
    for name, outcome, simulated in [
        ("first", "success", False),
        ("second", "unknown", False),
        ("failed", "failure", False),
        ("fake", "success", True),
    ]:
        directory = store.root / ("runs/demonstration_" + name)
        directory.mkdir(parents=True)
        for camera in ("top", "left_wrist", "right_wrist"):
            (directory / (camera + ".mp4")).write_bytes(b"synthetic fixture")
        (directory / "demonstration.json").write_text(
            json.dumps(
                dict(
                    kind="human_demonstration",
                    mode="human_teleop",
                    closed=True,
                    error="",
                    simulated=simulated,
                    outcome=outcome,
                    title=name,
                    limitations="另一腕被持物挡住，抓夹深度未知",
                )
            )
        )
    context = store.context("新任务")
    assert {d["title"] for d in context["demonstrations"]} == {"first", "second"}
    assert all(d["limitations"] == "另一腕被持物挡住，抓夹深度未知" for d in context["demonstrations"])
    assert next(d for d in context["demonstrations"] if d["title"] == "second")["success_review_required"]
    assert not context["maps"]
    assert "无需手动选择" in (store.directory / "task_maps.md").read_text()
    assert all(d['success_review_required'] for d in context['demonstrations'])  # A bare success flag is insufficient.


def test_transition_cannot_join_nodes_from_unrelated_success_cases(library):
    store, _ = library
    data = graph()
    second = copy.deepcopy(data['cases'][0])
    second['id'] = 'two'
    data['cases'].append(second)
    data['nodes'][1]['case_ids'] = ['two']
    with pytest.raises(ValueError, match='same successful'):
        store.put(data)


def test_touching_identical_small_evidence_does_not_invalidate_success_map(library):
    import os
    store, _ = library
    store.put(graph())
    os.utime(store.root/'image.jpg', None)
    store.compact()
    assert len(store.read()['maps']) == 1


def test_multiline_names_cannot_break_bounded_memory_index(library):
    store, _ = library
    data = graph()
    data['name'] = '名称\n'*40
    data['task'] = '任务\n'*40
    store.put(data)
    assert len((store.directory/'task_maps.md').read_text().splitlines()) < 20


def test_automatic_discovery_can_page_through_old_demos_without_manual_selection(library):
    store, _ = library
    for i in range(35):
        path = store.root/f'runs/demonstration_{i:02}'
        path.mkdir(parents=True)
        for camera in ('top', 'left_wrist', 'right_wrist'):
            (path/(camera+'.mp4')).write_bytes(b'synthetic video fixture')
        (path/'demonstration.json').write_text(json.dumps(dict(kind='human_demonstration', mode='human_objects',
                                                             closed=True, simulated=False, error='')))
    first = store.context('新指令')
    assert len(first['demonstrations']) == 32 and first['next_demo_offset'] == 32
    second = store.context('新指令', demo_offset=first['next_demo_offset'])
    assert len(second['demonstrations']) == 3 and second['next_demo_offset'] is None
    assert not {d['source'] for d in first['demonstrations']} & {d['source'] for d in second['demonstrations']}
