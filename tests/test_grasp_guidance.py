"""Incomplete or stale authored observations must not appear ready for closure."""

import pytest

from astra_vla.grasp_guidance import guide, review_stage
from astra_vla.operation_memory import record_problem
from astra_vla.task_memory import TaskMemory


def review():
    stage = next(s for s in guide()['stages'] if s['id'] == 'preclose')
    return dict(arm='right', observed_at=100., facts={
        name: dict(status='confirmed', observation='合成观察，仅测试完整性',
                   views=['top', 'left_wrist', 'right_wrist', 'sdk'])
        for name in stage['required']})


@pytest.mark.parametrize('missing', ['closure_direction', 'entry_depth', 'auxiliary_clearance'])
def test_closure_cannot_appear_ready_without_orientation_depth_or_other_arm(missing):
    data = review()
    data['facts'].pop(missing)
    result = review_stage('preclose', data, clock=lambda: 100)
    assert not result['ready_for_next_local_action']
    assert missing in [b['fact'] for b in result['blockers']]
    assert not result['motion_authorized']


@pytest.mark.parametrize('arm,auxiliary', [('right', 'left_wrist'), ('left', 'right_wrist')])
def test_depth_requires_correct_auxiliary_wrist_and_unknown_is_not_confirmed(arm, auxiliary):
    data = review()
    data['arm'] = arm
    data['facts']['entry_depth']['views'].remove(auxiliary)
    assert not review_stage('preclose', data, clock=lambda: 100)['ready_for_next_local_action']
    data['facts']['entry_depth']['views'].append(auxiliary)
    assert review_stage('preclose', data, clock=lambda: 100)['ready_for_next_local_action']
    data['facts']['entry_depth']['status'] = 'unknown'
    assert not review_stage('preclose', data, clock=lambda: 100)['ready_for_next_local_action']


@pytest.mark.parametrize('stamp', [39, 101, float('nan'), None, True])
def test_stage_review_rejects_stale_future_or_invalid_observation_time(stamp):
    data = review()
    data['observed_at'] = stamp
    with pytest.raises(ValueError, match='60 seconds'):
        review_stage('preclose', data, clock=lambda: 100)


def test_planning_exposes_order_and_failure_lessons_without_success_maps(tmp_path):
    record_problem('抓取', '未确认方向及夹深', change='补辅助腕', directory=tmp_path/'memory')
    context = TaskMemory(tmp_path/'memory', root=tmp_path).context('继续夹取羽毛球')
    assert not context['maps']
    assert context['problem_lessons'][0]['observed'] == '未确认方向及夹深'
    assert context['desktop_grasp_guide']['order'] == [
        'prepare_views', 'overhead', 'approach', 'align', 'preclose', 'lift_check']
    assert not context['desktop_grasp_guide']['motion_authorized']
