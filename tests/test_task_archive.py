"""Task/file lifecycle integration with temporary storage and no devices."""
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from astra_vla.operation_memory import record_task
from astra_vla.task_archive import TaskArchive
from astra_vla.teleop import Policy


def test_two_arms_and_restart_share_one_task_and_memory_item(tmp_path, monkeypatch):
    from astra_vla.workers import teleop
    monkeypatch.setattr(teleop, 'ROOT', tmp_path)
    archive = TaskArchive(tmp_path)
    task = archive.begin('收纳羽毛球', '左扶筒，右夹球')
    for index, arm in enumerate(('left', 'right', 'right')):
        trace = teleop.Trace(str(index)*8, arm, Policy())
        trace.activate()
        trace.record('command', method='JointCtrl')
        trace.close({'stop_confirmed': True, 'reason': 'desktop switch'})
        assert trace.path.parent == archive.path(task['task_id'])/'segments'
        assert not (trace.path/'memory_error.json').exists()
    # Reconstructing a client/worker does not create another task.
    status = TaskArchive(tmp_path).status()
    assert status['task_id'] == task['task_id']
    state = json.loads((tmp_path/'docs/memory/state.json').read_text())
    assert len(state['sessions']) == 1 and state['sessions'][0]['session'] == task['task_id']
    assert state['sessions'][0]['commands'] == {'JointCtrl': 3}
    assert '控制分段3' in state['sessions'][0]['summary']
    assert '任务结果=unknown' in state['sessions'][0]['summary']
    assert len(list((tmp_path/'runs').iterdir())) == 1
    finished = archive.finish()
    record_task(archive, finished)
    result = json.loads((archive.path(task['task_id'])/'result.json').read_text())
    assert result['outcome'] == 'unknown' and not result['physical_success_verified']
    assert result['segment_count'] == 3 and result['arms'] == ['left', 'right']
    assert archive.status() is None
    assert archive.begin()['task_id'] != task['task_id']


def test_concurrent_recorders_create_one_task_and_live_writers_block_finish(tmp_path):
    def reserve(index):
        return TaskArchive(tmp_path).reserve('window_'+str(index), 'window')
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(reserve, range(8)))
    assert len({path.parent.parent for path in paths}) == 1
    archive = TaskArchive(tmp_path)
    assert len(archive.status()['artifacts']) == 8
    with pytest.raises(ValueError, match='收尾'):
        archive.finish()
    with pytest.raises(ValueError, match='已有'):
        archive.begin()
    for path in paths:
        archive.update(path, state='closed')
    assert archive.finish()['outcome'] == 'unknown'


def test_no_command_session_leaves_no_segment_or_operation_experience(tmp_path, monkeypatch):
    from astra_vla.workers import teleop
    monkeypatch.setattr(teleop, 'ROOT', tmp_path)
    trace = teleop.Trace('static', 'left', Policy())
    trace.close({'stop_confirmed': True})
    archive = TaskArchive(tmp_path)
    assert not trace.path and not archive.status()['artifacts']
    data = archive.finish()
    record_task(archive, data)
    assert not (tmp_path/'docs/memory/state.json').exists()


def test_success_requires_physical_review_and_failed_task_purges_after_lessons(tmp_path):
    archive = TaskArchive(tmp_path)
    path = archive.reserve('teleop_test', 'teleop', arm='left')
    path.mkdir(parents=True)
    (path/'events.jsonl').write_text('private raw evidence')
    archive.update(path, state='closed', commands={'JointCtrl': 2}, result={'stop_confirmed': True})
    with pytest.raises(ValueError, match='实物'):
        archive.finish(outcome='success')
    with pytest.raises(ValueError, match='知识提炼'):
        archive.finish(outcome='failure', summary='浅夹滑落')
    with pytest.raises(ValueError, match='240'):
        archive.finish(outcome='failure', summary='事实与教训'*60, reviewed=True)
    data = archive.finish(outcome='failure', summary='观察：浅夹滑落。改进：试提前交叉复核接触。夹深未标定。',
                          reviewed=True)
    assert not archive.path(data['task_id']).exists()
    state = json.loads((tmp_path/'docs/memory/state.json').read_text())
    assert len(state['sessions']) == 1 and state['sessions'][0]['evidence'] == ''
    assert not state['sessions'][0]['commands'] and '浅夹滑落' in state['sessions'][0]['summary']


def test_memory_write_failure_keeps_task_active_and_raw_material(tmp_path, monkeypatch):
    from astra_vla import operation_memory
    archive = TaskArchive(tmp_path)
    path = archive.reserve('teleop_test', 'teleop', arm='left')
    path.mkdir(parents=True)
    (path/'events.jsonl').write_text('raw evidence')
    archive.update(path, state='closed')
    def fail(*_args, **_kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(operation_memory, 'record_task', fail)
    with pytest.raises(OSError, match='disk full'):
        archive.finish(outcome='failure', summary='复核后的事实和教训', reviewed=True)
    assert archive.status()['state'] == 'active' and (path/'events.jsonl').exists()


def test_task_id_cannot_escape_runs(tmp_path):
    archive = TaskArchive(tmp_path)
    with pytest.raises(ValueError):
        archive.path('../../outside')


def test_dead_writer_is_interrupted_and_never_automatically_successful(tmp_path, monkeypatch):
    import os
    archive = TaskArchive(tmp_path)
    archive.reserve('teleop_crashed', 'teleop', arm='right')
    def dead(_pid, _signal):
        raise ProcessLookupError()
    monkeypatch.setattr(os, 'kill', dead)
    data = archive.finish()
    assert data['artifacts'][0]['state'] == 'interrupted'
    assert data['outcome'] == 'unknown'
    assert archive.summary(data)['last_stop_confirmed'] is None
    assert not archive.summary(data)['physical_success_verified']


def test_success_result_preserves_explicit_review_instead_of_inferencing_from_stop(tmp_path):
    archive = TaskArchive(tmp_path)
    archive.begin('模拟归档校验')
    data = archive.finish(outcome='success', summary='测试提供的复核文字，仅验证文件逻辑', basis='visual_review')
    result = json.loads((archive.path(data['task_id'])/'result.json').read_text())
    assert result['outcome'] == 'success' and result['basis'] == 'visual_review'
    assert result['summary'] == data['summary']


def test_window_video_joins_same_task_and_blocks_finish_until_encoded(tmp_path, monkeypatch):
    import numpy as np
    from astra_vla import window_recording
    monkeypatch.setattr(window_recording, 'ROOT', tmp_path)
    archive = TaskArchive(tmp_path)
    task = archive.begin('录像与控制同任务')
    recording = window_recording.WindowRecording()
    recording.start((32, 24))
    try:
        assert recording.path.parent.parent == archive.path(task['task_id'])/'videos'
        with pytest.raises(ValueError, match='收尾'):
            archive.finish()
        recording.submit(np.zeros((24, 32, 3), dtype=np.uint8).tobytes(), (32, 24))
    finally:
        recording.close()
    assert recording.status()['state'] == 'saved'
    finished = archive.finish()
    assert archive.summary(finished)['recording_count'] == 1
    assert archive.summary(finished)['segment_count'] == 0
    assert not archive.summary(finished)['physical_success_verified']
