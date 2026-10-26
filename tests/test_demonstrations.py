"""Synthetic RGB and SDL only; no hardware acceptance or real RPC."""
import json
import threading
import time

import av
import numpy as np
import pytest

from astra_vla.demonstrations import (
    CAMERAS, DemonstrationRecording, available, export_reference, select_reference,
)


def wait_samples(recorder, count):
    deadline = time.monotonic()+5
    while recorder.samples < count:
        assert time.monotonic() < deadline, recorder.status()
        time.sleep(.01)


@pytest.mark.parametrize('mode', ['human_objects', 'human_teleop'])
def test_three_sources_timestamps_trace_reference_and_clear(tmp_path, mode):
    recorder = DemonstrationRecording(tmp_path)
    recorder.start(mode)
    try:
        for index in range(3):
            recorder.next_frame = 0
            cameras = {name: (recorder.started+.1+index*.2+camera*.01, (64, 48),
                              np.full((48, 64, 3), 30+index*50, np.uint8).tobytes())
                       for camera, name in enumerate(CAMERAS)}
            recorder.submit(cameras, dict(requested_keys=['w', 'q'], arms={'right': {'joints_rad': [index]*6}}))
            wait_samples(recorder, index+1)
        # Repeated cached source stamps are not fabricated as new camera frames.
        recorder.next_frame = 0
        recorder.submit(cameras, dict(requested_keys=[]))
        wait_samples(recorder, 4)
    finally:
        recorder.close()
    assert recorder.status()['state'] == 'saved'
    data = json.loads((recorder.path/'demonstration.json').read_text())
    assert data['closed'] and data['camera_frames'] == dict.fromkeys(CAMERAS, 3)
    assert not data['robot_motion_verified']
    trace = [json.loads(line) for line in (recorder.path/'samples.jsonl').read_text().splitlines()]
    assert trace[0]['context']['requested_keys'] == ['w', 'q']
    assert trace[-1]['cameras'] == {}
    for index, name in enumerate(CAMERAS):
        with av.open(str(recorder.path/(name+'.mp4'))) as container:
            frames = list(container.decode(video=0))
        assert len(frames) == 3
        assert float(frames[-1].pts*frames[-1].time_base) == pytest.approx(.5+index*.01, abs=.002)
    assert available(tmp_path) == [recorder.path]
    target = tmp_path/'reference.json'
    select_reference(recorder.path, destination=target)
    assert json.loads(target.read_text())['enabled']
    index = export_reference(recorder.path, 3)
    assert '不重放' in index.read_text()
    assert len(list(index.parent.glob('*.jpg'))) >= 6
    select_reference(destination=target)
    assert not json.loads(target.read_text())['enabled']


def test_missing_camera_or_encoder_failure_is_not_selectable(tmp_path, monkeypatch):
    recorder = DemonstrationRecording(tmp_path)
    recorder.start('human_objects')
    recorder.close()
    assert recorder.status()['state'] == 'error'
    assert not available(tmp_path)
    with pytest.raises(ValueError):
        select_reference(recorder.path, destination=tmp_path/'reference.json')
    def failed(*_args, **_kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(av, 'open', failed)
    recorder.start('human_teleop')
    recorder.submit({'top': (time.time(), (2, 2), bytes(12))}, {})
    recorder.close()
    assert 'disk full' in recorder.status()['error']
    assert not json.loads((recorder.path/'demonstration.json').read_text())['closed']


def test_bounded_queue_and_simulated_reference_rejection(tmp_path):
    recorder = DemonstrationRecording(tmp_path)
    release = threading.Event()
    recorder.encode = lambda: release.wait(3)
    recorder.start('human_objects', simulated=True)
    try:
        for index in range(20):
            recorder.next_frame = 0
            recorder.submit({}, {'index': index})
        assert recorder.pending.qsize() == 2
        assert recorder.dropped == 18
    finally:
        release.set()
        recorder.close()
    data = json.loads((recorder.path/'demonstration.json').read_text())
    data.update(closed=True, error='')
    (recorder.path/'demonstration.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='离线'):
        select_reference(recorder.path, destination=tmp_path/'reference.json')


def test_codex_names_and_reviews_recordings_without_renaming_ids_or_assuming_success(tmp_path):
    from astra_vla.demonstrations import annotate, demonstration_label
    directory = tmp_path/'runs/demonstration_20261025_test'
    directory.mkdir(parents=True)
    for name in CAMERAS:
        (directory/(name+'.mp4')).write_bytes(b'synthetic fixture')
    (directory/'demonstration.json').write_text(json.dumps(dict(
        kind='human_demonstration', mode='human_teleop', simulated=False, closed=True, error='')))
    (tmp_path/'observed.jpg').write_bytes(b'synthetic fixture')
    with pytest.raises(ValueError, match='成功需要'):
        annotate(directory, task='收纳', outcome='success', root=tmp_path)
    data = annotate(directory, task='物体归位', title='遥控示教·物体归位', description='接近、低位释放、撤开',
                    tags=['收纳'], outcome='success', basis='visual_review', evidence=['observed.jpg'],
                    limitations='仅合成元数据测试', root=tmp_path)
    assert directory.exists() and data['outcome'] == 'success'
    assert demonstration_label(directory) == '遥控示教·物体归位 · 成功已复核'
    catalog = json.loads((tmp_path/'docs/memory/task_maps.json').read_text())
    assert catalog['maps'] == []  # Naming/success annotation never invents a task route.
    assert '遥控示教·物体归位' in (tmp_path/'docs/memory/task_maps.md').read_text()
    data = annotate(directory, task='不同的任务', description='不同的完成条件', root=tmp_path)
    assert data['outcome'] == 'unknown' and 'outcome_evidence' not in data
    annotate(directory, task='不同的任务', outcome='failure', root=tmp_path)
    with pytest.raises(ValueError, match='失败示范'):
        select_reference(directory, destination=tmp_path/'reference.json')
