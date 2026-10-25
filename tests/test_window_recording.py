"""Encode/decode synthetic complete windows. No camera or robot connection."""
import json
import threading
import time

import av
import numpy as np

from astra_vla.window_recording import WindowRecording


def test_window_video_preserves_full_frame_resize_pts_and_normal_close(tmp_path):
    clock = [0.]
    recorder = WindowRecording(tmp_path, clock=lambda: clock[0])
    assert not recorder.due() and recorder.status()['state'] == 'idle'
    recorder.start((320, 240))
    try:
        for index, (stamp, size) in enumerate(((0., (320, 240)), (.117, (640, 240)), (.713, (320, 240)))):
            clock[0] = stamp
            assert recorder.due()
            image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            image[:, :size[0]//2] = [220, 30, 30]
            image[:, size[0]//2:] = [30, 220, 30]  # synthetic right sidebar must be included
            recorder.submit(image.tobytes(), size)
            assert not recorder.due()
            deadline = time.monotonic()+5
            while recorder.status()['frames'] < index+1:
                assert time.monotonic() < deadline, recorder.status()
                time.sleep(.01)
        recorder.stop()
        assert recorder.status()['state'] in ('finalizing', 'saved')
    finally:
        recorder.close()
    status = recorder.status()
    assert status['state'] == 'saved' and status['frames'] == 3
    metadata = json.loads(recorder.path.with_suffix('.json').read_text())
    assert metadata['closed'] and metadata['kind'] == 'desktop_window'
    with av.open(status['path']) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 3
    assert abs(float(frames[-1].pts*frames[-1].time_base)-.713) < .002
    for frame in frames:
        pixels = frame.to_ndarray(format='rgb24')
        assert pixels.shape == (240, 320, 3)
        assert pixels[120, 30, 0] > 180 and pixels[120, 290, 1] > 180
    # A wide resized window is fitted completely, never cropped to the camera.
    assert frames[1].to_ndarray(format='rgb24')[5].max() < 10


def test_encoder_backpressure_drops_old_frames_without_blocking_or_reordering(tmp_path):
    clock = [0.]
    recorder = WindowRecording(tmp_path, clock=lambda: clock[0])
    ready, release = threading.Event(), threading.Event()
    def stalled_encoder():
        ready.set()
        release.wait(3)
    recorder.encode = stalled_encoder
    recorder.start((2, 2))
    try:
        assert ready.wait(1)
        for i in range(20):
            clock[0] += .05
            recorder.submit(bytes([i]*12), (2, 2))
        assert recorder.pending.qsize() == 2
        assert recorder.status()['dropped'] == 18
        samples = [recorder.pending.get_nowait(), recorder.pending.get_nowait()]
        assert [sample[2][0] for sample in samples] == [18, 19]
    finally:
        release.set()
        recorder.close()


def test_encoder_failure_is_visible_and_does_not_throw_in_render_thread(tmp_path, monkeypatch):
    def failed(*_a, **_k):
        raise OSError('disk full simulation')
    monkeypatch.setattr(av, 'open', failed)
    recorder = WindowRecording(tmp_path)
    recorder.start((320, 240))
    recorder.close()
    assert recorder.status()['state'] == 'error'
    assert 'disk full simulation' in recorder.status()['error']
    assert not json.loads(recorder.path.with_suffix('.json').read_text())['closed']
