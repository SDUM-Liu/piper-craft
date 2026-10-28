"""Color identity and source-age checks; never opens a camera or CAN."""
from fractions import Fraction
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from astra_vla.clock_mapping import CameraClock
from astra_vla.workers import rgb


def test_only_matching_usb_color_interface_is_selected(tmp_path):
    bus = tmp_path/'usb'
    bus.mkdir()
    (bus/'serial').write_text('wrist')
    nodes = tmp_path/'video'
    nodes.mkdir()
    for number, interface, index in [(0, '00', '0'), (1, '03', '1'), (2, '03', '0')]:
        device = bus/str(number)
        device.mkdir()
        (device/'bInterfaceNumber').write_text(interface)
        node = nodes/f'video{number}'
        node.mkdir()
        (node/'device').symlink_to(device)
        (node/'index').write_text(index)
    assert rgb.color_node('wrist', nodes, tmp_path/'dev') == tmp_path/'dev/video2'
    with pytest.raises(RuntimeError, match='exactly one'):
        rgb.color_node('other-wrist', nodes)


@pytest.mark.parametrize('pts', [None, 0, 99_000_000, 101_000_000])
def test_arrival_cannot_refresh_missing_stale_or_future_source(pts):
    frame = SimpleNamespace(pts=pts, time_base=Fraction(1, 1_000_000))
    with pytest.raises(ValueError, match='source timestamp'):
        rgb.source_time(frame, 1000., 100., CameraClock(), 1)


def test_capture_discards_empty_startup_and_rejects_repeated_source(monkeypatch):
    monkeypatch.setattr(rgb.time, 'monotonic', lambda: 100.)
    monkeypatch.setattr(rgb.time, 'time', lambda: 1000.)
    def frame(pts):
        return SimpleNamespace(width=640, height=480, pts=pts, time_base=Fraction(1, 1_000_000),
                               to_ndarray=lambda **_: np.zeros((480, 640, 3), dtype=np.uint8))
    packets = [SimpleNamespace(size=0),
               SimpleNamespace(size=614400, decode=lambda: [frame(0)]),
               SimpleNamespace(size=614400, decode=lambda: [frame(99_980_000)]),
               SimpleNamespace(size=614400, decode=lambda: [frame(99_980_000)])]
    closed = []
    container = SimpleNamespace(demux=lambda **_: iter(packets), close=lambda: closed.append(True))
    def open_color(path, **kwargs):
        assert path == '/fake/color'
        assert kwargs['options']['input_format'] == 'yuyv422'
        assert kwargs['format'] == 'v4l2'
        return container
    monkeypatch.setattr(rgb.av, 'open', open_color)
    camera = rgb.RGB.__new__(rgb.RGB)
    camera.config = {'cameras': {'left_wrist': {'serial': 'wrist'}}}
    camera.lock, camera.frames, camera.errors, camera.raw_frames = threading.Lock(), {}, {}, {}
    camera.discover = lambda role: '/fake/color'
    camera.capture('left_wrist', threading.Event())
    assert camera.frames['left_wrist']['sequence'] == 1
    assert camera.frames['left_wrist']['stamp'] == pytest.approx(999.98)
    assert camera.frames['left_wrist']['timestamp_source'] == 'kernel_monotonic'
    assert 'did not advance' in camera.errors['left_wrist']
    assert closed == [True]


def test_transport_delay_is_preserved_and_host_clock_jump_is_rejected():
    clock = CameraClock()
    frame = SimpleNamespace(pts=99_980_000, time_base=Fraction(1, 1_000_000))
    assert rgb.source_time(frame, 1000., 100., clock, 1) == pytest.approx(999.98)
    frame.pts = 100_013_000
    assert rgb.source_time(frame, 1000.1, 100.1, clock, 2) == pytest.approx(1000.013)
    frame.pts = 100_046_000
    with pytest.raises(RuntimeError, match='host clock discontinuity'):
        rgb.source_time(frame, 1001.2, 100.2, clock, 3)
