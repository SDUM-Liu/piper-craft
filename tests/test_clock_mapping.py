import pytest

from astra_vla.clock_mapping import CameraClock


def test_clock_drift_does_not_accumulate_future_timestamps():
    clock = CameraClock()
    for i in range(100):
        wall = 10000 + i * 10
        stamp = clock.project(100 + i * 10.002, i + 1, wall, 200 + i * 10)
        assert stamp == pytest.approx(wall)


def test_delayed_and_repeated_frames_do_not_become_fresh():
    clock = CameraClock()
    clock.project(100, 1, 10000, 200)
    assert clock.project(100.05, 2, 10000.25, 200.25) == pytest.approx(10000.05)
    assert clock.project(100.05, 2, 10001, 201) == pytest.approx(10000.05)


def test_clock_wrap_retains_source_elapsed_time():
    clock = CameraClock()
    clock.project(clock.PERIOD_S - .02, 1, 10000, 200)
    assert clock.project(.03, 2, 10000.05, 200.05) == pytest.approx(10000.05)
    assert clock.wraps == 1


@pytest.mark.parametrize("raw,sequence,wall,mono", [
    (99, 2, 10000.05, 200.05), (100.05, 0, 10000.05, 200.05),
    (101, 2, 10000.05, 200.05), (100.05, 2, 10002.05, 200.05),
])
def test_clock_reset_jump_and_host_step_cannot_refresh_evidence(raw, sequence, wall, mono):
    clock = CameraClock()
    clock.project(100, 1, 10000, 200)
    with pytest.raises(RuntimeError):
        clock.project(raw, sequence, wall, mono)
    assert clock.previous[2] == 10000
