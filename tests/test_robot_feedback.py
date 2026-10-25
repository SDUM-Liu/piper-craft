"""Pure CAN-byte fixtures; no SDK, sockets or devices."""
import struct
from types import SimpleNamespace

import pytest

from astra_vla.workers.feedback import CoherentArmFeedback


def packet(index, stamp=100., value=0):
    return SimpleNamespace(arbitration_id=0x2A2 + index, timestamp=stamp,
                           data=struct.pack('>ii', value, -value))


def cycle(collector, stamp=100., value=10):
    for i in range(6):
        collector.feed(packet(i, stamp+i*.00012, value))


def test_complete_cycle_never_exposes_partial_next_cycle():
    c = CoherentArmFeedback()
    cycle(c)
    before = c.snapshot()
    assert before['generation'] == 1 and before['span_s'] == pytest.approx(.0006)
    assert before['ee_pose']['position_m'] == [1e-5, -1e-5, 1e-5]
    for i in range(5):
        c.feed(packet(i, 100.005+i*.00012, 20))
        assert c.snapshot() == before
    c.feed(packet(5, 100.0056, 20))
    after = c.snapshot()
    assert after['generation'] == 2 and after['ee_pose']['position_m'][0] == 2e-5
    after['joints_rad'][0] = 999
    assert c.snapshot()['joints_rad'][0] != 999
    cycle(c, 99.)
    assert c.snapshot()['generation'] == 2


@pytest.mark.parametrize('problem', ['missing', 'order', 'old', 'nan', 'span', 'short', 'remote'])
def test_incomplete_cycles_never_refresh_timestamp(problem):
    c = CoherentArmFeedback()
    cycle(c)
    before = c.snapshot()
    for i in range(6):
        f = packet(i, 101.+i*.00012)
        if i == 2:
            if problem == 'missing':
                continue
            if problem == 'order':
                f.arbitration_id += 1
            if problem == 'old':
                f.timestamp = 100.
            if problem == 'nan':
                f.timestamp = float('nan')
            if problem == 'span':
                f.timestamp += .004
            if problem == 'short':
                f.data = b''
            if problem == 'remote':
                f.is_remote_frame = True
        c.feed(f)
    assert c.snapshot() == before
    cycle(c, 102.)
    assert c.snapshot()['generation'] == 2
