"""Disabled MOVE_P paths and immutable local sessions; no CAN access."""
import pytest

from astra_vla.teleop_desktop import Client
import test_teleop as fixtures

controller = fixtures.controller


@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('profile', ['slow', 'normal', 'fast'])
def test_native_request_is_rejected_without_preparation_or_sdk(controller, arm, profile):
    with pytest.raises(ValueError, match='已禁用'):
        controller.manager.open({'arm': arm, 'speed_profile': profile, 'execution_backend': 'controller_move_p'})
    assert not controller.commands and not controller.manager.active


def test_invalid_backend_request_cannot_replace_live_local_owner(controller):
    session = controller.manager.open({'arm': 'right'})['session']
    before = list(controller.commands)
    with pytest.raises(RuntimeError):
        controller.manager.open({'arm': 'right', 'execution_backend': 'controller_move_p'})
    assert controller.manager.session == session and controller.manager.active
    assert controller.commands == before
    with pytest.raises(ValueError, match='execution backend'):
        controller.manager.input({'session': session, 'sequence': 0, 'keys': ['w'],
                                  'execution_backend': 'controller_move_p'})
    assert controller.commands == before


def test_desktop_native_backend_cannot_be_requested_or_opened():
    calls = []
    client = Client(lambda _p, method, *a, **kw: calls.append(method))
    with pytest.raises(ValueError, match='已禁用'):
        client.request_backend('controller_move_p')
    client.execution_backend = 'controller_move_p'
    with pytest.raises(RuntimeError, match='已禁用'):
        client.execute('open', 'right')
    assert calls == [] and not client.control_active


def test_coordinate_change_cancels_a_queued_open():
    calls = []
    client = Client(lambda _p, method, *a, **kw: calls.append(method) or {'stop_confirmed': True})
    client.request('open', 'left')
    action, arm = client.commands.get_nowait()
    client.request_coordinate_mode('motion')
    for thread in client.threads:
        thread.join(1)
    client.execute(action, arm)
    assert calls == ['stop'] and not client.opening and not client.session


def test_legacy_worker_mode_mismatch_closes_owner_before_accepting_input():
    calls = []
    def rpc(_p, method, args=None, **kwargs):
        calls.append((method, args))
        return {'session': 'legacy', 'execution_backend': 'regularized_move_j'}
    client = Client(rpc)
    with pytest.raises(RuntimeError, match='坐标模式'):
        client.execute('open', 'right')
    assert [method for method, _ in calls] == ['teleop_open', 'teleop_close']
    assert not client.session and not client.control_active
