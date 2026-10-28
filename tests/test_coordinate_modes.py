"""Direction math and stop-before-switch contract; no hardware connections."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from astra_vla.teleop import Integrator, Policy, keyset
from astra_vla.teleop_desktop import Client
import test_teleop as fixtures

controller = fixtures.controller
DIRECTIONS = {'w': [1,0,0], 's': [-1,0,0], 'a': [0,1,0], 'd': [0,-1,0],
              'space': [0,0,1], 'lshift': [0,0,-1]}
TURNS = {'i': [0,-1,0], 'k': [0,1,0], 'j': [0,0,1], 'l': [0,0,-1],
         'o': [-1,0,0], 'p': [1,0,0]}


@pytest.mark.parametrize('mode', ['base', 'motion'])
@pytest.mark.parametrize('key', list(DIRECTIONS)+list(TURNS))
def test_all_six_axes_share_one_complete_frame_and_pin_tcp(mode, key):
    state = fixtures.state()
    state['ee_pose']['rpy_deg'] = [24, -38, 73]
    tcp = np.eye(4)
    tcp[:3, 3] = [.02, -.015, .145]
    tcp[:3, :3] = Rotation.from_euler('xyz', [12, 9, -17], degrees=True).as_matrix()
    policy = Policy(axis_convention='piper_tool', coordinate_mode=mode)
    integrator = Integrator(state, tcp, policy)
    start = integrator.target.copy()
    # Piper forward/right/up form this right-handed motion triad.
    basis = np.eye(3) if mode == 'base' else start[:3, :3] @ np.array([[0,0,-1],[0,1,0],[1,0,0]])
    proposal = integrator.propose(state, keyset([key]), .05, .065)
    if key in DIRECTIONS:
        np.testing.assert_allclose(proposal['tcp_target'][:3, 3]-start[:3, 3],
            basis@DIRECTIONS[key]*policy.translation_m_s*.05*.2, atol=1e-12)
        np.testing.assert_allclose(proposal['tcp_target'][:3, :3], start[:3, :3], atol=1e-12)
    else:
        turn = Rotation.from_matrix(proposal['tcp_target'][:3,:3]@start[:3,:3].T).as_rotvec()
        np.testing.assert_allclose(turn, basis@TURNS[key]*np.deg2rad(policy.rotation_deg_s)*.05*.2, atol=1e-12)
        np.testing.assert_allclose((proposal['ee_target']@tcp)[:3,3], start[:3,3], atol=1e-12)


@pytest.mark.parametrize('confirmed', [False, True])
def test_client_switch_requires_stop_confirmation_and_cannot_reuse_held_input(confirmed):
    calls = []
    def rpc(path, method, args, **kw):
        calls.append(method)
        return {'stop_confirmed': confirmed}
    client = Client(rpc)
    client.session, client.control_active, client.keys = 'existing', True, ['w', 'q']
    client.request_coordinate_mode('motion')
    for thread in client.threads:
        thread.join(1)
    assert calls == ['stop'] and client.keys == [] and not client.control_active
    assert client.coordinate_mode == ('motion' if confirmed else 'base')
    if confirmed:
        assert client.session is None
    else:
        assert '未切换' in client.error


def test_worker_rejects_mid_session_coordinate_reinterpretation_without_writes(controller):
    session = controller.manager.open({'arm': 'right', 'coordinate_mode': 'motion'})['session']
    assert controller.manager.view()['coordinate_mode'] == 'motion'
    before = list(controller.commands)
    with pytest.raises(ValueError, match='coordinate mode'):
        controller.manager.input({'session': session, 'sequence': 0, 'keys': ['w'], 'coordinate_mode': 'base'})
    assert controller.commands == before
    assert controller.manager.keys == frozenset()


def test_coordinate_switch_cancelled_by_stop_does_not_change_mode():
    import threading
    entered, release = threading.Event(), threading.Event()
    def rpc(*args, **kwargs):
        entered.set()
        release.wait(1)
        return {'stop_confirmed': True}
    client = Client(rpc)
    client.request_coordinate_mode('motion')
    assert entered.wait(1)
    client.request('stop')
    release.set()
    for thread in client.threads:
        thread.join(1)
    assert client.coordinate_mode == 'base' and not client.control_active


def test_observation_only_worker_reports_frame_before_first_control_entry(controller):
    view = controller.manager.view()
    assert not view['active'] and view['coordinate_mode'] == 'base'


def test_debug_forward_matches_selected_mode_instead_of_tool_alone():
    from astra_vla.teleop_desktop import debug_pose_lines
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler('y', 180, degrees=True).as_matrix()
    assert any('Facing: +X' in line for line in debug_pose_lines(tcp, 'piper_tool', coordinate_mode='base'))
    assert any('Facing: -Z' in line for line in debug_pose_lines(tcp, 'piper_tool', coordinate_mode='motion'))
