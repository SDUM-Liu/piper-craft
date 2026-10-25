"""Recorded-target calculations only; no SDK connection or robot commands."""
import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from astra_vla.config import ROOT
from astra_vla.native_pose import (NativePoseUnavailable, adapt_target, controller_bounds,
                                  placement, turn, warm_up)
from astra_vla.teleop import Policy


def limits():
    return {'source': 'controller_query_0x472_reply_0x473', 'complete': True,
            'limits': [dict(joint=i+1, min_deg=lo, max_deg=hi) for i, (lo, hi) in enumerate(
                zip([-150, 0, -170, -100, -70, -180], [150, 180, 0, 100, 70, 180]))]}


@pytest.mark.parametrize('name', ['teleop_20261023_195411_051c1cfa', 'teleop_20261023_192446_4b7a6cde'])
def test_recorded_boundaries_continue_translation_with_progressive_orientation(name):
    trace = ROOT/'runs'/name/'events.jsonl'
    if not trace.exists():
        pytest.skip('recorded evidence unavailable')
    targets = [row for row in map(json.loads, trace.read_text().splitlines()) if row['type'] == 'target']
    state = copy.deepcopy(targets[-1]['state']['right'])
    start = np.asarray(targets[-2]['tcp_target'])
    direction = np.asarray(targets[-1]['tcp_target'])[:3, 3]-start[:3, 3]
    direction /= np.linalg.norm(direction)
    lower, upper = controller_bounds(limits())
    warm_up()
    adjustments = []
    policy = Policy(rotation_deg_s=5.)
    origin = start[:3, 3].copy()
    for _ in range(8):
        target = start.copy()
        target[:3, 3] += direction*.0005
        result = adapt_target(state, {'tcp_target': target, 'start_tcp': start}, np.eye(4), limits(),
                              policy, True, budget_s=.1)
        chosen = np.asarray(result['tcp_target'])
        witness = np.asarray(result['model_witness_joints_rad'])
        assert np.all(witness >= lower) and np.all(witness <= upper)
        assert np.linalg.norm(turn(chosen, start)) <= np.deg2rad(.25)+1e-10
        assert np.linalg.norm(chosen[:3, 3]-(start[:3, 3]+result['progress']*(target[:3, 3]-start[:3, 3]))) < 2e-6
        assert np.dot(chosen[:3, 3]-start[:3, 3], direction) > 0
        # Every result is checked by an independent nonlinear forward evaluation.
        evaluated, _ = placement(witness, np.eye(4))
        assert np.linalg.norm(evaluated[:3, 3]-chosen[:3, 3]) < 2e-6
        assert np.linalg.norm(turn(evaluated, chosen)) < np.deg2rad(.002)
        adjustments.append(result['adjustment_deg'])
        start = chosen
        state['joints_rad'] = witness.tolist()
        state['ee_pose'] = {'position_m': chosen[:3, 3].tolist(), 'rpy_deg':
                           Rotation.from_matrix(chosen[:3, :3]).as_euler('xyz', degrees=True).tolist()}
    assert max(adjustments) > .02
    assert np.dot(start[:3, 3]-origin, direction) > .001


def test_wide_controller_limits_keep_orientation_and_explicit_rotation_is_not_overridden():
    trace = ROOT/'runs/teleop_20261023_195411_051c1cfa/events.jsonl'
    if not trace.exists():
        pytest.skip('recorded evidence unavailable')
    targets = [row for row in map(json.loads, trace.read_text().splitlines()) if row['type'] == 'target']
    state = targets[-1]['state']['right']
    proposal = {'tcp_target': np.asarray(targets[-1]['tcp_target']),
                'start_tcp': np.asarray(targets[-2]['tcp_target'])}
    warm_up()
    with pytest.raises(NativePoseUnavailable, match='当前朝向'):
        adapt_target(state, proposal, np.eye(4), limits(), Policy(), False, budget_s=.1)
    report = limits()
    report['limits'][4]['max_deg'] = 80.
    result = adapt_target(state, proposal, np.eye(4), report, Policy(), True, budget_s=.1)
    assert result['adjustment_deg'] == 0 and result['progress'] == 1
    assert np.rad2deg(result['model_witness_joints_rad'][4]) > 70


def test_mdh_jacobian_matches_finite_differences_with_offset_tcp():
    q = np.deg2rad([-52., 12., -8., 72., 69., -64.])
    tcp = np.eye(4)
    tcp[:3, 3] = [.02, 0., .08]
    base, jac = placement(q, tcp, True)
    for axis in range(6):
        sample = q.copy()
        sample[axis] += 1e-7
        value, _ = placement(sample, tcp)
        numeric = np.r_[value[:3, 3]-base[:3, 3], turn(value, base)]/1e-7
        np.testing.assert_allclose(jac[:, axis], numeric, atol=1e-7)


def test_unknown_or_incomplete_controller_bounds_are_never_urdf_defaults():
    with pytest.raises(ValueError, match='fresh controller'):
        controller_bounds({'complete': False})
    report = limits()
    report['limits'][4]['max_deg'] = float('nan')
    with pytest.raises(ValueError, match='invalid controller'):
        controller_bounds(report)
