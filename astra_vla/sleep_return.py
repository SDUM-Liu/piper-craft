"""Configured sleep joint targets and offline path checks; no SDK access."""
import json
import time

import numpy as np

from astra_vla.config import ROOT
from astra_vla.teleop_detour import check_joint_route


def sleep_pose_path():
    local = ROOT / 'config/sleep_pose.json'
    return local if local.exists() else ROOT.parent / 'sleep_pose.json'


def load_sleep_target(arm):
    if arm not in ('left', 'right'):
        raise ValueError('select left/right arm')
    path = sleep_pose_path()
    if not path.is_file():
        raise ValueError('请先配置 config/sleep_pose.json 的双臂 sleep 关节位置')
    data = json.loads(path.read_text())
    if data.get('version') != 1:
        raise ValueError('unsupported sleep pose version')
    arms = data.get('arms', {})
    row = arms.get(arm, arms.get(arm + '_piper', {})).get('joints_raw')
    if not isinstance(row, list) or len(row) != 6 or any(type(v) is not int for v in row):
        raise ValueError('sleep joints_raw 必须是六个整数，单位为 0.001 度；请配置所选臂')
    goal = np.deg2rad(np.array(row, dtype=float) / 1000)
    if not np.isfinite(goal).all():
        raise ValueError('invalid sleep joint target')
    return {'source': str(path), 'joint_sdk': row, 'goal_joints_rad': goal.tolist()}


def plan_sleep_return(model, state, target, geometry, options):
    """Check the fixed joint route, without deriving an SDK pose from FK."""
    start = np.asarray(state['joints_rad'], dtype=float)
    goal = np.asarray(target['goal_joints_rad'], dtype=float)
    if start.shape != (6,) or goal.shape != (6,) or not np.isfinite([start, goal]).all():
        raise ValueError('invalid sleep joint plan')
    width = max(state['gripper']['width_m'], geometry.get('unverified_gripper_max_m') or 0)
    envelope = min(.1, max(.001, np.ceil(width * 1000) / 1000))
    samples = check_joint_route(model, state, start, goal, envelope,
                               deadline=time.monotonic() + options['planning_timeout_s'])
    return {'start_joints_rad': start.tolist(), 'goal_joints_rad': goal.tolist(),
            'envelope_max_m': float(envelope), 'samples': samples, 'kind': 'sleep',
            'scope': 'robot_only', 'environment_guard': False, 'other_arm_guard': False}
