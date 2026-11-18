"""Explicit absolute joint targets in degrees; no robot connections."""
import math
import numbers


def joint_target(values):
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError('请输入 J1–J6 六个绝对角度（度）')
    encoded = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError('关节角度必须为有限数值')
        scaled = float(value) * 1000
        if not math.isfinite(scaled) or not -(2**31) <= scaled <= 2**31-1:
            raise ValueError('关节角度超出 CAN 整数编码范围')
        encoded.append(round(scaled))
    # Physical angle limits belong to the controller; never clip to URDF.
    return {'source': 'explicit_joint_degrees', 'joint_sdk': encoded,
            'goal_joints_rad': [math.radians(v / 1000) for v in encoded]}
