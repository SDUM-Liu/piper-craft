"""Desktop teleoperation policy and geometry. No device connections on import."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from astra_vla.config import ROOT


JOINT_KEYS = frozenset(f'joint_{axis}_{sign}' for axis in range(1, 7) for sign in ('minus', 'plus'))
KEYS = frozenset(('w', 's', 'a', 'd', 'space', 'lshift', 'i', 'k', 'j', 'l', 'o', 'p', 'q', 'e')) | JOINT_KEYS
TRANSLATION_KEYS = frozenset(('w', 's', 'a', 'd', 'space', 'lshift'))
GRIPPER_KEYS = frozenset(('q', 'e'))
SPEED_PROFILES = ('slow', 'normal', 'fast')
EXECUTION_BACKENDS = ('regularized_move_j',)
BACKEND_LABELS = {'regularized_move_j': '本地 IK', 'controller_move_p': '控制器原生'}
COORDINATE_MODES = ('base', 'motion')
COORDINATE_LABELS = {'base': '基坐标', 'motion': '运动坐标'}


def control_axes(actual, policy):
    """Forward/right/up in the selected arm's base, independent of camera view.

    Base: +X/-Y/+Z. Motion: the measured tool triad, including its vertical
    direction. Rotation uses these same axes; there is no hybrid base lift.
    """
    if policy.coordinate_mode == 'base':
        return np.array([1., 0., 0.]), np.array([0., -1., 0.]), np.array([0., 0., 1.])
    if policy.axis_convention == 'piper_tool':
        return actual[:3, 2], -actual[:3, 1], -actual[:3, 0]
    return actual[:3, 0], -actual[:3, 1], actual[:3, 2]


class RateLimitError(ValueError):
    """A smaller increment suggestion, never a validated motion permission."""
    def __init__(self, message, suggested_scale=None):
        super().__init__(message)
        self.suggested_scale = suggested_scale

    def __reduce__(self):
        return type(self), (str(self), self.suggested_scale)


def rate_rejection(message, velocity, previous_velocity, acceleration_step):
    low, high = 0., 1.
    for value, previous in zip(velocity, previous_velocity):
        if abs(value) < 1e-12:
            if abs(previous) > acceleration_step:
                return RateLimitError(message)
            continue
        bounds = sorted(((previous - acceleration_step) / value,
                         (previous + acceleration_step) / value))
        low = max(low, bounds[0])
        high = min(high, bounds[1])
    # Leave room for IK nonlinearity/SDK quantization. The owning worker must
    # solve and validate the new candidate, under the same deadline.
    hint = high * .98 if 0 < high < 1 and low <= high * .98 else None
    return RateLimitError(message, hint)


@dataclass(frozen=True)
class Policy:
    speed_profile: str = 'normal'
    control_hz: float = 20
    input_timeout_s: float = .25
    geometry_timeout_s: float = .045
    translation_m_s: float = .005
    rotation_deg_s: float = 2
    gripper_m_s: float = .005
    gripper_effort: float = .5
    speed_percent: int = 5
    position_lead_m: float = .002
    rotation_lead_deg: float = .5
    gripper_opening_lead_m: float = .001
    joint_acceleration_deg_s2: float = 20
    joint_lead_deg: float = .25
    tcp_drift_m: float = .002
    follow_mode: str = 'position_velocity'
    axis_convention: str = 'xyz'
    vertical_frame: str = 'tcp'
    coordinate_mode: str = 'base'
    ik_solver: str = 'casadi_ipopt'

    def __post_init__(self):
        if self.speed_profile not in SPEED_PROFILES:
            raise ValueError('unknown teleop speed profile')
        fast = self.speed_profile == 'fast'
        if fast and (self.follow_mode != 'position_velocity' or self.ik_solver not in ('casadi_ipopt', 'controller')):
            raise ValueError('fast profile requires ordinary MOVE_J and regularized IK')
        if self.follow_mode != 'position_velocity':
            raise ValueError('unknown teleop follow mode')
        if self.axis_convention not in ('xyz', 'piper_tool'):
            raise ValueError('unknown teleop axis convention')
        if self.coordinate_mode not in COORDINATE_MODES:
            raise ValueError('unknown teleop coordinate mode')
        if self.vertical_frame not in ('tcp', 'base') or self.ik_solver not in ('casadi_ipopt', 'controller'):
            raise ValueError('unknown teleop vertical frame/IK solver')
        maxima = dict(control_hz=50 if fast else 20,
                      input_timeout_s=.1 if fast else .25, geometry_timeout_s=.018 if fast else .05,
                      gripper_effort=.8, speed_percent=100, position_lead_m=.008 if fast else .002,
                      rotation_lead_deg=.5, gripper_opening_lead_m=.003,
                      joint_acceleration_deg_s2=120, joint_lead_deg=6 if fast else 2, tcp_drift_m=.002)
        for field in fields(self):
            if field.name in ('follow_mode', 'axis_convention', 'vertical_frame', 'coordinate_mode', 'ik_solver', 'speed_profile'):
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('teleop policy must contain finite numbers: ' + field.name)
            # Commanded rates have no application-level overspeed ceiling.
            # SDK percentage still has its protocol range; motion guards stay.
            if not 0 < value <= maxima.get(field.name, math.inf):
                raise ValueError('teleop policy outside permitted range: ' + field.name)
        if int(self.speed_percent) != self.speed_percent or self.gripper_effort < .1:
            raise ValueError('invalid teleop speed/effort')
        if self.geometry_timeout_s > 1 / self.control_hz:
            raise ValueError('geometry deadline exceeds control period')

    def for_speed(self, name):
        """Manual profiles keep the same mechanical/joint/collision constraints."""
        if name not in SPEED_PROFILES:
            raise ValueError('unknown teleop speed profile')
        if self.speed_profile != 'normal':
            raise ValueError('speed profiles must derive from the normal base policy')
        defaults = {'normal': {},
                    'slow': dict(translation_m_s=.005, rotation_deg_s=2., control_hz=20.,
                                 geometry_timeout_s=.045, position_lead_m=.002),
                    'fast': dict(translation_m_s=.1, rotation_deg_s=10., control_hz=50.,
                                 geometry_timeout_s=.018, position_lead_m=.008, joint_lead_deg=6.,
                                 input_timeout_s=.1)}
        path = ROOT / 'config/teleop_speeds.json'
        profiles = json.loads(path.read_text()) if path.exists() else defaults
        allowed = set(defaults['fast'])
        if (not isinstance(profiles, dict) or set(profiles) != set(SPEED_PROFILES) or
                any(not isinstance(v, dict) or not set(v) <= allowed for v in profiles.values())):
            raise ValueError('invalid teleop speed profile configuration')
        return replace(self, speed_profile=name, **dict(defaults[name], **profiles[name]))

    @classmethod
    def load(cls):
        path = ROOT / 'config/teleop.json'
        return cls(**json.loads(path.read_text())) if path.exists() else cls()


def keyset(value):
    if not isinstance(value, list) or len(value) > len(KEYS) or any(
        not isinstance(k, str) or k not in KEYS for k in value
    ):
        raise ValueError('unknown teleoperation key')
    result = frozenset(value)
    if result & JOINT_KEYS and len(result) != 1:
        raise ValueError('joint jog requires exactly one direction without TCP/gripper input')
    return result


def pose(state):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', state['ee_pose']['rpy_deg'], degrees=True).as_matrix()
    transform[:3, 3] = state['ee_pose']['position_m']
    return transform


def rotation_error(a, b):
    return float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


class Integrator:
    """Robot-local intents integrated from controller-reported pose and configured TCP."""
    def __init__(self, state, tcp, policy=None):
        self.tcp = np.asarray(tcp, dtype=float)
        self.inverse_tcp = np.linalg.inv(self.tcp)
        self.policy = policy or Policy()
        self.target = pose(state) @ self.tcp
        self.width = max(0., float(state['gripper']['width_m']))
        self.heading = self.target[:3, 0].copy()
        self.rotating = False
        self.speed_fraction = 0.

    def reset(self, state):
        self.target = pose(state) @ self.tcp
        self.width = max(0., float(state['gripper']['width_m']))
        self.rotating = False
        self.speed_fraction = 0.

    def propose(self, state, keys, dt, maximum_width, previous_keys=None, *, motion_scale=1.):
        if not 0 < motion_scale <= 1:
            raise ValueError('invalid teleop motion scale')
        dt = min(max(0., dt), 1 / self.policy.control_hz)
        actual = pose(state) @ self.tcp
        reported = actual
        changed = previous_keys is not None and keys != previous_keys
        gain = min(1., (0. if changed else self.speed_fraction) + dt / .25)
        motion_gain = gain * motion_scale
        joint = keys & JOINT_KEYS
        if joint:
            return {'tcp_target': actual.copy(), 'start_tcp': actual.copy(),
                    'ee_target': pose(state), 'width_m': max(0., float(state['gripper']['width_m'])),
                    'heading': self.heading.copy(), 'rotating': False, 'speed_fraction': motion_gain,
                    'arm_motion': True, 'gripper_motion': False, 'joint_jog': next(iter(joint))}
        forward, right, up = control_axes(actual, self.policy)
        roll = -forward
        velocity = (forward * (('w' in keys) - ('s' in keys)) +
                    right * (('d' in keys) - ('a' in keys)) +
                    up * (('space' in keys) - ('lshift' in keys)))
        velocity /= max(1., np.linalg.norm(velocity))
        angular = (right * (('i' in keys) - ('k' in keys)) +
                   up * (('j' in keys) - ('l' in keys)) +
                   roll * (('o' in keys) - ('p' in keys)))
        angular /= max(1., np.linalg.norm(angular))
        translating, rotating = np.linalg.norm(velocity) > 0, np.linalg.norm(angular) > 0
        target = self.target.copy()
        if previous_keys is not None and keys != previous_keys:
            # Changed intentions start at feedback, never complete a released direction.
            target = actual.copy()
            if self.rotating and rotating and not translating:
                target[:3, 3] = self.target[:3, 3]
        if rotating and not translating and not self.rotating:
            target[:3, 3] = actual[:3, 3]
        start = target.copy()
        target[:3, 3] += velocity * self.policy.translation_m_s * dt * motion_gain
        target[:3, :3] = Rotation.from_rotvec(angular * math.radians(self.policy.rotation_deg_s) * dt * motion_gain
                                            ).as_matrix() @ target[:3, :3]
        distance = np.linalg.norm(target[:3, 3] - reported[:3, 3])
        if rotating and not translating and distance > self.policy.tcp_drift_m:
            raise RuntimeError('fixed TCP drift exceeds 2 mm')
        if distance > self.policy.position_lead_m or rotation_error(reported, target) > math.radians(
            self.policy.rotation_lead_deg
        ):
            if (np.linalg.norm(start[:3, 3] - reported[:3, 3]) > self.policy.position_lead_m or
                    rotation_error(reported, start) > math.radians(self.policy.rotation_lead_deg)):
                return None
            # Consume only the remaining tracking budget instead of skipping a
            # whole tick when a full increment would exceed it. Scale this NEW
            # increment; never clamp an existing TCP anchor or accrue debt.
            delta = target[:3, 3] - start[:3, 3]
            turn = Rotation.from_matrix(target[:3, :3] @ start[:3, :3].T).as_rotvec()
            low, high = 0., 1.
            for _ in range(16):
                fraction = (low + high) / 2
                candidate = start.copy()
                candidate[:3, 3] += fraction * delta
                candidate[:3, :3] = Rotation.from_rotvec(fraction * turn).as_matrix() @ start[:3, :3]
                if (np.linalg.norm(candidate[:3, 3] - reported[:3, 3]) <= self.policy.position_lead_m and
                        rotation_error(reported, candidate) <= math.radians(self.policy.rotation_lead_deg)):
                    low = fraction
                else:
                    high = fraction
            if low <= 1e-4:
                return None
            target = start.copy()
            target[:3, 3] += .95 * low * delta
            target[:3, :3] = Rotation.from_rotvec(.95 * low * turn).as_matrix() @ start[:3, :3]
        grip = ('e' in keys) - ('q' in keys)
        actual_width = max(0., float(state['gripper']['width_m']))
        previous_grip = (('e' in previous_keys) - ('q' in previous_keys)) if previous_keys is not None else grip
        width = self.width if grip == previous_grip else actual_width
        width = min(maximum_width, max(0., width + grip * self.policy.gripper_m_s * dt * gain))
        # Q is a closing-position request, not a 1 mm feedback follower. Keep
        # advancing toward zero while held; effort still limits applied torque.
        if grip > 0 and width - actual_width > self.policy.gripper_opening_lead_m:
            width = self.width
        return {'tcp_target': target, 'start_tcp': start, 'ee_target': target @ self.inverse_tcp, 'width_m': width,
                'heading': forward, 'rotating': bool(rotating and not translating),
                'speed_fraction': motion_gain,
                'arm_motion': bool(translating or rotating), 'gripper_motion': bool(grip)}

    def commit(self, proposal):
        self.target = proposal['tcp_target'].copy()
        self.width = proposal['width_m']
        self.heading = proposal['heading'].copy()
        self.rotating = proposal['rotating']
        self.speed_fraction = proposal['speed_fraction']
