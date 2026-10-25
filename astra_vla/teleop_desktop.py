"""Minecraft-inspired real camera desktop. Startup is observation-only."""
from __future__ import annotations

import argparse
import base64
import copy
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from astra_vla.codex_messages import CodexMessageReader
from astra_vla.public_session import PublicSessionFeed, bind_current_session
from astra_vla.config import ROOT, RUNTIME, tcp_configuration
from astra_vla.rpc import call
from astra_vla.teleop import BACKEND_LABELS, EXECUTION_BACKENDS, JOINT_KEYS, KEYS, SPEED_PROFILES, Integrator, Policy, pose
from astra_vla.teleop_preferences import BINDING_HELP, DEFAULT_BINDINGS, DesktopPreferences
from astra_vla.window_recording import WindowRecording
from astra_vla.demonstrations import DemonstrationRecording, available, select_reference

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')


def fresh_frame(frame, now=None):
    now = time.time() if now is None else now
    stamp, age = frame.get('stamp'), frame.get('age_s', 0.)
    return (frame.get('available', True) and isinstance(stamp, (int, float))
            and isinstance(age, (int, float)) and math.isfinite(stamp) and math.isfinite(age)
            and 0 <= now - stamp <= .6 and 0 <= age <= .6)


def control_message(message, telemetry=None):
    """Keep the measured cause visible when entry fails before a session exists."""
    message = str(message)
    if 'teleop release displacement exceeds ' in message:
        telemetry = telemetry or {}
        actual = telemetry.get('release_extra_mm')
        detail = f'，本次记录{actual:.2f} mm' if isinstance(actual, (int, float)) else ''
        confirmed = '已确认静止' if telemetry.get('stop_confirmed') else '已请求停止，尚未确认静止'
        return f'历史停止原因：旧版松键位移阈值（现已取消）{detail}。{confirmed}。F3查看原始记录。'
    if message.startswith('teleop hold requires joints inside hard limits'):
        return '起姿超限，需先普通模式恢复起姿：' + message.partition('recovery: ')[2]
    if message == 'measured pose has URDF self collision':
        return '起姿自碰撞检查未通过；需先建立无自碰撞起姿'
    if message == 'automatic gravity recovery requires fully disabled standby':
        return '自动下垂恢复只支持完全失能的待机起姿；当前模式不满足条件'
    if message.startswith('automatic gravity recovery only accepts J2/J3 sag up to 3 degrees'):
        return '自动下垂恢复仅处理 J2/J3 不超过3°的越界；' + message.partition(': ')[2]
    if message == 'automatic gravity recovery is disabled':
        return '自动下垂恢复已在配置中关闭'
    if message == 'automatic gravity recovery requires updated desktop cancellation support':
        return '请重启桌面窗口，加载支持准备期间取消的新版后再进入控制'
    if message.startswith('automatic gravity recovery has no validated path:'):
        return '自动下垂恢复路径未通过校验：' + message.partition(': ')[2]
    if message == 'automatic folded recovery requires J2/J3 within 3.5 degrees of sleep':
        return '自动接触脱离仅支持接近折叠位的起姿；请先建立可控制起姿'
    if message == 'execution stopped; fresh verification and acknowledgement required':
        return '软件停止锁：观察现场后点击“现场复核后确认停止”，再进入控制'
    if message.startswith('teleop session is not active'):
        return '控制会话已停止：' + (message.partition(': ')[2] or '请查看停止原因，复核后重新进入控制')
    translations = (
        ('teleop control deadline exceeded', '控制计算超时；本轮目标已取消，请查看F3耗时'),
        ('teleop SDK send deadline exceeded', '目标发送超时；已请求停止，请核对静止结果'),
        ('teleop input heartbeat expired', '按键心跳已断开；已取消运动输入'),
        ('teleop input timeout', '按键心跳已断开；已取消运动输入'),
        ('fixed TCP drift exceeds', '固定TCP转向偏移超限'),
        ('teleop sampled path has URDF self collision', '目标路径自碰撞检查未通过'),
        ('desktop close', '已退出控制'),
        ('desktop stop', '已请求停止'),
        ('desktop escape', '已请求停止'),
    )
    for original, translated in translations:
        if original in message:
            detail = message.partition(original)[2].strip()
            return translated + ('：' + detail if detail else '')
    return message


def debug_pose_lines(tcp, convention, forward_label='W'):
    """Describe measured TCP in its own arm base; F5 never changes this frame."""
    if tcp is None:
        return ['XYZ: 等待反馈（绝对坐标 / 本臂基座 / mm）', 'Facing: 等待反馈']
    matrix = np.asarray(tcp, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return ['XYZ: 无效反馈（绝对坐标 / 本臂基座 / mm）', 'Facing: 无效反馈']
    rotation = matrix[:3, :3]
    forward = rotation[:, 2] if convention == 'piper_tool' else rotation[:, 0]
    dominant = int(np.argmax(np.abs(forward)))
    facing = ('+' if forward[dominant] >= 0 else '-') + 'XYZ'[dominant]
    horizontal = math.hypot(forward[0], forward[1])
    yaw = '—' if horizontal < 1e-8 else '%.2f' % math.degrees(math.atan2(forward[1], forward[0]))
    pitch = math.degrees(math.atan2(forward[2], horizontal))
    # Euler angles are display-only; integrate rotations with matrices as before.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected.*', category=UserWarning)
        rpy = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
    return [
        'XYZ: ' + ' / '.join('%.3f' % v for v in matrix[:3, 3]*1000) + ' mm',
        '绝对坐标: 本臂基座 · 反馈TCP',
        'Facing: %s · %s前进方向（基座轴）' % (facing, forward_label),
        'Yaw / Pitch: %s / %.2f deg' % (yaw, pitch),
        'Direction: ' + ' / '.join('%+.3f' % v for v in forward),
        'RPY: ' + ' / '.join('%.2f' % v for v in rpy) + ' deg',
    ]


def wrap_text(value, font, width):
    lines, line = [], ''
    for char in value:
        if line and font.size(line + char)[0] > width:
            lines.append(line)
            line = ''
        line += char
    if line:
        lines.append(line)
    return lines


class Controls:
    """Input model also used by headless tests; repeat events never create motion."""
    def __init__(self):
        self.arm = 'left'
        self.third_person = False
        self.keys = set()
        self.speed_profile = 'normal'

    @property
    def camera(self):
        return 'top' if self.third_person else self.arm + '_wrist'

    def press(self, name, repeat=False):
        if repeat:
            return None
        if name == 'f5':
            self.third_person = not self.third_person
        elif name == 'tab':
            self.speed_profile = SPEED_PROFILES[(SPEED_PROFILES.index(self.speed_profile)+1) % len(SPEED_PROFILES)]
            self.keys.clear()
        elif name in ('1', '2'):
            arm = 'left' if name == '1' else 'right'
            if arm != self.arm:
                self.keys.clear()
                self.arm = arm
                return 'switch'
        elif name == 'f9':
            self.keys.clear()
            return 'stop'
        elif name == 'escape':
            self.keys.clear()
            return 'menu'
        elif name in JOINT_KEYS:
            self.keys = {name}
        elif name in KEYS:
            self.keys.difference_update(JOINT_KEYS)
            self.keys.add(name)
        return None

    def release(self, name):
        self.keys.discard(name)

    def blur(self):
        self.keys.clear()


class Client:
    def __init__(self, rpc=call):
        self.rpc = rpc
        self.policy = Policy.load()
        self.speed_profile = 'normal'
        self.execution_backend = 'regularized_move_j'
        self.pending_backend = None
        self.requested_open_epoch = 0
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.input_changed = threading.Event()
        self.commands = queue.Queue(maxsize=8)
        self.robot = {}
        self.frames = {}
        self.error = ''
        self.camera_error = ''
        self.session = None
        self.control_active = False
        self.session_started = 0.
        self.cancel_epoch = 0
        self.stop_pending = False
        self.sequence = 0
        self.keys = []
        self.focused = True
        self.ui_tick = time.monotonic()
        self.threads = []
        self.opening = False

    def start(self):
        for function in (self.images, self.feedback, self.control):
            thread = threading.Thread(target=function, daemon=True, name='desktop-' + function.__name__)
            thread.start()
            self.threads.append(thread)

    def intent(self, controls, focused):
        with self.lock:
            keys = sorted(controls.keys) if focused and self.pending_backend is None else []
            changed = (keys != self.keys or focused != self.focused or
                       self.speed_profile != controls.speed_profile)
            self.speed_profile = controls.speed_profile
            self.keys = keys
            self.focused, self.ui_tick = focused, time.monotonic()
            if changed:
                self.input_changed.set()
            cancel_preparation = self.opening and not focused and changed
        if cancel_preparation:
            self.request('close')

    def request(self, action, arm=None):
        with self.lock:
            self.keys = []
            if action in ('stop', 'close', 'switch'):
                self.focused = False
                self.cancel_epoch += 1
                self.control_active = False
            if action == 'open':
                if self.session or self.opening or self.stop_pending or self.pending_backend is not None:
                    return
                self.requested_open_epoch = self.cancel_epoch
                self.opening = True
                self.error = ''
            priority = action in ('stop', 'close', 'switch') and (action == 'stop' or self.session or self.opening)
            if priority:
                if not self.stop_pending:
                    self.stop_pending = True
                    thread = threading.Thread(target=self.priority_stop, args=(action,), daemon=True,
                                              name='desktop-stop')
                    thread.start()
                    self.threads.append(thread)
                return
        try:
            self.commands.put_nowait((action, arm))
        except queue.Full:
            self.error = '控制请求繁忙；已清空按键'

    def request_backend(self, backend):
        if backend not in EXECUTION_BACKENDS:
            raise ValueError('unknown execution backend')
        with self.lock:
            if self.stop_pending or self.pending_backend is not None or backend == self.execution_backend:
                return
            self.keys, self.control_active = [], False
            self.cancel_epoch += 1
            epoch = self.cancel_epoch
            self.pending_backend, self.stop_pending = backend, True
            self.input_changed.set()
            thread = threading.Thread(target=self.complete_backend_switch, args=(backend, epoch),
                                      daemon=True, name='desktop-backend-switch')
            thread.start()
            self.threads.append(thread)

    def complete_backend_switch(self, backend, epoch):
        try:
            result = self.rpc(RUNTIME / 'robot.sock', 'stop',
                              {'reason': 'desktop backend switch', 'manual_tcp': True}, timeout=5)
            if result.get('stop_confirmed') is not True:
                raise RuntimeError('切换未完成：尚未确认静止，请再次请求停止')
            with self.lock:
                if epoch != self.cancel_epoch or self.done.is_set():
                    return
                self.execution_backend = backend
                self.session, self.keys, self.control_active = None, [], False
                if 'arms' in result:
                    self.robot = result
                self.error = ''
        except Exception as exc:
            with self.lock:
                self.error = '方案切换：'+str(exc)
        finally:
            with self.lock:
                self.pending_backend, self.stop_pending = None, False

    def priority_stop(self, reason):
        try:
            with self.lock:
                session = self.session
            method = 'teleop_close' if session else 'stop'
            result = self.rpc(RUNTIME / 'robot.sock', method,
                              {'session': session, 'reason': 'desktop ' + reason, 'manual_tcp': True}, timeout=5)
            with self.lock:
                self.control_active = False
                if result.get('stop_confirmed') or result.get('owner') is None:
                    self.session = None
                else:
                    self.error = '已请求停止，尚未确认静止；请再次退出控制核验'
        except Exception as exc:
            with self.lock:
                self.error = '停止确认：' + str(exc)
                self.control_active = False
        finally:
            self.stop_pending = False

    def update_images(self, snapshot):
        import pygame
        frames = {}
        now = time.time()
        names = {'top': '顶视', 'left_wrist': '左腕', 'right_wrist': '右腕'}
        missing = []
        for name, label in names.items():
            frame = snapshot.get('cameras', {}).get(name, {})
            if not fresh_frame(frame, now) or not frame.get('jpeg'):
                missing.append(label)
                continue
            surface = pygame.image.load(io.BytesIO(base64.b64decode(frame['jpeg'])), 'frame.jpg')
            frames[name] = {'surface': surface, 'stamp': frame['stamp'],
                            'age_s': frame.get('age_s', 0.),
                            'source': frame.get('source', 'rgb')}
        detail = snapshot.get('error')
        message = ('、'.join(missing) + '相机无新帧；可点击“恢复相机”') if missing else ''
        if detail:
            message += ('；' if message else '') + str(detail)
        with self.lock:
            self.frames, self.camera_error = frames, message

    def images(self):
        while not self.done.is_set():
            tick = time.monotonic()
            try:
                snapshot = self.rpc(RUNTIME / 'cameras.sock', 'snapshot', {'depth': False}, timeout=1)
                self.update_images(snapshot)
            except Exception as exc:
                with self.lock:
                    self.frames = {}
                    self.camera_error = '相机：' + str(exc)
            self.done.wait(max(0, 1 / 15 - (time.monotonic() - tick)))

    def feedback(self):
        while not self.done.is_set():
            tick = time.monotonic()
            try:
                result = self.rpc(RUNTIME / 'robot.sock', 'status', timeout=.5)
                self.accept_feedback(result, tick)
            except Exception as exc:
                with self.lock:
                    self.error = self.error or '机器人：' + str(exc)
            self.done.wait(.1)

    @staticmethod
    def session_reason(status):
        teleop = status.get('teleop', {})
        return (teleop.get('termination_reason') or teleop.get('blocker') or
                (status.get('fault') if status.get('stopped') else None))

    def accept_feedback(self, result, requested_at):
        with self.lock:
            self.robot = result
            if not self.session or requested_at < self.session_started:
                return
            teleop = result.get('teleop', {})
            if teleop.get('session') != self.session:
                self.session, self.keys, self.control_active = None, [], False
                self.error = self.error or '控制会话已结束或更换；请重新进入控制'
                return
            # Status may revoke control, never reactivate a locally cancelled
            # input stream or adopt another client's session.
            was_active = self.control_active
            self.control_active = was_active and teleop.get('active', False) and not self.stop_pending
            if not self.control_active:
                self.keys = []
                reason = self.session_reason(result)
                if reason and (was_active or self.error):
                    self.error = reason
                if not teleop.get('owner'):
                    self.session = None

    def control_error(self, exc, input_session=None):
        """Report the first worker cause, not its subsequent expired-input error."""
        with self.lock:
            session = input_session or self.session
            if self.session is not None and session != self.session:
                return  # A late error from a previous session is not current.
            self.control_active, self.keys = False, []
            self.error = str(exc)
        if session is None:
            return
        diagnostics = getattr(exc, 'diagnostics', None)
        reason = (diagnostics.get('reason') if isinstance(diagnostics, dict) and
                  diagnostics.get('session') == session else None)
        should_close = True
        try:
            result = self.rpc(RUNTIME / 'robot.sock', 'status', timeout=.5)
            teleop = result.get('teleop', {})
            with self.lock:
                self.robot = result
            if teleop.get('session') == session:
                reason = self.session_reason(result) or reason
                # The worker already requested stop. Do not issue a second
                # close which can hide its cause or interrupt stop confirmation.
                should_close = bool(teleop.get('active'))
                if not teleop.get('owner'):
                    with self.lock:
                        if self.session == session:
                            self.session = None
            elif teleop.get('session') is not None:
                should_close = False
                with self.lock:
                    if self.session == session:
                        self.session = None
        except Exception:
            pass  # A failed status read cannot cancel the best-effort stop.
        if reason:
            with self.lock:
                self.error = reason
        if should_close:
            try:
                result = self.rpc(RUNTIME / 'robot.sock', 'teleop_close',
                                  {'session': session, 'reason': 'desktop input failed: ' + str(exc)}, timeout=5)
                with self.lock:
                    if self.session == session:
                        if result.get('stop_confirmed') or not result.get('owner'):
                            self.session = None
                        else:
                            self.error += '；已请求停止，尚未确认静止'
            except Exception as close_exc:
                with self.lock:
                    self.error += '；停止确认：' + str(close_exc)

    def execute(self, action, arm):
        with self.lock:
            session = self.session
        if action == 'open':
            try:
                if self.robot and 'teleop' not in self.robot:
                    raise RuntimeError('现有 robot worker 尚未加载遥操作接口；请按 M06 完成控制权交接并重启')
                with self.lock:
                    epoch = self.requested_open_epoch
                    backend = self.execution_backend
                    if epoch != self.cancel_epoch or self.stop_pending:
                        return
                supported = self.robot.get('motion_control', {}).get('execution_backends', [])
                if backend == 'controller_move_p' and backend not in supported:
                    raise RuntimeError('机器人服务尚未加载控制器原生方案，请重启匹配版本')
                result = self.rpc(RUNTIME / 'robot.sock', 'teleop_open',
                                  {'arm': arm, 'preparation_cancel_supported': True,
                                   'speed_profile': self.speed_profile,
                                   'execution_backend': backend}, timeout=75)
                if (self.speed_profile != 'normal' and
                        result.get('speed_profile_active') != self.speed_profile):
                    self.rpc(RUNTIME / 'robot.sock', 'teleop_close',
                             {'session': result['session'], 'reason': 'worker does not support requested speed profile'}, timeout=5)
                    raise RuntimeError('机器人服务尚未加载快慢档，请重新启动匹配版本')
                if result.get('execution_backend', 'regularized_move_j') != backend:
                    self.rpc(RUNTIME / 'robot.sock', 'teleop_close',
                             {'session': result['session'], 'reason': 'worker execution backend mismatch'}, timeout=5)
                    raise RuntimeError('机器人服务未加载所选控制方案，请重启匹配版本')
                if epoch != self.cancel_epoch:
                    self.rpc(RUNTIME / 'robot.sock', 'teleop_close',
                             {'session': result['session'], 'reason': 'desktop preparation cancelled'}, timeout=5)
                    return
                with self.lock:
                    self.session, self.sequence = result['session'], 0
                    self.session_started = time.monotonic()
                    self.control_active = True
                    self.keys, self.error = [], ''
            finally:
                self.opening = False
        elif action in ('stop', 'close', 'switch'):
            if session:
                self.rpc(RUNTIME / 'robot.sock', 'teleop_close',
                         {'session': session, 'reason': 'desktop ' + action}, timeout=5)
            elif action == 'stop':
                self.rpc(RUNTIME / 'robot.sock', 'stop', {'reason': 'desktop stop', 'manual_tcp': True}, timeout=5)
            with self.lock:
                self.session, self.keys = None, []
                self.control_active = False
        elif action == 'acknowledge':
            snapshot = self.rpc(RUNTIME / 'cameras.sock', 'snapshot', timeout=1)
            if any(not f.get('available') or time.time() - f.get('stamp', 0) > .6
                   for f in snapshot['cameras'].values()) or len(snapshot['cameras']) != 3:
                raise RuntimeError('确认停止需要三路新鲜图像')
            self.rpc(RUNTIME / 'robot.sock', 'acknowledge', {'manual_tcp': True}, timeout=5)
            self.error = ''
        elif action == 'retry_cameras':
            status = self.rpc(RUNTIME / 'robot.sock', 'status', timeout=1)
            if session or self.opening or status.get('busy') or status.get('teleop', {}).get('active'):
                raise RuntimeError('请先退出控制，再恢复相机')
            if not status.get('stopped') or not status.get('stop_confirmed'):
                raise RuntimeError('请先请求停止并确认静止，再恢复相机')
            snapshot = self.rpc(RUNTIME / 'cameras.sock', 'snapshot', {'depth': False}, timeout=2)
            now = time.time()
            for camera in ('left_wrist', 'right_wrist'):
                if not fresh_frame(snapshot.get('cameras', {}).get(camera, {}), now):
                    self.rpc(RUNTIME / 'cameras.sock', 'retry', {'camera': camera}, timeout=25)
            self.error = ''

    def control(self):
        while not self.done.is_set():
            self.input_changed.clear()
            tick = time.monotonic()
            input_session = None
            try:
                try:
                    action, arm = self.commands.get_nowait()
                except queue.Empty:
                    action = None
                if action:
                    self.execute(action, arm)
                with self.lock:
                    session, keys, focused = self.session, list(self.keys), self.focused
                    profile = self.speed_profile
                    backend = self.execution_backend
                    control_active = self.control_active
                    ui_age = time.monotonic() - self.ui_tick
                    self.sequence += 1
                    sequence = self.sequence
                if session and control_active:
                    if not focused or ui_age > .1:
                        self.execute('close', None)
                    else:
                        input_session = session
                        self.rpc(RUNTIME / 'robot.sock', 'teleop_input',
                                 {'session': session, 'sequence': sequence, 'keys': keys,
                                  'focused': True, 'ui_age_s': ui_age, 'speed_profile': profile,
                                  'execution_backend': backend}, timeout=.2)
            except Exception as exc:
                # On input failure never keep replaying a stale intent. The worker
                # watchdog independently pauses even if this close cannot get through.
                self.control_error(exc, input_session)
            interval = .02 if self.speed_profile == 'fast' else .05
            self.input_changed.wait(max(0, interval - (time.monotonic() - tick)))

    def shutdown(self):
        with self.lock:
            self.keys = []
            self.focused = False
        self.request('close')
        deadline = time.monotonic() + 2
        while (self.session or self.opening or self.stop_pending or not self.commands.empty()) and time.monotonic() < deadline:
            time.sleep(.02)
        self.done.set()
        for thread in self.threads:
            thread.join(timeout=.5)


class DemoClient:
    """Interactive TCP framework simulation; no RPC/device entrypoint exists."""
    def __init__(self, clock=time.monotonic, policy=None):
        import pygame
        self.clock = clock
        self.base_policy = self.policy = policy or Policy.load()
        self.speed_profile = 'normal'
        self.execution_backend = 'regularized_move_j'
        self.pending_backend = None
        self.tcp = tcp_configuration()['arms']
        self.arm, self.integrator = 'left', None
        self.last_tick, self.last_keys = clock(), frozenset()
        self.lock = threading.Lock()
        self.session, self.opening, self.error = None, False, ''
        self.robot = {'teleop': {'active': False, 'phase': 'observing', 'control_frame': 'tcp'},
                      'arms': {arm: {'ee_pose': {'position_m': [0., 0., .3], 'rpy_deg': [0., 0., 0.]},
                                    'joints_rad': [0.] * 6,
                                    'gripper': {'width_m': .03}} for arm in ('left', 'right')}}
        self.robot['teleop']['policy'] = vars(self.policy).copy()
        for arm, state in self.robot['arms'].items():
            state['tcp_matrix'] = (pose(state) @ np.asarray(self.tcp[arm]['T_ee_tcp'])).tolist()
        surface = pygame.Surface((640, 480))
        surface.fill((90, 131, 146))
        pygame.draw.rect(surface, (63, 111, 76), (0, 170, 640, 310))
        for x in range(0, 640, 80):
            pygame.draw.line(surface, (77, 127, 84), (x, 170), (x, 480), 2)
        for y in range(170, 480, 50):
            pygame.draw.line(surface, (77, 127, 84), (0, y), (640, y), 2)
        pygame.draw.rect(surface, (198, 150, 88), (275, 235, 90, 80))
        pygame.draw.rect(surface, (153, 107, 65), (290, 248, 60, 55), 3)
        self.frames = {n: {'surface': surface, 'stamp': time.time(), 'depth_available': False}
                       for n in ('top', 'left_wrist', 'right_wrist')}

    def start(self):
        pass

    def intent(self, controls, focused):
        if not focused:
            if self.session:
                self.request('close')
            return
        if not self.session or controls.arm != self.arm:
            self.speed_profile = controls.speed_profile
            self.policy = self.base_policy.for_speed(self.speed_profile)
            return
        keys = frozenset(controls.keys)
        now = self.clock()
        with self.lock:
            state = self.robot['arms'][self.arm]
            if controls.speed_profile != self.speed_profile:
                self.speed_profile = controls.speed_profile
                self.policy = self.base_policy.for_speed(self.speed_profile)
                self.integrator.policy = self.policy
                self.integrator.reset(state)
                self.last_keys = frozenset()
                self.robot['teleop'].update(policy=asdict(self.policy), speed_profile=self.speed_profile)
            if self.last_keys - keys:
                self.integrator.reset(state)
            if now - self.last_tick < 1 / self.policy.control_hz:
                return
            self.last_tick = now
            proposal = self.integrator.propose(state, keys, 1 / self.policy.control_hz,
                                               .065 if self.arm == 'left' else .0855, self.last_keys)
            if proposal is not None:
                if proposal.get('joint_jog'):
                    _, axis, sign = proposal['joint_jog'].split('_')
                    state['joints_rad'][int(axis)-1] += (1 if sign == 'plus' else -1)*math.radians(
                        self.policy.rotation_deg_s)*proposal['speed_fraction']/self.policy.control_hz
                self.integrator.commit(proposal)
                state['ee_pose'] = {'position_m': proposal['ee_target'][:3, 3].tolist(),
                    'rpy_deg': Rotation.from_matrix(proposal['ee_target'][:3, :3]).as_euler(
                        'xyz', degrees=True).tolist()}
                state['gripper']['width_m'] = proposal['width_m']
                if proposal['gripper_motion']:
                    self.robot['teleop']['gripper_target_mm'] = proposal['width_m']*1000
                state['tcp_matrix'] = (pose(state) @ np.asarray(self.tcp[self.arm]['T_ee_tcp'])).tolist()
            self.last_keys = keys
            self.robot['teleop']['phase'] = 'moving' if keys else 'ready'

    def request(self, action, arm=None):
        with self.lock:
            telemetry = self.robot['teleop']
            if action == 'open':
                if self.session:
                    return
                self.arm = arm or 'left'
                self.integrator = Integrator(self.robot['arms'][self.arm], self.tcp[self.arm]['T_ee_tcp'],
                                             policy=self.policy)
                self.session = 'offline-' + self.arm
                self.last_tick, self.last_keys = self.clock(), frozenset()
                telemetry.update(active=True, phase='ready', arm=self.arm)
                self.error = ''
            elif action in ('close', 'stop', 'switch'):
                self.session, self.last_keys = None, frozenset()
                telemetry.update(active=False, phase='stopped', stop_confirmed=True)
                self.error = '离线模拟已停止；未连接设备'
            elif action == 'acknowledge':
                self.error = ''

    def request_backend(self, backend):
        if backend not in EXECUTION_BACKENDS:
            raise ValueError('unknown execution backend')
        self.request('close')
        self.execution_backend = backend
        self.robot['teleop'].update(execution_backend=backend, ik_solver=(
            'controller' if backend == 'controller_move_p' else 'casadi_ipopt'))

    def shutdown(self):
        self.request('close')


class Window:
    def __init__(self, client, *, demo=False, size=(1280, 800), always_on_top=None,
                 preferences=None, preferences_path=None, codex_messages_path=None):
        import pygame
        self.pg, self.client, self.demo = pygame, client, demo
        self.preferences_path = preferences_path or (None if demo else ROOT/'config/teleop_desktop.json')
        self.settings_notice = ''
        try:
            self.preferences = preferences or (DesktopPreferences.load(self.preferences_path)
                                               if self.preferences_path else DesktopPreferences())
        except (OSError, ValueError, TypeError) as exc:
            self.preferences = DesktopPreferences()
            self.settings_notice = '界面配置读取失败，已使用默认：'+str(exc)
        self.controls = Controls()
        self.controls.arm = self.preferences.startup_arm
        self.controls.speed_profile = self.preferences.startup_speed
        self.controls.third_person = self.preferences.startup_view == 'top'
        self.client.speed_profile = self.preferences.startup_speed
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption('PIPER CRAFT · 真实机械臂遥操作')
        pygame.key.set_repeat()
        font_path = next((p for p in (
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc') if Path(p).exists()), None)
        self.font = pygame.font.Font(font_path, 19)
        self.small = pygame.font.Font(font_path, 15)
        self.menu_font = pygame.font.Font(font_path, 26)
        title_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
        self.title_font = pygame.font.Font(title_path if Path(title_path).exists() else font_path, 26)
        self.focused, self.show_hud, self.show_debug = True, True, False
        self.menu_open, self.was_active = True, False
        self.menu_page, self.capture_binding = 'main', None
        self.buttons = {}
        self.hitboxes = []
        self.mouse_joint = None
        self.joint_rects = {}
        self.fullscreen, self.windowed_size = False, size
        pin = self.preferences.always_on_top if always_on_top is None else always_on_top
        self.always_on_top, self.topmost_requested, self.window_notice = False, pin, ''
        self.tool_tick, self.tool = 0., {'arms': {}}
        self.started_at = time.monotonic()
        self.render_clock = pygame.time.Clock()
        self.chat = deque(maxlen=12)
        self.chat_current = {}
        self.chat_state = None
        # Public external prose has its own queue: it must not evict live faults
        # or enter the input/RPC path. Demos never read the real source implicitly.
        self.codex_chat = deque(maxlen=32)
        self.codex_session = None if demo else PublicSessionFeed()
        self.codex_messages = (CodexMessageReader(codex_messages_path, max_age_s=20.)
                               if not demo or codex_messages_path is not None else None)
        self.recording = WindowRecording()
        self.recording_seen = 'idle'
        self.recording_notice = ''
        self.demonstration = DemonstrationRecording(ROOT/'runs/offline_demonstrations' if demo else None)
        self.demonstration_mode = 'human_objects'
        self.demonstration_notice = ''
        self.demonstration_candidates = []
        self.demonstration_index = 0
        self.demonstration_reference = ''
        self.demonstration_seen = 'idle'
        self.set_topmost(pin)
        if self.preferences.fullscreen:
            self.toggle_fullscreen()

    def key_label(self, action):
        key = self.preferences.bindings[action]
        return {'escape': 'Esc', 'lshift': '左Shift', 'space': 'Space', 'tab': 'Tab'}.get(key, key.upper())

    def save_preferences(self):
        try:
            if self.preferences_path:
                self.preferences.save(self.preferences_path)
            self.settings_notice = '设置已保存；启动默认项下次打开窗口生效' if self.preferences_path else '演示设置已应用，本次不写入文件'
        except (OSError, ValueError) as exc:
            self.settings_notice = '设置保存失败：'+str(exc)

    def rebind(self, physical):
        action = self.capture_binding
        try:
            updated = replace(self.preferences, bindings={**self.preferences.bindings, action: physical})
        except ValueError:
            self.settings_notice = '此键已占用或属于保留键，请选择其他按键'
            return
        self.preferences, self.capture_binding = updated, None
        self.save_preferences()

    def set_topmost(self, enabled):
        self.topmost_requested = bool(enabled)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Please use Window.get_surface.*', category=DeprecationWarning)
                native = self.pg.Window.from_display_module()
            native.minimum_size = (1100, 700)
            native.always_on_top = bool(enabled)
            self.always_on_top = bool(native.always_on_top)
            self.window_notice = '' if self.always_on_top == enabled else '当前桌面未应用置顶请求'
        except (AttributeError, RuntimeError, self.pg.error) as exc:
            self.always_on_top = False
            self.window_notice = '置顶不可用：' + str(exc)

    def toggle_fullscreen(self):
        self.controls.blur()
        if not self.fullscreen:
            self.windowed_size = self.screen.get_size()
        try:
            entering = not self.fullscreen
            self.screen = self.pg.display.set_mode((0, 0) if entering else self.windowed_size,
                                                  self.pg.FULLSCREEN if entering else self.pg.RESIZABLE)
            self.fullscreen = entering
            self.set_topmost(self.topmost_requested)
        except self.pg.error as exc:
            self.window_notice = '全屏切换失败：' + str(exc)

    def text(self, value, position, color=(244, 244, 244), font=None):
        font = font or self.font
        self.screen.blit(font.render(str(value), False, (28, 28, 28)), (position[0]+2, position[1]+2))
        self.screen.blit(font.render(str(value), False, color), position)

    def panel(self, rect, color=(91, 91, 91), border=(36, 36, 36)):
        pg = self.pg
        pg.draw.rect(self.screen, border, rect)
        pg.draw.rect(self.screen, color, rect.inflate(-4, -4))
        pg.draw.line(self.screen, (180, 180, 180), (rect.x+2, rect.y+2), (rect.right-3, rect.y+2), 2)
        pg.draw.line(self.screen, (150, 150, 150), (rect.x+2, rect.y+2), (rect.x+2, rect.bottom-3), 2)
        pg.draw.line(self.screen, (48, 48, 48), (rect.x+2, rect.bottom-3), (rect.right-3, rect.bottom-3), 2)

    def shade(self, rect, alpha=175, color=(12, 12, 12)):
        surface = self.pg.Surface(rect.size, self.pg.SRCALPHA)
        surface.fill((*color, alpha))
        self.screen.blit(surface, rect)

    def button(self, name, label, rect, color=(91, 91, 91)):
        hover = rect.collidepoint(self.pg.mouse.get_pos())
        self.panel(rect, tuple(min(255, c+22) for c in color) if hover else color)
        if hover:
            self.pg.draw.rect(self.screen, (235, 235, 235), rect, 1)
        self.text(label, (rect.centerx-self.small.size(label)[0]//2,
                         rect.y+(rect.height-self.small.get_height())//2-1), font=self.small)
        self.buttons[name] = rect
        self.hitboxes.append((name, rect))

    @staticmethod
    def key_name(pg, key):
        return {pg.K_SPACE: 'space', pg.K_LSHIFT: 'lshift', pg.K_ESCAPE: 'escape',
                pg.K_F5: 'f5'}.get(key, pg.key.name(key))

    def publish_intent(self):
        # A menu is a pause in input, not loss of the physical window focus.
        # Keep the empty-key heartbeat alive without closing the session.
        if self.menu_open:
            self.controls.blur()
        if self.mouse_joint is not None:
            rect = self.joint_rects.get(self.mouse_joint)
            if (self.mouse_joint not in self.controls.keys or not self.focused or self.menu_open or not self.show_hud or
                    not self.client.session or not self.client.robot.get('teleop', {}).get('active') or
                    rect is None or not rect.collidepoint(self.pg.mouse.get_pos()) or
                    self.client.execution_backend != 'regularized_move_j' or
                    not self.pg.mouse.get_pressed()[0]):
                self.controls.keys.discard(self.mouse_joint)
                self.mouse_joint = None
        self.client.intent(self.controls, self.focused)

    def toggle_menu(self):
        self.controls.blur()
        self.menu_open = not self.menu_open
        self.menu_page, self.capture_binding = 'main', None
        self.publish_intent()

    def activate(self, name):
        if name == 'open' and self.demonstration.status()['state'] in ('recording', 'finalizing') and (
                self.demonstration.mode == 'human_objects'):
            self.demonstration_notice = '请先结束亲手操作示范，再进入机械臂控制'
            return
        if name.startswith('demonstration'):
            self.controls.blur()
            self.publish_intent()
            try:
                if self.demo and name in ('demonstration_select', 'demonstration_clear'):
                    raise ValueError('离线演示不能修改真实示范参考')
                if name == 'demonstrations':
                    self.menu_open, self.menu_page = True, 'demonstrations'
                    self.demonstration_candidates = available(self.demonstration.destination)
                    self.demonstration_index = 0
                    reference = RUNTIME/'demonstration_reference.json'
                    data = json.loads(reference.read_text()) if not self.demo and reference.exists() else {}
                    self.demonstration_reference = Path(data['directory']).name if data.get('enabled') else '关闭'
                elif name == 'demonstration_mode':
                    if self.demonstration.status()['state'] not in ('recording', 'finalizing'):
                        self.demonstration_mode = ('human_teleop' if self.demonstration_mode == 'human_objects'
                                                   else 'human_objects')
                elif name == 'demonstration_record':
                    if self.demonstration.status()['state'] == 'recording':
                        self.demonstration.stop()
                    elif self.demonstration.status()['state'] != 'finalizing':
                        if self.demonstration_mode == 'human_objects' and (
                                self.client.session or self.client.opening or
                                self.client.robot.get('teleop', {}).get('active') or
                                self.client.robot.get('busy')):
                            raise ValueError('请先退出控制并确认静止，再录亲手操作示范')
                        if self.demonstration_mode == 'human_objects' and not self.demo and (
                                not self.client.robot.get('stop_confirmed') or
                                not all(fresh_frame(self.client.robot.get('arms', {}).get(arm, {}))
                                        for arm in ('left', 'right'))):
                            raise ValueError('亲手示范需要新鲜双臂反馈和停止确认')
                        if not all(fresh_frame(self.client.frames.get(camera, {})) for camera in
                                   ('top', 'left_wrist', 'right_wrist')):
                            raise ValueError('开始示范需要全部三路新鲜 RGB')
                        self.demonstration.start(self.demonstration_mode, simulated=self.demo)
                        self.demonstration_notice = '录制中；返回画面后示范，再到此处结束保存'
                elif name == 'demonstration_next':
                    self.demonstration_candidates = available(self.demonstration.destination)
                    self.demonstration_index += 1
                elif name == 'demonstration_select':
                    paths = self.demonstration_candidates
                    if not paths:
                        raise ValueError('暂无完整保存的真实示范')
                    path = paths[self.demonstration_index % len(paths)]
                    select_reference(path)
                    self.demonstration_reference = path.name
                    self.demonstration_notice = 'Codex 参考已启用：'+path.name
                elif name == 'demonstration_clear':
                    select_reference()
                    self.demonstration_reference = '关闭'
                    self.demonstration_notice = 'Codex 示范参考已关闭'
            except Exception as exc:
                self.demonstration_notice = str(exc)
            return
        if name in JOINT_KEYS:
            self.controls.blur()
            active = self.client.robot.get('teleop', {}).get('active', False)
            if (active and self.client.session and self.focused and not self.menu_open and
                    self.client.execution_backend == 'regularized_move_j' and not self.client.pending_backend):
                self.controls.press(name)
                self.mouse_joint = name
            self.publish_intent()
            return
        if (name.startswith('bind:') or name == 'restore_keys') and (
                self.client.session or self.client.opening or self.client.robot.get('teleop', {}).get('active')):
            self.settings_notice = '请先退出控制，再修改按键绑定'
            return
        if name in ('keys', 'defaults'):
            self.controls.blur()
            self.menu_page, self.capture_binding = name, None
            self.publish_intent()
        elif name == 'settings_back':
            self.menu_page, self.capture_binding = 'main', None
        elif name.startswith('bind:'):
            self.capture_binding = name.partition(':')[2]
            self.settings_notice = ''
        elif name == 'restore_keys':
            self.preferences = replace(self.preferences, bindings=dict(DEFAULT_BINDINGS))
            self.capture_binding = None
            self.save_preferences()
        elif name == 'restore_defaults':
            self.preferences = DesktopPreferences(bindings=dict(self.preferences.bindings))
            self.save_preferences()
        elif name.startswith('default:'):
            field = name.partition(':')[2]
            choices = {'startup_arm': ('left', 'right'), 'startup_speed': ('normal', 'fast', 'slow'),
                       'startup_view': ('wrist', 'top'), 'always_on_top': (True, False),
                       'fullscreen': (False, True), 'show_minimap': (True, False)}[field]
            value = choices[(choices.index(getattr(self.preferences, field))+1) % len(choices)]
            self.preferences = replace(self.preferences, **{field: value})
            self.save_preferences()
        elif name in ('left', 'right'):
            action = self.controls.press('1' if name == 'left' else '2')
            if action:
                self.client.request(action, self.controls.arm)
        elif name == 'backend':
            self.controls.blur()
            self.publish_intent()
            backend = ('controller_move_p' if self.client.execution_backend == 'regularized_move_j'
                       else 'regularized_move_j')
            self.client.request_backend(backend)
        elif name == 'speed':
            self.controls.press('tab')
            self.publish_intent()
        elif name == 'menu':
            self.toggle_menu()
        elif name == 'record_window':
            if self.recording.status()['state'] == 'recording':
                self.recording.stop()
            else:
                try:
                    self.recording_notice = ''
                    self.recording.start(self.screen.get_size())
                except Exception as exc:
                    self.recording_notice = '窗口录像无法开始：'+str(exc)
        elif name == 'view':
            self.controls.press('f5')
        elif name == 'debug':
            self.show_debug = not self.show_debug
        elif name == 'topmost':
            self.set_topmost(not self.topmost_requested)
        elif name == 'fullscreen':
            self.toggle_fullscreen()
        elif name == 'resume_view':
            if self.menu_open:
                self.toggle_menu()
        else:
            self.controls.blur()
            self.client.request(name, self.controls.arm)
            if name in ('stop', 'close'):
                self.menu_open = True

    def handle(self, event):
        pg = self.pg
        if event.type == pg.QUIT:
            self.controls.blur()
            self.client.request('close')
            return False
        if event.type == pg.WINDOWFOCUSLOST:
            self.focused = False
            self.controls.blur()
            self.client.request('close')
            self.menu_open = True
            self.capture_binding = None
        elif event.type == pg.WINDOWFOCUSGAINED:
            self.focused = True
        elif event.type == pg.KEYDOWN:
            physical = self.key_name(pg, event.key)
            repeat = getattr(event, 'repeat', False)
            name = self.preferences.action(physical)
            if not repeat and physical == 'f9':
                self.capture_binding, self.menu_page = None, 'main'
                self.activate('stop')
            elif not repeat and physical == 'escape':
                if self.capture_binding:
                    self.capture_binding = None
                elif self.menu_open and self.menu_page != 'main':
                    self.menu_page = 'main'
                else:
                    self.toggle_menu()
            elif self.capture_binding:
                if not repeat:
                    self.rebind(physical)
            elif name is None:
                pass
            elif not repeat and name == 'escape':
                self.toggle_menu()
            elif not repeat and name == 'f9':
                self.activate('stop')
            elif not repeat and name in ('f1', 'f3', 'f8', 'f11'):
                if self.mouse_joint is not None and name in ('f1', 'f3', 'f11'):
                    self.controls.keys.discard(self.mouse_joint)
                    self.mouse_joint = None
                    self.publish_intent()
                if name == 'f1':
                    self.show_hud = not self.show_hud
                elif name == 'f3':
                    self.show_debug = not self.show_debug
                elif name == 'f8':
                    self.activate('topmost')
                else:
                    self.toggle_fullscreen()
            elif self.menu_open and name in KEYS:
                pass  # Typing or holding movement keys over the menu never moves the robot.
            elif not repeat and name in ('1', '2'):
                self.activate('left' if name == '1' else 'right')
            else:
                action = self.controls.press(name, repeat)
                if action:
                    self.client.request(action, self.controls.arm)
        elif event.type == pg.KEYUP:
            self.controls.release(self.preferences.action(self.key_name(pg, event.key)))
        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            if self.mouse_joint is not None:
                self.controls.keys.discard(self.mouse_joint)
                self.mouse_joint = None
                self.publish_intent()
        elif event.type == pg.MOUSEMOTION and self.mouse_joint is not None:
            rect = self.joint_rects.get(self.mouse_joint)
            if rect is None or not rect.collidepoint(event.pos):
                self.controls.keys.discard(self.mouse_joint)
                self.mouse_joint = None
                self.publish_intent()
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            for name, rect in reversed(self.hitboxes):
                if rect.collidepoint(event.pos):
                    self.activate(name)
                    break
        return True

    def draw_scene(self, frame):
        pg = self.pg
        width, height = self.screen.get_size()
        self.screen.fill((16, 17, 18))
        self.minimap_rect = None
        self.sidebar_rect = None
        gutter = min(224, width//6)+20 if self.show_hud and not self.show_debug else 0
        viewport = pg.Rect(0, 0, width-gutter, height)
        self.main_image_rect = None
        if frame and (self.demo or fresh_frame(frame)):
            source = frame['surface']
            scale = min(viewport.width/source.get_width(), viewport.height/source.get_height())
            size = (max(1, int(source.get_width()*scale)), max(1, int(source.get_height()*scale)))
            self.main_image_rect = pg.Rect((viewport.width-size[0])//2, (height-size[1])//2, *size)
            self.screen.blit(pg.transform.smoothscale(source, size), self.main_image_rect)
        else:
            self.text('真实画面不可用 / 已过期', (24, height//2), (255, 182, 100))
        x, y = viewport.center
        for delta in (-1, 0, 1):
            color = (22, 22, 22) if delta else (255, 255, 255)
            pg.draw.line(self.screen, color, (x-9, y+delta), (x+9, y+delta))
            pg.draw.line(self.screen, color, (x+delta, y-9), (x+delta, y+9))
        if gutter:
            self.sidebar_rect = pg.Rect(viewport.right+8, 8, gutter-16, height-16)
            if self.preferences.show_minimap:
                extra = 44 if self.client.execution_backend == 'controller_move_p' else 0
                extra += 25 if self.client.robot.get('teleop', {}).get('speed_profile_pending') else 0
                self.draw_minimap(pg.Rect(self.sidebar_rect.x, 8, self.sidebar_rect.width,
                                          min(int(self.sidebar_rect.width*.75)+28, max(80, height-590-extra))))

    def draw_minimap(self, rect):
        pg = self.pg
        name = self.controls.arm+'_wrist' if self.controls.third_person else 'top'
        frame = self.client.frames.get(name)
        self.minimap_camera, self.minimap_rect = name, rect
        self.panel(rect, (25, 27, 29))
        label = '顶视' if name == 'top' else ('左腕' if self.controls.arm == 'left' else '右腕')
        self.text(label+' · 小窗', (rect.x+7, rect.y+3), font=self.small)
        area = pg.Rect(rect.x+4, rect.y+27, rect.width-8, rect.height-31)
        if frame and (self.demo or fresh_frame(frame)):
            source = frame['surface']
            scale = min(area.width/source.get_width(), area.height/source.get_height())
            size = (max(1, int(source.get_width()*scale)), max(1, int(source.get_height()*scale)))
            image_rect = pg.Rect(0, 0, *size)
            image_rect.center = area.center
            self.screen.blit(pg.transform.smoothscale(source, size), image_rect)
        else:
            self.text('暂无新帧', (area.x+8, area.centery-10), (255, 190, 100), self.small)

    def draw_toolbar(self):
        pg = self.pg
        area = self.sidebar_rect
        x, y, width = area.x+6, (self.minimap_rect.bottom+10 if self.minimap_rect else area.y), area.width-12
        profile = {'slow': '慢档', 'normal': '常规档', 'fast': '快档'}[self.controls.speed_profile]
        title = ('左臂' if self.controls.arm == 'left' else '右臂')+' · '+profile
        detour = self.client.robot.get('teleop', {}).get('detour', {})
        if detour.get('state') != 'idle' and detour.get('state'):
            title += ' · '+('绕行 %.0f%%' % (100*detour.get('progress', 0))
                            if detour['state'] == 'executing' else '后台规划')
        self.text(title, (x, y), font=self.small)
        y += 27
        if self.client.robot.get('teleop', {}).get('speed_profile_pending'):
            self.text('等待静止后切档', (x, y), (255, 220, 100), self.small)
            y += 25
        recording = self.recording.status()
        elapsed = int(recording['elapsed_s'])
        record_label = {'recording': f'● 停止录像 {elapsed//60:02}:{elapsed%60:02}',
                        'finalizing': '录像保存中…'}.get(recording['state'], '● 录制完整窗口')
        demonstration = self.demonstration.status()
        demonstration_label = {'recording': '● 示范录制中 / 结束', 'finalizing': '示范保存中…'}.get(
            demonstration['state'], '人类示范 / Codex 参考')
        backend_label = ('确认停止后切换…' if self.client.pending_backend else
                         '方案：'+BACKEND_LABELS[self.client.execution_backend])
        rows = [(('left', self.key_label('1')+' 左臂'), ('right', self.key_label('2')+' 右臂')),
                (('open', '进入控制'), ('acknowledge', '现场确认')),
                (('backend', backend_label),),
                (('speed', self.key_label('tab')+' 档位：'+profile),),
                (('record_window', record_label),),
                (('demonstrations', demonstration_label),),
                (('retry_cameras', '恢复相机'), ('menu', 'Esc 菜单'))]
        for items in rows:
            size = (width-6*(len(items)-1))//len(items)
            for index, (name, label) in enumerate(items):
                rect = pg.Rect(x+index*(size+6), y, size, 26)
                self.button(name, label, rect)
                if name == self.controls.arm:
                    pg.draw.rect(self.screen, (247, 247, 228), rect, 2)
                if name == 'record_window' and recording['state'] == 'recording':
                    pg.draw.rect(self.screen, (235, 80, 70), rect, 2)
                if name == 'demonstrations' and demonstration['state'] == 'recording':
                    pg.draw.rect(self.screen, (235, 80, 70), rect, 2)
            y += 31
        if self.client.execution_backend == 'controller_move_p':
            self.text('平移：允许自动调姿', (x, y), (255, 220, 110), self.small)
            y += 22
            self.text('原生：仅当前自碰撞检查', (x, y), (255, 220, 110), self.small)
            y += 22
        self.keys_top = y+5

    def draw_keys(self):
        pg = self.pg
        area = self.sidebar_rect
        x, y, width = area.x+6, self.keys_top, area.width-12
        cell = min(30, (width-92)//3)
        pitch = cell+4
        side_x, side_width = x+3*cell+16, width-3*cell-16
        self.key_rects = {}
        def cap(key, x, y, w=cell):
            rect = pg.Rect(x, y, w, 24)
            self.key_rects[key] = rect
            held = key in self.controls.keys
            self.panel(rect, (145, 119, 52) if held else (65, 65, 65))
            label = self.key_label(key)
            label += {'space': ' ↑', 'lshift': ' ↓', 'q': ' 合', 'e': ' 开', 'o': ' 逆', 'p': ' 顺'}.get(key, '')
            while label and self.small.size(label)[0] > w-8:
                label = label[:-1]
            self.text(label, (x+(w-self.small.size(label)[0])//2, y+2),
                      (255, 242, 173) if held else (225, 225, 225), self.small)
        self.text('平移 / 基座升降', (x, y), font=self.small)
        y += 20
        cap('w', x+pitch, y)
        for i, key in enumerate(('a', 's', 'd')):
            cap(key, x+i*pitch, y+26)
        cap('space', side_x, y, side_width)
        cap('lshift', side_x, y+26, side_width)
        y += 54
        cap('q', x, y, (width-6)//2)
        cap('e', x+(width-6)//2+6, y, (width-6)//2)
        y += 28
        self.text('转向 / 侧倾', (x, y), font=self.small)
        y += 20
        cap('i', x+pitch, y)
        for i, key in enumerate(('j', 'k', 'l')):
            cap(key, x+i*pitch, y+26)
        cap('o', side_x, y, side_width)
        cap('p', side_x, y+26, side_width)
        self.text('F9 停止 · Esc 菜单', (x, y+54), (220, 220, 220), self.small)

    def draw_joint_buttons(self):
        pg = self.pg
        width, height = self.screen.get_size()
        area = self.sidebar_rect or pg.Rect(width-232, 0, 224, height)
        panel = pg.Rect(area.x, height-118, area.width, 110)
        self.shade(panel, 225, (24, 26, 30))
        local = self.client.execution_backend == 'regularized_move_j'
        active = bool(self.client.session and self.client.robot.get('teleop', {}).get('active'))
        label = '关节 · 按住点动' if local else '关节 · 请切本地方案'
        self.text(label, (panel.x+6, panel.y+2), font=self.small)
        self.joint_rects = {}
        cell = (panel.width-18)//4
        for axis in range(1, 7):
            row, pair = divmod(axis-1, 2)
            for sign, suffix in enumerate(('minus', 'plus')):
                key = f'joint_{axis}_{suffix}'
                rect = pg.Rect(panel.x+6+(pair*2+sign)*(cell+2), panel.y+26+row*26, cell, 24)
                held = key in self.controls.keys and active and local
                color = (145, 119, 52) if held else ((75, 82, 93) if active and local else (48, 50, 54))
                self.button(key, f'J{axis}'+('−' if sign == 0 else '+'), rect, color)
                self.joint_rects[key] = rect

    def draw_menu(self, opening):
        pg = self.pg
        width, height = self.screen.get_size()
        self.shade(pg.Rect(0, 0, width, height), 115)
        if self.menu_page != 'main':
            if self.menu_page == 'demonstrations':
                self.draw_demonstrations()
            else:
                self.draw_settings()
            return
        rect = pg.Rect(width//2-230, (height-500)//2, 460, 500)
        self.text('准备中…' if opening else '遥操作菜单', (rect.x+147, rect.y+4), font=self.menu_font)
        rows = [(('resume_view', '返回画面'),),
                (('open', '进入控制'), ('close', '退出控制')),
                (('acknowledge', '现场复核后确认停止'),),
                (('backend', '确认停止后切换…' if self.client.pending_backend else
                  '控制方案：'+BACKEND_LABELS[self.client.execution_backend]),),
                (('view', self.key_label('f5')+' 视角：'+('顶视' if self.controls.third_person else '腕部')),
                 ('topmost', self.key_label('f8')+' 置顶：'+('开' if self.always_on_top else '关'))),
                (('fullscreen', self.key_label('f11')+' 全屏：'+('开' if self.fullscreen else '关')),
                 ('debug', self.key_label('f3')+' 调试：'+('开' if self.show_debug else '关'))),
                (('keys', '按键 / 设置'), ('demonstrations', '人类示范 / 参考')),
                (('defaults', '启动默认设置'),),
                (('stop', 'F9 · 请求停止'),)]
        for row, items in enumerate(rows):
            size = (rect.width-10*(len(items)-1))//len(items)
            for col, (name, label) in enumerate(items):
                self.button(name, label, pg.Rect(rect.x+col*(size+10), rect.y+48+row*43, size, 36))
        self.text('Esc 菜单 · F9 请求停止 · '+self.key_label('tab')+' 切档', (rect.x+65, rect.y+451), font=self.small)
        self.text('启动只观察 · 菜单打开时暂停按键输入', (rect.x+60, rect.y+476), font=self.small)

    def draw_demonstrations(self):
        pg = self.pg
        width, height = self.screen.get_size()
        rect = pg.Rect(width//2-310, max(8, (height-440)//2), 620, 440)
        self.shade(rect, 235)
        status = self.demonstration.status()
        mode = '亲手操作物体' if self.demonstration_mode == 'human_objects' else '遥控机械臂示教'
        self.text('人类动作示范 · 可选 Codex 参考', (rect.x+18, rect.y+12), font=self.menu_font)
        rows = [('demonstration_mode', '类型：'+mode),
                ('demonstration_record', '结束并保存示范' if status['state'] == 'recording' else
                 ('示范保存中…' if status['state'] == 'finalizing' else '开始录制三路 RGB / 反馈')),
                ('resume_view', '返回实时画面 / 继续示教'),
                ('demonstration_next', '浏览已保存示范（下一条）'),
                ('demonstration_select', '将当前示范选为 Codex 参考'),
                ('demonstration_clear', '关闭 Codex 示范参考'),
                ('settings_back', '返回菜单')]
        for index, (name, label) in enumerate(rows):
            self.button(name, label, pg.Rect(rect.x+18, rect.y+56+index*37, rect.width-36, 31))
        paths = self.demonstration_candidates
        current = paths[self.demonstration_index % len(paths)].name if paths else '暂无已保存真实示范'
        state_label = {'idle': '尚未开始录制', 'recording': '录制中', 'finalizing': '保存中',
                       'saved': '已保存', 'error': '录制失败'}[status['state']]
        lines = [current, status['error'] or self.demonstration_notice or state_label,
                 '亲手示范须先退出控制并确认静止；不会自动失能。',
                 '当前参考：'+self.demonstration_reference]
        for index, line in enumerate(lines):
            self.text(line[:65], (rect.x+18, rect.y+325+index*24), font=self.small)

    def capture_demonstration(self):
        recorder = self.demonstration
        if not recorder.due():
            return
        with self.client.lock:
            robot = self.client.robot
            if recorder.mode == 'human_objects' and (self.client.session or self.client.opening or
                                                     robot.get('teleop', {}).get('active') or robot.get('busy')):
                recorder.stop('亲手示范期间检测到控制会话，已结束录制')
                return
            cameras = {}
            for name, frame in self.client.frames.items():
                if fresh_frame(frame):
                    surface = frame['surface']
                    cameras[name] = (frame['stamp'], surface.get_size(), self.pg.image.tobytes(surface, 'RGB'))
            context = dict(arms=copy.deepcopy(robot.get('arms', {})), teleop=copy.deepcopy(robot.get('teleop', {})),
                           requested_keys=sorted(self.controls.keys), sent_keys=list(getattr(self.client, 'keys', [])),
                           selected_arm=self.controls.arm, backend=self.client.execution_backend,
                           speed_profile=self.controls.speed_profile, focused=self.focused,
                           menu_open=self.menu_open, robot_error=self.client.error,
                           stop_confirmed=robot.get('stop_confirmed'), fault=robot.get('fault'),
                           feedback_note='SDK snapshot; input queues do not prove command execution')
        recorder.submit(cameras, context)

    def draw_settings(self):
        pg = self.pg
        width, height = self.screen.get_size()
        if self.menu_page == 'keys':
            rect = pg.Rect(width//2-450, max(128, (height-520)//2), 900, 520)
            self.text('按键说明与修改', (rect.centerx-100, rect.y), font=self.menu_font)
            self.text('点击按键后按新键；冲突不会覆盖。控制中可查看，退出控制后可修改。',
                      (rect.x+25, rect.y+37), font=self.small)
            for index, (action, help_text) in enumerate(BINDING_HELP.items()):
                x, y = rect.x+(index//12)*450, rect.y+70+(index % 12)*29
                self.text(help_text, (x+14, y+2), font=self.small)
                label = '按下新键…' if self.capture_binding == action else self.key_label(action)
                self.button('bind:'+action, label, pg.Rect(x+270, y, 160, 27))
            self.text('F9始终请求停止；Esc取消改键 / 返回菜单；长按持续运动，松键请求暂停。',
                      (rect.x+20, rect.bottom-87), font=self.small)
            self.button('restore_keys', '恢复默认按键', pg.Rect(rect.x+200, rect.bottom-45, 220, 34))
            self.button('settings_back', '返回菜单', pg.Rect(rect.x+440, rect.bottom-45, 220, 34))
        else:
            rect = pg.Rect(width//2-230, max(128, (height-450)//2), 460, 450)
            self.text('启动默认设置', (rect.x+145, rect.y), font=self.menu_font)
            rows = [('startup_arm', '默认机械臂', {'left': '左臂', 'right': '右臂'}),
                    ('startup_speed', '默认档位', {'slow': '慢档', 'normal': '常规档', 'fast': '快档'}),
                    ('startup_view', '默认视角', {'wrist': '腕部', 'top': '顶视'}),
                    ('always_on_top', '默认置顶', {True: '开', False: '关'}),
                    ('fullscreen', '默认全屏', {True: '开', False: '关'}),
                    ('show_minimap', '辅助相机小窗', {True: '开', False: '关'})]
            for index, (field, label, values) in enumerate(rows):
                self.button('default:'+field, label+'：'+values[getattr(self.preferences, field)],
                            pg.Rect(rect.x, rect.y+55+index*45, rect.width, 36))
            self.text('单击切换并保存，下次启动生效；始终只观察。', (rect.x+28, rect.bottom-100), font=self.small)
            self.button('restore_defaults', '恢复启动默认值', pg.Rect(rect.x, rect.bottom-50, 225, 36))
            self.button('settings_back', '返回菜单', pg.Rect(rect.x+235, rect.bottom-50, 225, 36))

    def draw_debug(self, robot, telemetry, state):
        pg = self.pg
        width, height = self.screen.get_size()
        now, tick = datetime.now().astimezone(), time.monotonic()
        policy = telemetry.get('policy') or asdict(self.client.policy)
        tcp = state.get('tcp_matrix')
        if not self.demo and time.monotonic()-self.tool_tick > .5:
            self.tool, self.tool_tick = tcp_configuration(), time.monotonic()
        if state.get('ee_pose') and self.controls.arm in self.tool.get('arms', {}):
            tcp = (pose(state) @ self.tool['arms'][self.controls.arm]['T_ee_tcp']).tolist()
        uptime = max(0, int(tick-self.started_at))
        arm = '左臂' if self.controls.arm == 'left' else '右臂'
        left = ['PIPER CRAFT · '+BACKEND_LABELS[self.client.execution_backend],
                '%.0f fps · Display: %d x %d' % (self.render_clock.get_fps(), width, height),
                'Selected: %s · View: %s' % (arm, '顶视' if self.controls.third_person else '腕部'), '',
                *debug_pose_lines(tcp, policy.get('axis_convention'), self.key_label('w')),
                'Axes: %s %s / %s -Y_tcp' % (self.key_label('w'), '+Z_tcp' if policy.get('axis_convention') == 'piper_tool' else '+X_tcp', self.key_label('d')),
                'Vertical: %s +Z_base / %s -Z_base' % (self.key_label('space'), self.key_label('lshift')) if policy.get('vertical_frame') == 'base'
                else 'Vertical: %s %s / %s 反向' % (self.key_label('space'), '-X_tcp' if policy.get('axis_convention') == 'piper_tool' else '+Z_tcp', self.key_label('lshift')),
                '', 'Local Time: ' + now.strftime('%Y-%m-%d %H:%M:%S'),
                'Timezone: ' + now.strftime('%Z %z'),
                'Uptime: %02d:%02d:%02d' % (uptime//3600, uptime//60 % 60, uptime % 60)]
        session_started = getattr(self.client, 'session_started', 0.)
        left += ['Session: %.1f s' % max(0., tick-session_started) if self.client.session and session_started else
                 'Session: ' + ('OFFLINE' if self.demo and self.client.session else '未进入控制')]
        joints = state.get('joints_rad')
        if joints is not None:
            left += ['', 'Joints (deg):']
            left += ['J%d: %+.3f' % (i+1, math.degrees(q)) for i, q in enumerate(joints[:6])]
        else:
            left += ['', 'Joints (deg): 等待反馈']
        grip = state.get('gripper', {})
        frame = self.client.frames.get(self.controls.camera)
        feedback_age = state.get('age_s')
        phase = telemetry.get('phase', 'observing')
        phase_label = {'observing': '只观察', 'ready': '等待按键', 'moving': '连续控制',
                       'detour_planning': '后台规划绕行', 'detour_executing': '绕行中（松键取消）',
                       'entry_blocked': '进入控制被阻止', 'waiting_feedback': '等待机械臂跟随',
                       'pausing': '正在确认静止', 'stopping': '已请求停止', 'stopped': '停止已确认',
                       'stop_unconfirmed': '停止尚未确认'}.get(phase, phase)
        stop = ('已确认静止' if robot.get('stop_confirmed') else '已请求停止，尚未确认静止') if robot.get('stopped') else '未请求全局停止'
        right = ['Python: '+sys.version.split()[0]+' / SDL: '+'.'.join(map(str, pg.get_sdl_version())),
                 'Controller: '+str(telemetry.get('execution_backend', self.client.execution_backend) if telemetry.get('active') else self.client.execution_backend),
                 'IK: '+str(telemetry.get('ik_solver') or ('controller' if self.client.execution_backend == 'controller_move_p' else policy.get('ik_solver', '—'))),
                 '检查: '+('仅当前自碰撞检查' if self.client.execution_backend == 'controller_move_p' else '完整关节路径'),
                 'Control: '+str(telemetry.get('phase', 'observing'))+' / '+str(telemetry.get('owner') or '无控制者'),
                 'Status: '+('准备中…' if self.client.opening else phase_label),
                 'Stop: '+stop, '',
                 '有效目标 %.1f Hz · 周期p95 %s ms' % (telemetry.get('target_hz', 0),
                    '%.1f' % telemetry['cycle_p95_ms'] if telemetry.get('cycle_p95_ms') is not None else '—'),
                 '循环 %.1f Hz · 截止超时 %d' % (telemetry.get('control_hz', 0), telemetry.get('deadline_misses', 0)),
                 '反馈帧龄: '+('%.1f ms' % (feedback_age*1000) if feedback_age is not None else '等待反馈'),
                 '画面帧龄: '+('OFFLINE' if self.demo else '%.1f ms' % (max(0., time.time()-frame['stamp'])*1000) if frame else '无新帧'),
                 '平移目标 %.0f · 反馈 %.2f mm/s' % (policy.get('translation_m_s', .005)*1000,
                    telemetry.get('actual_tcp_speed_mm_s', 0)),
                 '夹爪实读 %s mm · 指令 %s mm' % ('%.2f' % (grip['width_m']*1000) if 'width_m' in grip else '—',
                    '%.2f' % telemetry['gripper_target_mm'] if telemetry.get('gripper_target_mm') is not None else '—'),
                 '夹爪SDK力矩 %s · 上限 %.3f' % ('%+.3f' % grip['effort'] if 'effort' in grip else '—',
                    telemetry.get('gripper_effort_limit') or self.client.policy.gripper_effort),
                 '松键额外 %.2f mm · 仅记录' % telemetry.get('release_extra_mm', 0)]
        if not telemetry.get('gripper_control_available', True):
            right += ['夹爪已隔离，Q/E禁用']
        controller = state.get('controller', {})
        if controller:
            enabled = controller.get('motors_enabled', [])
            right += ['Motors: %d/%d enabled · Error: %s' %
                      (sum(enabled), len(enabled), controller.get('err_code', '—'))]
        reason = self.client.error or telemetry.get('termination_reason') or telemetry.get('blocker')
        if reason:
            right += ['', '原始原因：', str(reason)]
        if getattr(self.client, 'camera_error', ''):
            right += ['相机原始原因：', self.client.camera_error]
        if telemetry.get('trace_path'):
            right += ['记录：', telemetry['trace_path']]
        # F3 uses per-line backgrounds, aligned to both screen edges; leave the
        # bottom strip for chat and controls. Long raw causes wrap, not ellipsize.
        column_width, step = (width-36)//2, self.small.get_linesize()
        for lines, align_right in ((left, False), (right, True)):
            if height < 760:
                lines = [line for line in lines if line]
            y = 6
            for line in lines:
                for part in wrap_text(line, self.small, column_width) or ['']:
                    if y+step > height-165:
                        break
                    if part:
                        size = self.small.size(part)[0]
                        x = width-6-size if align_right else 6
                        self.shade(pg.Rect(x-2, y, size+4, step), 150, (50, 50, 50))
                        self.text(part, (x, y-1), (224, 224, 224), self.small)
                    y += step

    def draw_chat(self, robot, telemetry, error, camera_error):
        """Timestamped system chat; active failures persist even with F1 hidden."""
        tick = time.monotonic()
        public_messages = self.codex_messages.poll() if self.codex_messages else []
        if self.codex_session:
            public_messages += self.codex_session.poll()
        if public_messages:
            for message in public_messages:
                age = max(0., time.time()-message['stamp'])
                self.codex_chat.append({'stamp': datetime.fromtimestamp(message['stamp']).strftime('%H:%M:%S'),
                                        'tick': tick-age, 'source': 'Codex', 'text': message['text'],
                                        'color': {'info': (160, 220, 255), 'warning': (255, 220, 110),
                                                  'error': (255, 150, 150)}[message['level']]})
        stop_history = telemetry.get('phase') == 'stopped' and not robot.get('stopped') and not robot.get('fault')
        blocker = error or (None if stop_history else telemetry.get('blocker'))
        current = {}
        if blocker:
            normal_exit = str(blocker) in ('desktop close', 'desktop control closed', 'desktop escape', 'desktop stop',
                                           '离线模拟已停止；未连接设备')
            current['控制'] = (('' if normal_exit else '无法执行：') + control_message(blocker, telemetry).replace('F3', self.key_label('f3')),
                               (190, 190, 190) if normal_exit else (255, 100, 100))
        if camera_error:
            current['相机'] = ('画面源异常：' + camera_error, (255, 170, 80))
        if robot.get('stopped') and not robot.get('stop_confirmed'):
            current['停止'] = ('已请求停止，尚未确认静止', (255, 220, 100))
        if not telemetry.get('gripper_control_available', True):
            current['夹爪'] = ('夹爪已隔离，Q/E禁用', (255, 220, 100))
        if self.window_notice:
            current['窗口'] = (self.window_notice, (255, 220, 100))
        if self.settings_notice:
            current['设置'] = (self.settings_notice, (255, 220, 100))
        recording_error = self.recording_notice or self.recording.status()['error']
        if recording_error:
            current['录像'] = ('窗口录像失败：'+recording_error, (255, 100, 100))
        if self.codex_messages and self.codex_messages.error:
            current['Codex消息源'] = ('公开消息读取失败：'+self.codex_messages.error, (255, 220, 110))
        if self.codex_session and self.codex_session.error:
            current['Codex会话源'] = ('公开会话读取失败：'+self.codex_session.error, (255, 220, 110))
        state = (bool(telemetry.get('active')), self.controls.arm, self.controls.camera, self.controls.speed_profile)
        if state != self.chat_state:
            active, arm, camera, profile = state
            self.chat.append({'stamp': datetime.now().strftime('%H:%M:%S'), 'tick': tick, 'source': '系统',
                              'text': '%s · %s · %s · %s' % ('已进入控制' if active else '只观察 / 未进入控制',
                                '左臂' if arm == 'left' else '右臂', '顶视画面' if camera == 'top' else '腕部画面',
                                {'slow': '慢档', 'normal': '常规档', 'fast': '快档'}[profile]),
                              'color': (190, 190, 190)})
            self.chat_state = state
        for source, (message, color) in current.items():
            if self.chat_current.get(source) != message:
                self.chat.append({'stamp': datetime.now().strftime('%H:%M:%S'), 'tick': tick,
                                  'source': source, 'text': message, 'color': color})
        self.chat_current = {source: message for source, (message, _) in current.items()}
        # Current failures take priority over fading informational history.
        visible = [entry for entry in self.chat if entry['source'] == '系统' and tick-entry['tick'] < 8]
        visible += [entry for entry in self.codex_chat if tick-entry['tick'] < 20]
        visible.sort(key=lambda entry: entry['tick'])
        latest = {entry['source']: entry for entry in self.chat
                  if self.chat_current.get(entry['source']) == entry['text']}
        visible += [latest[source] for source in current if source in latest]
        width, height = self.screen.get_size()
        line_width, step = min(width-36, 940), self.small.get_linesize()
        if self.sidebar_rect:
            line_width = min(line_width, self.sidebar_rect.left-24)
        settings_page = self.menu_open and self.menu_page != 'main'
        if self.menu_open and not settings_page:
            line_width = min(line_width, max(200, (width-460)//2-24))
        rendered = []
        max_lines = 5 if settings_page else max(3, min(8, (height//3)//step))
        per_alert = max(1, max_lines//max(1, len(current)))
        for entry in visible:
            value = '[%s] [%s] %s' % (entry['stamp'], entry['source'], entry['text'])
            parts = wrap_text(value, self.small, line_width)
            if entry['source'] == 'Codex' and len(parts) > 3:
                parts = parts[:3]
                parts[-1] = parts[-1][:-1] + '…'
            if entry['source'] in current and len(parts) > per_alert:
                # Keep every active source visible; the unmodified full cause
                # remains in F3, rather than hiding a second simultaneous fault.
                parts = parts[:per_alert]
                parts[-1] = parts[-1][:-12] + '… ['+self.key_label('f3')+' 查看原文]'
            rendered += [(part, entry['color']) for part in parts]
        lines = rendered[-max_lines:]
        y = 6 if settings_page else height-150-len(lines)*step
        for line, color in lines:
            self.shade(self.pg.Rect(8, y, min(line_width+10, width-16), step), 145)
            self.text(line, (12, y-1), color, self.small)
            y += step

    def draw(self):
        pg = self.pg
        self.render_clock.tick()
        width, height = self.screen.get_size()
        self.buttons = {}
        self.hitboxes = []
        with self.client.lock:
            robot = self.client.robot
            frame = self.client.frames.get(self.controls.camera)
            error, camera_error = self.client.error, getattr(self.client, 'camera_error', '')
            opening = self.client.opening
        telemetry = robot.get('teleop', {})
        active = bool(telemetry.get('active'))
        if active and not self.was_active:
            self.menu_open = False
        elif self.was_active and not active:
            self.menu_open = True
        self.was_active = active
        self.draw_scene(frame)
        state = robot.get('arms', {}).get(self.controls.arm, {})
        self.key_rects = {}
        if self.show_hud and self.sidebar_rect:
            self.draw_toolbar()
            self.draw_keys()
        if self.demo and not self.show_debug:
            self.text('OFFLINE DEMO · TCP 模拟 · 不连接设备', (16, 16), (255, 205, 96), self.small)
        if self.show_debug:
            self.draw_debug(robot, telemetry, state)
            if self.demo:
                self.text('OFFLINE DEMO · 不连接设备', (width-265, height-145), (255, 205, 96), self.small)
        self.joint_rects = {}
        if self.show_hud:
            self.draw_joint_buttons()
        if self.menu_open:
            self.draw_menu(opening)
        self.draw_chat(robot, telemetry, error, camera_error)
        recording = self.recording.status()
        if recording['state'] != self.recording_seen:
            self.recording_seen = recording['state']
            if recording['state'] == 'saved':
                self.chat.append({'stamp': datetime.now().strftime('%H:%M:%S'), 'tick': time.monotonic(),
                                  'source': '系统', 'text': '窗口录像已保存：'+recording['path'],
                                  'color': (180, 240, 170)})
        if self.recording.due():
            self.recording.submit(pg.image.tobytes(self.screen, 'RGB'), self.screen.get_size())
        self.capture_demonstration()
        status = self.demonstration.status()
        if status['state'] != self.demonstration_seen:
            self.demonstration_seen = status['state']
            if status['state'] == 'saved':
                self.demonstration_notice = '示范已保存：'+Path(status['path']).name
                self.demonstration_candidates = available(self.demonstration.destination)
                self.demonstration_index = 0
        pg.display.flip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='interactive offline TCP simulation; never connect devices')
    parser.add_argument('--screenshot', type=Path, help='save one offline preview frame (requires --demo)')
    parser.add_argument('--frames', type=int, help='exit after N display frames (preview/testing)')
    parser.add_argument('--no-always-on-top', action='store_true', help='start without the default always-on-top flag')
    parser.add_argument('--codex-messages', type=Path,
                        help='public Codex JSONL source (default: .runtime/codex_messages.jsonl; demo reads only if explicit)')
    args = parser.parse_args()
    if args.screenshot and not args.demo:
        parser.error('--screenshot requires --demo')
    try:
        import pygame
    except ImportError:
        parser.exit(1, '请先运行 uv sync --extra desktop 安装桌面依赖。\n')
    pygame.display.init()
    pygame.font.init()
    if not args.demo:
        try:
            bind_current_session()
        except (ValueError, OSError):
            pass
    client = DemoClient() if args.demo else Client()
    window = Window(client, demo=args.demo, always_on_top=False if args.no_always_on_top else None,
                    codex_messages_path=args.codex_messages)
    client.start()
    clock = pygame.time.Clock()
    exit_requested = threading.Event()
    old_signals = {sig: signal.signal(sig, lambda *_: exit_requested.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    count, running = 0, True
    try:
        while running and not exit_requested.is_set():
            for event in pygame.event.get():
                running = window.handle(event) and running
            window.publish_intent()
            window.draw()
            count += 1
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(window.screen, str(args.screenshot))
                running = False
            if args.frames and count >= args.frames:
                running = False
            clock.tick(60)
    finally:
        try:
            client.shutdown()
        finally:
            try:
                try:
                    window.recording.close()
                finally:
                    window.demonstration.close()
            finally:
                pygame.quit()
                for sig, handler in old_signals.items():
                    signal.signal(sig, handler)


if __name__ == '__main__':
    main()
