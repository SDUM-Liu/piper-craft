"""Operator-enabled Agent tools routed through the real desktop control client."""
from dataclasses import dataclass, field
import threading
import time

from astra_vla.config import ROOT
from astra_vla.joint_target import joint_target
from astra_vla.task_archive import TaskArchive
from .session import LiveBackend, compact_device, task_session


@dataclass
class DesktopRequest:
    action: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: dict = field(default_factory=dict)
    requested_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    started_wall: float | None = None
    session: str | None = None
    key: str | None = None

    def finish(self, status, reason=None, **feedback):
        self.result = dict(status=status, arrived=feedback.pop('arrived', None), reason=reason, **feedback)
        self.done.set()


class DesktopBackend(LiveBackend):
    """Only the focused desktop may consume a request. No raw motion RPC here."""
    mode = 'hardware_assisted'
    actions_enabled = True

    def __init__(self, client, interrupted):
        super().__init__(rpc=client.rpc)
        self.client, self.interrupted = client, interrupted
        self.lock = threading.Lock()
        self.request = None
        self.cancel_requested = threading.Event()

    def execute(self, action, *, cancelled):
        request = DesktopRequest(action)
        with self.lock:
            if self.request or cancelled():
                return dict(status='rejected', arrived=None, reason='desktop_not_armed_or_busy')
            self.request = request
        while not request.done.wait(.05):
            if cancelled():
                self.cancel_requested.set()
            if time.monotonic()-request.requested_at > 35:
                self.cancel_requested.set()
                # A dead UI loses the ordinary worker input heartbeat independently.
                return dict(status='unknown', arrived=None, reason='no_terminal_desktop_feedback')
        return request.result

    def cancel(self):
        self.cancel_requested.set()

    def disable(self):
        self.interrupted.set()
        self.cancel_requested.set()

    def pump(self, *, focused, menu, arm, manual_keys):
        with self.lock:
            request = self.request
            if request is None:
                return None
            client = self.client
            now, wall = time.monotonic(), time.time()
            with client.lock:
                session, robot, frames = client.session, client.robot, client.frames
                available = (client.control_active and not client.stop_pending and not client.opening
                             and client.pending_coordinate_mode is None and not client.cancel_detour
                             and client.execution_backend == 'regularized_move_j')
            telemetry = robot.get('teleop', {})
            ready = (focused and not menu and not manual_keys and available and session
                     and session == telemetry.get('session') and telemetry.get('active')
                     and arm == telemetry.get('arm') == request.action['arm']
                     and not robot.get('fault') and not robot.get('stopped')
                     and all(name in frames and 0 <= wall-frames[name]['stamp'] <= self.max_age
                             for name in ('top', 'left_wrist', 'right_wrist')))
            device = compact_device(robot, wall, self.state_age)
            ready = ready and all(row['valid'] and row['error'] == 0 and row['arm_status'] == 0
                                  for row in device['arms'].values())
            cancelled = self.interrupted.is_set() or self.cancel_requested.is_set()
            if request.started_at is None:
                if (cancelled or not ready or now-request.requested_at > .5 or
                        not 0 <= wall-request.action['observation_time'] <= self.max_age):
                    request.finish('rejected', 'desktop_focus_control_arm_or_source_unavailable')
                else:
                    request.session, request.started_at, request.started_wall = session, now, wall
                    if request.action['kind'] == 'move_arm':
                        client.request_joint_target(arm, request.action['joints_deg'],
                                                    request_id=request.action['id'])
                    else:
                        request.key = {'close': 'q', 'open': 'e'}[request.action['command']]
                        return request.key
            elif not ready or session != request.session or cancelled or now-request.started_at > 30:
                if session == request.session:
                    client.request('close')
                request.finish('unknown', 'desktop_interrupted_or_feedback_unavailable')
            elif request.key is not None:
                if now-request.started_at < .35:
                    return request.key
                self.gripper_result(request, telemetry)
            else:
                self.move_result(request, telemetry, device['arms'][arm])
            if request.done.is_set():
                self.request = None
                self.cancel_requested.clear()
            return None

    @staticmethod
    def gripper_result(request, telemetry):
        for row in reversed(telemetry.get('step_feedback', [])):
            if (row.get('session') != request.session or row.get('started_at', 0) < request.started_wall
                    or row.get('keys') != [request.key]):
                continue
            if row['settled'] and row['feedback_status'] == 'measured' and row['targets_sent']:
                request.finish('completed', source='desktop_sdk_key_feedback', settled=True,
                               source_time=row['source_stamp'], gripper=row['gripper'])
            elif row['finalized']:
                request.finish('unknown', 'gripper_step_not_confirmed')
            return

    @staticmethod
    def move_result(request, telemetry, arm):
        result = telemetry.get('joint_target', {})
        if result.get('request_id') != request.action['id']:
            return
        if (result['state'] == 'completed' and telemetry.get('stop_confirmed') and
                result.get('joint_sdk') == joint_target(request.action['joints_deg'])['joint_sdk']):
            request.finish('completed', arrived=True, source='desktop_sdk_joint_target',
                           source_time=arm['source_time'], settled=True)
        elif result['state'] in ('blocked', 'cancelled'):
            request.finish('unknown', 'joint_route_'+result['state'])


class DesktopAgent:
    def __init__(self, client, *, model_name, account, contract, task_id):
        self.client, self.model_name, self.account = client, model_name, account
        self.contract, self.task_id = contract, task_id
        self.thread, self.backend = None, None
        self.notice = 'Agent 待启用 · F7'

    def toggle(self):
        if self.thread and self.thread.is_alive():
            self.interrupt()
            return
        self.backend = DesktopBackend(self.client, threading.Event())
        self.thread = threading.Thread(target=self.run, args=(self.backend,), name='desktop-agent', daemon=True)
        self.notice = 'Agent 已启用 · F7 停止'
        self.thread.start()

    def interrupt(self):
        if self.backend:
            self.backend.disable()
        self.notice = 'Agent 已停止 · F7 重新启动'

    def run(self, backend):
        from .__main__ import authenticated_model
        from .chatgpt import ChatGPT

        try:
            model = authenticated_model(ChatGPT(), self.account, self.model_name)
            if backend.interrupted.is_set():
                return
            archive = TaskArchive(ROOT)
            with task_session(archive, self.task_id, model, self.contract, backend,
                              cancelled=backend.interrupted.is_set) as session:
                result = session.run()
            self.notice = 'Agent '+result['status']+' · F7'
        except (ValueError, RuntimeError, OSError) as exc:
            self.notice = 'Agent '+str(exc)[:88]+' · F7'
        finally:
            backend.disable()
