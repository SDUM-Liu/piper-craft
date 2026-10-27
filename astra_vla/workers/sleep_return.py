"""Explicit selected-arm sleep return inside an existing local control session."""
from astra_vla.sleep_return import load_sleep_target


class SleepReturn:
    def return_sleep(self, args):
        # TX then session lock matches pause/write ordering. A delayed RPC may
        # never start in a different session or override a menu/stop heartbeat.
        with self.robot.tx_lock, self.mutex:
            if (not self.active or args.get('session') != self.session or
                    args.get('arm') != self.arm):
                raise RuntimeError('请先为所选臂进入控制，再请求回 Sleep')
            if self.native_control:
                raise ValueError('回 Sleep 使用普通 MOVE_J，请先切换到本地方案')
            if getattr(self, 'cancel_automatic', False):
                raise RuntimeError('请先关闭菜单，再请求回 Sleep')
            self.check_watchdog()
            if self.robot.stopped.is_set() or self.robot.fault:
                raise RuntimeError('stopped before sleep return')
            if self.token_value() != self.token:
                raise RuntimeError('configuration changed before sleep return')
            target = load_sleep_target(self.arm)
            if self.detour_context is not None:
                raise RuntimeError('已有自动动作，请先取消后再请求回 Sleep')
            self.keys = frozenset()
            self.generation += 1
            generation = self.generation
            ended_step = self.key_feedback.change([], self.clock.monotonic(), self.clock.time(), 'sleep return')
        self.pause(generation)
        if ended_step:
            self.capture_key_feedback()
        with self.mutex:
            if not self.active or generation != self.generation or self.robot.stopped.is_set():
                return self.view()
            self.sleep_return = dict(target, state='waiting', arm=self.arm)
            self.detour_context = {'state': 'starting', 'kind': 'sleep', 'generation': generation,
                                   'session': self.session, 'keys': frozenset(), 'sleep_target': target,
                                   'speed_percent': self.policy.speed_percent,
                                   'joint_lead_deg': self.policy.joint_lead_deg}
            self.reason = '回 Sleep：等待新鲜静止反馈，再检查完整关节路径；Esc或F9取消'
            self.trace.record('sleep_requested', generation=generation, target=target)
        self.input_changed.set()
        return self.view()
