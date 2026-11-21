"""Explicit selected-arm sleep return inside an existing local control session."""
from astra_vla.sleep_return import load_sleep_target


class SleepReturn:
    def return_sleep(self, args):
        return self.request_joint_route(args, None, 'sleep')

    def set_joint_target(self, args):
        from astra_vla.joint_target import joint_target
        target = joint_target(args.get('joints_deg'))
        request_id = args.get('request_id')
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id.isascii() or not request_id.isidentifier() or len(request_id) > 64:
                raise ValueError('invalid joint target request ID')
            target['request_id'] = request_id
        return self.request_joint_route(args, target, 'joint_target')

    def request_joint_route(self, args, target, kind):
        label = '回 Sleep' if kind == 'sleep' else '关节角度控制'
        # TX then session lock matches pause/write ordering. A delayed RPC may
        # never start in a different session or override a menu/stop heartbeat.
        with self.robot.tx_lock, self.mutex:
            if (not self.active or args.get('session') != self.session or
                    args.get('arm') != self.arm):
                raise RuntimeError('请先为所选臂进入控制，再请求' + label)
            if self.native_control:
                raise ValueError(label + '使用普通 MOVE_J，请先切换到本地方案')
            if getattr(self, 'cancel_automatic', False):
                raise RuntimeError('请先关闭菜单，再请求' + label)
            if kind == 'joint_target' and (type(args.get('input_sequence')) is not int or
                    args['input_sequence'] != self.sequence or
                    args['input_sequence'] <= getattr(self, 'last_joint_target_sequence', -1) or
                    self.keys or self.pending_speed_profile is not None):
                raise RuntimeError('角度请求已过期或仍有输入；请松键后重新执行')
            self.check_watchdog()
            if self.robot.stopped.is_set() or self.robot.fault:
                raise RuntimeError('stopped before sleep return')
            if self.token_value() != self.token:
                raise RuntimeError('configuration changed before sleep return')
            if kind == 'sleep':
                target = load_sleep_target(self.arm)
            if self.detour_context is not None:
                raise RuntimeError('已有自动动作，请先取消后再请求' + label)
            if kind == 'joint_target':
                self.last_joint_target_sequence = args['input_sequence']
            self.keys = frozenset()
            self.generation += 1
            generation = self.generation
            ended_step = self.key_feedback.change([], self.clock.monotonic(), self.clock.time(), kind)
        self.pause(generation)
        if ended_step:
            self.capture_key_feedback()
        with self.mutex:
            if not self.active or generation != self.generation or self.robot.stopped.is_set():
                return self.view()
            setattr(self, 'sleep_return' if kind == 'sleep' else 'joint_target',
                    dict(target, state='waiting', arm=self.arm))
            self.detour_context = {'state': 'starting', 'kind': kind, 'generation': generation,
                                   'session': self.session, 'keys': frozenset(), 'sleep_target': target,
                                   'speed_percent': self.policy.speed_percent,
                                   'joint_lead_deg': self.policy.joint_lead_deg}
            self.reason = label + '：等待新鲜静止反馈；Esc或F9取消'
            self.trace.record(kind + '_requested', generation=generation, target=target)
        self.input_changed.set()
        return self.view()
