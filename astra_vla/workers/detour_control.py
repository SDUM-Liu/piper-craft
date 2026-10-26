"""Live detour lifecycle, hosted only by the exclusive teleoperation owner."""
from dataclasses import asdict
import math

import numpy as np

from astra_vla.teleop import GRIPPER_KEYS, JOINT_KEYS, pose, rotation_error
from astra_vla.teleop_detour import DetourPolicy, endpoint_intent


class DetourControl:
    def reset_detour(self):
        context = getattr(self, 'detour_context', None)
        if context and context.get('future'):
            context['future'].cancel()
        self.detour_context = None
        self.detour_slow_since = None

    def detour_status(self):
        context = getattr(self, 'detour_context', None)
        return {'enabled': bool(getattr(self, 'detour_policy', DetourPolicy(enabled=False)).enabled
                                and hasattr(self.robot, 'detour_pool') and not getattr(self, 'native_control', False)),
                'state': context['state'] if context else 'idle',
                'target_tcp': context.get('target') if context else None,
                'goal_joints_rad': context.get('plan', {}).get('goal_joints_rad') if context else None,
                'progress': context.get('progress', 0.) if context else None,
                'temporary_tcp_departure': 'unrestricted', 'collision_scope': 'robot_only'}

    def consider_detour(self, keys, generation, result):
        if keys & JOINT_KEYS:
            return False
        config = self.detour_policy
        if self.detour_context is not None:
            # Do not push an already boundary-directed local candidate farther
            # while the background planner is finding another branch. The last
            # validated target remains in force; no pause/settling transaction.
            return bool(result.get('detour_recommended'))
        if not config.enabled or not hasattr(self.robot, 'detour_pool') or not keys or keys & GRIPPER_KEYS:
            self.detour_slow_since = None
            return False
        progress = result.get('ik', {}).get('progress_fraction', 1.)
        if not result.get('detour_recommended') and progress >= config.low_progress:
            self.detour_slow_since = None
            return False
        now = self.clock.monotonic()
        if self.detour_slow_since is None:
            self.detour_slow_since = now
        if not result.get('detour_recommended') and now-self.detour_slow_since < config.trigger_s:
            return False
        with self.mutex:
            if not self.active or generation != self.generation or keys != self.keys:
                return True
            self.detour_context = {'state': 'starting', 'generation': generation,
                                   'session': self.session, 'keys': keys}
            self.reason = '正在后台规划绕行；松键或改键取消'
            self.trace.record('detour_requested', generation=generation,
                              progress_fraction=progress, boundary_hint=bool(result.get('detour_recommended')))
        return True

    def detour_blocked(self, generation, reason):
        self.pause(generation)
        with self.mutex:
            if self.active and generation == self.generation:
                self.blocked_generation = generation
                self.reason = reason+'；松键后重试'
                self.trace.record('detour_blocked', generation=generation, reason=reason)

    def service_detour(self, keys, generation, before, started):
        from astra_vla.workers.teleop import (geometry_detour_plan, geometry_detour_step,
                                            CalculationExpired, SendWindowExpired)
        context = self.detour_context
        if not context:
            return False
        if context['session'] != self.session or context['generation'] != generation or keys != context['keys']:
            self.pause(generation)
            return True
        state = before[self.arm]
        now = self.clock.monotonic()
        if context['state'] == 'starting':
            target = endpoint_intent(state, self.geometry['tcp'], self.policy, keys,
                                     self.maximum_width, self.detour_policy.goal_horizon_s)
            future = self.robot.detour_pool.submit(geometry_detour_plan, state, target.tolist(),
                                                   self.geometry, asdict(self.detour_policy))
            with self.mutex:
                if context is not self.detour_context or generation != self.generation:
                    future.cancel()
                    return True
                context.update(state='planning', future=future, started=now, before=before,
                               target=target.tolist())
                self.phase = 'detour_planning'
                self.reason = '正在搜索绕行路径；继续按住原键，松键取消'
                self.trace.record('detour_planning', generation=generation, target_tcp=target.tolist())
            return False
        if context['state'] == 'planning':
            if now-context['started'] > self.detour_policy.planning_timeout_s+1:
                self.detour_blocked(generation, '绕行规划超时')
                return True
            if not context['future'].done():
                self.phase = 'detour_planning'
                self.reason = '正在搜索绕行路径；继续按住原键，松键取消'
                return False
            try:
                plan = context['future'].result()
                for field in ('start_joints_rad', 'goal_joints_rad'):
                    vector = np.asarray(plan[field], dtype=float)
                    if vector.shape != (6,) or not np.isfinite(vector).all():
                        raise ValueError('invalid detour joint plan')
                if not np.allclose(np.asarray(plan['target_tcp']), context['target'], atol=1e-12, rtol=0):
                    raise ValueError('detour target changed during planning')
            except Exception as error:
                self.detour_blocked(generation, str(error))
                return True
            with self.mutex:
                if context is not self.detour_context or generation != self.generation or not self.active:
                    return True
            if (np.max(np.abs(np.array(state['joints_rad'])-plan['start_joints_rad'])) > math.radians(self.policy.joint_lead_deg)
                    or abs(state['gripper']['width_m']-context['before'][self.arm]['gripper']['width_m']) > .00025):
                self.detour_blocked(generation, '规划期间起姿变化，已丢弃绕行路径')
                return True
            for arm in before:
                old, latest = context['before'][arm], before[arm]
                reference = dict(old, joints_rad=latest['joints_rad'], ee_pose=latest['ee_pose']) if arm == self.arm else old
                self.robot.check_motion_unchanged(arm, reference, latest, 'during detour planning')
            with self.mutex:
                if context is not self.detour_context or generation != self.generation or not self.active:
                    return True
                context.update(state='executing', plan=plan, started=now, progress=0., last_progress=now)
                self.phase = 'detour_executing'
                self.reason = '正在绕行到固定TCP目标；松键或改键取消'
                self.trace.record('detour_plan', generation=generation, plan=plan)
            return True
        plan = context['plan']
        if now-context['started'] > self.detour_policy.execution_timeout_s:
            self.detour_blocked(generation, '绕行执行超时')
            return
        distance = float(np.max(np.abs(np.array(plan['goal_joints_rad'])-state['joints_rad'])))
        initial = max(1e-9, float(np.max(np.abs(np.array(plan['goal_joints_rad'])-plan['start_joints_rad']))))
        progress = max(0., min(1., 1-distance/initial))
        if progress > context['progress']+.0001:
            context['progress'], context['last_progress'] = progress, now
        if distance <= math.radians(.1):
            actual = pose(state) @ self.integrator.tcp
            target = np.array(plan['target_tcp'])
            if state['controller']['motion_status'] == 0:
                if (np.linalg.norm(actual[:3, 3]-target[:3, 3]) > self.policy.position_lead_m
                        or rotation_error(actual, target) > math.radians(self.policy.rotation_lead_deg)):
                    self.detour_blocked(generation, '绕行关节到点但SDK末端反馈未到TCP目标')
                    return
                self.trace.record('detour_completed', generation=generation, state=before,
                                  target_tcp=plan['target_tcp'])
                with self.mutex:
                    if context is self.detour_context and generation == self.generation and self.active:
                        self.reset_detour()
                        self.integrator.reset(self.intent_state(state))
                        self.phase, self.reason = 'moving', None
                return
        if now-context['last_progress'] > 2.:
            self.detour_blocked(generation, '绕行关节跟随停滞')
            return
        # Geometry has its own budget inside the frame. Feedback, input and
        # configuration rechecks plus SDK TX share the full control period,
        # just as in the ordinary jog path.
        geometry_deadline = started+self.policy.geometry_timeout_s
        send_deadline = started+1/self.policy.control_hz
        checkpoint = self.clock.monotonic()
        def stage(name):
            nonlocal checkpoint
            current = self.clock.monotonic()
            self.cycle_stages[name] = (current-checkpoint)*1000
            checkpoint = current
        args = (state, plan, asdict(self.policy), self.last_goal_joints, self.velocity)
        future = self.robot.geometry_pool.submit(geometry_detour_step, *args)
        try:
            result = future.result(timeout=max(.001, geometry_deadline-self.clock.monotonic()))
        except TimeoutError as error:
            future.cancel()
            raise CalculationExpired('detour step geometry deadline exceeded') from error
        stage('detour_geometry')
        if self.clock.monotonic() > geometry_deadline:
            raise CalculationExpired('detour step geometry deadline exceeded')
        if result is None:
            self.phase = 'detour_executing'
            self.reason = '绕行等待关节跟随；松键取消'
            return
        refreshed = {arm: self.robot.healthy(arm, arm == self.arm) for arm in before}
        for arm, old in before.items():
            latest = refreshed[arm]
            reference = dict(old, joints_rad=latest['joints_rad'], ee_pose=latest['ee_pose']) if arm == self.arm else old
            self.robot.check_motion_unchanged(arm, reference, latest, 'during detour step')
        if np.max(np.abs(np.array(refreshed[self.arm]['joints_rad'])-state['joints_rad'])) > math.radians(.05):
            try:
                result = self.robot.recheck_detour_target(refreshed[self.arm], plan, asdict(self.policy),
                                                          self.last_goal_joints, self.velocity, result)
            except TimeoutError as error:
                raise CalculationExpired('detour fresh path deadline exceeded') from error
            if self.clock.monotonic() > geometry_deadline:
                raise CalculationExpired('detour fresh path deadline exceeded')
        stage('detour_feedback_recheck')
        state = refreshed[self.arm]
        if abs(state['gripper']['width_m']-context['before'][self.arm]['gripper']['width_m']) > .00025:
            self.detour_blocked(generation, '绕行期间夹爪开度变化')
            return
        with self.robot.tx_lock, self.mutex:
            if (not self.active or self.robot.stopped.is_set() or context is not self.detour_context
                    or generation != self.generation or keys != self.keys
                    or self.clock.monotonic()-self.last_input > self.policy.input_timeout_s):
                return
            if self.token_value() != self.token:
                raise RuntimeError('configuration changed before detour write')
            if self.camera_error:
                raise RuntimeError(self.camera_error)
            self.fresh_cameras(self.cameras)
            stage('detour_configuration')
            if self.clock.monotonic() > send_deadline:
                raise CalculationExpired('detour send deadline exceeded')
            self.trace.activate()
            commands = []
            if self.paused:
                commands += [('MotionCtrl_1', (0, 4, 0)),
                             ('JointCtrl', tuple(round(math.degrees(q)*1000) for q in state['joints_rad']))]
            commands += [('MotionCtrl_2', (1, 1, int(self.policy.speed_percent), 0)),
                         ('JointCtrl', tuple(result['joint_sdk']))]
            if self.paused:
                commands += [('MotionCtrl_1', (0, 2, 0))]
            for method, arguments in commands:
                if self.robot.stopped.is_set() or self.robot.fault:
                    raise RuntimeError('stopped before detour write')
                if (context is not self.detour_context or generation != self.generation
                        or self.clock.monotonic()-self.last_input > self.policy.input_timeout_s):
                    return
                if self.clock.monotonic() > send_deadline:
                    raise SendWindowExpired('detour send deadline exceeded')
                self.trace.record('command', method=method, arguments=list(arguments), generation=generation)
                self.cycle_writes += 1
                self.robot.actuator_commands_sent = True
                getattr(self.robot.arms[self.arm], method)(*arguments)
            if self.clock.monotonic() > send_deadline:
                raise SendWindowExpired('detour SDK send deadline exceeded')
            stage('detour_SDK_send')
            self.velocity = result['velocity']
            self.last_goal_joints = np.deg2rad(np.array(result['joint_sdk'])/1000).tolist()
            self.paused, self.phase, self.last_confirmation = False, 'detour_executing', None
            self.reason = '正在绕行到固定TCP目标；松键或改键取消'
            self.deadline_misses = 0
            self.accepted_keys = keys
            self.key_feedback.sent(keys)
            self.target_stamps.append(self.clock.monotonic())
            self.trace.record('detour_target', generation=generation, keys=sorted(keys),
                              joints_sdk=result['joint_sdk'], state=refreshed,
                              target_tcp=plan['target_tcp'], geometry_samples=result['samples'], progress=progress)
