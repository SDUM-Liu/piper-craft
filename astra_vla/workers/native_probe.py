"""Native MOVE_P backend and bounded legacy probe inside the existing CAN owner."""

from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from astra_vla.teleop import EXECUTION_BACKENDS, TRANSLATION_KEYS, pose, rotation_error


def pose_sdk(transform):
    value = np.asarray(transform, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("invalid native probe pose")
    xyz = np.rint(value[:3, 3] * 1e6)
    rpy = np.rint(Rotation.from_matrix(value[:3, :3]).as_euler("xyz", degrees=True) * 1000)
    return tuple(int(x) for x in np.r_[xyz, rpy])


class NativeProbe:
    def configure_native_probe(self, args, arm):
        enabled = args.get("native_pose_probe", False)
        if not isinstance(enabled, bool):
            raise ValueError("native probe flag must be boolean")
        if enabled and (arm != "right" or args.get("speed_profile") != "slow"):
            raise ValueError("native probe requires right arm and explicit slow profile")
        backend = args.get('execution_backend', 'controller_move_p' if enabled else 'regularized_move_j')
        if backend not in EXECUTION_BACKENDS or (enabled and backend != 'controller_move_p'):
            raise ValueError('unknown or conflicting execution backend')
        self.native_pose_probe = enabled
        self.execution_backend = backend
        self.native_control = backend == 'controller_move_p'

    def begin_native_probe(self, state):
        self.native_origin = pose(state) @ self.integrator.tcp
        self.native_joints = np.asarray(state["joints_rad"]).copy()
        self.native_started = self.clock.monotonic()

    def native_probe_policy(self):
        speed = min(self.policy.speed_percent, 5) if self.policy.speed_profile == 'slow' else self.policy.speed_percent
        self.policy = replace(self.policy, speed_percent=speed, ik_solver='controller')

    def native_probe_bounds(self, state, target=None):
        if not self.native_pose_probe:
            return
        if self.clock.monotonic() - self.native_started > 20:
            raise RuntimeError("native probe 20 second session budget expired")
        actual = pose(state) @ self.integrator.tcp
        for candidate in (actual,) if target is None else (actual, target):
            if np.linalg.norm(candidate[:3, 3] - self.native_origin[:3, 3]) > 0.005 or rotation_error(
                self.native_origin, candidate
            ) > np.deg2rad(2):
                raise RuntimeError("native probe 5 mm / 2 degree envelope exceeded")
        if np.max(np.abs(np.asarray(state["joints_rad"]) - self.native_joints)) > np.deg2rad(8):
            raise RuntimeError("native probe joint departure exceeds 8 degrees")

    def send_native_probe(self, proposal, before, keys, generation, started, mark):
        # Native MOVE_P deliberately does not query local URDF collisions.
        # Feedback, target feasibility, cancellation and tracking remain checked.
        from astra_vla.workers.teleop import native_feedback, geometry_native_step, CalculationExpired, SendWindowExpired
        from astra_vla.native_pose import NativePoseUnavailable
        from dataclasses import asdict

        self.native_probe_bounds(before[self.arm], proposal["tcp_target"])
        if self.native_pose_probe or not proposal['arm_motion']:
            job = self.robot.geometry_pool.submit(native_feedback, before[self.arm], self.geometry)
        else:
            relax = bool(keys & TRANSLATION_KEYS) and not keys & {'i', 'k', 'j', 'l', 'o', 'p'}
            job = self.robot.geometry_pool.submit(geometry_native_step, before[self.arm], self.geometry,
                proposal, self.native_limits, asdict(self.policy), relax)
        try:
            selected = job.result(timeout=self.policy.geometry_timeout_s)
            if isinstance(selected, dict):
                chosen = np.asarray(selected['tcp_target'], dtype=float)
                if (chosen.shape != (4, 4) or not np.isfinite(chosen).all() or
                        not 0 < selected['progress'] <= 1 or not np.isfinite(selected['adjustment_deg'])):
                    raise RuntimeError('invalid native adapted target')
                proposal = dict(proposal, tcp_target=chosen, ee_target=chosen@self.integrator.inverse_tcp,
                                speed_fraction=proposal['speed_fraction']*selected['progress'])
                self.native_adjustment_deg = selected['adjustment_deg']
            elif selected is not True:
                raise RuntimeError("native feedback validation rejected")
        except NativePoseUnavailable as exc:
            with self.mutex:
                if not self.active or generation != self.generation:
                    return
            self.pause(generation)
            with self.mutex:
                if generation == self.generation:
                    self.reason, self.blocked_generation = str(exc), generation
            self.trace.record('native_pose_blocked', generation=generation, reason=str(exc))
            return
        except (TimeoutError, FutureTimeoutError) as exc:
            job.cancel()
            raise CalculationExpired("native validation timeout") from exc
        mark("native_validation")
        target = pose_sdk(proposal["ee_target"])
        with self.robot.tx_lock, self.mutex:
            if (
                not self.active
                or generation != self.generation
                or self.clock.monotonic() - self.last_input > self.policy.input_timeout_s
            ):
                return
            self.fresh_cameras(self.cameras)
            if self.camera_error or self.token_value() != self.token:
                raise RuntimeError(self.camera_error or "configuration changed before native probe write")
            for arm in before:
                latest = self.robot.healthy(arm, arm == self.arm)
                self.robot.check_motion_state(latest)
                old = before[arm]
                reference = dict(old, joints_rad=latest['joints_rad'], ee_pose=latest['ee_pose'],
                                 gripper=dict(old['gripper'], width_m=latest['gripper']['width_m']))
                requested = self.gripper_enable_request_at
                if (arm == self.arm and not old['gripper']['enabled'] and latest['gripper']['enabled'] and
                        requested is not None and 0 <= self.clock.monotonic()-requested <= .3):
                    reference['gripper']['enabled'] = True
                self.robot.check_motion_unchanged(arm, reference, latest, 'during native calculation')
                if arm == self.arm:
                    self.native_probe_bounds(latest, proposal['tcp_target'])
                    actual_tcp = pose(latest) @ self.integrator.tcp
                    if (np.linalg.norm(proposal['tcp_target'][:3, 3]-actual_tcp[:3, 3]) > self.policy.position_lead_m or
                            rotation_error(actual_tcp, proposal['tcp_target']) > np.deg2rad(self.policy.rotation_lead_deg)):
                        self.trace.record('discarded', reason='native target outside fresh tracking lead')
                        return
            state = before[self.arm]
            commands = []
            if self.paused:
                commands.append(("MotionCtrl_1", (0, 4, 0)))
            commands.append(("MotionCtrl_2", (1, 0, int(self.policy.speed_percent), 0)))
            if self.paused:
                commands.append(("EndPoseCtrl", pose_sdk(pose(state))))
            commands.append(("EndPoseCtrl", target))
            if self.paused:
                commands.append(("MotionCtrl_1", (0, 2, 0)))
            if proposal['gripper_motion']:
                commands.append(('GripperCtrl', (round(proposal['width_m']*1e6),
                                                  round(self.policy.gripper_effort*1000), 1, 0)))
            self.trace.activate()
            for method, arguments in commands:
                if self.robot.stopped.is_set() or self.robot.fault:
                    raise RuntimeError("execution stopped before native probe write")
                if self.clock.monotonic() - started > 1 / self.policy.control_hz:
                    raise SendWindowExpired("native probe control deadline exceeded")
                self.trace.record("command", method=method, arguments=list(arguments), generation=generation)
                self.cycle_writes += 1
                self.robot.actuator_commands_sent = True
                if method == 'GripperCtrl' and not state['gripper']['enabled']:
                    self.gripper_enable_request_at = self.clock.monotonic()
                getattr(self.robot.arms[self.arm], method)(*arguments)
                if method == 'GripperCtrl':
                    self.gripper_commanded = True
                    self.gripper_target_m = arguments[0]/1e6
            if self.clock.monotonic() - started > 1 / self.policy.control_hz:
                raise SendWindowExpired("native probe SDK send deadline exceeded")
            self.integrator.commit(proposal)
            self.accepted_keys = keys
            self.paused, self.phase, self.last_confirmation = False, "moving", None
            self.deadline_misses, self.waiting_since = 0, None
            self.target_stamps.append(self.clock.monotonic())
            self.trace.record(
                "target",
                generation=generation,
                keys=sorted(keys),
                pose_sdk=list(target),
                tcp_target=proposal["tcp_target"].tolist(),
                state=before,
                width_m=proposal["width_m"],
                ik={"solver": "controller"},
                geometry_scope="none",
                automatic_orientation=not self.native_pose_probe,
                orientation_adjustment_deg=getattr(self, 'native_adjustment_deg', 0.),
            )
            mark("native_SDK_send")
