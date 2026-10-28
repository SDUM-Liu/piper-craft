"""Real SDK worker. Startup connects listeners only; never disables/reset arms."""
import fcntl
import math
import signal
import sys
import threading
import time
import multiprocessing
import uuid
from concurrent.futures import ProcessPoolExecutor

from piper_sdk import C_PiperInterface_V2
from astra_vla.config import RUNTIME, atomic_json, settings
from astra_vla.rpc import serve
from astra_vla.kinematics import Kinematics
from astra_vla.workers.feedback import CoherentArmFeedback


def initialize_geometry(config):
    global geometry_model
    geometry_model = Kinematics(config)
    from astra_vla.workers.teleop import initialize_teleop_geometry
    initialize_teleop_geometry(geometry_model)


GRIPPER_POSITION_TOLERANCE_M = .002


class TargetRejected(RuntimeError):
    """The controller rejected a pose; it is not an emergency-stop request."""


def enum(value):
    return int(getattr(value, "value", value))


def rotation_distance_deg(actual, target):
    def matrix(rpy):
        r, p, y = map(math.radians, rpy)
        sr, cr, sp, cp, sy, cy = math.sin(r), math.cos(r), math.sin(p), math.cos(p), math.sin(y), math.cos(y)
        return (cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, -sp, cp*sr, cp*cr)
    cosine = (sum(a*b for a, b in zip(matrix(actual), matrix(target))) - 1) / 2
    return math.degrees(math.acos(max(-1, min(1, cosine))))


def clock_stamp():
    """Wall time aligns with CAN logs; monotonic time measures local delays."""
    return {"wall_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


class Robot:
    def __init__(self):
        self.config = settings()
        self.lock = threading.Lock()
        # Action RPCs hold self.lock, but stop must bypass it. Serialize only
        # SDK writes so an in-flight write finishes before the stop packet.
        self.tx_lock = threading.Lock()
        self.stop_state_lock = threading.Lock()
        self.stop_epoch = 0
        self.stop_completed_epoch = 0
        self.stopped = threading.Event()
        self.stopped.set()
        self.stop_confirmation = None
        self.fault = 'startup observation: confirm scene before control'
        self.last_rejection = None
        self.arms = {}
        self.files = []
        self.coherent_feedback = {}
        self.kinematics = Kinematics(self.config)
        # Only quantized robot-local path rechecks run here, never IPOPT. Keep
        # separate mutable Pinocchio data from feedback and diagnostic RPCs.
        self.teleop_recheck_kinematics = Kinematics(self.config)
        # Mesh distance/IK calls can hold the GIL; never starve live CAN readers.
        self.geometry_pool = ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_geometry, initargs=(self.config,))
        # Long endpoint searches must never occupy the streaming geometry queue.
        self.detour_pool = ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_geometry, initargs=(self.config,))
        for name, port in self.config["arms"].items():
            file = open(RUNTIME / (port + ".lock"), "w")
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.files.append(file)
            arm = C_PiperInterface_V2(port)
            arm.ConnectPort(can_init=False, piper_init=False, start_thread=True)
            # SDK getters return objects mutated after their locks are released,
            # and each CAN fragment updates the shared array independently.
            # Publish only complete pose+joint cycles on this same SDK reader.
            collector = CoherentArmFeedback()
            bus = arm.GetCanBus()
            original = bus.callback_function
            def receive(frame, original=original, collector=collector):
                original(frame)
                collector.feed(frame)
            bus.callback_function = receive
            self.coherent_feedback[name] = collector
            arm.SearchPiperFirmwareVersion()  # Read-only query; no initialization or motor commands.
            self.arms[name] = arm
        from astra_vla.workers.teleop import TeleopController
        self.teleop = TeleopController(self)

    def state(self, name):
        p = self.arms[name]
        pose_message, joints_message = p.GetArmEndPoseMsgs(), p.GetArmJointMsgs()
        status_message, grip_message = p.GetArmStatus(), p.GetArmGripperMsgs()
        pose, joints, status, grip = (pose_message.end_pose, joints_message.joint_state,
                                     status_message.arm_status, grip_message.gripper_state)
        # Copy scalars: SDK feedback instances are continuously mutated in place.
        xyz = [int(getattr(pose, axis)) for axis in ("X_axis", "Y_axis", "Z_axis")]
        rpy = [int(getattr(pose, axis)) for axis in ("RX_axis", "RY_axis", "RZ_axis")]
        timestamp = float(getattr(pose_message, "time_stamp", 0))
        # SDK time_stamp is Unix seconds; reject missing/stale rather than invent freshness.
        stamps = [float(getattr(m, "time_stamp", 0)) for m in
                  (pose_message, joints_message, status_message, grip_message)]
        age = max(0, time.time() - min(stamps)) if all(stamps) else 1e9
        rates = {"pose": pose_message.Hz, "joint": joints_message.Hz,
                 "status": status_message.Hz, "gripper": grip_message.Hz}
        gripper_faults = [attr for attr in ("voltage_too_low", "motor_overheating", "driver_overcurrent",
                                           "driver_overheating", "sensor_status", "driver_error_status")
                          if getattr(grip.foc_status, attr, False)]
        enabled = [bool(v) for v in p.GetArmEnableStatus()]
        firmware = p.GetPiperFirmwareVersion()
        ee_pose = {"position_m": [v / 1e6 for v in xyz], "rpy_deg": [v / 1000 for v in rpy]}
        joint_values = [math.radians(getattr(joints, "joint_" + str(i)) / 1000) for i in range(1, 7)]
        collector = getattr(self, 'coherent_feedback', {}).get(name)
        coherent = collector.snapshot() if collector else None
        if coherent:
            ee_pose, joint_values = coherent['ee_pose'], coherent['joints_rad']
            timestamp = stamps[0] = coherent['pose_stamp']
            stamps[1] = coherent['joint_stamp']
            age = max(0, time.time() - min(stamps)) if all(stamps) else 1e9
        elif collector:
            age = 1e9  # No mixed-frame fallback before the first complete cycle.
        return {"arm": name, "stamp": timestamp, "age_s": age, "rates_hz": rates,
                "feedback_coherent": bool(coherent),
                "feedback_cycle_span_s": coherent['span_s'] if coherent else None,
                "feedback_stamps": dict(zip(("pose", "joint", "status", "gripper"), stamps)),
                "ee_pose": ee_pose,
                "joints_rad": joint_values,
                "gripper": {"width_m": grip.grippers_angle / 1e6,
                            "effort": grip.grippers_effort / 1000,
                            "enabled": bool(grip.foc_status.driver_enable_status),
                            "faults": gripper_faults},
                "controller": {"ctrl_mode": enum(status.ctrl_mode),
                               "ctrl_mode_label": str(status.ctrl_mode),
                               "move_mode": enum(status.mode_feed),
                               "move_mode_label": str(status.mode_feed),
                               "firmware": firmware if isinstance(firmware, str) else None,
                               "arm_status": enum(status.arm_status),
                               "arm_status_label": str(status.arm_status),
                               "motion_status": enum(status.motion_status),
                               "motion_status_label": str(status.motion_status),
                               "err_code": int(status.err_code), "motors_enabled": enabled}}

    def status(self):
        arms = {name: self.state(name) for name in self.arms}
        teleop = getattr(self, 'teleop', None)
        if teleop and teleop.session:
            from astra_vla.teleop import pose
            for name, state in arms.items():
                if name in teleop.tool['arms']:
                    import numpy as np
                    tcp = pose(state) @ np.asarray(teleop.tool['arms'][name]['T_ee_tcp'])
                    state['tcp_matrix'] = tcp.tolist()
        native = bool(teleop and teleop.active and getattr(teleop, 'native_control', False))
        return {"arms": arms,
                "fault": self.fault, "stopped": self.stopped.is_set(), "busy": self.lock.locked(),
                "stop_requested": self.stopped.is_set(),
                "stop_confirmed": bool(getattr(self, "stop_confirmation", None) and
                                       self.stop_confirmation["confirmed"]),
                "stop_confirmation": getattr(self, "stop_confirmation", None),
                "teach_exit": getattr(self, "teach_exit", None),
                "last_rejection": getattr(self, "last_rejection", None),
                "motion_control": {"controller": "controller_move_p" if native else "regularized_move_j",
                                   "ik_solver": "controller" if native else "casadi_ipopt",
                                   "motion_mode": "MOVE_P" if native else "MOVE_J", "pose_source": "controller_sdk",
                                   "joint_position_limits": "controller",
                                   "execution_backends": ["regularized_move_j"],
                                   "native_control_enabled": False,
                                   "coordinate_modes": ["base", "motion"]},
                "teleop": teleop.view() if teleop else {"active": False}}

    def command(self, p, method, *args):
        # Check stop/fault at each write after acquiring the TX lock. Stop
        # packets use the same lock, so later targets cannot overtake them.
        with self.tx_lock:
            if self.stopped.is_set() or self.fault:
                raise RuntimeError("execution stopped before actuator command")
            return getattr(p, method)(*args)

    def healthy(self, name, require_enabled=True, gripper_only=False, allow_target_rejection=False):
        state = self.state(name)
        if state["age_s"] > self.config["limits"]["state_age_s"] or any(
            h <= 0 for h in state["rates_hz"].values()
        ):
            raise RuntimeError("Piper feedback missing/stale")
        control = state["controller"]
        # Explicit operator gripper maintenance may read a rejected arm target. This
        # exception never applies to prepare/move or model gripper commands.
        allowed_status = (0, 4) if (gripper_only or allow_target_rejection) and control["ctrl_mode"] in (0, 1) else (0,)
        if (control["arm_status"] == 4 and 4 not in allowed_status and
                not control["err_code"] and not state["gripper"]["faults"]):
            raise TargetRejected("TARGET_POS_EXCEEDS_LIMIT: controller rejected the end-pose target; "
                               "fresh path verification required before a smaller target can execute")
        if control["err_code"] or control["arm_status"] not in allowed_status or state["gripper"]["faults"]:
            raise RuntimeError("controller/gripper reports fault")
        if require_enabled and (control["ctrl_mode"] != 1 or not all(control["motors_enabled"])):
            raise RuntimeError("arm not in enabled CAN control; use explicit device preparation")
        if self.fault or self.stopped.is_set():
            raise RuntimeError("execution stopped; fresh verification and acknowledgement required")
        return state

    def stationary_feedback(self, require_fault_free=False, reference_states=None, poll_s=.05,
                            joint_tolerance_deg=.1, timeout_s=1.5, feedback_check=None):
        """Observe a new quiet window; a sent pause is never proof of stopping."""
        if not .005 <= poll_s <= .05:
            raise ValueError('stationary poll interval must be 5..50 ms')
        if isinstance(joint_tolerance_deg, bool) or not 0 < joint_tolerance_deg <= .15:
            raise ValueError('stationary joint tolerance must be within 0..0.15 degrees')
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or not .5 <= timeout_s <= 5:
            raise ValueError('stationary timeout must be within 0.5..5 seconds')
        started, baseline, previous, samples = time.monotonic(), None, {}, 0
        stable_since, reason = None, "no fresh stationary feedback window"
        controller_feedback = {}
        teleop = getattr(self, 'teleop', None)
        require_gripper = bool(teleop and teleop.owns_lock)
        while time.monotonic() - started < timeout_s:
            # Recovery keeps checking actual geometry, cancellation and RGB
            # throughout a longer settling window. These errors are fatal,
            # unlike movement that simply restarts the quiet-window timer.
            if feedback_check is not None:
                feedback_check()
            current = {}
            try:
                for name in self.arms:
                    state = self.state(name)
                    stamps = state.get("feedback_stamps", {})
                    pose_stamp, joint_stamp = stamps.get("pose", 0), stamps.get("joint", 0)
                    values = (*state["joints_rad"], *state["ee_pose"]["position_m"],
                              *state["ee_pose"]["rpy_deg"])
                    control = state["controller"]
                    # motion_status is target arrival, not a velocity estimate.
                    # Enabled feedback also does not measure holding torque.
                    controller_feedback[name] = {"stamp": state["stamp"], "age_s": state["age_s"],
                        "motion_status": control["motion_status"],
                        "motors_enabled": list(control["motors_enabled"])}
                    if (state["age_s"] > min(.3, self.config["limits"]["state_age_s"]) or
                            not all(math.isfinite(v) for v in values) or
                            pose_stamp <= previous.get(name, (0, 0))[0] or
                            joint_stamp <= previous.get(name, (0, 0))[1] or
                            any(h <= 0 for h in state["rates_hz"].values())):
                        raise RuntimeError("feedback missing, stale, or not advancing")
                    grip_stamp = stamps.get('gripper', 0)
                    if require_gripper and (not math.isfinite(state['gripper']['width_m']) or
                            grip_stamp <= previous.get(name, (0, 0, 0))[2]):
                        raise RuntimeError('gripper feedback missing, stale, or not advancing')
                    previous[name] = (pose_stamp, joint_stamp, grip_stamp)
                    if require_fault_free and (control["err_code"] or control["arm_status"] != 0 or
                                               state["gripper"]["faults"]):
                        raise RuntimeError("fresh fault-free feedback required; no automatic reset")
                    current[name] = state
                    if reference_states and name in reference_states:
                        reference = reference_states[name]
                        if (max(abs(a-b) for a, b in zip(state['joints_rad'], reference['joints_rad'])) >
                                math.radians(.1) or
                                math.dist(state['ee_pose']['position_m'], reference['ee_pose']['position_m']) > .0005 or
                                rotation_distance_deg(state['ee_pose']['rpy_deg'], reference['ee_pose']['rpy_deg']) > .2 or
                                abs(state['gripper']['width_m'] - reference['gripper']['width_m']) > .0005):
                            return {'confirmed': False, 'stamp': time.time(), 'samples': samples,
                                    'stable_window_s': 0, 'controller_feedback': controller_feedback,
                                    'holding_torque_verified': False,
                                    'reason': name + ' moved from fixed preparation reference'}
                if not current:
                    raise RuntimeError("no arm feedback")
                if baseline is not None:
                    for name, state in current.items():
                        reference = baseline[name]
                        if require_gripper and abs(state['gripper']['width_m'] -
                                                   reference['gripper']['width_m']) > .0005:
                            raise RuntimeError('gripper feedback still moving')
                        if (max(abs(a-b) for a, b in zip(state["joints_rad"], reference["joints_rad"])) >
                                math.radians(joint_tolerance_deg) or
                                math.dist(state["ee_pose"]["position_m"], reference["ee_pose"]["position_m"]) > .0005 or
                                rotation_distance_deg(state["ee_pose"]["rpy_deg"], reference["ee_pose"]["rpy_deg"]) > .2):
                            raise RuntimeError("joint/end-effector feedback still moving")
                if baseline is None:
                    baseline, stable_since, samples = current, time.monotonic(), 0
                samples += 1
                if samples >= 10 and time.monotonic() - stable_since >= .5:
                    return {"confirmed": True, "stamp": time.time(), "samples": samples,
                            "stable_window_s": time.monotonic() - stable_since,
                            "controller_feedback": controller_feedback, "holding_torque_verified": False,
                            "joint_tolerance_deg": joint_tolerance_deg,
                            "reason": "fresh joint and end-effector feedback stationary within tolerances"}
            except Exception as exc:
                baseline, stable_since, samples = None, None, 0
                reason = str(exc)
            time.sleep(poll_s)
        return {"confirmed": False, "stamp": time.time(), "samples": samples,
                "stable_window_s": 0 if stable_since is None else time.monotonic() - stable_since,
                "controller_feedback": controller_feedback, "holding_torque_verified": False,
                "reason": reason}

    def stop(self, reason="operator stop", *, manual_tcp=False):
        if not isinstance(manual_tcp, bool):
            raise ValueError('manual TCP flag must be boolean')
        timeline = {"request": clock_stamp(), "arms": {}}
        # Latch before waiting for an in-flight SDK send. Never take the action
        # lock here: the action thread may itself be waiting for this stop.
        with self.stop_state_lock:
            self.stop_epoch += 1
            epoch = self.stop_epoch
            self.stopped.set()
            self.fault = reason
            self.stop_confirmation = None
            timeline["latched"] = clock_stamp()
        timeline["stop_epoch"] = epoch
        teleop = getattr(self, 'teleop', None)
        if teleop and teleop.owns_lock:
            teleop.invalidate(reason)
        failures = []
        timeline["tx_lock_wait_start"] = clock_stamp()
        with self.tx_lock:
            timeline["tx_lock_acquired"] = clock_stamp()
            if teleop and teleop.owns_lock:
                try:
                    teleop.freeze_gripper()
                except Exception as exc:
                    failures.append('teleop gripper hold: ' + str(exc))
            for name, p in self.arms.items():
                # Request planner pause. Actual stopping/holding is not guaranteed by
                # delivery of this packet and must be assessed from fresh feedback.
                # emergency_stop=1 is an emergency stop and changed these arms to
                # STANDBY with gravity droop during the recorded 2026-09-29 trial.
                attempt = {"sdk_enter": clock_stamp()}
                timeline["arms"][name] = attempt
                try:
                    p.MotionCtrl_1(0x00, 0x01, 0x00)
                except Exception as exc:
                    failures.append(name + ": " + str(exc))
                    attempt["sdk_exception"] = str(exc)
                else:
                    # This SDK returns None even if its CAN send failed. A
                    # normal return is not evidence of bus or controller receipt.
                    attempt["sdk_return"] = clock_stamp()
                    attempt["send_status"] = "unverified"
                finally:
                    attempt["sdk_exit"] = clock_stamp()
        timeline["tx_lock_released"] = clock_stamp()
        timeline["stationary_start"] = clock_stamp()
        confirmation = (self.stationary_feedback(joint_tolerance_deg=.15) if manual_tcp
                        else self.stationary_feedback())
        timeline["stationary_end"] = clock_stamp()
        confirmation["timeline"] = timeline
        confirmation["command_failures"] = failures
        if failures:
            confirmation["confirmed"] = False
        with self.stop_state_lock:
            # A later stop request owns the latch and its confirmation. An
            # older observer must never overwrite the newer result.
            if epoch == self.stop_epoch:
                self.stop_confirmation = confirmation
                self.stop_completed_epoch = epoch
        if teleop and teleop.owns_lock:
            teleop.complete_stop(confirmation)
        if failures:
            raise RuntimeError("stop command failed: " + "; ".join(failures))
        return self.status()


    def check_prepare(self, name):
        return self.check_prepare_mode(name)

    def controller_diagnostics(self, args):
        """Query controller-stored limits through this CAN owner, without motion.

        Read every axis separately: the SDK's all-axis cache has a single
        timestamp, so one fresh response cannot prove that all six are fresh.
        This never clears a fault or writes controller configuration.
        """
        arm = args.get('arm', 'right')
        if arm not in self.arms:
            raise ValueError('select left/right arm')
        if not self.lock.acquire(False):
            raise RuntimeError('controller diagnostics require idle stopped control')
        try:
            if not self.stopped.is_set() or not (self.stop_confirmation or {}).get('confirmed'):
                raise RuntimeError('controller diagnostics require confirmed stop')
            return self.read_controller_limits(arm)
        finally:
            self.lock.release()

    def read_controller_limits(self, arm):
        """Called under the action lock at entry, or by stopped diagnostics."""
        device, limits = self.arms[arm], []
        for index in range(1, 7):
            requested = time.time()
            with self.tx_lock:
                device.SearchMotorMaxAngleSpdAccLimit(index, 0x01)
            deadline = time.monotonic() + .5
            result = None
            while time.monotonic() < deadline:
                message = device.GetCurrentMotorAngleLimitMaxVel()
                stamp = float(message.time_stamp)
                value = message.current_motor_angle_limit_max_vel
                values = (value.motor_num, value.min_angle_limit, value.max_angle_limit, value.max_joint_spd)
                if (values[0] == index and requested <= stamp <= time.time() and
                        message.time_stamp == stamp):
                    result = {'joint': index, 'stamp': stamp,
                              'min_deg': values[1] / 10., 'max_deg': values[2] / 10.,
                              'max_speed_rad_s': values[3] / 1000.}
                    break
                time.sleep(.005)
            limits.append(result or {'joint': index, 'available': False})
        return {'arm': arm, 'source': 'controller_query_0x472_reply_0x473',
                'complete': all('stamp' in item for item in limits), 'limits': limits,
                'state': self.state(arm), 'motion_commands_sent': False}

    def check_teleop_prepare(self, name, follow_mode, *, arm_only=False):
        if follow_mode != 'position_velocity':
            raise ValueError('PIPER CRAFT supports ordinary MOVE_J only')
        return self.check_prepare_mode(name, arm_only=arm_only)

    def check_prepare_mode(self, name, *, arm_only=False):
        if self.fault or self.stopped.is_set():
            raise RuntimeError("latched stop requires explicit acknowledge")
        state = self.healthy(name, False)
        mode = state["controller"]["ctrl_mode"]
        if mode not in (0, 1):
            raise RuntimeError("teach/MIT/master-slave state requires manual mode handoff")
        if state['controller']['move_mode'] in (4, 5):
            raise RuntimeError('direct MIT/CPV controller mode requires manual recovery')
        p = self.arms[name]
        mode_message = p.GetArmModeCtrl()
        if getattr(mode_message, 'Hz', 0) > 0 and enum(mode_message.mode_ctrl.mit_mode) == 0xAD:
            raise RuntimeError('MIT/high-follow controller mode requires manual recovery')
        maximum = self.config["limits"].get("maintenance_gripper_max_m", {}).get(name, .065)
        if not -GRIPPER_POSITION_TOLERANCE_M <= state["gripper"]["width_m"] <= maximum:
            # Desktop arm-only control can bound the complete physical jaw
            # stroke without claiming this encoder has a verified zero.
            width = state['gripper']['width_m']
            if (arm_only and math.isfinite(width) and -maximum <= width < -.002 and
                    not state['gripper']['enabled']):
                return state
            raise RuntimeError("gripper encoder outside -2..65mm; physical zero/opening verification required")
        return state


    def exit_teach_mode(self, epoch):
        """Explicit scene acknowledgement only: end teaching, never replay/reset/enable.

        Called with the action lock held. Only the documented 0x150 end-record
        byte is sent; firmware that keeps mode=2 must be recovered separately.
        """
        reference = {name: self.state(name) for name in self.arms}
        affected = [name for name, state in reference.items() if state['controller']['ctrl_mode'] == 2]
        if not affected:
            return
        report = {'state': 'exiting', 'arms': affected, 'attempts': dict.fromkeys(affected, 0),
                  'before': reference, 'after': {}, 'error': '', 'command': 'MotionCtrl_1(0, 0, 2)',
                  'reset_sent': False, 'motor_enable_sent': False, 'motion_target_sent': False}
        path = RUNTIME.parent/'runs'/('teach_exit_'+uuid.uuid4().hex[:12])/'result.json'
        report['evidence'] = str(path)
        self.teach_exit = report
        camera_identity = None

        def guarded_feedback():
            nonlocal camera_identity
            with self.stop_state_lock:
                if epoch != self.stop_epoch or self.stop_completed_epoch != epoch:
                    raise RuntimeError('现场确认期间收到停止请求；已取消退出示教')
            cameras = self.teleop.camera_read()['cameras']
            if set(cameras) != {'top', 'left_wrist', 'right_wrist'}:
                raise RuntimeError('退出示教需要三路新鲜 RGB')
            for name, frame in cameras.items():
                stamp, age = frame.get('stamp', 0), frame.get('age_s', float('inf'))
                if (not frame.get('available') or not math.isfinite(stamp) or not math.isfinite(age) or
                        not 0 <= time.time()-stamp <= .6 or not 0 <= age <= .6):
                    raise RuntimeError(name+' RGB missing/stale during teach exit')
            identity = {name: tuple(frame.get(k) for k in ('serial', 'width', 'height'))
                        for name, frame in cameras.items()}
            if camera_identity is not None and identity != camera_identity:
                raise RuntimeError('退出示教期间相机配置变化')
            camera_identity = identity
            states = {name: self.state(name) for name in self.arms}
            for name, state in states.items():
                self.check_motion_state(state)
                control = state['controller']
                status_stamp = state.get('feedback_stamps', {}).get('status', 0)
                if (not state.get('feedback_coherent', True) or
                        not math.isfinite(status_stamp) or not 0 <= time.time()-status_stamp <= .3 or
                        control['err_code'] or control['arm_status'] != 0 or state['gripper']['faults']):
                    raise RuntimeError(name+' 示教退出需要新鲜且无故障的控制器反馈')
                if control['ctrl_mode'] not in (0, 1, 2) or control['move_mode'] in (4, 5):
                    raise RuntimeError(name+' 处于其他控制模式；现场确认不能自动接管')
                message = self.arms[name].GetArmModeCtrl()
                if getattr(message, 'Hz', 0) > 0 and enum(message.mode_ctrl.mit_mode) == 0xAD:
                    raise RuntimeError(name+' 处于 MIT 高跟随模式；现场确认不能自动接管')
                old = reference[name]
                if (max(abs(a-b) for a, b in zip(state['joints_rad'], old['joints_rad'])) > math.radians(.15) or
                        math.dist(state['ee_pose']['position_m'], old['ee_pose']['position_m']) > .0005 or
                        rotation_distance_deg(state['ee_pose']['rpy_deg'], old['ee_pose']['rpy_deg']) > .2 or
                        abs(state['gripper']['width_m']-old['gripper']['width_m']) > .0005 or
                        state['gripper']['enabled'] != old['gripper']['enabled'] or
                        control['motors_enabled'] != old['controller']['motors_enabled']):
                    raise RuntimeError(name+' 退出示教期间姿态或使能状态变化；请重新观察')
                if name not in affected and control['ctrl_mode'] != old['controller']['ctrl_mode']:
                    raise RuntimeError(name+' 控制模式变化；已取消退出示教')
            report['after'] = states
            return states

        with self.stop_state_lock:
            if epoch != self.stop_epoch or self.stop_completed_epoch != epoch:
                raise RuntimeError('现场确认期间收到停止请求；已取消退出示教')
            self.stopped.set()
            self.fault = '现场确认：正在退出示教模式'
        try:
            # Evidence creation and all-arm checks must succeed before the first SDK write.
            atomic_json(path, report)
            deadline = time.monotonic()+2.
            stable_since, previous_stamps, next_send = None, {}, 0.
            while time.monotonic() < deadline:
                states = guarded_feedback()
                stamps = {name: state['feedback_stamps']['status'] for name, state in states.items()}
                advancing = all(stamp > previous_stamps.get(name, 0) for name, stamp in stamps.items())
                previous_stamps = stamps
                exited = all(states[name]['controller']['ctrl_mode'] in (0, 1) for name in affected)
                if exited and advancing:
                    stable_since = time.monotonic() if stable_since is None else stable_since
                    if time.monotonic()-stable_since >= .2:
                        confirmation = self.stationary_feedback(
                            require_fault_free=True, reference_states=reference, joint_tolerance_deg=.15,
                            feedback_check=guarded_feedback)
                        if not confirmation['confirmed']:
                            raise RuntimeError('退出示教后尚未确认静止：'+confirmation['reason'])
                        final_states = guarded_feedback()
                        if any(final_states[name]['controller']['ctrl_mode'] not in (0, 1) for name in affected):
                            raise RuntimeError('退出示教后模式再次变化；请重新观察')
                        with self.stop_state_lock:
                            if epoch != self.stop_epoch:
                                raise RuntimeError('现场确认期间收到停止请求；已取消退出示教')
                            self.stop_confirmation = confirmation
                        report['stop_confirmation'] = confirmation
                        report['state'] = 'exited'
                        return
                else:
                    stable_since = None
                if not exited and time.monotonic() >= next_send:
                    for name in affected:
                        # Refresh both arms and RGB before each write, then serialize with stop.
                        states = guarded_feedback()
                        if time.monotonic() >= deadline:
                            break
                        if states[name]['controller']['ctrl_mode'] != 2:
                            continue
                        with self.tx_lock:
                            with self.stop_state_lock:
                                if epoch != self.stop_epoch or self.stop_completed_epoch != epoch:
                                    raise RuntimeError('现场确认期间收到停止请求；已取消退出示教')
                            report['attempts'][name] += 1
                            self.arms[name].MotionCtrl_1(0x00, 0x00, 0x02)
                    next_send = time.monotonic()+.2
                time.sleep(.05)
            raise RuntimeError('退出示教未确认：控制器仍为示教模式或模式反馈未稳定更新；'
                               '本次未复位，请按有可靠支撑的人工恢复流程处理')
        except Exception as exc:
            report['state'], report['error'] = 'failed', str(exc)
            with self.stop_state_lock:
                owns_epoch = epoch == self.stop_epoch
                if owns_epoch:
                    self.fault = str(exc)
                    self.stopped.set()
            if owns_epoch and any(report['attempts'].values()):
                # A failed mode transition must retain the ordinary pause/stop
                # guard, even though this path never submits a motion target.
                try:
                    stopped = self.stop('现场确认退出示教失败：'+str(exc), manual_tcp=True)
                    report['stop_confirmation'] = stopped.get('stop_confirmation')
                except Exception as stop_error:
                    report['stop_error'] = str(stop_error)
            raise
        finally:
            atomic_json(path, report)

    def acknowledge(self, *, manual_tcp=False, exit_teach=False):
        if not isinstance(manual_tcp, bool):
            raise ValueError('manual TCP flag must be boolean')
        if not isinstance(exit_teach, bool):
            raise ValueError('exit teach flag must be boolean')
        if not self.lock.acquire(False):
            raise RuntimeError("previous command still running")
        try:
            with self.stop_state_lock:
                epoch = self.stop_epoch
                if self.stop_completed_epoch != epoch:
                    raise RuntimeError("stop request still being sent or observed")
            def confirm_stationary():
                confirmation = (self.stationary_feedback(require_fault_free=True, joint_tolerance_deg=.15)
                                if manual_tcp else self.stationary_feedback(require_fault_free=True))
                with self.stop_state_lock:
                    if epoch != self.stop_epoch or self.stop_completed_epoch != epoch:
                        raise RuntimeError("new stop request or arm action during acknowledgement")
                    self.stop_confirmation = confirmation
                    if not confirmation['confirmed']:
                        raise RuntimeError('fresh sustained stationary fault-free feedback required; stop unconfirmed: '
                                           +confirmation['reason'])
                return confirmation
            confirm_stationary()
            if exit_teach:
                self.exit_teach_mode(epoch)
            with self.stop_state_lock:
                if epoch != self.stop_epoch or self.stop_completed_epoch != epoch:
                    raise RuntimeError("new stop request or arm action during acknowledgement")
                # Never resumes a trajectory or clears physical/controller faults.
                self.fault = None
                self.stopped.clear()
            acknowledged = getattr(getattr(self, 'teleop', None), 'scene_acknowledged', None)
            if acknowledged:
                acknowledged()
        finally:
            self.lock.release()
        return self.status()


    def check_motion_state(self, state):
        """Fail closed on malformed feedback, including NaNs in drift comparisons."""
        values = (state["joints_rad"], state["ee_pose"]["position_m"], state["ee_pose"]["rpy_deg"])
        if (any(len(row) != size for row, size in zip(values, (6, 3, 3))) or
                not all(math.isfinite(v) for row in values for v in row) or
                not math.isfinite(state["gripper"]["width_m"])):
            raise RuntimeError("invalid joint/pose/gripper feedback during motion validation")
        age = state["age_s"]
        if (not math.isfinite(age) or age < 0 or age > self.config["limits"]["state_age_s"] or
                not state["rates_hz"] or any(not math.isfinite(h) or h <= 0 for h in state["rates_hz"].values())):
            raise RuntimeError("Piper feedback missing/stale during motion validation")

    def teleop_feedback(self, state):
        return self.kinematics.feedback_pose(state)

    def recheck_teleop_target(self, *args):
        if args[3].get('scope') != 'tcp_jog':
            raise ValueError('inline recheck is restricted to robot-local teleoperation')
        return self.teleop_recheck_kinematics.teleop_step(*args[:-1], validated_goal=args[-1])

    def recheck_detour_target(self, *args):
        from astra_vla.teleop_detour import detour_joint_step
        return detour_joint_step(self.teleop_recheck_kinematics, *args[:-1], validated=args[-1])

    def check_gravity_recovery_target(self, *args):
        from astra_vla.gravity_recovery import check_recovery_target
        return check_recovery_target(self.teleop_recheck_kinematics, *args)

    def check_motion_unchanged(self, arm, previous, latest, stage):
        self.check_motion_state(latest)
        if (max(abs(a-b) for a, b in zip(latest["joints_rad"], previous["joints_rad"])) > math.radians(.5) or
                math.dist(latest["ee_pose"]["position_m"], previous["ee_pose"]["position_m"]) > .003 or
                rotation_distance_deg(latest["ee_pose"]["rpy_deg"], previous["ee_pose"]["rpy_deg"]) > 3 or
                abs(latest["gripper"]["width_m"] - previous["gripper"]["width_m"]) > .002 or
                latest["gripper"]["enabled"] != previous["gripper"]["enabled"] or
                any(latest["controller"][key] != previous["controller"][key] for key in
                    ("ctrl_mode", "move_mode", "arm_status", "err_code", "motors_enabled"))):
            raise RuntimeError(arm + " arm/gripper changed " + stage + "; observe again")


    def dispatch(self, method, args):
        if method == 'status':
            return self.status()
        if method == 'controller_diagnostics':
            return self.controller_diagnostics(args)
        if method == 'stop':
            return self.stop(args.get('reason', 'operator stop'), manual_tcp=True)
        if method == 'acknowledge':
            return self.acknowledge(manual_tcp=True, exit_teach=True)
        if method == 'preflight':
            return {name: self.check_prepare(name) for name in args.get('arms', list(self.arms))}
        methods = {'teleop_open': self.teleop.open, 'teleop_input': self.teleop.input,
                   'teleop_close': self.teleop.close, 'teleop_sleep': self.teleop.return_sleep}
        if method not in methods:
            raise ValueError('unknown robot method; PIPER CRAFT teleop only')
        return methods[method](args)

    def close(self):
        if self.lock.locked():
            self.stop('robot worker shutdown during control', manual_tcp=True)
        for arm in self.arms.values():
            arm.DisconnectPort()
        self.geometry_pool.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, 'detour_pool'):
            self.detour_pool.shutdown(wait=False, cancel_futures=True)
        if getattr(self, 'teleop', None):
            self.teleop.shutdown_camera()


def main():
    # CAN readers, stop/input RPCs and the 20 Hz loop share Python's GIL.
    # A 5 ms default timeslice caused multi-slice latency on short read checks.
    sys.setswitchinterval(.001)
    RUNTIME.mkdir(mode=0o700, exist_ok=True)
    robot = Robot()
    def shutdown(*_):
        if robot.close() is False:
            return
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        serve(RUNTIME / "robot.sock", robot.dispatch)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
