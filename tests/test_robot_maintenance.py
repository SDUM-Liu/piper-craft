import copy
import importlib.util
import math
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import numpy as np

from astra_vla.config import ROOT


@pytest.fixture
def worker(monkeypatch):
    # Isolated SDK command spies; importing tests never connects CAN or loads meshes.
    sdk = ModuleType("piper_sdk")
    sdk.C_PiperInterface_V2 = object
    kine = ModuleType("astra_vla.kinematics")
    kine.Kinematics = object
    monkeypatch.setitem(sys.modules, "piper_sdk", sdk)
    monkeypatch.setitem(sys.modules, "astra_vla.kinematics", kine)
    spec = importlib.util.spec_from_file_location("isolated_robot_worker", ROOT / "astra_vla/workers/robot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = [100.0]
    nanosecond_tick = [0]
    def sleep(seconds):
        clock[0] += seconds
    def nanoseconds():
        nanosecond_tick[0] += 1
        return round(clock[0] * 1_000_000_000) + nanosecond_tick[0]
    module.time = SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0], sleep=sleep,
                                  time_ns=nanoseconds, monotonic_ns=nanoseconds)
    robot = module.Robot.__new__(module.Robot)
    state = {"ee_pose": {"position_m": [.1, 0, .2], "rpy_deg": [0, 0, 0]},
             "joints_rad": [0.]*6, "age_s": .01, "rates_hz": {"pose": 200, "gripper": 200},
             "controller": {"ctrl_mode": 1, "move_mode": 0, "arm_status": 0, "err_code": 0,
                            "motors_enabled": [True]*6},
             "gripper": {"width_m": -.0037, "enabled": False, "faults": []}}
    def feedback(*_):
        return {**copy.deepcopy(state), "stamp": clock[0],
                "feedback_stamps": {key: clock[0] for key in ("pose", "joint", "status", "gripper")}}
    robot.state = feedback
    robot.healthy = lambda *args, **__: robot.state(args[0])
    robot.config = {"limits": {"state_age_s": .3, "collision_margin_m": .015, "table_clearance_m": .003}}
    robot.fault = None
    robot.stopped = threading.Event()
    robot.tx_lock = threading.Lock()
    robot.stop_state_lock = threading.Lock()
    robot.stop_epoch = 0
    robot.stop_completed_epoch = 0
    robot.kinematics = SimpleNamespace(state_pose_matrix=state_pose_matrix, check_hold=lambda _: None)
    install_geometry(robot)
    state["controller"]["motion_status"] = 0
    return robot, state


def test_state_uses_complete_cycle_not_sdk_mutable_arrays(worker):
    from astra_vla.workers.feedback import CoherentArmFeedback
    import struct
    robot, state = worker
    pose = SimpleNamespace(**{k: 999999 for k in ('X_axis','Y_axis','Z_axis','RX_axis','RY_axis','RZ_axis')})
    joints = SimpleNamespace(**{'joint_'+str(i): 999999 for i in range(1,7)})
    foc = SimpleNamespace(driver_enable_status=False)
    grip = SimpleNamespace(grippers_angle=0,grippers_effort=0,foc_status=foc)
    status = SimpleNamespace(ctrl_mode=0,mode_feed=0,arm_status=0,motion_status=0,err_code=0)
    def message(**items):
        return SimpleNamespace(time_stamp=100.,Hz=200.,**items)
    robot.arms = {'left':SimpleNamespace(GetArmEndPoseMsgs=lambda:message(end_pose=pose),
        GetArmJointMsgs=lambda:message(joint_state=joints),GetArmStatus=lambda:message(arm_status=status),
        GetArmGripperMsgs=lambda:message(gripper_state=grip),GetArmEnableStatus=lambda:[False]*6,
        GetPiperFirmwareVersion=lambda:'fake')}
    c=CoherentArmFeedback()
    robot.coherent_feedback={'left':c}
    def read():
        return type(robot).state(robot,'left')
    assert read()['age_s']>.3  # Never authorize the initial mixed getter values.
    for i in range(6):
        c.feed(SimpleNamespace(arbitration_id=0x2A2+i,timestamp=99.999+i*.0001,
                               data=struct.pack('>ii',1000,0)))
    actual=read()
    assert actual['feedback_coherent'] and actual['age_s']<.3
    assert actual['ee_pose']['position_m']==[.001,0.,.001]
    assert actual['joints_rad'][0]==pytest.approx(math.radians(1))
    assert actual['feedback_stamps']['pose']==99.999
    assert actual['feedback_cycle_span_s']==pytest.approx(.0005)


def state_pose_matrix(state):
    roll, pitch, yaw = map(math.radians, state["ee_pose"]["rpy_deg"])
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                            math.sin(pitch), math.cos(yaw), math.sin(yaw))
    transform = np.eye(4)
    transform[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                        [-sp, cp*sr, cp*cr]]
    transform[:3, 3] = state["ee_pose"]["position_m"]
    return transform


def install_geometry(robot, result=None, complete=None):
    """Record geometry inputs without running IK, meshes, a process pool, or CAN."""
    calls = []
    def submit(*args, **kwargs):
        calls.append((copy.deepcopy(args[1:]), copy.deepcopy(kwargs)))
        def finish(**_):
            if complete:
                complete()
            if isinstance(result, Exception):
                raise result
            return copy.deepcopy(result if result is not None else {
                "reachable": True, "start_joints_rad": args[1]["joints_rad"],
                "goal_joints_rad": args[1]["joints_rad"]})
        return SimpleNamespace(result=finish)
    robot.geometry_pool = SimpleNamespace(submit=submit)
    return calls


@pytest.mark.parametrize("method", ["status", "preflight", "acknowledge", "stop"])
def test_read_only_and_stop_rpc_send_only_expected_commands(worker, method):
    robot, state = worker
    commands = []
    robot.lock = threading.Lock()
    state["gripper"]["width_m"] = .02
    robot.arms = {"right": SimpleNamespace(GetArmModeCtrl=lambda: SimpleNamespace(Hz=0),
        MotionCtrl_1=lambda *args: commands.append(args))}
    calls = install_geometry(robot)
    robot.dispatch(method, {"arm": "right"})
    assert commands == ([(0, 1, 0)] if method == "stop" else [])
    assert calls == []


def test_stop_attempts_second_arm_when_first_send_fails(worker):
    robot, _ = worker
    stopped = []
    def broken(*_):
        stopped.append("left")
        raise ConnectionError("CAN send failed")
    robot.arms = {"left": SimpleNamespace(MotionCtrl_1=broken),
                  "right": SimpleNamespace(MotionCtrl_1=lambda *_: stopped.append("right"))}
    robot.stopped = threading.Event()
    with pytest.raises(RuntimeError, match="stop command failed"):
        robot.stop()
    assert stopped == ["left", "right"]
    assert robot.stopped.is_set()
    attempts = robot.stop_confirmation["timeline"]["arms"]
    assert attempts["left"]["sdk_exception"] == "CAN send failed"
    assert "sdk_return" not in attempts["left"]
    assert attempts["left"]["sdk_exit"]["monotonic_ns"] > attempts["left"]["sdk_enter"]["monotonic_ns"]
    assert attempts["right"]["send_status"] == "unverified"
    assert not robot.stop_confirmation["confirmed"]


def test_regular_stop_pauses_planner_without_emergency_stop_or_release(worker):
    robot, _ = worker
    robot.lock = threading.Lock()
    commands = []
    robot.arms = {arm: SimpleNamespace(MotionCtrl_1=lambda *args, a=arm: commands.append((a, args)))
                  for arm in ("left", "right")}
    result = robot.stop("operator stop")
    assert commands == [("left", (0, 1, 0)), ("right", (0, 1, 0))]
    assert result["stopped"] and result["stop_confirmed"]
    assert result["stop_confirmation"]["stable_window_s"] >= .5
    assert result["stop_confirmation"]["samples"] >= 10
    timeline = result["stop_confirmation"]["timeline"]
    stamps = [timeline[key] for key in ("request", "latched", "tx_lock_wait_start", "tx_lock_acquired")]
    for arm in ("left", "right"):
        attempt = timeline["arms"][arm]
        stamps.extend((attempt["sdk_enter"], attempt["sdk_return"], attempt["sdk_exit"]))
        assert attempt["send_status"] == "unverified"
    stamps.extend(timeline[key] for key in ("tx_lock_released", "stationary_start", "stationary_end"))
    assert timeline["stop_epoch"] == 1
    assert all(set(stamp) == {"wall_ns", "monotonic_ns"} for stamp in stamps)
    assert all(stamps[i]["monotonic_ns"] < stamps[i+1]["monotonic_ns"] for i in range(len(stamps)-1))
    assert all(stamps[i]["wall_ns"] < stamps[i+1]["wall_ns"] for i in range(len(stamps)-1))


def test_stop_packet_follows_inflight_sdk_write_and_blocks_followup_target(worker):
    robot, _ = worker
    robot.lock = threading.Lock()
    entered, release = threading.Event(), threading.Event()
    packets, outcomes = [], []
    def pose(*_):
        entered.set()
        assert release.wait(2)
        packets.append("pose")
    arm = SimpleNamespace(EndPoseCtrl=pose,
        MotionCtrl_2=lambda *_: packets.append("mode"),
        MotionCtrl_1=lambda *args: packets.append(("stop", args)))
    robot.arms = {"right": arm}
    def action():
        try:
            robot.command(arm, "EndPoseCtrl", 1, 2, 3, 4, 5, 6)
            robot.command(arm, "MotionCtrl_2", 1, 0, 5, 0)
        except Exception as exc:
            outcomes.append(str(exc))
    action_thread = threading.Thread(target=action)
    stop_thread = threading.Thread(target=lambda: robot.stop("operator stop"))
    action_thread.start()
    assert entered.wait(1)
    stop_thread.start()
    assert robot.stopped.wait(1)
    release.set()
    action_thread.join(2)
    stop_thread.join(2)
    assert not action_thread.is_alive() and not stop_thread.is_alive()
    assert packets == ["pose", ("stop", (0, 1, 0))]
    assert outcomes == ["execution stopped before actuator command"]


def test_stop_latched_while_waiting_for_tx_lock_blocks_sdk_write(worker):
    robot, _ = worker
    robot.lock = threading.Lock()
    entered = threading.Event()
    packets, outcomes = [], []
    arm = SimpleNamespace(EndPoseCtrl=lambda *_: packets.append("pose"),
                          MotionCtrl_1=lambda *_: packets.append("stop"))
    robot.arms = {"right": arm}
    def action():
        entered.set()
        try:
            robot.command(arm, "EndPoseCtrl", 1, 2, 3, 4, 5, 6)
        except Exception as exc:
            outcomes.append(str(exc))
    action_thread = threading.Thread(target=action)
    stop_thread = threading.Thread(target=lambda: robot.stop("operator stop"))
    with robot.tx_lock:
        action_thread.start()
        assert entered.wait(1)
        stop_thread.start()
        assert robot.stopped.wait(1)
    action_thread.join(2)
    stop_thread.join(2)
    assert not action_thread.is_alive() and not stop_thread.is_alive()
    assert packets == ["stop"]
    assert outcomes == ["execution stopped before actuator command"]


def test_concurrent_stop_cannot_be_cleared_by_older_acknowledgement(worker):
    robot, _ = worker
    robot.lock = threading.Lock()
    robot.arms = {"right": SimpleNamespace(MotionCtrl_1=lambda *_: None)}
    robot.stopped.set()
    robot.fault = "old stop"
    entered, release = threading.Event(), threading.Event()
    outcomes = []
    def feedback(require_fault_free=False):
        if require_fault_free:
            entered.set()
            assert release.wait(2)
        return {"confirmed": True, "reason": "stationary"}
    robot.stationary_feedback = feedback
    def acknowledge():
        try:
            robot.acknowledge()
        except Exception as exc:
            outcomes.append(str(exc))
    thread = threading.Thread(target=acknowledge)
    thread.start()
    assert entered.wait(1)
    robot.stop("new stop")
    release.set()
    thread.join(2)
    assert not thread.is_alive()
    assert outcomes == ["new stop request or arm action during acknowledgement"]
    assert robot.stopped.is_set() and robot.fault == "new stop"
    assert robot.stop_confirmation["confirmed"]
    assert robot.stop_confirmation["reason"] == "stationary"
    assert robot.stop_confirmation["command_failures"] == []
    assert robot.stop_confirmation["timeline"]["stop_epoch"] == 1


@pytest.mark.parametrize("motion_status", [0, 1])
@pytest.mark.parametrize("enabled", [False, True])
def test_stationary_feedback_does_not_equate_target_arrival_or_enable_with_motion_or_torque(worker, motion_status, enabled):
    robot, state = worker
    robot.arms = {"right": SimpleNamespace()}
    state["controller"].update(motion_status=motion_status, motors_enabled=[enabled]*6)
    result = robot.stationary_feedback()
    assert result["confirmed"] and result["samples"] >= 10
    assert result["stable_window_s"] >= .5
    assert result["controller_feedback"]["right"]["motion_status"] == motion_status
    assert result["controller_feedback"]["right"]["motors_enabled"] == [enabled]*6
    assert not result["holding_torque_verified"]


@pytest.mark.parametrize('problem', ['moving', 'duplicate', 'nonfinite'])
def test_teleop_stop_confirmation_requires_stationary_advancing_gripper(worker, problem):
    robot, state = worker
    robot.arms = {'right': SimpleNamespace()}
    robot.teleop = SimpleNamespace(owns_lock=True)
    state['gripper']['width_m'] = .02
    read = robot.state
    reads = [0]
    def measured(arm):
        current = read(arm)
        reads[0] += 1
        if problem == 'moving':
            current['gripper']['width_m'] += reads[0] * .0006
        elif problem == 'duplicate':
            current['feedback_stamps']['gripper'] = 100.
        else:
            current['gripper']['width_m'] = float('nan')
        return current
    robot.state = measured
    result = robot.stationary_feedback()
    assert not result['confirmed']
    assert 'gripper' in result['reason']
    robot.state = read
    assert robot.stationary_feedback()['confirmed']


@pytest.mark.parametrize('movement', ['joint', 'position', 'orientation', 'gripper'])
def test_preparation_rejects_initial_drift_before_a_later_quiet_window(worker, movement):
    robot, state = worker
    robot.arms = {'right': SimpleNamespace()}
    reference = robot.state('right')
    read = robot.state
    reads = [0]
    def moved_then_quiet(arm):
        current = read(arm)
        reads[0] += 1
        if reads[0] >= 2:
            if movement == 'joint':
                current['joints_rad'][0] += math.radians(.2)
            elif movement == 'position':
                current['ee_pose']['position_m'][2] -= .001
            elif movement == 'orientation':
                current['ee_pose']['rpy_deg'][0] += .3
            else:
                current['gripper']['width_m'] += .001
        return current
    robot.state = moved_then_quiet
    result = robot.stationary_feedback(reference_states={'right': reference}, poll_s=.02)
    assert not result['confirmed'] and 'fixed preparation reference' in result['reason']
    assert reads[0] == 2
    assert result['stamp'] - reference['stamp'] == pytest.approx(.02)
    assert not result['holding_torque_verified']


@pytest.mark.parametrize("movement", ["joint", "position", "orientation", "duplicate", "stale"])
def test_sent_pause_does_not_prove_stopping_and_ack_does_not_clear_moving_arm(worker, movement):
    robot, state = worker
    state["controller"]["motion_status"] = 1  # A paused target may remain unreached.
    robot.lock = threading.Lock()
    commands, reads = [], [0]
    robot.arms = {"right": SimpleNamespace(MotionCtrl_1=lambda *args: commands.append(args))}
    read = robot.state
    def feedback(*args):
        current = read(*args)
        reads[0] += 1
        if movement == "joint":
            current["joints_rad"][3] = reads[0] * .001
        elif movement == "position":
            current["ee_pose"]["position_m"][2] += reads[0] * .00015
        elif movement == "orientation":
            current["ee_pose"]["rpy_deg"][2] += reads[0] * .08
        elif movement == "duplicate":
            current["feedback_stamps"] = {"pose": 100., "joint": 100.}
        else:
            current["age_s"] = .4
        return current
    robot.state = feedback
    result = robot.stop("trajectory guard")
    assert result["stopped"] and not result["stop_confirmed"]
    assert result["stop_confirmation"]["controller_feedback"]["right"]["motion_status"] == 1
    assert not result["stop_confirmation"]["holding_torque_verified"]
    assert robot.fault == "trajectory guard"
    with pytest.raises(RuntimeError, match="sustained stationary"):
        robot.acknowledge()
    assert robot.stopped.is_set() and robot.fault == "trajectory guard"
    # Neither a failed pause nor a refused acknowledgement sends another type
    # of actuator packet, reset, disable, target, or gripper command.
    assert commands == [(0, 1, 0)]
    robot.state = read
    recovered = robot.acknowledge()
    assert not recovered["stopped"] and recovered["fault"] is None
    assert recovered["stop_confirmation"]["stable_window_s"] >= .5
    assert recovered["stop_confirmation"]["samples"] >= 10
    assert commands == [(0, 1, 0)]


def test_acknowledge_rejects_single_fresh_frame_after_unconfirmed_stop(worker):
    robot, _ = worker
    robot.lock = threading.Lock()
    robot.arms = {"right": SimpleNamespace()}
    robot.fault = "continued motion after pause"
    robot.stopped.set()
    robot.stop_confirmation = {"confirmed": False}
    only_frame = robot.state("right")
    robot.state = lambda *_: copy.deepcopy(only_frame)
    with pytest.raises(RuntimeError, match="sustained stationary"):
        robot.acknowledge()
    assert robot.stopped.is_set() and robot.fault == "continued motion after pause"
    assert not robot.stop_confirmation["confirmed"]


@pytest.fixture
def teaching_worker(worker, monkeypatch, tmp_path):
    robot, state = worker
    robot.lock = threading.Lock()
    robot.stopped.set()
    robot.fault = 'startup observation'
    commands, modes = [], {'left': 2, 'right': 2}
    after_send = [lambda name: modes.update({name: 0})]
    read = robot.state
    def feedback(name):
        result = read(name)
        result['controller']['ctrl_mode'] = modes[name]
        return result
    robot.state = feedback
    def send(name, args):
        commands.append((name, args))
        after_send[0](name)
    robot.arms = {name: SimpleNamespace(
        MotionCtrl_1=lambda *args, name=name: send(name, args),
        GetArmModeCtrl=lambda: SimpleNamespace(Hz=0)) for name in modes}
    def camera_read():
        stamp = read('left')['stamp']
        return {'cameras': {name: dict(available=True, stamp=stamp, age_s=0, serial=name, width=640, height=480)
                            for name in ('top', 'left_wrist', 'right_wrist')}}
    robot.teleop = SimpleNamespace(session=None, active=False, owns_lock=False,
                                   camera_read=camera_read, view=lambda: {'active': False})
    monkeypatch.setitem(type(robot).acknowledge.__globals__, 'RUNTIME', tmp_path/'.runtime')
    return robot, state, commands, modes, after_send


@pytest.mark.parametrize('other_mode', [1, 2])
def test_scene_acknowledgement_ends_only_teaching_without_targets_enable_or_reset(teaching_worker, other_mode):
    robot, _, commands, modes, _ = teaching_worker
    modes['right'] = other_mode
    result = robot.dispatch('acknowledge', {})
    assert commands == [('left', (0, 0, 2))]+([('right', (0, 0, 2))] if other_mode == 2 else [])
    assert result['teach_exit']['state'] == 'exited'
    assert result['stop_confirmed'] and not result['stopped'] and result['fault'] is None
    assert not result['teleop']['active'] and not robot.lock.locked()
    assert all(modes[name] in (0, 1) for name in modes)
    import json
    from pathlib import Path
    evidence = json.loads(Path(result['teach_exit']['evidence']).read_text())
    assert evidence['state'] == 'exited'
    assert not any(evidence[key] for key in ('reset_sent', 'motor_enable_sent', 'motion_target_sent'))


def test_scene_acknowledgement_retains_stop_when_firmware_does_not_exit(teaching_worker):
    robot, _, commands, _, after_send = teaching_worker
    after_send[0] = lambda _: None
    with pytest.raises(RuntimeError, match='退出示教未确认'):
        robot.dispatch('acknowledge', {})
    assert 4 <= len(commands) <= 22
    assert all(args == (0, 0, 2) for _, args in commands[:-2])
    assert commands[-2:] == [('left', (0, 1, 0)), ('right', (0, 1, 0))]
    assert robot.stopped.is_set() and robot.teach_exit['state'] == 'failed'
    assert not robot.lock.locked()


@pytest.mark.parametrize('failure', ['rgb', 'feedback', 'fault', 'joint', 'gripper', 'enable', 'mit', 'stop'])
def test_teach_exit_checks_both_arms_and_cancels_before_second_write(teaching_worker, failure):
    robot, state, commands, _, after_send = teaching_worker
    def interrupt(_):
        if failure == 'rgb':
            robot.teleop.camera_read = lambda: {'cameras': {}}
        elif failure == 'feedback':
            state['age_s'] = 1.
        elif failure == 'fault':
            state['controller']['err_code'] = 63
        elif failure == 'joint':
            state['joints_rad'][0] += math.radians(1)
        elif failure == 'gripper':
            state['gripper']['width_m'] += .003
        elif failure == 'enable':
            state['controller']['motors_enabled'] = [False]*6
        elif failure == 'mit':
            robot.arms['right'].GetArmModeCtrl = lambda: SimpleNamespace(
                Hz=200, mode_ctrl=SimpleNamespace(mit_mode=0xAD))
        else:
            robot.stop_epoch += 1
            robot.stop_completed_epoch = robot.stop_epoch
            robot.fault = 'new stop'
    after_send[0] = interrupt
    with pytest.raises(RuntimeError):
        robot.dispatch('acknowledge', {})
    assert commands == [('left', (0, 0, 2))]+(
        [] if failure == 'stop' else [('left', (0, 1, 0)), ('right', (0, 1, 0))])
    assert robot.stopped.is_set() and robot.teach_exit['state'] == 'failed'
    if failure == 'stop':
        assert robot.fault == 'new stop'


@pytest.mark.parametrize('failure', ['rgb', 'other_mode', 'mit', 'moving', 'evidence'])
def test_teach_exit_refuses_unsafe_initial_conditions_before_any_write(teaching_worker, monkeypatch, failure):
    robot, state, commands, modes, _ = teaching_worker
    if failure == 'rgb':
        robot.teleop.camera_read = lambda: {'cameras': {}}
    elif failure == 'other_mode':
        modes['right'] = 4
    elif failure == 'mit':
        robot.arms['right'].GetArmModeCtrl = lambda: SimpleNamespace(
            Hz=200, mode_ctrl=SimpleNamespace(mit_mode=0xAD))
    elif failure == 'moving':
        read = robot.state
        def moving(name):
            state['joints_rad'][0] += .01
            return read(name)
        robot.state = moving
    else:
        def failed(*_):
            raise OSError('evidence disk full')
        monkeypatch.setitem(type(robot).acknowledge.__globals__, 'atomic_json', failed)
    with pytest.raises((RuntimeError, OSError)):
        robot.dispatch('acknowledge', {})
    assert not commands
    assert robot.stopped.is_set()


def test_legacy_acknowledgement_does_not_attempt_teach_exit(teaching_worker):
    robot, _, commands, modes, _ = teaching_worker
    robot.acknowledge()
    assert not commands and modes == {'left': 2, 'right': 2}


def test_teach_exit_sdk_exception_still_requests_stop_for_both_arms(teaching_worker):
    robot, _, commands, _, _ = teaching_worker
    def failing(*args):
        commands.append(('left', args))
        if args == (0, 0, 2):
            raise OSError('CAN send failed')
    robot.arms['left'].MotionCtrl_1 = failing
    with pytest.raises(OSError, match='CAN send failed'):
        robot.dispatch('acknowledge', {})
    assert commands == [('left', (0, 0, 2)), ('left', (0, 1, 0)), ('right', (0, 1, 0))]
    assert robot.stopped.is_set() and not robot.lock.locked()


@pytest.mark.parametrize('width,enabled,accepted', [(-.0021, False, True), (-.0037, False, True),
    (-.0021, True, False), (-.066, False, False), (.066, False, False), (float('nan'), False, False)])
def test_only_manual_arm_preparation_can_quarantine_disabled_negative_gripper(worker, width, enabled, accepted):
    robot, measured = worker
    measured['gripper'].update(width_m=width, enabled=enabled)
    robot.arms = {'right': SimpleNamespace(GetArmModeCtrl=lambda: SimpleNamespace(Hz=0))}
    with pytest.raises(RuntimeError, match='physical zero/opening verification'):
        robot.check_prepare('right')
    if accepted:
        result = robot.check_teleop_prepare('right', 'position_velocity', arm_only=True)
        assert result['gripper']['width_m'] == width
        assert not result['gripper']['enabled']
    else:
        with pytest.raises(RuntimeError, match='physical zero/opening verification'):
            robot.check_teleop_prepare('right', 'position_velocity', arm_only=True)


def test_manual_stationary_band_accepts_recorded_servo_jitter_but_keeps_other_checks(worker):
    robot, state = worker
    robot.arms = {'right': SimpleNamespace()}
    base = robot.state
    reads = [0]
    def jitter(arm):
        current = base(arm)
        reads[0] += 1
        current['joints_rad'][4] = math.radians(.12 if reads[0] % 2 else 0.)
        return current
    robot.state = jitter
    assert not robot.stationary_feedback()['confirmed']
    result = robot.stationary_feedback(joint_tolerance_deg=.15)
    assert result['confirmed'] and result['stable_window_s'] >= .5
    assert result['joint_tolerance_deg'] == .15
    reference = base('right')
    assert not robot.stationary_feedback(reference_states={'right': reference}, joint_tolerance_deg=.15)['confirmed']
    def translating(arm):
        current = jitter(arm)
        current['ee_pose']['position_m'][0] += reads[0]*.0001
        return current
    robot.state = translating
    assert not robot.stationary_feedback(joint_tolerance_deg=.15)['confirmed']
    with pytest.raises(ValueError):
        robot.stationary_feedback(joint_tolerance_deg=.151)


def test_recovery_quiet_window_can_wait_for_late_settling_without_widening_tolerances(worker):
    robot, state = worker
    robot.arms = {'right': SimpleNamespace()}
    base = robot.state
    start = base('right')['stamp']
    checks = []
    def settling(arm):
        current = base(arm)
        elapsed = current['stamp'] - start
        current['joints_rad'][1] = math.radians(min(elapsed, 2.) * 2.)
        return current
    robot.state = settling
    assert not robot.stationary_feedback()['confirmed']
    def monitor():
        checks.append(base('right')['stamp'])
    result = robot.stationary_feedback(timeout_s=5., feedback_check=monitor)
    assert result['confirmed'] and result['stable_window_s'] >= .5
    assert result['joint_tolerance_deg'] == .1
    assert len(checks) >= 10 and max(np.diff(checks)) <= .0500001


@pytest.mark.parametrize('failure', ['stop', 'camera stale', 'actual collision'])
def test_recovery_quiet_window_monitor_errors_abort_immediately(worker, failure):
    robot, _ = worker
    robot.arms = {'right': SimpleNamespace()}
    def monitor():
        raise RuntimeError(failure)
    with pytest.raises(RuntimeError, match=failure):
        robot.stationary_feedback(timeout_s=5., feedback_check=monitor)


@pytest.mark.parametrize('timeout', [0, .49, 5.01, float('nan'), float('inf'), True])
def test_stationary_wait_budget_is_bounded(worker, timeout):
    robot, _ = worker
    with pytest.raises(ValueError, match='stationary timeout'):
        robot.stationary_feedback(timeout_s=timeout)


@pytest.mark.parametrize('fresh', [True, False])
def test_controller_limit_query_is_read_only_and_requires_fresh_per_axis_reply(worker, fresh):
    robot, _ = worker
    robot.lock = threading.Lock()
    robot.stopped.set()
    robot.stop_confirmation = {'confirmed': True}
    calls = []
    response = SimpleNamespace(time_stamp=0., current_motor_angle_limit_max_vel=SimpleNamespace(
        motor_num=0, min_angle_limit=-700, max_angle_limit=700, max_joint_spd=3000))
    def query(index, content):
        calls.append((index, content))
        response.current_motor_angle_limit_max_vel.motor_num = index
        response.time_stamp = 100. if fresh else 99.
    robot.arms = {'right': SimpleNamespace(SearchMotorMaxAngleSpdAccLimit=query,
        GetCurrentMotorAngleLimitMaxVel=lambda: response)}
    result = robot.controller_diagnostics({'arm': 'right'})
    assert calls == [(index, 1) for index in range(1, 7)]
    assert result['motion_commands_sent'] is False
    assert result['complete'] is fresh
    assert not robot.lock.locked() and robot.stopped.is_set()
    if fresh:
        assert result['limits'][4]['max_deg'] == 70
    else:
        assert all(item['available'] is False for item in result['limits'])


def test_controller_diagnostics_cannot_query_during_active_control(worker):
    robot, _ = worker
    robot.lock = threading.Lock()
    robot.arms = {'right': object()}
    robot.stop_confirmation = {'confirmed': True}
    with pytest.raises(RuntimeError, match='confirmed stop'):
        robot.controller_diagnostics({'arm': 'right'})
    robot.stopped.set()
    robot.lock.acquire()
    with pytest.raises(RuntimeError, match='idle stopped'):
        robot.controller_diagnostics({'arm': 'right'})
    robot.lock.release()
