"""SDL dummy display and simulated RPC only; never reads live workers."""
import os
import base64
import io
import threading
import time

import pytest
import numpy as np

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
pg = pytest.importorskip('pygame')

from astra_vla.teleop_desktop import Client, Controls, DemoClient, Window, fresh_frame  # noqa: E402
from astra_vla.teleop_desktop import control_message, debug_pose_lines, wrap_text  # noqa: E402


@pytest.fixture
def window():
    pg.display.init()
    pg.font.init()
    result = Window(DemoClient(), demo=True)
    yield result
    result.recording.close()
    result.demonstration.close()
    pg.quit()


def test_f5_never_changes_selected_arm_or_motion():
    controls = Controls()
    controls.press('w')
    controls.press('f5')
    assert controls.camera == 'top' and controls.arm == 'left' and controls.keys == {'w'}
    controls.press('f5')
    assert controls.camera == 'left_wrist'


def test_entry_error_keeps_measured_cause_readable_in_hud(window):
    original = 'teleop hold requires joints inside hard limits; use explicit recovery: J2=-2.505° [0.000,180.000]°'
    message = control_message(original)
    assert '起姿超限' in message and 'J2=-2.505°' in message
    lines = wrap_text(message, window.small, 300)
    assert ''.join(lines) == message
    assert all(window.small.size(line)[0] <= 300 for line in lines)
    window.client.error = original
    window.draw()


def test_repeat_switch_and_blur_clear_keys():
    controls = Controls()
    controls.press('w', repeat=True)
    assert not controls.keys
    controls.press('q')
    assert controls.press('2') == 'switch'
    assert controls.arm == 'right' and not controls.keys
    controls.press('d')
    controls.blur()
    assert not controls.keys


def test_press_and_release_wake_rpc_sender_without_waiting_for_heartbeat():
    waiting, received = threading.Event(), threading.Event()
    inputs = []
    def rpc(_path, method, args, **_):
        assert method == 'teleop_input'
        inputs.append(args['keys'])
        received.set()
        return {'accepted': True}
    client = Client(rpc=rpc)
    client.session, client.control_active = 'offline', True
    class LongWait(threading.Event):
        def wait(self, _timeout=None):
            waiting.set()
            return super().wait(1.)
    client.input_changed = LongWait()
    thread = threading.Thread(target=client.control)
    thread.start()
    controls = Controls()
    try:
        assert received.wait(.2) and waiting.wait(.2)
        for keys in (['w'], []):
            waiting.clear()
            received.clear()
            controls.keys = set(keys)
            client.intent(controls, True)
            assert received.wait(.2) and waiting.wait(.2)
            assert inputs[-1] == keys
    finally:
        client.done.set()
        client.input_changed.set()
        thread.join(1)
    assert not thread.is_alive()


def test_sdl_events_draw_hud_and_stop(window):
    window.menu_open = False
    assert window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_w, repeat=False))
    window.draw()
    assert window.controls.keys == {'w'}
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F5, repeat=False))
    assert window.controls.camera == 'top'
    window.handle(pg.event.Event(pg.KEYUP, key=pg.K_w))
    assert not window.controls.keys
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE, repeat=False))
    assert window.menu_open and not window.client.error
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F9, repeat=False))
    assert '离线' in window.client.error
    window.handle(pg.event.Event(pg.WINDOWFOCUSLOST))
    assert not window.focused and not window.controls.keys
    window.draw()
    assert set(window.buttons) >= {'open', 'close', 'stop', 'acknowledge', 'left', 'right', 'view'}


def test_interactive_demo_hold_f5_release_rotation_and_gripper_never_call_rpc(monkeypatch):
    from astra_vla import teleop_desktop
    def forbidden(*_args, **_kwargs):
        raise AssertionError('offline TCP framework touched a real RPC or calibration')
    monkeypatch.setattr(teleop_desktop, 'call', forbidden)
    monkeypatch.setattr('astra_vla.config.verification', forbidden, raising=False)
    clock = [0.]
    from astra_vla.teleop import Policy
    demo = DemoClient(clock=lambda: clock[0], policy=Policy(translation_m_s=.02))
    controls = Controls()
    initial = np.asarray(demo.robot['arms']['left']['tcp_matrix'])
    demo.request('open', 'left')
    controls.press('w')
    controls.press('e')
    for tick in range(200):
        if tick == 100:
            controls.press('f5')
        clock[0] += .050001
        demo.intent(controls, True)
    state = demo.robot['arms']['left']
    np.testing.assert_allclose(np.asarray(state['tcp_matrix'])[:3, 3] - initial[:3, 3],
                               [.198, 0, 0], atol=1e-10)
    assert state['gripper']['width_m'] == pytest.approx(.065)
    controls.blur()
    anchor = np.asarray(state['tcp_matrix'])[:3, 3].copy()
    clock[0] += .050001
    demo.intent(controls, True)
    assert demo.robot['teleop']['phase'] == 'ready'
    controls.press('i')
    for _ in range(100):
        clock[0] += .050001
        demo.intent(controls, True)
    np.testing.assert_allclose(np.asarray(state['tcp_matrix'])[:3, 3], anchor, atol=1e-12)
    demo.intent(controls, False)
    assert demo.session is None and demo.robot['teleop']['stop_confirmed']
    demo.request('open', 'right')
    assert demo.session == 'offline-right'
    demo.shutdown()


def test_client_startup_and_shutdown_only_read_workers():
    calls = []
    def rpc(_path, method, args=None, **kwargs):
        calls.append(method)
        if method == 'snapshot':
            return {'cameras': {}}
        return {'teleop': {'active': False}}
    client = Client(rpc)
    client.start()
    time.sleep(.08)
    client.shutdown()
    assert calls and set(calls) <= {'snapshot', 'status'}


def test_esc_bypasses_blocked_prepare_and_does_not_reactivate():
    calls = []
    blocked, release = threading.Event(), threading.Event()
    def rpc(_path, method, args=None, **kwargs):
        calls.append(method)
        if method == 'teleop_open':
            blocked.set()
            release.wait(2)
            return {'session': 'new'}
        if method == 'stop':
            release.set()
            return {'stop_confirmed': True}
        return {'stop_confirmed': True}
    client = Client(rpc)
    client.opening = True
    opener = threading.Thread(target=client.execute, args=('open', 'left'))
    opener.start()
    assert blocked.wait(1)
    client.request('stop')
    opener.join(1)
    for thread in client.threads:
        thread.join(1)
    assert not opener.is_alive() and 'stop' in calls
    assert calls[-1] == 'teleop_close'
    assert client.session is None and not client.control_active


def test_unconfirmed_stop_keeps_session_available_for_retry():
    client = Client(lambda *_args, **_kwargs: {'stop_confirmed': False, 'owner': 'desktop'})
    client.session = 'active'
    client.control_active = True
    client.priority_stop('close')
    assert client.session == 'active' and not client.control_active
    assert '尚未确认' in client.error


def test_focus_loss_cancels_blocking_preparation_on_priority_thread():
    calls, stopped = [], threading.Event()
    client = Client(lambda _path, method, *_args, **_kwargs:
                    (calls.append(method), stopped.set(), {'stop_confirmed': True})[-1])
    client.opening = True
    client.intent(Controls(), False)
    assert stopped.wait(1)
    for thread in client.threads:
        thread.join(1)
    assert calls == ['stop'] and client.cancel_epoch == 1


def test_stale_capture_reports_source_error_and_recovers_without_hiding_control_error(window):
    buffer = io.BytesIO()
    pg.image.save(pg.Surface((4, 4)), buffer, 'frame.png')
    encoded = base64.b64encode(buffer.getvalue()).decode()
    now = time.time()
    snapshot = {'cameras': {n: {'available': True, 'stamp': now, 'age_s': .01, 'jpeg': encoded}
                           for n in ('top', 'left_wrist', 'right_wrist')}, 'error': None}
    snapshot['cameras']['left_wrist']['stamp'] = now - 60000
    snapshot['error'] = 'wait_for_frames cannot be called before start()'
    client = Client()
    client.error = '关节限位拒绝'
    client.update_images(snapshot)
    assert set(client.frames) == {'top', 'right_wrist'}
    assert '左腕' in client.camera_error and snapshot['error'] in client.camera_error
    assert client.error == '关节限位拒绝'
    snapshot['cameras']['left_wrist']['stamp'] = time.time()
    snapshot['error'] = None
    client.update_images(snapshot)
    assert len(client.frames) == 3 and not client.camera_error
    assert client.error == '关节限位拒绝'
    window.client = client
    window.demo = False
    window.draw()
    assert 'retry_cameras' in window.buttons


def test_camera_retry_only_restarts_missing_wrist_capture_and_never_writes_robot():
    calls = []
    now = time.time()
    def rpc(path, method, args=None, **kwargs):
        calls.append((path.name, method, args))
        if method == 'status':
            return {'stopped': True, 'stop_confirmed': True, 'busy': False, 'teleop': {'active': False}}
        if method == 'snapshot':
            return {'cameras': {'top': {'stamp': now}, 'right_wrist': {'stamp': now},
                                'left_wrist': {'available': True, 'stamp': now - 60000}}}
        assert method == 'retry' and args == {'camera': 'left_wrist'}
        return {'retry_started': True, 'hardware_reset': False}
    client = Client(rpc)
    client.execute('retry_cameras', None)
    assert calls == [('robot.sock', 'status', None), ('cameras.sock', 'snapshot', {'depth': False}),
                     ('cameras.sock', 'retry', {'camera': 'left_wrist'})]


@pytest.mark.parametrize('status', [
    {'stopped': True, 'stop_confirmed': True, 'busy': True},
    {'stopped': True, 'stop_confirmed': True, 'teleop': {'active': True}},
    {'stopped': True, 'stop_confirmed': False},
    {'stopped': False, 'stop_confirmed': True},
])
def test_camera_retry_refuses_active_or_unconfirmed_stop(status):
    def rpc(path, method, **kwargs):
        assert path.name == 'robot.sock' and method == 'status'
        return status
    with pytest.raises(RuntimeError):
        Client(rpc).execute('retry_cameras', None)


@pytest.mark.parametrize('frame', [
    {'stamp': float('nan')}, {'stamp': 100., 'age_s': float('nan')},
    {'stamp': 100., 'age_s': 1.}, {'stamp': 100., 'available': False},
    {'stamp': 101.}, {'stamp': 99.}, {},
])
def test_invalid_or_stale_frame_cannot_be_displayed_as_live(frame):
    assert not fresh_frame(frame, now=100.)


@pytest.mark.parametrize('stopped,fault,visible', [(False, None, False), (True, 'fault', True)])
def test_acknowledged_normal_stop_history_does_not_look_like_current_entry_error(window, stopped, fault, visible):
    texts = []
    window.client.robot['stopped'] = stopped
    window.client.robot['fault'] = fault
    window.client.robot['teleop'] = {'phase': 'stopped', 'blocker': 'old exit reason'}
    window.text = lambda value, *args, **kwargs: texts.append(str(value))
    window.draw()
    assert any('fault' in text for text in texts) == visible
    assert not any('old exit reason' in text for text in texts)


def test_tab_speed_cycle_clears_held_motion_and_ignores_repeat():
    controls = Controls()
    controls.press('space')
    controls.press('tab')
    assert controls.speed_profile == 'fast' and not controls.keys
    controls.press('tab', repeat=True)
    assert controls.speed_profile == 'fast'
    controls.press('tab')
    assert controls.speed_profile == 'slow'
    controls.press('tab')
    assert controls.speed_profile == 'normal'


def test_fast_client_rejects_legacy_worker_which_ignores_profile_instead_of_claiming_fast_control():
    calls = []
    def rpc(path, method, args=None, **kwargs):
        calls.append((method, args))
        return {'session': 'legacy'} if method == 'teleop_open' else {'stop_confirmed': True}
    c = Client(rpc=rpc)
    c.speed_profile = 'fast'
    with pytest.raises(RuntimeError, match='尚未加载快慢档'):
        c.execute('open', 'left')
    assert [name for name, _ in calls] == ['teleop_open', 'teleop_close']
    assert not c.session and not c.control_active


def test_topmost_toggle_and_key_repeat_are_local_only(window, monkeypatch):
    from types import SimpleNamespace
    native = SimpleNamespace(always_on_top=False)
    monkeypatch.setattr(pg, 'Window', SimpleNamespace(from_display_module=lambda: native))
    window.client.request = lambda *_args: pytest.fail('pinning submitted a robot action')
    window.controls.press('w')
    window.set_topmost(True)
    assert native.always_on_top and window.always_on_top and not window.window_notice
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F8, repeat=False))
    assert not native.always_on_top and not window.always_on_top
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F8, repeat=True))
    assert not native.always_on_top and window.controls.keys == {'w'}


def test_unsupported_topmost_reports_failure_instead_of_claiming_pinned(window, monkeypatch):
    from types import SimpleNamespace
    def unsupported():
        raise pg.error('unsupported window manager')
    monkeypatch.setattr(pg, 'Window', SimpleNamespace(from_display_module=unsupported))
    window.set_topmost(True)
    assert not window.always_on_top and '置顶不可用' in window.window_notice


def test_fullscreen_clears_keys_and_reapplies_pin_without_robot_request(window, monkeypatch):
    from types import SimpleNamespace
    calls = []
    native = SimpleNamespace(always_on_top=False)
    monkeypatch.setattr(pg, 'Window', SimpleNamespace(from_display_module=lambda: native))
    def mode(size, flags):
        calls.append((size, flags))
        return pg.Surface((1920, 1080) if size == (0, 0) else size)
    monkeypatch.setattr(pg.display, 'set_mode', mode)
    window.controls.press('space')
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F11, repeat=False))
    assert window.fullscreen and not window.controls.keys and native.always_on_top
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F11, repeat=False))
    assert not window.fullscreen and calls == [((0, 0), pg.FULLSCREEN), ((1280, 800), pg.RESIZABLE)]


def test_single_speed_button_cycles_and_arm_view_shortcuts_never_auto_open(window):
    calls = []
    window.client.request = lambda *args: calls.append(args)
    window.menu_open = False
    window.controls.press('w')
    window.draw()
    for profile in ('fast', 'slow', 'normal'):
        window.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=window.buttons['speed'].center))
        assert window.controls.speed_profile == profile and not window.controls.keys
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_TAB, repeat=False))
    assert window.controls.speed_profile == 'fast'
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_TAB, repeat=True))
    assert window.controls.speed_profile == 'fast'
    window.activate('speed')
    assert window.controls.speed_profile == 'slow' and not window.controls.keys
    window.handle(pg.event.Event(pg.MOUSEWHEEL, y=-1, x=0))
    assert window.controls.speed_profile == 'slow' and not calls
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F5, repeat=False))
    assert window.controls.third_person
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F5, repeat=False))
    assert not window.controls.third_person
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_2, repeat=False))
    assert window.controls.arm == 'right' and calls == [('switch', 'right')]


def test_menu_clears_live_intent_blocks_motion_and_f9_is_the_global_stop(window):
    def forbidden(*_args, **_kwargs):
        pytest.fail('opening the menu wrote a robot RPC')
    client = Client(rpc=forbidden)
    client.session, client.control_active = 'offline-test', True
    window.client, window.menu_open = client, False
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_w, repeat=False))
    window.publish_intent()
    assert client.keys == ['w']
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE, repeat=False))
    assert window.menu_open and client.keys == [] and client.session == 'offline-test'
    assert client.focused and client.control_active
    for key in (pg.K_w, pg.K_SPACE, pg.K_q, pg.K_i):
        window.handle(pg.event.Event(pg.KEYDOWN, key=key, repeat=False))
    window.publish_intent()
    assert not window.controls.keys and not client.keys
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE, repeat=True))
    assert window.menu_open
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE, repeat=False))
    window.publish_intent()
    assert not window.menu_open and not client.keys
    calls = []
    client.request = lambda *args: calls.append(args)
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F9, repeat=False))
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F9, repeat=True))
    assert calls == [('stop', 'left')] and window.menu_open


@pytest.mark.parametrize('size', [(1280, 800), (1100, 700)])
def test_controls_are_below_minimap_and_menu_actions_stay_in_menu(window, size):
    window.screen = pg.display.set_mode(size)
    window.menu_open = False
    window.draw()
    from astra_vla.teleop import JOINT_KEYS
    assert set(window.buttons) == {'left', 'right', 'open', 'acknowledge', 'backend', 'speed', 'record_window',
                                   'demonstrations', 'retry_cameras', 'menu'} | JOINT_KEYS
    assert all(window.sidebar_rect.contains(rect) and rect.top > window.minimap_rect.bottom
               for rect in [*window.buttons.values(), *window.key_rects.values()])
    assert all(not rect.colliderect(window.main_image_rect)
               for rect in [*window.buttons.values(), *window.key_rects.values()])
    assert set(window.key_rects) == {'w', 's', 'a', 'd', 'space', 'lshift', 'q', 'e', 'i', 'k', 'j', 'l', 'o', 'p'}
    assert all(not a.colliderect(b) for i, a in enumerate(window.buttons.values())
               for b in list(window.buttons.values())[i+1:])
    window.activate('menu')
    window.draw()
    assert {'open', 'close', 'stop', 'acknowledge', 'view', 'topmost', 'fullscreen', 'debug'} <= set(window.buttons)


def test_hiding_hud_keeps_errors_and_stop_confirmation_visible(window):
    texts = []
    window.menu_open = False
    window.client.error = 'actual controller fault'
    window.client.robot.update(stopped=True, stop_confirmed=False)
    window.text = lambda value, *_args, **_kwargs: texts.append(str(value))
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F1, repeat=False))
    window.draw()
    assert not window.show_hud and not window.buttons
    assert any('[控制] 无法执行：actual controller fault' in text for text in texts)
    assert any('尚未确认静止' in t for t in texts)
    assert any('OFFLINE DEMO' in t for t in texts)


def test_human_objects_recording_blocks_control_and_requires_fresh_views(window, tmp_path):
    window.demonstration.destination = tmp_path
    window.activate('demonstrations')
    window.client.frames.pop('top')
    window.activate('demonstration_record')
    assert window.demonstration.status()['state'] == 'idle'
    assert '三路' in window.demonstration_notice
    window.client.frames['top'] = window.client.frames['left_wrist']
    window.activate('demonstration_record')
    assert window.demonstration.status()['state'] == 'recording'
    window.activate('open')
    assert window.client.session is None
    assert '先结束' in window.demonstration_notice
    window.draw()
    window.demonstration.close()
    assert window.demonstration.status()['state'] == 'saved'
    window.activate('demonstrations')
    assert not window.demonstration_candidates  # simulated evidence is never a real reference


def test_teleop_demonstration_allows_explicit_control_entry_and_keeps_feedback(window, tmp_path):
    window.demonstration.destination = tmp_path
    window.activate('demonstration_mode')
    window.activate('demonstration_record')
    window.activate('open')
    assert window.client.session
    window.menu_open = False
    window.controls.press('w')
    window.draw()
    window.demonstration.close()
    import json
    trace = json.loads((window.demonstration.path/'samples.jsonl').read_text().splitlines()[0])
    assert trace['context']['requested_keys'] == ['w']
    assert set(trace['context']['arms']) == {'left', 'right'}


def test_object_demo_needs_stop_and_fresh_feedback_and_aborts_external_control(window, tmp_path):
    window.demo = False  # Still uses DemoClient: no real workers or motion.
    window.demonstration.destination = tmp_path
    window.activate('demonstration_record')
    assert window.demonstration.status()['state'] == 'idle'
    assert '停止确认' in window.demonstration_notice
    window.client.robot['stop_confirmed'] = True
    window.activate('demonstration_record')
    assert window.demonstration.status()['state'] == 'idle'  # no arm timestamps
    for arm in window.client.robot['arms'].values():
        arm['stamp'], arm['age_s'] = time.time(), 0.
    window.activate('demonstration_record')
    assert window.demonstration.status()['state'] == 'recording'
    window.client.robot['teleop']['active'] = True
    window.capture_demonstration()
    window.demonstration.close()
    assert window.demonstration.status()['state'] == 'error'
    assert '控制会话' in window.demonstration.status()['error']


def test_offline_demonstration_never_changes_real_reference(window, monkeypatch):
    from astra_vla import teleop_desktop
    def forbidden(*_args, **_kwargs):
        raise AssertionError('offline demo wrote real reference')
    monkeypatch.setattr(teleop_desktop, 'select_reference', forbidden)
    for action in ('demonstration_select', 'demonstration_clear'):
        window.activate(action)
        assert '离线演示不能修改' in window.demonstration_notice


def test_f3_shows_original_cause_trace_and_displacement_without_limit(window):
    texts = []
    window.menu_open = False
    window.client.robot['teleop'].update(release_extra_mm=25., termination_reason='original cause',
                                        trace_path='runs/actual_trace')
    window.text = lambda value, *_args, **_kwargs: texts.append(str(value))
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F3, repeat=False))
    window.draw()
    assert window.show_debug and 'original cause' in texts and 'runs/actual_trace' in texts
    assert any('25.00 mm · 仅记录' in t for t in texts)
    assert not any(' / 2.00 mm' in t for t in texts)


def test_old_release_limit_error_is_marked_as_history_after_limit_removal():
    message = control_message('teleop control: teleop release displacement exceeds normal profile budget',
                              {'release_extra_mm': 2.175, 'stop_confirmed': True})
    assert '现已取消' in message and '2.17 mm' in message and '已确认静止' in message


def test_gripper_target_and_actual_are_separate_and_effort_is_visible(window):
    texts = []
    window.client.robot['teleop'].update(gripper_target_mm=0., gripper_effort_limit=.5)
    window.client.robot['arms']['left']['gripper'].update(width_m=.05166, effort=-.126)
    window.text = lambda value, *_args, **_kwargs: texts.append(str(value))
    window.show_debug = True
    window.draw()
    assert any('夹爪实读 51.66 mm · 指令 0.00 mm' in t for t in texts)
    assert any('SDK力矩 -0.126 · 上限 0.500' in t for t in texts)


def test_routine_status_is_only_in_f3_but_unconfirmed_stop_stays_visible(window):
    texts = []
    window.menu_open = False
    window.client.robot.update(stopped=True, stop_confirmed=True)
    window.client.robot['teleop'].update(phase='stopped', release_extra_mm=2.175)
    window.text = lambda value, *_args, **_kwargs: texts.append(str(value))
    window.draw()
    assert not any('停止已确认' in t or '已确认静止' in t or '松键额外' in t or '夹爪实读' in t for t in texts)
    texts.clear()
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F3, repeat=False))
    window.draw()
    assert 'Status: 停止已确认' in texts and 'Stop: 已确认静止' in texts
    assert any('松键额外 2.17 mm' in t for t in texts)
    texts.clear()
    window.show_debug = False
    window.client.robot['stop_confirmed'] = False
    window.draw()
    assert any('已请求停止，尚未确认静止' in t for t in texts)


@pytest.mark.parametrize('size', [(1280, 800), (1100, 700)])
def test_f3_uses_measured_base_coordinates_and_tool_forward_not_camera_selection(window, size):
    from scipy.spatial.transform import Rotation
    window.screen = pg.display.set_mode(size)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    matrix[:3, 3] = [.123456, -.02, .35]
    state = window.client.robot['arms']['left']
    state.update(tcp_matrix=matrix.tolist(), joints_rad=np.deg2rad([1, -2, 30, -40, 50, -60]).tolist())
    state.pop('ee_pose')  # Test the TCP actually supplied in this status.
    texts = []
    window.text = lambda value, *_args, **_kwargs: texts.append(str(value))
    window.show_debug = True
    window.draw()
    assert any('XYZ: 123.456 / -20.000 / 350.000 mm' in t for t in texts)
    assert any('Facing: +X' in t for t in texts)
    assert 'J1: +1.000' in texts and 'J6: -60.000' in texts
    assert any(t.startswith('Local Time: ') for t in texts)
    assert any(t.startswith('Uptime: ') for t in texts)
    texts.clear()
    window.controls.press('f5')
    window.draw()
    assert any('Facing: +X' in t for t in texts)
    assert any('XYZ: 123.456 / -20.000 / 350.000 mm' in t for t in texts)


def test_f3_vertical_heading_and_absent_feedback_do_not_invent_coordinates():
    lines = debug_pose_lines(np.eye(4), 'piper_tool')
    assert any('Facing: +Z' in t for t in lines)
    assert 'Yaw / Pitch: — / 90.00 deg' in lines
    assert '等待反馈' in debug_pose_lines(None, 'piper_tool')[0]
    assert '无效反馈' in debug_pose_lines(np.full((4, 4), np.nan), 'piper_tool')[0]


def test_chat_keeps_simultaneous_faults_without_spam_and_without_robot_writes(window):
    window.client.request = lambda *_a, **_k: pytest.fail('drawing sent a robot command')
    window.client.error = 'unknown controller fault'
    window.client.camera_error = 'missing wrist camera'
    window.client.robot.update(stopped=True, stop_confirmed=False)
    texts = []
    window.text = lambda value, *_args, **_kwargs: texts.append(str(value))
    window.show_hud = False
    window.draw()
    count = len(window.chat)
    for _ in range(3):
        window.draw()
    assert len(window.chat) == count
    assert any('[控制] 无法执行：unknown controller fault' in t for t in texts)
    assert any('[相机] 画面源异常：missing wrist camera' in t for t in texts)
    assert any('[停止] 已请求停止，尚未确认静止' in t for t in texts)
    window.client.error = ''
    texts.clear()
    window.draw()
    assert not any('unknown controller fault' in t for t in texts)


@pytest.mark.parametrize('size', [(1280, 800), (1100, 700)])
def test_minimap_switches_complementary_camera_without_covering_main_image(window, size):
    window.screen = pg.display.set_mode(size)
    window.menu_open = False
    window.draw()
    assert window.minimap_camera == 'top'
    assert window.main_image_rect and not window.minimap_rect.colliderect(window.main_image_rect)
    assert window.minimap_rect.right <= size[0] and window.minimap_rect.width <= 224
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F5, repeat=False))
    window.draw()
    assert window.minimap_camera == 'left_wrist'
    window.activate('right')
    window.draw()
    assert window.minimap_camera == 'right_wrist'
    window.show_debug = True
    window.draw()
    assert window.minimap_rect is None


def test_minimap_never_presents_stale_auxiliary_frame_as_live(window):
    window.menu_open = False
    window.demo = False
    window.client.frames['top']['stamp'] = time.time()-5
    texts = []
    window.text = lambda value, *_a, **_k: texts.append(str(value))
    window.draw()
    assert '暂无新帧' in texts and window.main_image_rect


def test_key_rebinding_conflicts_persistence_release_and_reserved_stop(window, tmp_path):
    from astra_vla.teleop_preferences import DesktopPreferences
    window.preferences_path = tmp_path/'desktop.json'
    window.activate('keys')
    window.activate('bind:w')
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_s, repeat=False))
    assert window.capture_binding == 'w' and window.preferences.bindings['w'] == 'w'
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_b, repeat=False))
    assert window.capture_binding is None
    assert DesktopPreferences.load(window.preferences_path).bindings['w'] == 'b'
    window.activate('resume_view')
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_w, repeat=False))
    assert not window.controls.keys
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_b, repeat=False))
    assert window.controls.keys == {'w'}
    window.handle(pg.event.Event(pg.KEYUP, key=pg.K_b))
    assert not window.controls.keys
    window.activate('menu')
    window.activate('keys')
    window.activate('bind:q')
    calls = []
    window.client.request = lambda *args: calls.append(args)
    window.handle(pg.event.Event(pg.KEYDOWN, key=pg.K_F9, repeat=False))
    assert calls == [('stop', 'left')] and window.capture_binding is None
    assert window.preferences.bindings['q'] == 'q'
    window.activate('restore_keys')
    assert DesktopPreferences.load(window.preferences_path).bindings['w'] == 'w'


def test_startup_defaults_save_without_changing_live_selection_or_entering_control(window, tmp_path):
    from astra_vla.teleop_preferences import DesktopPreferences
    path = tmp_path/'desktop.json'
    window.preferences_path = path
    window.activate('defaults')
    for field in ('startup_arm', 'startup_speed', 'startup_view', 'always_on_top'):
        window.activate('default:'+field)
    assert window.controls.arm == 'left' and window.controls.speed_profile == 'normal'
    assert not window.controls.third_person and window.client.session is None
    loaded = DesktopPreferences.load(path)
    assert loaded.startup_arm == 'right' and loaded.startup_speed == 'fast' and loaded.startup_view == 'top'
    other = Window(DemoClient(), demo=True, preferences_path=path)
    assert other.controls.arm == 'right' and other.controls.speed_profile == 'fast'
    assert other.client.speed_profile == 'fast' and other.controls.third_person
    assert other.client.session is None and not other.client.robot['teleop']['active']
    other.activate('restore_defaults')
    assert DesktopPreferences.load(path) == DesktopPreferences()


def test_binding_help_is_available_during_control_but_changes_are_blocked(window):
    window.client.session = 'offline-test'
    window.activate('keys')
    assert window.menu_page == 'keys'
    window.activate('bind:w')
    assert window.capture_binding is None and '退出控制' in window.settings_notice
    window.activate('restore_keys')
    assert window.preferences.bindings['w'] == 'w'


def test_persistent_entry_and_ack_buttons_work_even_with_duplicate_menu_buttons(window):
    calls = []
    window.client.request = lambda *args: calls.append(args)
    window.menu_open = True
    window.draw()
    for action in ('open', 'acknowledge'):
        rects = [rect for name, rect in window.hitboxes if name == action]
        assert len(rects) == 2
        for rect in rects:
            window.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=rect.center))
        assert calls[-2:] == [(action, 'left')]*2


def test_record_complete_window_button_and_pending_layout_without_robot_rpc(window, tmp_path):
    import av
    from astra_vla.window_recording import WindowRecording
    window.recording = WindowRecording(tmp_path)
    window.screen = pg.display.set_mode((1100, 700))
    window.menu_open = False
    window.client.robot['teleop']['speed_profile_pending'] = 'fast'
    calls = []
    window.client.request = lambda *args: calls.append(args)
    texts = []
    original_text = window.text
    def text(value, *args, **kwargs):
        texts.append((value, args[0]))
        return original_text(value, *args, **kwargs)
    window.text = text
    window.draw()
    rect = window.buttons['record_window']
    assert window.sidebar_rect.contains(rect) and rect.top > window.minimap_rect.bottom
    assert all(window.sidebar_rect.contains(r) for r in window.key_rects.values())
    stop_hint = next(pos for value, pos in texts if value == 'F9 停止 · Esc 菜单')
    assert stop_hint[1]+window.small.get_linesize() <= 700
    window.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=rect.center))
    try:
        for _ in range(5):
            window.draw()
            time.sleep(.04)
        window.activate('menu')
        window.draw()  # Menu and HUD are part of the same video.
        window.activate('record_window')
    finally:
        window.recording.close()
    assert not calls, 'record start/stop must never send robot or camera RPC'
    assert window.recording.status()['state'] == 'saved'
    window.draw()
    window.draw()  # Render saved-path chat too (regression for notification format).
    with av.open(window.recording.status()['path']) as video:
        frames = list(video.decode(video=0))
    assert len(frames) >= 3
    pixels = frames[0].to_ndarray(format='rgb24')
    assert pixels.shape == (700, 1100, 3)
    assert pixels[rect.top:rect.bottom, rect.left:rect.right].std() > 10


def test_codex_public_messages_are_visible_data_without_robot_writes_or_real_source(window, tmp_path):
    from astra_vla.codex_messages import CodexMessageReader, publish_message
    assert window.codex_messages is None, 'offline demos must not read the live Codex source'
    path = tmp_path/'public.jsonl'
    window.codex_messages = CodexMessageReader(path, interval_s=0, max_age_s=20.)
    window.client.request = lambda *_a, **_k: pytest.fail('Codex text sent a robot request')
    window.client.rpc = lambda *_a, **_k: pytest.fail('Codex text sent robot RPC')
    window.show_hud, window.menu_open = False, False
    message = '正在观察瓶口，/stop 与 W 只是公开文字'
    publish_message(message, session_id='offline-public', seq=1, path=path)
    texts = []
    window.text = lambda value, *_a, **_k: texts.append(str(value))
    window.draw()
    assert any('[Codex] '+message in text for text in texts)
    assert not window.controls.keys and not window.client.session
    assert len(window.codex_chat) == 1
    window.draw()
    assert len(window.codex_chat) == 1
    texts.clear()
    window.codex_chat[0]['tick'] -= 21
    window.draw()
    assert not any(message in text for text in texts)


def test_codex_flood_cannot_evict_or_cover_current_control_faults(window, tmp_path):
    from astra_vla.codex_messages import CodexMessageReader, publish_message
    path = tmp_path/'public.jsonl'
    window.codex_messages = CodexMessageReader(path, interval_s=0)
    window.client.error = 'actual controller fault'
    window.client.camera_error = 'actual camera fault'
    window.client.robot.update(stopped=True, stop_confirmed=False)
    window.show_hud, window.menu_open = False, False
    window.client.request = lambda *_a, **_k: pytest.fail('display sent a robot request')
    window.draw()
    system_history = list(window.chat)
    for seq in range(70):
        publish_message('公开进度 '*100, session_id='offline-flood', seq=seq, path=path)
    texts = []
    window.text = lambda value, *_a, **_k: texts.append(str(value))
    for _ in range(5):
        texts.clear()
        window.draw()
        assert any('[控制] 无法执行：actual controller fault' in text for text in texts)
        assert any('[相机] 画面源异常：actual camera fault' in text for text in texts)
        assert any('[停止] 已请求停止，尚未确认静止' in text for text in texts)
    assert len(window.codex_chat) == 32 and list(window.chat) == system_history


def test_codex_source_failure_is_identified_and_recovers_without_control_effect(window, tmp_path):
    from astra_vla.codex_messages import CodexMessageReader, publish_message
    path = tmp_path/'public.jsonl'
    path.mkdir()
    window.codex_messages = CodexMessageReader(path, interval_s=0)
    window.menu_open = False
    window.client.request = lambda *_a, **_k: pytest.fail('source failure sent robot request')
    texts = []
    window.text = lambda value, *_a, **_k: texts.append(str(value))
    window.draw()
    assert any('[Codex消息源]' in text for text in texts)
    assert not window.client.error and not window.client.session
    path.rmdir()
    publish_message('消息源已恢复', session_id='offline-recover', seq=1, path=path)
    texts.clear()
    window.draw()
    assert any('[Codex] 消息源已恢复' in text for text in texts)
    assert not any('[Codex消息源]' in text for text in texts)


def test_codex_message_is_in_complete_window_frame_before_recording_submit(window, tmp_path):
    from astra_vla.codex_messages import CodexMessageReader, publish_message
    path = tmp_path/'public.jsonl'
    window.codex_messages = CodexMessageReader(path, interval_s=0)
    window.menu_open = False
    window.draw()
    before = pg.surfarray.array3d(window.screen).copy()
    publish_message('Codex 公开进度：观察瓶子与杯子', session_id='offline-capture', seq=1, path=path)
    captures = []
    window.recording.due = lambda: True
    window.recording.submit = lambda rgb, size: captures.append((rgb, size))
    window.draw()
    assert len(captures) == 1
    rgb, (width, height) = captures[0]
    captured = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
    after = pg.surfarray.array3d(window.screen).transpose(1, 0, 2)
    assert np.array_equal(captured, after)
    # Text occupies the main-view message area, and was already drawn at capture.
    changed = np.any(captured != before.transpose(1, 0, 2), axis=2)
    assert changed[height-320:height-140, :window.sidebar_rect.left].sum() > 100


def test_demo_explicit_codex_path_uses_only_temporary_source(window, tmp_path):
    path = tmp_path/'explicit-public.jsonl'
    other = Window(DemoClient(), demo=True, codex_messages_path=path)
    assert other.codex_messages.path == path
    assert other.codex_messages.poll() == [] and not path.exists()


@pytest.mark.parametrize('size', [(1280, 800), (1100, 700)])
def test_backend_button_clears_keys_exits_control_and_selects_without_auto_open(window, size):
    window.screen = pg.display.set_mode(size)
    window.menu_open = False
    window.client.request('open', 'left')
    window.controls.keys.add('w')
    window.draw()
    window.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=window.buttons['backend'].center))
    assert not window.controls.keys and not window.client.session
    assert window.client.execution_backend == 'controller_move_p'
    window.draw()
    texts = []
    draw_text = window.text
    window.text = lambda text, *a, **kw: (texts.append(str(text)), draw_text(text, *a, **kw))[1]
    window.show_debug = True
    window.draw()
    assert any('控制器原生' in text for text in texts)
    assert any('IK: controller' in text for text in texts)
    assert any('无本地自碰撞检查' in text for text in texts)
    for name, rect in window.hitboxes:
        if name == 'backend':
            assert window.screen.get_rect().contains(rect)
    window.activate('backend')
    assert window.client.execution_backend == 'regularized_move_j' and not window.client.session


def test_backend_button_is_available_in_menu_when_hud_is_hidden(window):
    window.show_hud = False
    window.menu_open = True
    window.draw()
    assert 'backend' in window.buttons
    window.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=window.buttons['backend'].center))
    assert window.client.execution_backend == 'controller_move_p'
    assert not window.client.session
