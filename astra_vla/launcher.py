"""Observation-only startup for the single PIPER CRAFT desktop control stack."""
import argparse
import fcntl
import os
import shlex
import signal
import subprocess
import sys
import time

from astra_vla.config import ROOT, RUNTIME, atomic_json, settings
from astra_vla.rpc import call


def reuse_worker(name):
    path = RUNTIME/(name+'.sock')
    if not path.exists():
        return False
    try:
        state = call(path, 'status' if name == 'robot' else 'snapshot', timeout=2)
    except (ConnectionRefusedError, FileNotFoundError):
        return False
    if name == 'robot' and state.get('motion_control', {}).get('pose_source') != 'controller_sdk':
        raise RuntimeError('existing robot owner uses an older control stack; stop and hand off before reload')
    if name == 'cameras' and state.get('pipeline') != 'rgb_only':
        raise RuntimeError('existing camera owner uses an older pipeline; close it before reload')
    return True


def main():
    parser = argparse.ArgumentParser(description='PIPER CRAFT: RGB desktop and ordinary MOVE_J')
    parser.add_argument('--no-window', action='store_true', help='start observation workers only')
    args, desktop_args = parser.parse_known_args()
    cfg = settings()
    RUNTIME.mkdir(mode=0o700, exist_ok=True)
    lock = (RUNTIME/'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = subprocess.check_output(['bash', '-c', 'source '+shlex.quote(cfg['ros_setup'])+' >/dev/null && env -0'])
    env = dict(os.environ)
    env.update(dict(row.decode().split('=', 1) for row in output.split(b'\0') if b'=' in row))
    env.update(PYTHONPATH=str(ROOT)+':'+env.get('PYTHONPATH', ''), ASTRA_ROOT=str(ROOT))
    children, logs = [], []
    exiting = False
    def close(*_):
        nonlocal exiting
        if exiting:
            return
        exiting = True
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    print('owner still exiting; ownership retained by PID', child.pid, flush=True)
        for log in logs:
            log.close()
    def interrupted(*_):
        close()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for name, python in (('robot', cfg['robot_python']), ('rgb', cfg['rgb_python']), ('cameras', cfg['ros_python'])):
            if reuse_worker(name):
                print('复用现有', name, 'worker', flush=True)
                continue
            log = (RUNTIME/(name+'.log')).open('ab')
            logs.append(log)
            children.append(subprocess.Popen([python, '-u', '-m', 'astra_vla.workers.'+name],
                cwd=ROOT, env=env, stdout=log, stderr=log))
        deadline = time.monotonic()+20
        while True:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('worker exited; see .runtime/*.log')
            try:
                for name in ('robot', 'rgb', 'cameras'):
                    call(RUNTIME/(name+'.sock'), 'status' if name == 'robot' else 'snapshot', timeout=1)
                break
            except (OSError, RuntimeError):
                if time.monotonic() >= deadline:
                    raise RuntimeError('workers did not become ready')
                time.sleep(.2)
        if not args.no_window:
            gui_env = dict(os.environ, PYTHONPATH=str(ROOT), ASTRA_ROOT=str(ROOT))
            for key in ('PYTHONHOME', 'LD_LIBRARY_PATH'):
                gui_env.pop(key, None)
            children.append(subprocess.Popen([sys.executable, '-m', 'astra_vla.teleop_desktop', *desktop_args],
                cwd=ROOT, env=gui_env))
        atomic_json(RUNTIME/'processes.json', {'launcher': os.getpid(), 'children': [p.pid for p in children]})
        print('PIPER CRAFT 已启动：仅观察，现场确认后显式进入控制。', flush=True)
        while not exiting:
            if any(child.poll() is not None for child in children):
                break
            time.sleep(.2)
    finally:
        close()


if __name__ == '__main__':
    main()
