"""Identity-bound wrist RGB capture. No depth streams, intrinsics or calibration."""
import base64
import fcntl
import math
from pathlib import Path
import signal
import threading
import time

import cv2
import av

from astra_vla.clock_mapping import CameraClock
from astra_vla.config import RUNTIME, settings
from astra_vla.rpc import serve


def usb_identity(port):
    path = Path(port)
    for parent in (path, *path.parents):
        serial = parent / 'serial'
        if serial.is_file():
            return serial.read_text().strip()
    for device in Path('/sys/bus/usb/devices').glob('*'):
        try:
            key = '-'.join((device / field).read_text().strip() for field in ('busnum', 'devpath', 'devnum'))
            if key == port:
                return (device / 'serial').read_text().strip()
        except OSError:
            continue
    return None


def color_node(serial, video_root=Path('/sys/class/video4linux'), dev_root=Path('/dev')):
    """D435 color interface only; never open its depth/metadata interfaces."""
    matches = []
    for node in video_root.glob('video*'):
        try:
            interface = (node/'device').resolve()
            if (usb_identity(str(interface)) == serial and
                    (interface/'bInterfaceNumber').read_text().strip() == '03' and
                    (node/'index').read_text().strip() == '0'):
                matches.append(dev_root/node.name)
        except OSError:
            continue
    if len(matches) != 1:
        raise RuntimeError('expected exactly one wrist color node for USB '+serial)
    return matches[0]


def source_time(frame, wall, mono, clock, sequence):
    """V4L2 UVC timestamps are host monotonic; arrival cannot refresh old data."""
    if frame.pts is None or frame.time_base is None:
        raise ValueError('RGB kernel source timestamp missing')
    raw = float(frame.pts*frame.time_base)
    if not math.isfinite(raw) or not 0 <= mono-raw <= .3:
        raise ValueError('RGB kernel source timestamp stale/invalid')
    # Kernel PTS already shares time.monotonic's epoch. Map its capture time,
    # rather than estimating an unknown device epoch from this packet's arrival.
    return clock.project(raw, sequence, wall-(mono-raw), raw)


class RGB:
    def __init__(self):
        self.config = settings()
        self.lock = threading.Lock()
        self.ownership = threading.Lock()
        self.frames, self.errors, self.threads, self.stops = {}, {}, {}, {}
        for role in ('left_wrist', 'right_wrist'):
            self.retry(role)

    def discover(self, role):
        return color_node(self.config['cameras'][role]['serial'])

    def retry(self, role):
        if role not in ('left_wrist', 'right_wrist'):
            raise ValueError('select a real wrist camera')
        with self.ownership:
            old = self.threads.get(role)
            if old and old.is_alive():
                self.stops[role].set()
                old.join(timeout=3)
                if old.is_alive():
                    raise RuntimeError('existing RGB reader has not released this device')
            stop = threading.Event()
            self.stops[role] = stop
            thread = threading.Thread(target=self.capture, args=(role, stop), daemon=True)
            self.threads[role] = thread
            thread.start()
        return {'started': True, 'camera': role}

    def capture(self, role, stop):
        container = None
        try:
            node = self.discover(role)
            container = av.open(str(node), format='v4l2', options={
                'input_format': 'yuyv422', 'video_size': '640x480', 'framerate': '30'},
                timeout=(3, 1))
            clock = CameraClock()
            previous, sequence, last_good = None, 0, time.monotonic()
            for packet in container.demux(video=0):
                if stop.is_set():
                    break
                if time.monotonic()-last_good > 1:
                    raise RuntimeError('RGB color stream has no valid new frame')
                # UVC may emit empty startup buffers. They are never pictures.
                if packet.size < 640*480*2:
                    continue
                for frame in packet.decode():
                    if frame.width != 640 or frame.height != 480:
                        raise ValueError('unexpected RGB color dimensions')
                    mono, wall = time.monotonic(), time.time()
                    # Zero PTS is an invalid startup buffer, not a new epoch.
                    if frame.pts == 0:
                        continue
                    stamp = source_time(frame, wall, mono, clock, sequence+1)
                    if previous is not None and stamp <= previous:
                        raise ValueError('RGB kernel source time did not advance')
                    previous, sequence, last_good = stamp, sequence+1, mono
                    ok, jpeg = cv2.imencode('.jpg', frame.to_ndarray(format='bgr24'),
                                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise RuntimeError('RGB JPEG encoding failed')
                    with self.lock:
                        self.frames[role] = {'available': True, 'stamp': stamp, 'sequence': sequence,
                            'serial': self.config['cameras'][role]['serial'], 'width': 640, 'height': 480,
                            'jpeg': base64.b64encode(jpeg).decode(), 'source': 'native_rgb',
                            'backend': 'v4l2_color', 'timestamp_source': 'kernel_monotonic'}
                        self.errors.pop(role, None)
            if not stop.is_set():
                raise RuntimeError('RGB color stream ended')
        except Exception as exc:
            with self.lock:
                self.errors[role] = str(exc)
        finally:
            if container is not None:
                container.close()

    def snapshot(self):
        with self.lock:
            now = time.time()
            return {'cameras': {role: dict(self.frames.get(role, {'available': False,
                        'serial': self.config['cameras'][role]['serial']}),
                        age_s=now-self.frames[role]['stamp'] if role in self.frames else None)
                    for role in ('left_wrist', 'right_wrist')}, 'errors': dict(self.errors)}

    def dispatch(self, method, args):
        if method == 'snapshot':
            return self.snapshot()
        if method == 'retry':
            return self.retry(args['camera'])
        raise ValueError('unknown RGB method')

    def close(self):
        for stop in self.stops.values():
            stop.set()
        for thread in self.threads.values():
            thread.join(timeout=2)


def main():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME/'rgb.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    camera = RGB()
    def close(*_):
        camera.close()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, close)
    signal.signal(signal.SIGINT, close)
    try:
        serve(RUNTIME/'rgb.sock', camera.dispatch)
    finally:
        camera.close()


if __name__ == '__main__':
    main()
