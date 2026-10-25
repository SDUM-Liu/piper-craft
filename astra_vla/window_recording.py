"""Bounded, asynchronous capture of the rendered desktop, independent of robot RPC."""
from datetime import datetime
from fractions import Fraction
import math
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np

from astra_vla.config import ROOT, atomic_json


class WindowRecording:
    fps = 30

    def __init__(self, destination=None, clock=time.monotonic):
        self.destination = Path(destination) if destination else ROOT/'runs'
        self.clock = clock
        self.lock = threading.Lock()
        self.thread = None
        self.state, self.error, self.path = 'idle', '', None
        self.started, self.next_frame = 0., 0.
        self.frames, self.dropped = 0, 0

    def status(self):
        with self.lock:
            return {'state': self.state, 'error': self.error, 'path': str(self.path) if self.path else None,
                    'elapsed_s': max(0., self.clock()-self.started) if self.state == 'recording' else 0.,
                    'frames': self.frames, 'dropped': self.dropped}

    def start(self, size):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.size = tuple(max(2, int(v)//2*2) for v in size)
            directory = self.destination/('window_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
            directory.mkdir(parents=True)
            self.path = directory/'window.mp4'
            self.started, self.wall_started = self.clock(), time.time()
            self.next_frame, self.frames, self.dropped = self.started, 0, 0
            self.pending = queue.Queue(maxsize=2)
            self.stopping = threading.Event()
            self.state, self.error = 'recording', ''
            self.thread = threading.Thread(target=self.encode, name='desktop-window-recording', daemon=True)
            self.thread.start()

    def due(self):
        with self.lock:
            return self.state == 'recording' and self.clock() >= self.next_frame

    def submit(self, rgb, size):
        """Only the latest two frames can wait; encoding never blocks input."""
        with self.lock:
            now = self.clock()
            if self.state != 'recording' or now < self.next_frame:
                return
            self.next_frame = self.started+(math.floor((now-self.started)*self.fps)+1)/self.fps
            sample = (now, size, rgb)
            try:
                self.pending.put_nowait(sample)
            except queue.Full:
                try:
                    self.pending.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
                self.pending.put_nowait(sample)

    def stop(self):
        with self.lock:
            if self.state == 'recording':
                self.state = 'finalizing'
                self.stopping.set()

    def close(self):
        self.stop()
        if self.thread:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError('窗口录像仍在收尾，请检查磁盘与编码器')

    def encode(self):
        container, failure, last_pts = None, '', -1
        try:
            import av
            import cv2
            container = av.open(str(self.path), mode='w')
            stream = container.add_stream('libx264', rate=self.fps)
            stream.width, stream.height = self.size
            stream.pix_fmt = 'yuv420p'
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)
            stream.codec_context.thread_count = 1
            stream.options = {'preset': 'ultrafast', 'crf': '23'}
            while not self.stopping.is_set() or not self.pending.empty():
                try:
                    stamp, (width, height), rgb = self.pending.get(timeout=.05)
                except queue.Empty:
                    continue
                pixels = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
                # Keep the complete window on resize/fullscreen, with letterboxing.
                scale = min(self.size[0]/width, self.size[1]/height)
                w, h = max(1, round(width*scale)), max(1, round(height*scale))
                canvas = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
                x, y = (self.size[0]-w)//2, (self.size[1]-h)//2
                canvas[y:y+h, x:x+w] = cv2.resize(pixels, (w, h), interpolation=cv2.INTER_AREA)
                frame = av.VideoFrame.from_ndarray(canvas, format='rgb24')
                frame.pts = max(last_pts+1, round((stamp-self.started)*1000))
                last_pts, frame.time_base = frame.pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
                with self.lock:
                    self.frames += 1
            for packet in stream.encode():
                container.mux(packet)
        except Exception as exc:
            failure = str(exc)
        finally:
            if container:
                try:
                    container.close()
                except Exception as exc:
                    failure = failure or str(exc)
            with self.lock:
                if not failure and not self.frames:
                    failure = '未采集到窗口帧；未生成有效视频'
                metadata = {'kind': 'desktop_window', 'source': 'pygame complete rendered client area',
                            'path': str(self.path), 'size': self.size, 'fps_limit': self.fps,
                            'start_stamp': self.wall_started, 'finish_stamp': time.time(),
                            'frames': self.frames, 'dropped_frames': self.dropped,
                            'last_frame_s': max(0, last_pts)/1000,
                            'closed': not failure, 'error': failure, 'audio': False,
                            'robot_motion_verified': False}
                try:
                    atomic_json(self.path.with_suffix('.json'), metadata)
                except Exception as exc:
                    failure = failure or str(exc)
                self.error, self.state = failure, 'error' if failure else 'saved'
