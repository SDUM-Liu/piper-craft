"""Three RGB views from existing ROS top RGB and identity-bound wrist RGB."""
import base64
import fcntl
import signal
import threading
import time

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image

from astra_vla.config import RUNTIME, settings
from astra_vla.rpc import call, serve
from astra_vla.rgb_frame import snapshot_frames


class Cameras:
    def __init__(self):
        self.config = settings()
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frames, self.errors = {}, {}
        self.raw_frames = {}
        self.subscribers = [rospy.Subscriber(cfg['color'], Image, self.color,
            callback_args=name, queue_size=1) for name, cfg in self.config['cameras'].items()]

    def color(self, message, name):
        try:
            stamp = message.header.stamp.to_sec()
            if stamp <= 0:
                raise ValueError('RGB source timestamp missing')
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError('RGB JPEG encoding failed')
            with self.lock:
                self.frames[name] = {'available': True, 'stamp': stamp,
                    'sequence': message.header.seq, 'serial': self.config['cameras'][name]['serial'],
                    'width': message.width, 'height': message.height,
                    'jpeg': base64.b64encode(jpeg).decode(), 'source': 'ros_rgb'}
                self.raw_frames[name] = (stamp, image.copy())
                self.errors.pop(name, None)
        except Exception as exc:
            with self.lock:
                self.errors[name] = str(exc)

    def snapshot(self, image_format='jpeg'):
        with self.lock:
            result = {name: dict(self.frames.get(name, {'available': False, 'serial': cfg['serial']}))
                      for name, cfg in self.config['cameras'].items()}
            errors = dict(self.errors)
            raw = dict(getattr(self, 'raw_frames', {}))
        result = snapshot_frames(result, raw, image_format)
        try:
            native = call(RUNTIME/'rgb.sock', 'snapshot', {'format': image_format}, timeout=.4)
            for name, frame in native['cameras'].items():
                if (name in result and frame.get('serial') == self.config['cameras'][name]['serial'] and
                        frame.get('available') and frame.get('stamp', 0) > result[name].get('stamp', 0)):
                    result[name] = frame
            errors.update(native.get('errors', {}))
        except (OSError, RuntimeError) as exc:
            errors['native_rgb'] = str(exc)
        now = time.time()
        for frame in result.values():
            frame['age_s'] = now-frame['stamp'] if frame.get('stamp') else None
        return {'pipeline': 'rgb_only', 'cameras': result, 'error': '; '.join(k+': '+v for k, v in errors.items()) or None}

    def dispatch(self, method, args):
        if method == 'snapshot':
            return self.snapshot(args.get('format', 'jpeg'))
        if method == 'retry':
            return call(RUNTIME/'rgb.sock', 'retry', {'camera': args['camera']}, timeout=5)
        raise ValueError('unknown RGB camera method')


def main():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME/'cameras.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rospy.init_node('piper_craft_cameras', disable_signals=True)
    camera = Cameras()
    def close(*_):
        rospy.signal_shutdown('PIPER CRAFT camera shutdown')
        raise SystemExit(0)
    signal.signal(signal.SIGINT, close)
    signal.signal(signal.SIGTERM, close)
    serve(RUNTIME/'cameras.sock', camera.dispatch)


if __name__ == '__main__':
    main()
