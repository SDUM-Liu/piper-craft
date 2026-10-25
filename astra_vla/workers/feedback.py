"""Immutable complete Piper pose/joint cycles, from the existing SDK reader.

No bus, SDK or actuator access. Partial, missing and out-of-order cycles never
replace the last complete sample; its original timestamp then becomes stale.
"""
import math
import struct
import threading


class CoherentArmFeedback:
    IDS = tuple(range(0x2A2, 0x2A8))

    def __init__(self):
        self.lock = threading.Lock()
        self.pending = []
        self.completed = None
        self.generation = 0

    def feed(self, frame):
        identifier = frame.arbitration_id
        if identifier not in self.IDS:
            return
        stamp = float(frame.timestamp)
        with self.lock:
            if identifier == self.IDS[0]:
                self.pending = []
                if self.completed and stamp <= self.completed[0][0][0]:
                    return
            if (getattr(frame, 'is_extended_id', False) or getattr(frame, 'is_error_frame', False) or
                    getattr(frame, 'is_remote_frame', False) or len(frame.data) != 8 or
                    not math.isfinite(stamp) or stamp <= 0 or
                    identifier != self.IDS[len(self.pending)] or
                    (self.pending and not self.pending[-1][0] <= stamp <= self.pending[0][0] + .003)):
                self.pending = []
                return
            self.pending.append((stamp, struct.unpack('>ii', frame.data)))
            if len(self.pending) == len(self.IDS):
                self.generation += 1
                self.completed = (tuple(self.pending), self.generation)
                self.pending = []

    def snapshot(self):
        with self.lock:
            completed = self.completed
        if completed is None:
            return None
        parts, generation = completed
        pose = sum((pair for _, pair in parts[:3]), ())
        joints = sum((pair for _, pair in parts[3:]), ())
        return {'pose_stamp': parts[0][0], 'joint_stamp': parts[3][0],
                'span_s': parts[-1][0] - parts[0][0], 'generation': generation,
                'ee_pose': {'position_m': [v / 1e6 for v in pose[:3]],
                            'rpy_deg': [v / 1000 for v in pose[3:]]},
                'joints_rad': [math.radians(v / 1000) for v in joints]}
