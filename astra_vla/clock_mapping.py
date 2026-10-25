"""Map real device time with a lower envelope of observed host arrival bounds."""


class CameraClock:
    PERIOD_S = 2**32 / 1e6

    def __init__(self):
        self.offset = None
        self.previous = None
        self.wraps = 0

    def project(self, raw, sequence, wall_bound, monotonic_bound):
        offset = wall_bound - raw if self.offset is None else self.offset
        wrapped = False
        if self.previous is not None:
            old_raw, old_sequence, old_stamp, old_mono, old_wall = self.previous
            if sequence == old_sequence and raw == old_raw:
                return old_stamp  # Polling the same frame never refreshes its source time.
            if sequence <= old_sequence:
                raise RuntimeError("RGB frame counter moved backwards")
            if raw < old_raw:
                if 0 <= raw < old_raw < self.PERIOD_S and old_raw - raw > self.PERIOD_S / 2:
                    offset += self.PERIOD_S
                    wrapped = True
                else:
                    raise RuntimeError("RGB device clock reset without rollover")
            elapsed = monotonic_bound - old_mono
            if abs((wall_bound - old_wall) - elapsed) > .5:
                raise RuntimeError("host clock discontinuity")
            if abs((raw + offset - old_stamp) - elapsed) > .5:
                raise RuntimeError("device/arrival clock discontinuity")
        # Clock drift and delayed startup frames can put a fixed epoch in the future.
        # Retain device elapsed time and the earliest observed arrival offset. Longer
        # transport delays cannot make old frames fresh. This is not hardware sync.
        offset = min(offset, wall_bound - raw)
        stamp = raw + offset
        self.offset = offset
        self.previous = (raw, sequence, stamp, monotonic_bound, wall_bound)
        self.wraps += int(wrapped)
        return stamp
