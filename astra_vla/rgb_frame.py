"""Lossless observation of decoded RGB pixels, without desktop rendering."""
import base64

import cv2


def snapshot_frames(frames, raw_frames, image_format):
    if image_format not in ('jpeg', 'png'):
        raise ValueError('RGB format must be jpeg or png')
    if image_format == 'jpeg':
        return frames
    result = {}
    for name, frame in frames.items():
        selected = dict(frame)
        selected.pop('jpeg', None)
        raw = raw_frames.get(name)
        if selected.get('available'):
            if raw is None or raw[0] != selected.get('stamp'):
                selected.update(available=False, error='lossless RGB source unavailable')
            else:
                ok, encoded = cv2.imencode('.png', raw[1])
                if not ok:
                    raise ValueError('RGB PNG encoding failed')
                selected.update(png=base64.b64encode(encoded).decode(), encoding='png',
                                lossless=True, resized=False, pixel_source='decoded_camera_rgb')
        result[name] = selected
    return result
