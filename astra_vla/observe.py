"""Read the three original-resolution RGB views; never send robot commands."""
import argparse
import base64
import json
from pathlib import Path
import time

from astra_vla.config import RUNTIME, settings
from astra_vla.rpc import call


def capture(destination, *, rpc=call, clock=time.time):
    config = settings()
    snapshot = rpc(RUNTIME/'cameras.sock', 'snapshot', {'format': 'png'}, timeout=3)
    frames = snapshot['cameras']
    now = clock()
    for name in ('top', 'left_wrist', 'right_wrist'):
        frame = frames.get(name, {})
        stamp = frame.get('stamp', 0)
        if (not frame.get('available') or not frame.get('png') or
                frame.get('serial') != config['cameras'][name]['serial'] or
                not 0 <= now-stamp <= config['limits']['camera_age_s']):
            raise RuntimeError(name+' original RGB missing/stale/identity mismatch')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, frame in frames.items():
        path = destination/(name+'.png')
        path.write_bytes(base64.b64decode(frame['png'], validate=True))
        metadata[name] = {k: v for k, v in frame.items() if k not in ('png', 'jpeg')}
        metadata[name]['path'] = str(path.resolve())
    (destination/'observation.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
