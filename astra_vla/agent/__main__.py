"""Read-only scene inspection or a fully isolated offline experiment."""
import argparse
import json
from pathlib import Path

from astra_vla.config import atomic_json
from .experiment import run_experiment
from .perception import ColorRegions, observe_scene
from .runtime import ReadOnlyPiper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='RGB/SDK read only; no motion or control-session entry')
    inspect.add_argument('--profiles', type=Path, required=True, help='JSON label -> [HSV lower, HSV upper]')
    inspect.add_argument('--observation', type=Path, help='existing observation.json for offline inspection')
    inspect.add_argument('--feedback', type=Path, help='existing SDK status JSON (offline inspection only)')
    inspect.add_argument('--output', type=Path, required=True, help='new directory, never overwrite source evidence')
    experiment = sub.add_parser('experiment')
    experiment.add_argument('--output', type=Path, required=True)
    experiment.add_argument('--pairs', type=int, default=8)
    args = parser.parse_args()
    if args.command == 'experiment':
        result = run_experiment(args.output, pairs=args.pairs)
    else:
        if args.feedback and not args.observation:
            parser.error('--feedback requires --observation; live SDK is read directly')
        args.output.mkdir(parents=True, exist_ok=False)
        provider = ColorRegions(json.loads(args.profiles.read_text()))
        if args.observation:
            metadata = json.loads(args.observation.read_text())
            feedback = json.loads(args.feedback.read_text()) if args.feedback else {}
            scene = observe_scene(metadata, feedback, provider)
        else:
            scene = ReadOnlyPiper(provider).observe(args.output/'source')
        atomic_json(args.output/'scene.json', scene.to_dict())
        result = {'scene': str((args.output/'scene.json').resolve()), 'observation_id': scene.observation_id,
                  'mode': scene.mode, 'object_count': len(scene.objects), 'motion_sent': False}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
