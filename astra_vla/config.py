"""Current PIPER CRAFT device, TCP and control configuration only."""
import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get('ASTRA_ROOT', Path(__file__).resolve().parents[1])).resolve()
RUNTIME = ROOT/'.runtime'


def settings():
    result = yaml.safe_load((ROOT/'config/devices.yaml').read_text())
    for name in ('robot_python', 'rgb_python', 'ros_python', 'ros_setup', 'urdf'):
        path = Path(os.path.expanduser(result[name]))
        result[name] = str((ROOT/path).resolve() if not path.is_absolute() else path)
    result['package_dirs'] = [str((ROOT/p).resolve()) for p in result['package_dirs']]
    return result


def tcp_configuration():
    path = ROOT/'config/tcp.json'
    if path.exists():
        return json.loads(path.read_text())
    return {'arms': settings()['tcp'], 'source': 'devices.yaml', 'configured': True,
            'projection_required': False, 'frame': 'joint6', 'units': 'metres'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.new')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)
