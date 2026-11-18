"""Small JSON contracts shared by perception, skills and experiment records."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

CAMERAS = ('top', 'left_wrist', 'right_wrist')


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}', value):
        raise ValueError('invalid identifier')
    return value


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('expected finite number')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(prefix):
    return prefix + '_' + uuid.uuid4().hex


@contextmanager
def locked(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def validate(value, schema):
    """Deliberately small schema vocabulary; unsupported schemas fail closed."""
    if set(schema) - {'type', 'properties', 'required', 'items', 'enum', 'minimum', 'maximum'}:
        raise ValueError('unsupported schema keyword')
    kind = schema['type']
    valid = {'object': isinstance(value, dict), 'array': isinstance(value, list),
             'string': isinstance(value, str), 'boolean': type(value) is bool,
             'number': type(value) in (int, float), 'integer': type(value) is int,
             'null': value is None}
    if kind not in valid or not valid[kind]:
        raise ValueError('schema type mismatch: ' + kind)
    if kind in ('number', 'integer'):
        number(value)
        if value < schema.get('minimum', -math.inf) or value > schema.get('maximum', math.inf):
            raise ValueError('number outside schema')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('unknown enum value')
    if kind == 'object':
        properties = schema.get('properties', {})
        if set(value) - set(properties) or set(schema.get('required', [])) - set(value):
            raise ValueError('missing or extra schema fields')
        for key, item in value.items():
            validate(item, properties[key])
    if kind == 'array':
        for item in value:
            validate(item, schema['items'])


@dataclass(frozen=True)
class ImageEvidence:
    camera_id: str
    observation_id: str
    timestamp: float
    sequence: int
    serial: str
    path: str
    sha256: str
    width: int
    height: int

    @classmethod
    def from_metadata(cls, camera, observation_id, metadata):
        if camera not in CAMERAS or metadata.get('available') is not True:
            raise ValueError('original RGB unavailable')
        if metadata.get('lossless') is not True or metadata.get('resized') is not False:
            raise ValueError('lossless unscaled source required')
        if number(metadata['stamp']) <= 0 or type(metadata['sequence']) is not int or metadata['sequence'] < 0:
            raise ValueError('source timestamp/sequence required')
        if any(type(metadata[k]) is not int or metadata[k] <= 0 for k in ('width', 'height')):
            raise ValueError('invalid image dimensions')
        if not metadata.get('serial'):
            raise ValueError('camera identity required')
        path = Path(metadata['path']).resolve()
        if path.suffix.lower() != '.png':
            raise ValueError('original PNG required')
        return cls(camera, identifier(observation_id), metadata['stamp'], metadata['sequence'],
                   str(metadata['serial']), str(path), file_digest(path), metadata['width'], metadata['height'])

    def verify(self):
        if file_digest(self.path) != self.sha256:
            raise ValueError('source image changed')


@dataclass
class SceneState:
    observation_id: str
    timestamp: float
    views: dict
    robot_state: dict
    gripper_state: dict
    objects: list = field(default_factory=list)
    object_features: dict = field(default_factory=dict)
    visibility: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    uncertainty: list = field(default_factory=lambda: ['metric_depth_unknown', 'contact_force_unknown'])
    spatial_relations: list = field(default_factory=list)
    provider_identity: dict = field(default_factory=dict)
    mode: str = 'offline'

    def to_dict(self):
        return asdict(self)

    def object(self, camera, label):
        matches = [o for o in self.objects if o['camera_id'] == camera and o['label'] == label]
        return matches[0] if len(matches) == 1 else None  # ambiguity is never resolved by list order
