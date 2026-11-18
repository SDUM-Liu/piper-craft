"""Trusted callable skills, immutable manifests, explicit dependencies and branches."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
import time

from astra_vla.config import atomic_json
from .contracts import digest, file_digest, identifier, locked, unique, validate


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    name: str
    version: str
    description: str
    input_schema: dict
    output_schema: dict
    preconditions: tuple
    completion_conditions: tuple
    failure_conditions: tuple
    parameters: dict
    implementation: str
    evidence: tuple
    applicability: tuple
    dependencies: tuple = ()


class SkillRegistry:
    """No exec/import of paths from manifests. Callables must be supplied by code.

    The source closure is explicit; built-ins bind all Agent sources. Fingerprints
    detect edits, not malicious code. Old source must be restored by the operator
    before an old implementation can execute; JSON alone cannot restore code.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.bindings = {}

    def register(self, spec, function, *, source_files=()):
        for value in (spec.skill_id, spec.version):
            identifier(value)
        if not all((spec.name, spec.description, spec.preconditions, spec.completion_conditions,
                    spec.failure_conditions, spec.applicability)):
            raise ValueError('skill contract and applicability required')
        expected = function.__module__ + ':' + function.__qualname__
        if spec.implementation != expected or not callable(function):
            raise ValueError('implementation binding differs')
        paths = set(map(lambda p: str(Path(p).resolve()), source_files))
        paths.add(str(Path(inspect.getsourcefile(function)).resolve()))
        record = {'spec': asdict(spec), 'sources': {p: file_digest(p) for p in sorted(paths)}}
        for dependency in spec.dependencies:
            self.load(dependency)
        revision = digest(record)
        with locked(self.directory):
            path = self.directory / (revision + '.json')
            if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(record)):
                raise ValueError('immutable skill record changed')
            if not path.exists():
                atomic_json(path, record)
        self.bindings[revision] = function
        return revision

    def load(self, revision):
        if not isinstance(revision, str) or len(revision) != 64 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('invalid skill revision')
        record = json.loads((self.directory/(revision+'.json')).read_text())
        if digest(record) != revision:
            raise ValueError('skill manifest changed')
        return record

    def execute(self, revision, value, context):
        record = self.load(revision)
        spec = record['spec']
        validate(value, spec['input_schema'])
        function = self.bindings.get(revision)
        if function is None:
            raise ValueError('trusted provider must be registered after restart')
        self.verify_sources(revision)
        started = time.monotonic()
        entry = dict(call_id=unique('skill_call'), skill_id=spec['skill_id'], version=spec['version'],
                     revision=revision, input=deepcopy(value), status='running')
        context.skill_history.append(entry)
        try:
            # Functions enforce executable pre/completion/failure predicates;
            # descriptions in the manifest explain them but do not grant motion.
            result = function(deepcopy(value), context, deepcopy(spec['parameters']))
            validate(result, spec['output_schema'])
            entry.update(status=result['status'], output=deepcopy(result))
            return result
        except Exception as exc:
            entry.update(status='error', error=str(exc))
            raise
        finally:
            entry['duration_s'] = time.monotonic()-started

    def verify_sources(self, revision):
        record = self.load(revision)
        for path, expected in record['sources'].items():
            if file_digest(path) != expected:
                raise ValueError('skill implementation changed since freezing')
        for dependency in record['spec']['dependencies']:
            self.verify_sources(dependency)


def run_composition(registry, nodes, start, context, *, budget=32):
    """Result-directed graph, bounded even when recovery edges form cycles."""
    if type(budget) is not int or not 1 <= budget <= 1000 or start not in nodes:
        raise ValueError('invalid composition/budget')
    for node in nodes.values():
        registry.load(node['revision'])
        if any(target is not None and target not in nodes for target in node['transitions'].values()):
            raise ValueError('unknown composition target')
    current = start
    for _ in range(budget):
        if context.cancelled():
            return {'status': 'cancelled'}
        node = nodes[current]
        result = registry.execute(node['revision'], node['input'], context)
        status = result['status']
        if status not in node['transitions']:
            return result  # unknown outcomes never fall through to the next operation
        following = node['transitions'][status]
        if following is None:
            return result
        if status in ('slippage_detected', 'needs_observation'):
            context.recovery_count += 1
        current = following
    return {'status': 'budget_exhausted'}
