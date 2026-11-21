"""Versioned procedures for interpreting the scene and planning a handoff."""
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path

from .contracts import CAMERAS
from .skills import SkillRegistry, SkillSpec, run_composition

STRING = {'type': 'string'}
ARRAY = {'type': 'array', 'items': STRING}
ARM = {'type': 'string', 'enum': ['left', 'right']}


def schema(**fields):
    return {'type': 'object', 'properties': fields, 'required': list(fields)}


def choose_views(value, context, parameters):
    operator = value['operator']+'_wrist'
    auxiliary = next(wrist for wrist in CAMERAS[1:] if wrist != operator)
    first_view = context.parameters.get('first_view', parameters['first_view'])
    wrists = [auxiliary, operator] if first_view == 'auxiliary' else [operator, auxiliary]
    return {'status': 'succeeded', 'cameras': [*wrists, 'top']}


def review(value, context, parameters):
    snapshot = context.snapshot
    pending = [name for name in value['conditions'] if
               snapshot['visual_judgments'].get(name, {}).get('answer') != 'yes' or
               snapshot['visual_judgments'].get(name, {}).get('state') != 'accepted']
    if snapshot['needs_current_observation']:
        pending.append('current_three_view_observation')
    return {'status': 'needs_observation' if pending else 'succeeded', 'pending': pending}


def register(directory):
    registry = SkillRegistry(directory)
    versions = {}
    definitions = (
        ('select_views', choose_views, schema(operator=ARM), schema(status=STRING, cameras=ARRAY),
         {'first_view': 'auxiliary'}, 'Choose wrist inspection order based on the operating arm.'),
        ('post_action_review', review, schema(conditions=ARRAY), schema(status=STRING, pending=ARRAY),
         {}, 'List post-motion visual conditions still requiring evidence.'),
    )
    source_files = (Path(__file__).parent/'skills.py', Path(__file__).parent/'task_state.py')
    for name, callback, inputs, outputs, params, description in definitions:
        spec = SkillSpec(name, name, '1', description, inputs, outputs,
                         ('current task and arm roles',), ('ordered views or explicit missing observations',),
                         ('required source missing or stale',), params,
                         f'{callback.__module__}:{callback.__qualname__}', ('source RGB references',),
                         ('task planning and observation; no motion authorization',))
        versions[name] = registry.register(spec, callback, source_files=source_files)
    return registry, versions


@dataclass
class TaskSkills:
    registry: SkillRegistry
    versions: dict
    policy_version: str = 'builtin_task_skills_v1'
    parameters: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    @classmethod
    def defaults(cls, directory):
        return cls(*register(directory))

    def context(self, snapshot, cancelled):
        return SimpleNamespace(snapshot=snapshot, cancelled=cancelled,
                               parameters=self.parameters, skill_history=self.history,
                               recovery_count=0)

    def execute(self, name, value, snapshot, cancelled):
        ctx = self.context(snapshot, cancelled)
        if name == 'handoff_review':
            # Handoff keeps the receiver's grasp, the donor's release clearance and
            # the object's support as named conditions from the task contract.
            nodes = {
                'review': dict(revision=self.versions['post_action_review'],
                               input={'conditions': value['conditions']}, transitions={'succeeded': 'views'}),
                'views': dict(revision=self.versions['select_views'],
                              input={'operator': value['receiver']}, transitions={}),
            }
            return run_composition(self.registry, nodes, 'review', ctx, budget=2)
        return self.registry.execute(self.versions[name], value, ctx)
