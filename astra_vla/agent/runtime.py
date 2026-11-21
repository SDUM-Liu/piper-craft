"""System 1: local skill execution and feedback; no model call per control cycle."""
from dataclasses import dataclass, field
from pathlib import Path
import time

from astra_vla.config import RUNTIME
from astra_vla.observe import capture
from astra_vla.rpc import call
from .contracts import unique
from .episode import Episode
from .memory import retrieve
from .perception import LocalVisualServo, grasp_relation, observe_scene, visual_error
from .skills import SkillSpec, run_composition


class ReadOnlyPiper:
    """Existing camera PNG and SDK status RPC only; no session adoption/input spoofing."""
    mode = 'live_readonly'

    def __init__(self, provider, *, rpc=call):
        self.provider, self.rpc = provider, rpc

    def observe(self, directory):
        metadata = capture(directory, rpc=self.rpc)
        feedback = self.rpc(RUNTIME/'robot.sock', 'status', timeout=1)
        return observe_scene(metadata, feedback, self.provider, mode='live_readonly')

    def move(self, delta):
        raise RuntimeError('phase-one live adapter is read-only; desktop control integration not qualified')

    def initial_measurements(self):
        return []  # no fabricated calibration and no exploratory physical motion


@dataclass
class ExecutionContext:
    adapter: object
    episode: Episode
    target: str
    camera: str
    desired_px: list
    cancelled: object
    scene: object = None
    servo: object = None
    skill_history: list = field(default_factory=list)
    recovery_count: int = 0
    error: dict = field(default_factory=dict)
    aligned_frames: int = 0
    previous_sequence: int = -1
    previous_stamp: float = -1
    previous_serial: str = ''
    max_actions: int = 20


def observe_skill(value, ctx, parameters):
    ctx.episode.record['observation_count'] += 1  # include failed/stale observation requests in cost
    ctx.episode.flush()
    scene = ctx.adapter.observe(ctx.episode.path/'observations'/unique('frame'))
    view = scene.views[ctx.camera]
    if (view.sequence <= ctx.previous_sequence or view.timestamp <= ctx.previous_stamp or
            (ctx.previous_serial and view.serial != ctx.previous_serial)):
        return {'status': 'needs_observation'}
    ctx.previous_sequence, ctx.previous_stamp, ctx.previous_serial = view.sequence, view.timestamp, view.serial
    ctx.scene = scene
    ctx.episode.observation(scene)
    return {'status': 'succeeded', 'observation_id': scene.observation_id}


def detect_skill(value, ctx, parameters):
    obj = ctx.scene.object(value['camera'], value['target']) if ctx.scene else None
    if obj is None:
        return {'status': 'needs_observation'}
    return {'status': 'succeeded', 'object_id': obj['object_id'], 'center_px': obj['center_px'],
            'observation_id': ctx.scene.observation_id}


def orientation_skill(value, ctx, parameters):
    obj = ctx.scene.object(value['camera'], value['target']) if ctx.scene else None
    if not obj or not obj['axis_2d']:
        return {'status': 'needs_observation'}
    return {'status': 'succeeded', 'axis_2d': obj['axis_2d'], 'observation_id': ctx.scene.observation_id}


def grasp_skill(value, ctx, parameters):
    if ctx.scene is None:
        return {'status': 'needs_observation'}
    ctx.episode.record['model_judgment'] = grasp_relation(ctx.scene, value['camera'], value['target'])
    # Projected visibility is not a verified grasp result.
    return {'status': 'needs_observation'}


def error_skill(value, ctx, parameters):
    if ctx.scene is None:
        return {'status': 'needs_observation'}
    ctx.error = visual_error(ctx.scene, value['camera'], value['target'], value['desired_px'])
    ctx.episode.record.setdefault('visual_errors', []).append(dict(ctx.error))
    if ctx.error['status'] != 'succeeded':
        ctx.aligned_frames = 0
        return {'status': 'needs_observation'}
    output = {key: ctx.error[key] for key in ('error_px', 'norm_px', 'observation_id')}
    if ctx.error['norm_px'] <= parameters['tolerance_px']:
        ctx.aligned_frames += 1
        return {'status': 'succeeded' if ctx.aligned_frames >= 2 else 'confirm_observation', **output}
    ctx.aligned_frames = 0
    return {'status': 'adjust', **output}


def align_skill(value, ctx, parameters):
    if ctx.cancelled():
        return {'status': 'cancelled'}
    if ctx.adapter.mode != 'offline':
        return {'status': 'unsupported'}
    if (ctx.scene is None or ctx.error.get('status') != 'succeeded' or
            ctx.error.get('observation_id') != ctx.scene.observation_id):
        return {'status': 'needs_observation'}
    if ctx.episode.record['action_count'] >= ctx.max_actions:
        return {'status': 'budget_exhausted'}
    proposal = ctx.servo.propose(ctx.error['error_px'])
    if proposal['status'] != 'succeeded':
        return {'status': 'needs_observation'}
    before = ctx.scene
    action = ctx.episode.action(proposal, None)
    try:
        feedback = ctx.adapter.move(proposal['delta'])
        action.update(feedback=feedback, status='feedback_received')
    except Exception as exc:
        action.update(status='error', error=str(exc))
        raise
    finally:
        ctx.episode.flush()
    # A new source observation is required before fitting the measured response.
    result = observe_skill({}, ctx, {})
    if result['status'] != 'succeeded':
        return result
    previous = before.object(ctx.camera, ctx.target)
    latest = ctx.scene.object(ctx.camera, ctx.target)
    if previous is None or latest is None:
        return {'status': 'needs_observation'}
    ctx.servo.update(feedback['actual_delta'],
                     [b-a for a, b in zip(previous['center_px'], latest['center_px'])],
                     evidence=[before.views[ctx.camera].sha256, ctx.scene.views[ctx.camera].sha256])
    return {'status': 'succeeded'}


BUILTINS = {'observe_scene': observe_skill, 'detect_object': detect_skill,
            'estimate_object_orientation': orientation_skill, 'inspect_grasp': grasp_skill,
            'estimate_visual_error': error_skill, 'align_image': align_skill}


def register_builtins(registry, *, tolerance_px=2.):
    from .contracts import number
    if number(tolerance_px) <= 0:
        raise ValueError('positive pixel tolerance required')
    result = {}
    for name, function in BUILTINS.items():
        properties = {}
        if name in ('detect_object', 'estimate_object_orientation', 'inspect_grasp', 'estimate_visual_error'):
            properties = {'target': {'type': 'string'},
                          'camera': {'type': 'string', 'enum': ['top', 'left_wrist', 'right_wrist']}}
        if name == 'estimate_visual_error':
            properties['desired_px'] = {'type': 'array', 'items': {'type': 'number'}}
        spec = SkillSpec(name, name.replace('_', ' '), '1.0.0', function.__name__,
                         {'type': 'object', 'properties': properties, 'required': list(properties)},
                         {'type': 'object', 'properties': {'status': {'type': 'string', 'enum': [
                             'succeeded', 'adjust', 'needs_observation', 'confirm_observation',
                             'unsupported', 'cancelled', 'budget_exhausted']},
                             'observation_id': {'type': 'string'}, 'object_id': {'type': 'string'},
                             'center_px': {'type': 'array', 'items': {'type': 'number'}},
                             'axis_2d': {'type': 'array', 'items': {'type': 'number'}},
                             'error_px': {'type': 'array', 'items': {'type': 'number'}},
                             'norm_px': {'type': 'number', 'minimum': 0}},
                          'required': ['status']},
                         ('required fresh features and compatible adapter',),
                         ('explicit result predicate, never SDK arrival alone',),
                         ('unknown geometry, stale feedback or execution exception',),
                         {'tolerance_px': tolerance_px} if name == 'estimate_visual_error' else {},
                         function.__module__+':'+function.__qualname__, (),
                         ('configured colour markers; offline image alignment or live read-only observation',))
        result[name] = registry.register(spec, function, source_files=Path(__file__).parent.glob('*.py'))
    return result


class Agent:
    def __init__(self, registry, archive, adapter, *, memory_root, skills=None):
        self.registry, self.archive, self.adapter = registry, archive, adapter
        self.memory_root = Path(memory_root)
        self.skills = skills or register_builtins(registry)

    def execute_task(self, *, task, target_object, destination=None, scene_id='unspecified',
                     camera='top', gain=.5, max_actions=20, policy_version='baseline', cancelled=lambda: False,
                     use_memory=True, exploration=False):
        from .contracts import CAMERAS, identifier, number
        identifier(scene_id)
        if (task not in ('inspect_scene', 'align_image') or not isinstance(target_object, str) or not target_object
                or camera not in CAMERAS or type(max_actions) is not int or not 1 <= max_actions <= 100):
            raise ValueError('phase one supports inspect_scene/align_image, a target and bounded budget')
        if task == 'align_image' and (not isinstance(destination, list) or len(destination) != 2):
            raise ValueError('image alignment requires destination [u,v] pixels')
        if destination is not None:
            for coordinate in destination:
                number(coordinate)
        if task == 'align_image' and self.adapter.mode != 'offline':
            raise ValueError('live image motion is outside phase-one qualification')
        servo = LocalVisualServo(gain=gain, max_step=.25)
        memory = retrieve(self.memory_root, task=task, objects=[target_object], scene=scene_id,
                          skills=list(self.skills)) if use_memory else {'revision': 'disabled', 'entries': []}
        episode = Episode(self.archive, task=task, scene_id=scene_id, policy_version=policy_version,
                          skill_versions=self.skills, memory_revision=memory['revision'],
                          model_identity='none/local-rules', mode=self.adapter.mode)
        episode.record['retrieved_memory'] = memory
        episode.record['policy_parameters'] = {'gain': gain}
        episode.record['execution_budget'] = {'max_actions': max_actions}
        episode.record['task_input'] = {'target_object': target_object, 'destination': destination, 'camera': camera}
        episode.record['environment_identity'] = self.adapter.identity() if hasattr(self.adapter, 'identity') else None
        ctx = ExecutionContext(self.adapter, episode, target_object, camera, destination or [0, 0], cancelled,
                               servo=servo, max_actions=max_actions)
        revisions = self.skills
        nodes = {
            'observe': dict(revision=revisions['observe_scene'], input={}, transitions={'succeeded': 'detect'}),
            'detect': dict(revision=revisions['detect_object'], input={'target': target_object, 'camera': camera}, transitions={
                'succeeded': 'error' if task == 'align_image' else None}),
            'error': dict(revision=revisions['estimate_visual_error'], input={
                'target': target_object, 'camera': camera, 'desired_px': destination or [0, 0]}, transitions={
                'adjust': 'align', 'needs_observation': 'observe', 'confirm_observation': 'observe'}),
            'align': dict(revision=revisions['align_image'], input={}, transitions={'succeeded': 'error'}),
        }
        if exploration:
            # Reobserve ambiguous/lost features; never explore motion without geometry.
            nodes['detect']['transitions']['needs_observation'] = 'observe'
        stage, reason = None, None
        started = time.monotonic()
        try:
            for sample in self.adapter.initial_measurements():
                ctx.servo.update(**sample)
            def bounded_cancel():
                return cancelled() or time.monotonic()-started > 30
            ctx.cancelled = bounded_cancel
            result = run_composition(self.registry, nodes, 'observe', ctx, budget=4*max_actions+8)
            status = result['status']
            if status != 'succeeded':
                stage = ctx.skill_history[-1]['skill_id'] if ctx.skill_history else 'before_observation'
                reason = status
        except Exception as exc:
            status, reason = 'error', str(exc)
            stage = ctx.skill_history[-1]['skill_id'] if ctx.skill_history else 'initialization'
        record = episode.close(status=status, reason=reason, stage=stage, skill_history=ctx.skill_history,
                               recovery_count=ctx.recovery_count)
        return dict(task_status=status, success=None, skill_versions=self.skills,
                    execution_metrics={key: record[key] for key in ('execution_time', 'model_calls', 'action_count',
                                                                  'observation_count', 'recovery_count')},
                    observations=[o['observation_id'] for o in record['observation_history']],
                    failure_reason=reason, episode_path=str(episode.path/'episode.json'),
                    task_outcome='unknown', physical_success=None)
