"""Held-out paired skill validation using labeled synthetic camera occlusions."""
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np

from astra_vla.config import atomic_json
from astra_vla.task_archive import TaskArchive
from .contracts import digest, file_digest
from .episode import Episode
from .harness import METRICS, PairedHarness, PolicyStore
from .perception import observe_scene
from .shuttle_sim import ShuttleSimulator
from .task_skills import TaskSkills

SCOPE = 'offline_view_selection'
EVALUATOR = 'synthetic_camera_visibility_v1'


def validate_views(output, *, proposal=None, regression=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = TaskArchive(output)
    skills = TaskSkills.defaults(output/'skills')
    policies = PolicyStore(output/'policies')
    harness = PairedHarness(output/'comparisons', policies)
    first_view = 'operator'
    proposal_record = None
    if proposal:
        proposal_record = json.loads(Path(proposal).read_text())
        first_view = proposal_record['first_view']
        source = Path(proposal_record['episode_path'])
        if first_view not in ('auxiliary', 'operator') or file_digest(source) != proposal_record['sha256']:
            raise ValueError('invalid proposal or changed development evidence')
        entry = json.loads(source.read_text())
        if entry['episode_id'] != proposal_record['episode_id'] or entry['state'] != 'closed':
            raise ValueError('an immutable closed development episode is required')
        atomic_json(output/'proposal.json', proposal_record)
    previous_view = 'auxiliary'
    if first_view == previous_view:
        raise ValueError('the proposed setting is already the parent default')
    parent = policies.freeze(name='parent', parameters={'first_view': previous_view},
                             skill_versions=skills.versions, scope=SCOPE)
    conditions = dict(model='fixed_skill_only', perception=digest(ShuttleSimulator(archive).provider.identity()),
                      task_distribution='synthetic_wrist_occlusions_v1', control_interface='read_only',
                      action_budget=1, evaluator=EVALUATOR)
    development = ('dev_01', 'dev_02')
    validation = tuple(f'holdout_{index:02}' for index in range(8))

    def trial(case, revision, index, preferred_role):
        task = archive.status() or archive.begin('Synthetic visibility test', 'Find visible shuttle contact.')
        sim = ShuttleSimulator(archive)
        operator = 'left' if index % 2 else 'right'
        auxiliary = 'right' if operator == 'left' else 'left'
        visible = (operator if preferred_role == 'operator' else auxiliary)+'_wrist'
        policy = policies.load(revision)
        episode = Episode(archive, task=task['instruction'], scene_id=case, policy_version=revision,
                          skill_versions=skills.versions, memory_revision=None,
                          model_identity=conditions['model'], mode='offline')
        scene = sim.observe(episode.path/'observations'/'initial')
        metadata = json.loads((episode.path/'observations'/'initial'/'observation.json').read_text())
        rng = np.random.default_rng(index)
        shade = tuple(int(v) for v in rng.integers(24, 64, 3))
        for camera, frame in scene.views.items():
            image = cv2.imread(frame.path)
            cv2.rectangle(image, (0, 129), (223, 143), shade, thickness=-1)
            if camera != visible:
                cv2.rectangle(image, (38, 30), (200, 126), shade, thickness=-1)
            if not cv2.imwrite(frame.path, image):
                raise OSError('cannot render occlusion fixture')
        scene = observe_scene(metadata, sim.status(), sim.provider, now=sim.clock())
        episode.record.update(task_input={'operating_arm': operator},
                              policy_parameters=policy['parameters'], execution_budget={'max_actions': 1},
                              environment_identity={k: conditions[k] for k in ('task_distribution',
                                                                                'control_interface')})
        episode.record['observation_count'] += 1
        episode.observation(scene)
        runner = TaskSkills(skills.registry, skills.versions, revision, deepcopy(policy['parameters']))
        result = runner.execute('select_views', {'operator': operator}, {}, lambda: False)
        actual = result['cameras'][0]
        # The chosen wrist and the independent visibility label are separate inputs.
        success = actual == visible
        receipt = dict(evaluator=EVALUATOR, episode_id=episode.record['episode_id'], task_success=success,
                       selected=actual, simulator_visible_camera=visible)
        path = episode.path/'evaluation.json'
        atomic_json(path, receipt)
        episode.record.update(task_outcome='success' if success else 'failure', evaluation_result=receipt)
        episode.close(status='completed', skill_history=runner.history)
        metrics = {name: None for name in METRICS}
        metrics.update(task_success=success, execution_failure=False, number_of_actions=0, number_of_observations=1,
                       model_calls=0, recovery_count=0, intervention_count=0)
        return episode.path/'episode.json', metrics, [{'path': str(path.resolve()), 'sha256': file_digest(path)}]

    evidence = []
    for index, case in enumerate(development):
        path, _, _ = trial(case, parent, index, 'operator')
        evidence.append(dict(path=str(path.resolve()), sha256=file_digest(path)))
    hypothesis = dict(status='hypothesis', modification_target='select_views.first_view', evidence=evidence,
                      rationale=(proposal_record['hypothesis'] if proposal_record else
                                 'The parent first view was occluded in two independent development fixtures.'))
    candidate = policies.freeze(name='candidate', parameters={'first_view': first_view},
                                skill_versions=skills.versions, parent=parent, hypothesis=hypothesis, scope=SCOPE)
    comparison_id = harness.preregister(parent=parent, candidate=candidate, development_cases=development,
                                        validation_cases=validation, conditions=conditions)
    for index, case in enumerate(validation, start=100):
        priority = 'auxiliary' if regression and index == 107 else 'operator'
        order = (parent, candidate) if index % 2 == 0 else (candidate, parent)
        for revision in order:
            path, metrics, evidence = trial(case, revision, index, priority)
            harness.record_trial(comparison_id, policy=revision, case=case, episode_path=path, metrics=metrics,
                                 evaluator=EVALUATOR, evidence=evidence)
    report = harness.select(comparison_id)
    if report['decision'] == 'inherit':
        harness.promote_offline(comparison_id)
    report['rollback_revision'] = parent
    report['rollback_parameters'] = policies.restore(parent)['parameters']
    atomic_json(output/'report.json', report)
    return report


def pinned_skills(directory, revision=None):
    directory = Path(directory)
    policies = PolicyStore(directory/'policies')
    selected = json.loads((policies.directory/'offline_default.json').read_text())
    harness = PairedHarness(directory/'comparisons', policies)
    report = harness.select(selected['comparison_id'])
    if (report['scope'] != SCOPE or report['decision'] != 'inherit'
            or selected['selection_sha256'] != digest(report) or selected['revision'] != report['candidate']):
        raise ValueError('selected view version is not backed by the held-out paired result')
    revision = revision or report['candidate']
    if revision not in (report['parent'], report['candidate']):
        raise ValueError('select the validated candidate or the original parent for rollback')
    policy = policies.restore(revision)
    registry = TaskSkills.defaults(directory/'skills')
    if registry.versions != policy['skill_versions']:
        raise ValueError('installed skill implementation differs from the pinned one')
    return TaskSkills(registry.registry, registry.versions, revision, policy['parameters'])
