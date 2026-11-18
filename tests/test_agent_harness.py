from copy import deepcopy
import json
from pathlib import Path

import pytest

from astra_vla.agent.contracts import digest, file_digest
from astra_vla.agent.experiment import ImagePlaneSimulator, run_experiment
from astra_vla.agent.harness import ABLATIONS, PairedHarness, PolicyStore, propose_improvement
from astra_vla.agent.runtime import Agent, register_builtins
from astra_vla.agent.skills import SkillRegistry
from astra_vla.config import atomic_json
from astra_vla.task_archive import TaskArchive


def setup_protocol(tmp_path, pairs=8):
    registry = SkillRegistry(tmp_path/'skills')
    skills = register_builtins(registry)
    store = PolicyStore(tmp_path/'policies')
    parent = store.freeze(name='parent', parameters={'gain': .18}, skill_versions=skills)
    archive = TaskArchive(tmp_path)
    archive.begin('offline')
    simulator = ImagePlaneSimulator(1)
    agent = Agent(registry, archive, simulator, memory_root=tmp_path, skills=skills)
    development = agent.execute_task(task='align_image', target_object='target', destination=[110., 90.],
                                     scene_id='dev_1', policy_version=parent, gain=.18, max_actions=8)
    simulator.evaluate(development['episode_path'], [110., 90.])
    hypothesis = propose_improvement([development['episode_path']])
    candidate = store.freeze(name='candidate', parameters={'gain': .65}, skill_versions=skills,
                             parent=parent, hypothesis=hypothesis)
    harness = PairedHarness(tmp_path/'experiments', store)
    conditions = dict(model='none/local-rules', perception=digest(simulator.provider.identity()),
                      task_distribution='image-plane-uniform-center-v1', control_interface='simulated_xy/v1',
                      action_budget=8, evaluator='image_plane_ground_truth/v1')
    val = [f'val_{i}' for i in range(pairs)]
    comparison_id = harness.preregister(parent=parent, candidate=candidate, development_cases=['dev_1'],
                                        validation_cases=val, conditions=conditions)
    return registry, archive, store, harness, parent, candidate, comparison_id, conditions


def trial(tmp_path, env, policy, case, *, override_gain=None):
    registry, archive, store, _, _, _, _, _ = env
    row = store.load(policy)
    simulator = ImagePlaneSimulator(101+int(case.split('_')[-1]))
    agent = Agent(registry, archive, simulator, memory_root=tmp_path, skills=row['skill_versions'])
    result = agent.execute_task(task='align_image', target_object='target', destination=[110., 90.],
                                scene_id=case, policy_version=policy, max_actions=8,
                                gain=override_gain if override_gain is not None else row['parameters']['gain'])
    metrics, evidence = simulator.evaluate(result['episode_path'], [110., 90.])
    return dict(policy=policy, case=case, episode_path=result['episode_path'], metrics=metrics,
                evaluator='image_plane_ground_truth/v1', evidence=evidence)


def test_end_to_end_paired_experiment_and_lineage(tmp_path):
    report = run_experiment(tmp_path, pairs=8)
    assert report['decision'] == 'inherit'
    assert report['n'] == 8 and report['mean_difference'] == 1
    assert report['p_value'] == pytest.approx(1/256)
    assert report['metrics']['task_success'] == [0, 1]
    assert report['metrics']['grasp_slippage'] == [None, None]
    assert report['metrics']['completion_time'][0] is None
    alias = json.loads((tmp_path/'policies/offline_default.json').read_text())
    assert alias['scope'] == 'offline_image_alignment'
    store = PolicyStore(tmp_path/'policies')
    assert store.restore(report['parent'])['parameters'] == {'gain': .18}
    assert store.load(alias['revision'])['parent'] == report['parent']
    assert not (tmp_path/'docs/memory/task_maps.json').exists() or not json.loads(
        (tmp_path/'docs/memory/task_maps.json').read_text())['maps']
    assert not TaskArchive(tmp_path).status()
    with pytest.raises(ValueError, match='empty output'):
        run_experiment(tmp_path)


def test_insufficient_samples_do_not_become_default(tmp_path):
    report = run_experiment(tmp_path, pairs=2)
    assert report['decision'] == 'inconclusive' and report['reason'] == 'insufficient_pairs'
    assert not (tmp_path/'policies/offline_default.json').exists()


def test_missing_pairs_and_duplicate_trials_are_not_silently_dropped(tmp_path):
    env = setup_protocol(tmp_path)
    _, _, _, harness, parent, _, comparison, _ = env
    row = trial(tmp_path, env, parent, 'val_0')
    harness.record_trial(comparison, **row)
    assert harness.select(comparison)['reason'] == 'incomplete_pairs'
    with pytest.raises(ValueError, match='duplicate trial'):
        harness.record_trial(comparison, **row)
    with pytest.raises(ValueError, match='not validated'):
        harness.promote_offline(comparison)


def test_evidence_tampering_rejected_at_admission_and_selection(tmp_path):
    env = setup_protocol(tmp_path, pairs=1)
    _, _, _, harness, parent, candidate, comparison, _ = env
    parent_row = trial(tmp_path, env, parent, 'val_0')
    candidate_row = trial(tmp_path, env, candidate, 'val_0')
    for row in (parent_row, candidate_row):
        harness.record_trial(comparison, **row)
    evidence_path = Path(candidate_row['evidence'][0]['path'])
    evidence_path.write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        harness.select(comparison)
    with pytest.raises(ValueError, match='evidence changed'):
        harness.record_trial(comparison, **candidate_row)


def test_wrong_parameters_costs_evaluator_and_mode_rejected(tmp_path):
    env = setup_protocol(tmp_path)
    _, _, _, harness, parent, _, comparison, _ = env
    row = trial(tmp_path, env, parent, 'val_0', override_gain=.9)
    with pytest.raises(ValueError, match='runtime parameters'):
        harness.record_trial(comparison, **row)
    row = trial(tmp_path, env, parent, 'val_0')
    bad = deepcopy(row)
    bad['metrics']['number_of_actions'] = 0
    with pytest.raises(ValueError, match='counters'):
        harness.record_trial(comparison, **bad)
    bad = dict(row, evaluator='policy_self_rating')
    with pytest.raises(ValueError, match='evaluator'):
        harness.record_trial(comparison, **bad)
    path = Path(row['episode_path'])
    episode = json.loads(path.read_text())
    episode['mode'] = 'hardware_assisted'
    atomic_json(path, episode)
    with pytest.raises(ValueError, match='episode does not bind'):
        harness.record_trial(comparison, **row)


def test_unknown_outcome_remains_inconclusive(tmp_path):
    env = setup_protocol(tmp_path, pairs=1)
    _, _, _, harness, parent, candidate, comparison, _ = env
    for policy in (parent, candidate):
        row = trial(tmp_path, env, policy, 'val_0')
        path = Path(row['episode_path'])
        episode = json.loads(path.read_text())
        episode['evaluation_result']['task_success'] = None
        episode['task_outcome'] = 'unknown'
        atomic_json(path, episode)
        receipt_path = Path(row['evidence'][0]['path'])
        atomic_json(receipt_path, episode['evaluation_result'])
        row['evidence'][0]['sha256'] = file_digest(receipt_path)
        row['metrics']['task_success'] = None
        harness.record_trial(comparison, **row)
    report = harness.select(comparison)
    assert report['decision'] == 'inconclusive' and report['reason'] == 'unknown_outcomes'


def test_validation_leakage_reuse_and_wrong_development_provenance_rejected(tmp_path):
    env = setup_protocol(tmp_path)
    _, _, _, harness, parent, candidate, _, conditions = env
    for dev, val, match in [(['dev_1'], ['dev_1'], 'leaked'),
                            (['dev_1'], ['val_0'], 'already claimed'),
                            (['other_dev'], ['new_val'], 'development episodes')]:
        with pytest.raises(ValueError, match=match):
            harness.preregister(parent=parent, candidate=candidate, development_cases=dev,
                                validation_cases=val, conditions=conditions)


def test_e_group_disallows_skill_changes_and_hypothesis_does_not_prove_cause(tmp_path):
    env = setup_protocol(tmp_path)
    _, _, store, _, parent, candidate, _, _ = env
    row = store.load(candidate)
    assert row['hypothesis']['causality'] == 'unproven'
    e_parent = store.freeze(name='fixed_e', parameters={'gain': .18}, skill_versions=row['skill_versions'],
                            ablation='E')
    with pytest.raises(ValueError, match='skill-changing'):
        store.freeze(name='forbidden', parameters={'gain': .8}, skill_versions=row['skill_versions'],
                     parent=e_parent, hypothesis=row['hypothesis'], ablation='E')
    assert ABLATIONS['E'] == dict(memory=True, fixed_skills=True, exploration=True, skill_changes=False)


def test_perception_failures_do_not_propose_grasp_or_gain_changes(tmp_path):
    path = tmp_path/'failed.json'
    atomic_json(path, dict(state='closed', failure_stage='detect_object', episode_id='episode',
                           failure_reason='target_missing'))
    proposal = propose_improvement([path])
    assert proposal['modification_target'] == 'perception_provider'
    assert proposal['cause_hypothesis'] == 'perception'
    assert proposal['causality'] == 'unproven'


def test_different_paired_start_states_rejected(tmp_path):
    env = setup_protocol(tmp_path, pairs=1)
    registry, archive, store, harness, parent, candidate, comparison, _ = env
    harness.record_trial(comparison, **trial(tmp_path, env, parent, 'val_0'))
    simulator = ImagePlaneSimulator(999)
    row = store.load(candidate)
    agent = Agent(registry, archive, simulator, memory_root=tmp_path, skills=row['skill_versions'])
    result = agent.execute_task(task='align_image', target_object='target', destination=[110., 90.],
                                scene_id='val_0', policy_version=candidate, gain=.65, max_actions=8)
    metrics, evidence = simulator.evaluate(result['episode_path'], [110., 90.])
    harness.record_trial(comparison, policy=candidate, case='val_0', episode_path=result['episode_path'],
                         metrics=metrics, evidence=evidence, evaluator='image_plane_ground_truth/v1')
    with pytest.raises(ValueError, match='initial conditions'):
        harness.select(comparison)


def test_candidate_failure_regressions_block_inheritance(tmp_path):
    env = setup_protocol(tmp_path)
    _, _, _, harness, parent, candidate, comparison, _ = env
    for index in range(8):
        for policy in (parent, candidate):
            row = trial(tmp_path, env, policy, f'val_{index}')
            # Supply an independent negative control result for both policies.
            path = Path(row['episode_path'])
            episode = json.loads(path.read_text())
            success = policy == parent
            episode['evaluation_result']['task_success'] = success
            episode['task_outcome'] = 'success' if success else 'failure'
            atomic_json(path, episode)
            evidence_path = Path(row['evidence'][0]['path'])
            atomic_json(evidence_path, episode['evaluation_result'])
            row['evidence'][0]['sha256'] = file_digest(evidence_path)
            row['metrics']['task_success'] = success
            harness.record_trial(comparison, **row)
    report = harness.select(comparison)
    assert report['decision'] == 'retain_parent' and len(report['regressions']) == 8
    with pytest.raises(ValueError, match='not validated'):
        harness.promote_offline(comparison)
