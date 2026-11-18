from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from astra_vla.agent.episode import Episode
from astra_vla.agent.experiment import ImagePlaneSimulator
from astra_vla.agent.memory import retrieve
from astra_vla.agent.runtime import Agent, ReadOnlyPiper, register_builtins
from astra_vla.agent.skills import SkillRegistry, SkillSpec, run_composition
from astra_vla.config import atomic_json
from astra_vla.task_archive import TaskArchive


def slip_skill(value, context, parameters):
    return {'status': 'slippage_detected'}


def recovery_skill(value, context, parameters):
    return {'status': 'recovered'}


def spec(function, skill_id='test', dependencies=()):
    return SkillSpec(skill_id, 'Test', '1.0.0', 'test callable contract',
                     {'type': 'object', 'properties': {}},
                     {'type': 'object', 'properties': {'status': {'type': 'string'}}, 'required': ['status']},
                     ('known observation',), ('explicit outcome',), ('unknown geometry',), {},
                     function.__module__+':'+function.__qualname__, (), ('offline only',), dependencies)


def test_skill_registry_restart_requires_trusted_binding_and_preserves_old_version(tmp_path):
    registry = SkillRegistry(tmp_path)
    old = registry.register(spec(slip_skill), slip_skill)
    new = registry.register(replace(spec(slip_skill), version='2.0.0'), slip_skill)
    assert old != new and registry.load(old)['spec']['version'] == '1.0.0'
    context = SimpleNamespace(skill_history=[])
    with pytest.raises(ValueError, match='trusted provider'):
        SkillRegistry(tmp_path).execute(old, {}, context)
    again = SkillRegistry(tmp_path)
    assert again.register(spec(slip_skill), slip_skill) == old
    assert again.execute(old, {}, context)['status'] == 'slippage_detected'
    assert context.skill_history[-1]['revision'] == old


def test_frozen_manifest_and_source_tampering_rejected(tmp_path):
    source = tmp_path/'closure.py'
    source.write_text('x = 1\n')
    registry = SkillRegistry(tmp_path/'registry')
    revision = registry.register(spec(slip_skill), slip_skill, source_files=[source])
    source.write_text('x = 2\n')
    with pytest.raises(ValueError, match='implementation changed'):
        registry.execute(revision, {}, SimpleNamespace(skill_history=[]))
    path = registry.directory/(revision+'.json')
    row = json.loads(path.read_text())
    row['spec']['version'] = 'tampered'
    atomic_json(path, row)
    with pytest.raises(ValueError, match='manifest changed'):
        registry.load(revision)


def test_slip_branches_to_recovery_not_transport_and_cycles_are_bounded(tmp_path):
    registry = SkillRegistry(tmp_path)
    slip = registry.register(spec(slip_skill), slip_skill)
    recover = registry.register(spec(recovery_skill, 'recover', (slip,)), recovery_skill)
    context = SimpleNamespace(skill_history=[], cancelled=lambda: False, recovery_count=0)
    graph = {'verify': dict(revision=slip, input={}, transitions={'slippage_detected': 'recover'}),
             'recover': dict(revision=recover, input={}, transitions={'recovered': None})}
    assert run_composition(registry, graph, 'verify', context)['status'] == 'recovered'
    assert context.recovery_count == 1
    assert [r['skill_id'] for r in context.skill_history] == ['test', 'recover']
    graph['recover']['transitions']['recovered'] = 'verify'
    assert run_composition(registry, graph, 'verify', context, budget=3)['status'] == 'budget_exhausted'
    context.cancelled = lambda: True
    assert run_composition(registry, graph, 'verify', context)['status'] == 'cancelled'


def setup_agent(tmp_path, adapter=None):
    archive = TaskArchive(tmp_path)
    archive.begin('offline test')
    registry = SkillRegistry(tmp_path/'registry')
    return Agent(registry, archive, adapter or ImagePlaneSimulator(5), memory_root=tmp_path)


def test_episode_task_ownership_unknown_outcomes_and_cleanup(tmp_path):
    archive = TaskArchive(tmp_path)
    with pytest.raises(ValueError, match='expected active'):
        archive.reserve('episode_1', 'agent', task_id='not_current')
    task = archive.begin('episode test')
    with pytest.raises(ValueError, match='expected active'):
        archive.reserve('episode_1', 'agent', task_id='not_current')
    episode = Episode(archive, task='inspect', scene_id='scene', policy_version='v1', skill_versions={},
                      memory_revision='none', model_identity='none', mode='offline')
    assert episode.record['task_id'] == task['task_id']
    assert TaskArchive.for_artifact(episode.path).root == tmp_path
    with pytest.raises(ValueError, match='收尾'):
        archive.finish()
    record = episode.close(status='succeeded')
    assert record['task_outcome'] == 'unknown' and record['model_judgment'] is None
    archive.finish(outcome='failure', reviewed=True, summary='离线测试；没有实体操作结论')
    assert not episode.path.exists()  # existing failure-retention policy still applies


def test_full_feedback_loop_reobserves_and_leaves_success_for_evaluator(tmp_path):
    agent = setup_agent(tmp_path)
    result = agent.execute_task(task='align_image', target_object='target', destination=[110., 90.],
                                gain=.65, max_actions=8)
    assert result['task_status'] == 'succeeded' and result['success'] is None
    episode = json.loads(open(result['episode_path']).read())
    assert episode['task_outcome'] == 'unknown'
    assert 1 < episode['action_count'] <= 8
    assert episode['observation_count'] > episode['action_count']
    assert episode['model_calls'] == 0 and episode['recovery_count'] == 0
    assert episode['policy_parameters'] == {'gain': .65}
    metrics, _ = agent.adapter.evaluate(result['episode_path'], [110., 90.])
    assert metrics['task_success'] is True
    assert metrics['placement_error'] is None and metrics['grasp_slippage'] is None


def test_action_budget_and_cancel_never_fabricate_outcome(tmp_path):
    agent = setup_agent(tmp_path)
    result = agent.execute_task(task='align_image', target_object='target', destination=[110., 90.],
                                gain=.1, max_actions=1)
    assert result['task_status'] == 'budget_exhausted' and result['success'] is None
    assert result['execution_metrics']['action_count'] == 1
    cancelled = agent.execute_task(task='inspect_scene', target_object='target', cancelled=lambda: True)
    assert cancelled['task_status'] == 'cancelled' and cancelled['observations'] == []


def test_execution_exception_is_recorded_and_task_can_finish(tmp_path):
    class Broken(ImagePlaneSimulator):
        def move(self, delta):
            raise RuntimeError('test feedback failure')
    agent = setup_agent(tmp_path, Broken(5))
    result = agent.execute_task(task='align_image', target_object='target', destination=[110., 90.])
    assert result['task_status'] == 'error' and result['success'] is None
    episode = json.loads(open(result['episode_path']).read())
    assert episode['failure_stage'] == 'align_image'
    assert episode['skill_history'][-1]['status'] == 'error'
    assert agent.archive.status()['artifacts'][-1]['state'] == 'closed'
    agent.archive.finish()


def test_readonly_adapter_does_not_allow_live_motion_or_pickle_state(tmp_path):
    adapter = ReadOnlyPiper(None, rpc=lambda *a, **k: pytest.fail('unexpected RPC'))
    with pytest.raises(RuntimeError, match='read-only'):
        adapter.move([.1, 0])
    agent = setup_agent(tmp_path, adapter)
    with pytest.raises(ValueError, match='qualification'):
        agent.execute_task(task='align_image', target_object='bottle', destination=[100, 80])
    with pytest.raises(ValueError, match='phase one'):
        agent.execute_task(task='pick_and_place', target_object='bottle', destination='cup')
    assert not agent.archive.status()['artifacts']


def test_memory_selective_retrieval_never_reads_demonstrations_or_mutates(tmp_path, monkeypatch):
    from astra_vla.task_memory import TaskMemory
    monkeypatch.setattr(TaskMemory, 'demonstrations', lambda *a, **k: pytest.fail('demos excluded'))
    path = tmp_path/'docs/memory/state.json'
    atomic_json(path, {'sessions': [], 'problems': [
        {'topic': 'bottle slippage', 'observed': 'relative motion', 'inference': 'grasp unknown',
         'change': 'reobserve', 'validation': 'not measured'},
        {'topic': 'unrelated firmware', 'observed': 'unrelated'}]})
    before = path.read_bytes()
    result = retrieve(tmp_path, task='bottle', objects=['bottle'], limit=1)
    assert len(result['entries']) == 1
    assert result['entries'][0]['content']['topic'] == 'bottle slippage'
    assert result['entries'][0]['applicability'] == 'assessment_required'
    assert path.read_bytes() == before and not result['demonstrations_included']


def test_register_builtin_descriptors_are_callable_and_bounded(tmp_path):
    registry = SkillRegistry(tmp_path)
    revisions = register_builtins(registry)
    assert {'observe_scene', 'detect_object', 'estimate_object_orientation', 'inspect_grasp',
            'estimate_visual_error', 'align_image'} == set(revisions)
    for revision in revisions.values():
        record = registry.load(revision)
        assert record['sources'] and record['spec']['input_schema'] and record['spec']['output_schema']


def test_repeated_frame_is_counted_but_never_drives_another_action(tmp_path):
    class Repeating(ImagePlaneSimulator):
        last = None
        def observe(self, directory):
            if self.last is None:
                self.last = super().observe(directory)
            return self.last
    agent = setup_agent(tmp_path, Repeating(5))
    result = agent.execute_task(task='align_image', target_object='target', destination=[110., 90.])
    episode = json.loads(open(result['episode_path']).read())
    assert result['task_status'] == 'needs_observation'
    assert episode['action_count'] == 1 and episode['observation_count'] == 2
    assert len(episode['observation_history']) == 1
    assert result['success'] is None
