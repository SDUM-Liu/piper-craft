"""Episode records belong to TaskArchive; independent outcomes stay separate."""
from copy import deepcopy
import time

from astra_vla.config import atomic_json
from .contracts import unique


class Episode:
    def __init__(self, archive, *, task, scene_id, policy_version, skill_versions,
                 memory_revision, model_identity, mode):
        active = archive.status()
        if active is None:
            raise ValueError('begin a TaskArchive before an Agent episode')
        if mode not in ('offline', 'live_readonly', 'hardware_autonomous', 'hardware_assisted'):
            raise ValueError('explicit execution mode required')
        self.archive = archive
        self.started = time.monotonic()
        episode_id = unique('episode')
        self.path = archive.reserve(episode_id, 'agent', task_id=active['task_id'])
        self.path.mkdir(parents=True, exist_ok=False)
        self.record = dict(schema_version=1, episode_id=episode_id, task_id=active['task_id'], task=task,
                           scene_id=scene_id, policy_version=policy_version, skill_versions=skill_versions,
                           memory_revision=memory_revision, model_identity=model_identity, mode=mode,
                           initial_observation=None, observation_history=[], action_history=[], robot_feedback=[],
                           skill_history=[], task_outcome='unknown', model_judgment=None, failure_stage=None,
                           failure_reason=None, execution_time=None, model_calls=0, action_count=0,
                           observation_count=0, recovery_count=0, intervention_count=0,
                           evidence_references=[], evaluation_result=None, state='open')
        self.flush()

    def flush(self):
        atomic_json(self.path/'episode.json', self.record)

    def observation(self, scene):
        row = scene.to_dict()
        self.record['initial_observation'] = self.record['initial_observation'] or row
        self.record['observation_history'].append(row)
        self.record['robot_feedback'].append(deepcopy(scene.robot_state))
        self.record['evidence_references'].extend(deepcopy(list(row['views'].values())))
        self.flush()

    def action(self, proposal, feedback):
        entry = dict(proposal=deepcopy(proposal), feedback=deepcopy(feedback), status='attempted')
        self.record['action_history'].append(entry)
        self.record['action_count'] += 1
        self.flush()
        return entry

    def close(self, *, status, reason=None, stage=None, skill_history=(), recovery_count=0):
        # Execution status NEVER supplies task outcome. Evaluator attaches it separately.
        self.record.update(state='closed', execution_status=status, failure_reason=reason, failure_stage=stage,
                           execution_time=time.monotonic()-self.started, skill_history=list(skill_history),
                           recovery_count=recovery_count)
        self.flush()
        self.archive.update(self.path, state='closed', episode_id=self.record['episode_id'])
        return deepcopy(self.record)
