"""Task-scoped, durable projection of observations, actions and deliverables."""
from dataclasses import asdict
import json
import time

from astra_vla.config import atomic_json
from .contracts import identifier, unique


class TaskState:
    def __init__(self, archive, task_id, contract, *, clock=time.time):
        self.archive, self.task_id, self.clock = archive, task_id, clock
        self.path = archive.path(task_id)/'agent_state.json'
        self.data = self._load(contract)

    def _load(self, contract):
        phases = contract['phases']
        facts = contract['visual_conditions']
        if any(not isinstance(item, list) for item in (phases, facts, contract['record_phases'])):
            raise ValueError('phases and requirements must be lists')
        if not phases or len(set(phases)) != len(phases) or any(not identifier(x) for x in (*phases, *facts)):
            raise ValueError('unique task phases and named visual conditions required')
        if any(phase not in phases for phase in contract['record_phases']):
            raise ValueError('unknown recording phase')
        phase_conditions = contract.get('phase_conditions', {})
        assigned = [condition for names in phase_conditions.values() for condition in names]
        if (set(phase_conditions) - set(phases) or set(assigned) - set(facts)
                or len(assigned) != len(set(assigned))):
            raise ValueError('invalid phase-specific visual conditions')
        for phase, actions in contract.get('action_conditions', {}).items():
            if phase not in phases or set(actions) - {'move_arm', 'gripper'} or any(
                    not isinstance(names, list) or not names or set(names)-set(facts)
                    for names in actions.values()):
                raise ValueError('invalid desktop motion conditions')
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data['task_id'] != self.task_id or data['contract'] != contract:
                raise ValueError('persisted task contract differs')
            return data
        data = dict(schema_version=1, task_id=self.task_id, contract=contract,
                    phase=phases[0], roles={'left': 'unassigned', 'right': 'unassigned'},
                    device={}, observation=None, facts={}, last_action=None, motion_intervals={},
                    video={'state': 'unknown', 'coverage': {}}, model_judgment=None,
                    independent_result='unknown', pending_action=None, resume_check=True)
        atomic_json(self.path, data)
        return data

    def save(self):
        atomic_json(self.path, self.data)

    def require_recheck(self, cause):
        self.invalidate(cause)
        self.data['resume_check'] = True
        if self.data['pending_action']:
            pending = self.data['pending_action']
            self.data['last_action'] = {**pending, 'status': 'unknown', 'arrived': None}
            self.track_interval(pending)
            self.data['pending_action'] = None
        self.save()

    def invalidate(self, cause):
        for fact in self.data['facts'].values():
            if fact['state'] != 'historical':
                fact.update(state='needs_review', reason=cause)

    def update_observation(self, scene, episode_id):
        self.data['observation'] = dict(id=scene.observation_id, source_time=scene.timestamp,
                                        episode=episode_id,
                                        views={name: asdict(frame) for name, frame in scene.views.items()})
        self.data['resume_check'] = False
        self.save()

    def set_plan(self, phase, left, right):
        phases = self.data['contract']['phases']
        current, requested = phases.index(self.data['phase']), phases.index(phase)
        if requested > current + 1:
            raise ValueError('phase transition skips an intermediate phase')
        if self.data['resume_check']:
            raise ValueError('check current observation and device state first')
        if any(not 0 < len(role.strip()) <= 120 for role in (left, right)):
            raise ValueError('both arm roles must be explicit')
        conditions = self.data['contract'].get('phase_conditions', {})
        if requested > current:
            for name in conditions.get(phases[current], []):
                if not self.confirmed(name):
                    raise ValueError('current phase still needs evidence: '+name)
            for name in conditions.get(phases[current], []):
                self.data['facts'][name]['state'] = 'historical'
        elif requested < current:
            for phase in phases[requested:current]:
                for name in conditions.get(phase, []):
                    if name in self.data['facts']:
                        self.data['facts'][name].update(state='needs_review', reason='returned_to_phase')
        self.data.update(phase=phase, roles={'left': left, 'right': right})
        self.save()

    def record_judgment(self, condition, answer, observation_id, evidence, explanation, max_age):
        if condition not in self.data['contract']['visual_conditions']:
            raise ValueError('unknown visual condition')
        if not 0 < len(explanation.strip()) <= 800:
            raise ValueError('visual judgment explanation must contain 1–800 characters')
        if self.data['facts'].get(condition, {}).get('state') == 'historical':
            raise ValueError('return to the phase before revising a historical judgment')
        obs = self.data['observation']
        if not obs or obs['id'] != observation_id or self.data['resume_check']:
            raise ValueError('use a current observation')
        if not evidence or len({item['camera'] for item in evidence}) != len(evidence):
            raise ValueError('cite at least one distinct source frame')
        refs = []
        for citation in evidence:
            frame = obs['views'][citation['camera']]
            if (citation['source_time'] != frame['timestamp'] or
                    not 0 <= self.clock()-frame['timestamp'] <= max_age):
                raise ValueError('source timestamp mismatch or expired frame')
            refs.append({k: frame[k] for k in ('camera_id', 'timestamp', 'sequence', 'sha256')})
        old = self.data['facts'].get(condition)
        conflict = (old and old['observation_id'] == observation_id and
                    (old.get('reason') == 'conflicting_judgment' or
                     old['answer'] not in ('unknown', answer) and answer != 'unknown'))
        self.data['facts'][condition] = dict(source='model_visual_judgment', answer=answer,
                                            state='needs_review' if conflict else 'accepted',
                                            reason='conflicting_judgment' if conflict else None,
                                            observation_id=observation_id, frames=refs, explanation=explanation,
                                            recorded_at=self.clock())
        self.save()
        return self.data['facts'][condition]

    def refresh(self, device, *, max_age):
        self.data['device'] = device
        for fact in self.data['facts'].values():
            if fact['state'] == 'accepted' and any(
                    not 0 <= self.clock()-frame['timestamp'] <= max_age for frame in fact['frames']):
                fact.update(state='needs_review', reason='expired')
        self._recording()
        self.save()

    def plan_action(self, kind, arguments, *, episode_id=None):
        if self.data['pending_action']:
            raise ValueError('previous action needs status reconciliation')
        oldest_frame = min(frame['timestamp'] for frame in self.data['observation']['views'].values())
        action = dict(id=unique('action'), phase=self.data['phase'], kind=kind, **arguments,
                      started=self.clock(), observation_time=oldest_frame, episode_id=episode_id)
        self.data['pending_action'] = action
        self.data['last_action'] = {**action, 'status': 'pending', 'arrived': None}
        self.invalidate('subsequent_action')
        self.data['resume_check'] = True
        self.save()
        return action

    def finish_action(self, action, result):
        if self.data['pending_action'] != action:
            raise ValueError('action result does not match the pending action')
        self.data['last_action'] = {**action, **result}
        self.track_interval(action)
        self.data['pending_action'] = None
        self.save()

    def track_interval(self, action):
        opened, closed = self.data['motion_intervals'].get(action['phase'], (action['started'], 0))
        self.data['motion_intervals'][action['phase']] = [min(opened, action['started']), max(closed, self.clock())]

    def confirmed(self, name, *, current_only=False):
        row = self.data['facts'].get(name, {})
        states = ('accepted',) if current_only else ('accepted', 'historical')
        return row.get('state') in states and row.get('answer') == 'yes'

    def _recording(self):
        rows = [row for row in self.archive.read(self.task_id)['artifacts'] if row['kind'] == 'window']
        closed = [(r['result']['start_stamp'] + r['result'].get('first_frame_s', 0.),
                   r['result']['start_stamp'] + r['result']['last_frame_s'])
                  for r in rows if r.get('result', {}).get('closed') is True and
                  r['result'].get('frames', 0) > 0 and 'last_frame_s' in r['result']]
        active = [r for r in rows if r['state'] == 'open']
        state = ('recording' if any(r.get('recording_state') == 'recording' for r in active)
                 else 'pending' if active else 'saved' if closed else 'idle' if not rows else 'error')
        coverage = {}
        for phase in self.data['contract']['record_phases']:
            interval = self.data['motion_intervals'].get(phase)
            if not interval:
                coverage[phase] = 'unknown'
                continue
            start, end = interval
            coverage[phase] = ('covered' if any(a <= start <= end <= b for a, b in closed)
                               else 'unknown' if any(r['started_at'] <= start for r in active)
                               else 'missing')
        self.data['video'] = dict(state=state, coverage=coverage, references=[row['path'] for row in rows[-4:]])

    def context(self):
        data = self.data
        obs = data['observation']
        pending = [name for name in data['contract']['visual_conditions'] if not self.confirmed(name)]
        by_phase = data['contract'].get('phase_conditions', {})
        assigned = {name for conditions in by_phase.values() for name in conditions}
        current = by_phase.get(data['phase'], ())
        pending_current = [name for name in pending if name in current or name not in assigned]
        return dict(task_id=self.task_id, phase=data['phase'], roles=data['roles'], device=data['device'],
                    last_action=data['last_action'], video=data['video'], visual_judgments=data['facts'],
                    pending_visual=pending, pending_current_visual=pending_current,
                    pending_video=[p for p, status in data['video']['coverage'].items()
                                                           if status != 'covered'],
                    observation={**obs, 'views': {camera: {k: v for k, v in frame.items() if k != 'path'}
                                                  for camera, frame in obs['views'].items()}} if obs else None,
                    needs_current_observation=data['resume_check'], independent_result=data['independent_result'])
