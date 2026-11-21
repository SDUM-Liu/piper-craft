"""Deterministic, isolated dual-arm workflow fixture, with rendered source evidence."""
from email.message import Message
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np

from astra_vla.config import atomic_json
from astra_vla.task_archive import TaskArchive
from .contracts import CAMERAS, canonical
from .perception import ColorRegions, observe_scene
from .responses import Responses
from .session import task_session

TASK = 'Pick up the shuttlecock from the desk, place it head-first in the tube, withdraw, and record both phases.'
SIM_GUIDE = (
    ' This is an isolated abstract replay simulator. All joints beyond J1 are zero. Left J1=10 supports the tube. '
    'Right J1=15 approaches the shuttle head; close the right gripper there. Right J1=20 lifts and briefly occludes '
    'the grasp. The pale ball and tail mark the shuttle, two pale blue bars are the gripper, and the green '
    'outline is the tube. At the lift, if a gray block hides the shuttle, request a fresh observation and fetch '
    'its three frames. If the shuttle is visibly between the two gripper bars, use judge for held_after_lift '
    'immediately, cite that observation and its source times, then advance to insertion. '
    'In insertion, right J1=30 aligns the head above the tube; then open the gripper. '
    'In verify, right J1=40 withdraws the empty gripper. Fetch current frames and use judge for shuttle_in_tube '
    'if they show the shuttle staying inside the green outline. The fixture starts video late; finish the '
    'physical simulation and report the missing pickup coverage in model_result. '
    'These are simulator command landmarks only.'
)
CONTRACT = {'phases': ['pickup', 'insertion', 'verify'],
            'visual_conditions': ['held_after_lift', 'shuttle_in_tube'],
            'phase_conditions': {'pickup': ['held_after_lift'], 'verify': ['shuttle_in_tube']},
            'record_phases': ['pickup', 'insertion']}


class ShuttleSimulator:
    mode = 'offline'
    actions_enabled = True
    max_age = 60.0
    state_age = 1.0

    def __init__(self, archive, *, autonomous_frames=False):
        self.archive = archive
        self.time = time.time()
        self.sequence = 1
        self.joints = {'left': [0.]*6, 'right': [0.]*6}
        self.closed = False
        self.location = 'desk'
        self.hidden = False
        self.autonomous_frames = autonomous_frames
        self.last_observed_sequence = None
        self.executed = []
        self.cancel_count = 0
        self.recording = None
        self.record_start = None
        self.provider = ColorRegions({})

    def clock(self):
        return self.time

    def advance(self):
        self.time += .25
        self.sequence += 1

    def status(self):
        return dict(source='shuttle_simulator', fault=None, stopped=False, stop_confirmed=True,
                    arms={arm: dict(stamp=self.time, feedback_coherent=True,
                                    controller={'ctrl_mode': 1, 'move_mode': 1, 'err_code': 0},
                                    joints_rad=np.radians(value).tolist(),
                                    gripper={'width_m': .0 if arm == 'right' and self.closed else .06})
                          for arm, value in self.joints.items()},
                    teleop={'active': True, 'coordinate_mode': 'base',
                            'execution_backend': 'simulated_joint_target', 'phase': 'control'})

    def observe(self, directory):
        if self.autonomous_frames and self.sequence == self.last_observed_sequence:
            self.reveal()
        self.last_observed_sequence = self.sequence
        directory.mkdir(parents=True, exist_ok=False)
        rows = {}
        for index, camera in enumerate(CAMERAS):
            img = np.full((144, 224, 3), (38, 61, 79), dtype=np.uint8)
            cv2.rectangle(img, (154, 49), (191, 108), (75, 153, 35), thickness=3)
            cv2.putText(img, camera, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (210, 210, 210), 1)
            gripper = {0: 61, 15: 72, 20: 100, 30: 170, 40: 114}[self.joints['right'][0]]
            cv2.line(img, (gripper-14, 30), (gripper-14, 78), (200, 190, 135), 5)
            cv2.line(img, (gripper+14, 30), (gripper+14, 78), (200, 190, 135), 5)
            if self.joints['left'][0]:
                cv2.line(img, (198, 36+index), (198, 110), (102, 207, 194), 6)
            if self.hidden:
                cv2.rectangle(img, (83, 52), (117, 101), (84, 85, 91), thickness=-1)
            else:
                center = {'desk': (72, 112), 'held': (gripper, 67), 'tube': (171, 83)}[self.location]
                cv2.circle(img, center, 6, (255, 255, 255), -1)
                cv2.line(img, (center[0]-6, center[1]-15), center, (216, 213, 203), 4)
            path = directory/(camera+'.png')
            if not cv2.imwrite(str(path), img):
                raise OSError('could not write synthetic source PNG')
            rows[camera] = dict(available=True, lossless=True, resized=False, width=224, height=144,
                                stamp=self.time, sequence=self.sequence, serial='sim_'+camera,
                                path=str(path.resolve()))
        atomic_json(directory/'observation.json', rows)
        return observe_scene(rows, self.status(), self.provider, now=self.time)

    def execute(self, action, *, cancelled):
        if cancelled():
            return {'status': 'cancelled', 'arrived': None}
        self.executed.append({key: action[key] for key in ('id', 'phase', 'kind', 'arm')})
        arm, kind = action['arm'], action['kind']
        if kind == 'move_arm':
            joints = action['joints_deg']
            valid = (0, 10) if arm == 'left' else (0, 15, 20, 30, 40)
            if joints[0] not in valid or any(joints[1:]):
                return {'status': 'rejected', 'arrived': False}
            self.joints[arm] = joints
        elif arm == 'right':
            self.closed = action['command'] == 'close'
            if self.closed and self.joints[arm][0] == 15 and self.location == 'desk':
                self.location = 'held'
            if not self.closed and self.location == 'held':
                self.location = ('tube' if self.joints[arm][0] == 30 and self.joints['left'][0] == 10 else 'desk')
        self.hidden = self.location == 'held' and self.joints['right'][0] == 20
        self.advance()
        if action['phase'] == 'pickup' and self.hidden and self.recording is None:
            self.recording = self.archive.reserve('window_simulated', 'window', simulated=True,
                                                  recording_state='recording', started_at=self.time)
            self.record_start = self.time
        if action['phase'] == 'verify' and self.recording:
            self.archive.update(self.recording, state='closed', result=dict(
                closed=True, frames=20, start_stamp=self.record_start, first_frame_s=0.,
                last_frame_s=self.time-self.record_start, simulated=True))
            self.recording = None
        return {'status': 'completed', 'arrived': True if kind == 'move_arm' else None,
                'source': 'shuttle_simulator', 'source_time': self.time}

    def reveal(self):
        self.hidden = False
        self.advance()

    def cancel(self):
        self.cancel_count += 1

    def evaluate(self, session):
        success = self.location == 'tube' and not self.closed and self.joints['right'][0] == 40
        evaluation = dict(evaluator='shuttle_simulator_ground_truth_v1', task_success=success,
                          episode_id=session.episode.record['episode_id'], final_location=self.location,
                          gripper_closed=self.closed, right_withdrawn=self.joints['right'][0] == 40)
        session.state.data['independent_result'] = 'success' if success else 'failure'
        session.state.save()
        session.episode.record.update(evaluation_result=evaluation,
                                      task_outcome=session.state.data['independent_result'])
        session.episode.flush()
        return evaluation


class ScriptedTransport:
    """Deterministic Responses/SSE double. It exercises the production stream parser."""

    def __init__(self, sim, steps, *, context_mode, break_after=None):
        self.sim, self.steps, self.context_mode, self.break_after = sim, list(steps), context_mode, break_after
        self.index = 0
        self.requests = []

    def __call__(self, url, *, body, **kwargs):
        assert url == 'https://api.openai.com/v1/responses'
        assert body['stream'] is True and body['store'] is False
        context = json.loads(body['input'][0]['content'])['state']
        self.requests.append(context)
        if self.index == self.break_after:
            events = [{'type': 'response.output_item.done',
                       'item': self.tool_call('move_arm', dict(arm='right', joints_deg=[99, 0, 0, 0, 0, 0],
                                                               purpose='must never reach backend'))}]
        else:
            if self.index < len(self.steps):
                name, value = self.steps[self.index]
                args = value(context, self.sim) if callable(value) else value
                output = [self.tool_call(name, args)]
            else:
                output = [{'type': 'message', 'role': 'assistant', 'content': [
                    {'type': 'output_text', 'text': 'Scripted fixture complete.'}]}]
            events = [{'type': 'response.completed', 'response': {'status': 'completed', 'output': output}}]
        self.index += 1
        response = io.BytesIO(b''.join(b'data: '+canonical(event)+b'\n\n' for event in events))
        response.headers = Message()
        response.headers['Content-Type'] = 'text/event-stream'
        return response

    def tool_call(self, name, args):
        return dict(type='function_call', call_id=f'fixture_{self.index}', namespace='piper', name=name,
                    arguments=json.dumps(args))


class FixtureGrant:
    def token(self, account):
        return 'fixture_only_no_network'


def move(arm, angle, purpose):
    return ('move_arm', dict(arm=arm, joints_deg=[angle, 0, 0, 0, 0, 0], purpose=purpose))


def plan(phase):
    return ('plan', dict(phase=phase, left='auxiliary observation and tube support', right='pick and place'))


def observe(fresh=False):
    return ('observe', dict(fresh=fresh))


def frames(context, sim):
    return dict(observation_id=context['observation']['id'], cameras=list(CAMERAS))


def judge(condition, answer, explanation):
    def args(context, sim):
        observation = context['observation']
        return dict(condition=condition, answer=answer, explanation=explanation, observation_id=observation['id'],
                    evidence=[dict(camera=name, source_time=observation['views'][name]['timestamp'])
                              for name in CAMERAS])
    return ('judge', args)


def reveal(context, sim):
    sim.reveal()
    return {'fresh': True}


def pickup_steps(context_mode):
    return [observe(context_mode == 'full_history'), plan('pickup'), move('left', 10, 'establish tube support'),
            observe(), move('right', 15, 'approach shuttle head'), observe(),
            ('gripper', dict(arm='right', command='close', purpose='simulated reviewed pickup')),
            observe(), move('right', 20, 'short lift')]


def insertion_steps(context_mode):
    return [observe(context_mode == 'full_history'), ('observe', reveal), ('frames', frames),
            judge('held_after_lift', 'yes', 'The simulator source views show the ball between the fingers.'),
            plan('insertion'), move('right', 30, 'align head over supported tube'), observe(),
            ('gripper', dict(arm='right', command='open', purpose='release into tube')), observe(),
            plan('verify'), move('right', 40, 'withdraw empty fingers'), observe(), ('frames', frames),
            judge('shuttle_in_tube', 'yes', 'The shuttle remains inside the tube after withdrawal.'),
            ('model_result', dict(outcome='succeeded', explanation='Visual result; pickup video remains missing.'))]


def run_fixture(output, *, remote_model=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for context_mode in ('snapshot', 'full_history') if remote_model is None else ('snapshot',):
        archive = TaskArchive(output/context_mode)
        task = archive.begin('Isolated shuttlecock workflow', TASK+(SIM_GUIDE if remote_model else ''))
        sim = ShuttleSimulator(archive, autonomous_frames=remote_model is not None)
        first = pickup_steps(context_mode)
        if remote_model is None:
            runs = [(first, len(first)), (insertion_steps(context_mode), None)]
        else:
            runs = [(None, None)]
        episodes, states, transport_states = [], [], []
        for steps, stop in runs:
            if episodes:
                sim.advance()  # next camera sequence while resuming; physical state unchanged
            if remote_model is None:
                transport = ScriptedTransport(sim, steps, context_mode=context_mode, break_after=stop)
                model = Responses(FixtureGrant(), 'fixture', 'scripted_responses_fixture', open_stream=transport)
                model.identity = {'provider': 'scripted_responses_fixture', 'model': 'deterministic'}
            else:
                model = remote_model
            with task_session(archive, task['task_id'], model, CONTRACT, sim, clock=sim.clock,
                              memory_root=archive.root) as session:
                result = session.run(context_mode=context_mode, max_turns=48 if remote_model else 32)
                states.append(result['state'])
                episodes.append(session.episode.record)
                if remote_model is None:
                    transport_states.append(transport.requests)
                if stop is None:
                    evaluation = sim.evaluate(session)
        end = states[-1]
        omissions = None
        recovery = None
        if remote_model is None:
            cutoff, resume = states[0], transport_states[1][0]
            recovery = dict(
                arrival_without_hold=(cutoff['last_action']['arrived'] is True
                                      and 'held_after_lift' in cutoff['pending_visual']),
                pickup_video_missing=end['video']['coverage']['pickup'] == 'missing',
                insertion_video_covered=end['video']['coverage']['insertion'] == 'covered',
                resume_phase=resume['phase'], resume_pending=resume['pending_visual'],
                resumed_with_fresh_source=resume['observation']['id'] != cutoff['observation']['id'],
                repeated_action_count=sum(row['phase'] == 'pickup' and row['kind'] == 'move_arm'
                                          for row in sim.executed) - 3)
            omissions = sum(not recovery[key] for key in ('arrival_without_hold', 'pickup_video_missing',
                                                          'insertion_video_covered', 'resumed_with_fresh_source'))
            omissions += int(recovery['resume_phase'] != 'pickup' or
                             'held_after_lift' not in recovery['resume_pending'])
        reports[context_mode] = dict(task_id=task['task_id'], execution_status=result['status'],
                                     model_identity=model.identity, independent_evaluation=evaluation,
                                     video=end['video'], recovery=recovery, omission_count=omissions,
                                     text_context_bytes_by_turn=[turn['context_bytes'] for ep in episodes
                                                                 for turn in ep['turns']],
                                     request_bytes_by_turn=[turn['input_bytes'] for ep in episodes for turn in ep['turns']],
                                     captures=sum(ep['metrics']['capture_requests'] for ep in episodes),
                                     duplicate_captures=sum(ep['metrics']['duplicate_captures'] for ep in episodes),
                                     cache_reuses=sum(ep['metrics']['cache_reuses'] for ep in episodes))
    atomic_json(output/'report.json', reports)
    return reports
