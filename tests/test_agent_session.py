import base64
import hashlib
import io
import json
import threading
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from astra_vla.agent import chatgpt
from astra_vla.agent.desktop_control import DesktopBackend, DesktopRequest
from astra_vla.agent.responses import Interrupted, Responses, completed_turn, sse_events
from astra_vla.agent.session import task_session
from astra_vla.agent.shuttle_sim import CONTRACT, FixtureGrant, ScriptedTransport, ShuttleSimulator, plan, run_fixture
from astra_vla.agent.view_validation import pinned_skills, validate_views
from astra_vla.task_archive import TaskArchive


def test_shuttle_session_compares_resume_coverage_and_context(tmp_path):
    report = run_fixture(tmp_path/'fixture')
    snapshot, full = report['snapshot'], report['full_history']
    assert snapshot['independent_evaluation']['task_success'] is True
    assert snapshot['video']['coverage'] == {'pickup': 'missing', 'insertion': 'covered'}
    assert snapshot['recovery'] == full['recovery']
    assert snapshot['recovery']['arrival_without_hold']
    assert snapshot['recovery']['resume_phase'] == 'pickup'
    assert snapshot['recovery']['resumed_with_fresh_source']
    assert snapshot['recovery']['repeated_action_count'] == 0
    assert snapshot['omission_count'] == full['omission_count'] == 0
    assert sum(snapshot['text_context_bytes_by_turn']) < sum(full['text_context_bytes_by_turn'])
    assert snapshot['duplicate_captures'] == 0 < full['duplicate_captures']
    task_dir = tmp_path/'fixture'/'snapshot'/'runs'/snapshot['task_id']
    state = json.loads((task_dir/'agent_state.json').read_text())
    assert state['independent_result'] == 'success'
    assert state['facts']['held_after_lift']['state'] == 'historical'
    for episode in (task_dir/'episodes').glob('*/episode.json'):
        record = json.loads(episode.read_text())
        assert record['skill_versions'] and record['state'] == 'closed'
        if record['execution_status'] == 'interrupted':
            assert 'missing_response_completed' in record['failure_reason']
            assert all(row['proposal'].get('joints_deg', [0])[0] != 99 for row in record['action_history'])


def make_session(tmp_path, steps, *, cancelled=lambda: False):
    archive = TaskArchive(tmp_path)
    task = archive.begin('test', 'visual task')
    sim = ShuttleSimulator(archive)
    transport = ScriptedTransport(sim, steps, context_mode='snapshot')
    model = Responses(FixtureGrant(), 'fixture', 'fixture', open_stream=transport)
    return sim, task_session(archive, task['task_id'], model, CONTRACT, sim, clock=sim.clock,
                             cancelled=cancelled)


def test_invalid_arm_target_and_visual_reference_are_rejected(tmp_path):
    steps = [plan('pickup'),
             ('move_arm', dict(arm='left', joints_deg=[1], purpose='invalid')),
             ('gripper', dict(arm='both', command='close', purpose='invalid')),
             ('judge', lambda state, sim: dict(condition='held_after_lift', answer='yes',
                                                observation_id=state['observation']['id'],
                                                evidence=[dict(camera='top', source_time=sim.clock())],
                                                explanation='image never delivered'))]
    sim, context = make_session(tmp_path, steps)
    with context as session:
        session.run()
        rows = session.episode.record['tool_events']
        assert [row['result']['status'] for row in rows] == ['saved', 'rejected', 'rejected', 'rejected']
        assert sim.executed == []
        assert session.state.data['facts'] == {}


def test_action_cancellation_preserves_unknown_receipt(tmp_path):
    flag = threading.Event()
    args = dict(arm='left', joints_deg=[10, 0, 0, 0, 0, 0], purpose='sim test')
    sim, context = make_session(tmp_path, [plan('pickup'), ('move_arm', args)], cancelled=flag.is_set)
    execute = sim.execute

    def interrupted(action, *, cancelled):
        receipt = execute(action, cancelled=cancelled)
        flag.set()
        return receipt

    sim.execute = interrupted
    with context as session:
        result = session.run()
        assert result['status'] == 'cancelled'
        assert result['state']['last_action']['status'] == 'unknown'
        assert result['state']['last_action']['arrived'] is None
        assert session.state.data['pending_action'] is None
        assert len(sim.executed) == 1 and sim.cancel_count == 1


def test_stale_and_conflicting_visual_judgments(tmp_path):
    sim, context = make_session(tmp_path, [])
    with context as session:
        session.refresh()
        session.observe(True)
        observation = session.state.data['observation']
        citations = [dict(camera='top', source_time=observation['views']['top']['timestamp'])]
        args = ['held_after_lift', observation['id'], citations, 'reviewed', 1.]
        def submit(answer):
            condition, observation_id, evidence, explanation, age = args
            return session.state.record_judgment(condition, answer, observation_id, evidence, explanation, age)
        assert submit('yes')['state'] == 'accepted'
        assert submit('no')['state'] == 'needs_review'
        assert submit('no')['state'] == 'needs_review'
        sim.time += 2
        with pytest.raises(ValueError, match='expired frame'):
            submit('yes')
        session.episode.close(status='completed')


def test_open_episode_recovery_refuses_unknown_action_repeat(tmp_path):
    sim, context = make_session(tmp_path, [])
    with context as initial:
        initial.refresh()
        initial.observe(True)
        initial.state.set_plan('pickup', 'support', 'pickup')
        args = dict(arm='left', joints_deg=[10, 0, 0, 0, 0, 0], purpose='sim test')
        pending = initial.state.plan_action('move_arm', args)
        initial.episode.action(pending, {})
        former = initial.episode.path
        task_id = initial.state.task_id
    sim.advance()
    model = Responses(FixtureGrant(), 'fixture', 'fixture', open_stream=ScriptedTransport(
        sim, [('move_arm', args)], context_mode='snapshot'))
    with task_session(sim.archive, task_id, model, CONTRACT, sim, clock=sim.clock) as resumed:
        result = resumed.run()
        assert result['state']['last_action']['status'] == 'unknown'
        assert resumed.episode.record['tool_events'][0]['result']['status'] == 'rejected'
        assert resumed.state.data['pending_action'] is None
    prior = json.loads((former/'episode.json').read_text())
    assert prior['execution_status'] == 'interrupted'
    assert prior['action_history'][0]['status'] == 'unknown'
    assert not sim.executed


def test_desktop_needs_actual_focus_and_correlated_arrival(tmp_path):
    from types import SimpleNamespace

    now = time.time()
    sim = ShuttleSimulator(TaskArchive(tmp_path))
    robot = sim.status()
    for arm in robot['arms'].values():
        arm['controller']['arm_status'] = 0
    robot['teleop'].update(session='existing_desktop', arm='left', active=True, stop_confirmed=True)
    sent = []
    client = SimpleNamespace(lock=threading.Lock(), rpc=lambda *a, **kw: pytest.fail('unexpected RPC'),
                             session='existing_desktop', robot=robot, frames={name: {'stamp': now} for name in
                                                                            ('top', 'left_wrist', 'right_wrist')},
                             control_active=True, stop_pending=False, opening=False, pending_coordinate_mode=None,
                             cancel_detour=False, execution_backend='regularized_move_j',
                             request_joint_target=lambda *a, **kw: sent.append((a, kw)))
    bridge = DesktopBackend(client, threading.Event())
    action = dict(id='action_a', kind='move_arm', arm='left', joints_deg=[10, 0, 0, 0, 0, 0],
                  observation_time=now)
    denied = DesktopRequest(action)
    bridge.request = denied
    bridge.pump(focused=False, menu=False, arm='left', manual_keys=set())
    assert denied.result['status'] == 'rejected' and not sent
    accepted = DesktopRequest(action)
    bridge.request = accepted
    bridge.pump(focused=True, menu=False, arm='left', manual_keys=set())
    assert sent == [(('left', action['joints_deg']), {'request_id': 'action_a'})]
    robot['teleop']['joint_target'] = dict(request_id='previous', state='completed', joint_sdk=[10000, 0, 0, 0, 0, 0])
    bridge.pump(focused=True, menu=False, arm='left', manual_keys=set())
    assert not accepted.done.is_set()
    robot['teleop']['joint_target']['request_id'] = 'action_a'
    bridge.pump(focused=True, menu=False, arm='left', manual_keys=set())
    assert accepted.result['arrived'] is True
    assert bridge.request is None


def test_view_skill_pins_passed_candidate_and_rejects_a_regression(tmp_path):
    passing = tmp_path/'passing'
    report = validate_views(passing)
    assert report['decision'] == 'inherit'
    assert pinned_skills(passing).policy_version == report['candidate']
    assert pinned_skills(passing, report['rollback_revision']).parameters == {'first_view': 'auxiliary'}
    failing = tmp_path/'failing'
    result = validate_views(failing, regression=True)
    assert result['decision'] == 'retain_parent'
    assert result['regressions'] == ['holdout_07']
    assert not (failing/'policies'/'offline_default.json').exists()


def test_episode_linked_candidate_stays_a_proposal_until_paired_validation(tmp_path):
    sim, context = make_session(tmp_path/'development', [])
    with context as initial:
        initial.run()
        previous = initial.episode.record['episode_id']
        task_id = initial.state.task_id
    arguments = dict(first_view='operator', episode_id=previous, hypothesis='The auxiliary view was occluded.')
    transport = ScriptedTransport(sim, [('episodes', {}), ('propose_view_revision', arguments)],
                                  context_mode='snapshot')
    model = Responses(FixtureGrant(), 'fixture', 'fixture', open_stream=transport)
    sim.advance()
    with task_session(sim.archive, task_id, model, CONTRACT, sim, clock=sim.clock) as resumed:
        resumed.run()
        proposal, = resumed.episode.record['candidate_proposals']
        assert resumed.episode.record['tool_events'][0]['result']['episodes'][0]['id'] == previous
        assert resumed.episode.record['tool_events'][1]['result']['status'] == 'proposed'
    path = tmp_path/'proposal.json'
    path.write_text(json.dumps(proposal))
    result = validate_views(tmp_path/'qualified', proposal=path)
    assert result['decision'] == 'inherit'
    assert result['rollback_parameters']['first_view'] == 'auxiliary'


def test_oauth_dynamic_registration_pkce_rotation_and_revocation(tmp_path, monkeypatch):
    client_id = 'oaiapp_TEST'
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    public.update(kid='test', use='sig')
    seen, grants = [], []
    now = time.time()

    def callback(params):
        assert params['client_id'] in (chatgpt.DYNAMIC, client_id)
        assert params['code_challenge_method'] == 'S256'
        if seen:
            assert params['ext_agent_host_id'] == seen[0]['ext_agent_host_id']
            assert params['state'] != seen[0]['state']
        params['redirect_uri'] = 'http://127.0.0.1:1455/auth/callback'
        seen.append(params.copy())
        values = {'state': params['state'], 'code': 'authorization-code'}
        if params['client_id'] == chatgpt.DYNAMIC:
            assert params['agent_name_hint'] == 'Piper Craft'
            values['client_id'] = client_id
        return values

    def request(url, **kwargs):
        if url == chatgpt.DISCOVERY:
            return dict(issuer=chatgpt.ISSUER, jwks_uri=chatgpt.ISSUER+'/jwks',
                        revocation_endpoint=chatgpt.ISSUER+'/revoke')
        if url.endswith('/jwks'):
            return {'keys': [public]}
        assert url == chatgpt.TOKEN
        form = kwargs['form']
        grants.append(form)
        assert form['client_id'] == client_id and form['resource'] == chatgpt.RESOURCE
        reply = dict(token_type='Bearer', expires_in=3600, access_token='access', refresh_token='refresh',
                     scope=chatgpt.SCOPES)
        if form['grant_type'] == 'refresh_token':
            assert 'scope' not in form and form['refresh_token'] == 'refresh'
            return {**reply, 'access_token': 'new-access', 'refresh_token': 'new-refresh'}
        pending = seen[-1]
        challenge = base64.urlsafe_b64encode(hashlib.sha256(form['code_verifier'].encode()).digest()).decode()
        assert challenge.rstrip('=') == pending['code_challenge']
        assert form['redirect_uri'] == pending['redirect_uri']
        reply['id_token'] = jwt.encode(dict(iss=chatgpt.ISSUER, aud=client_id, iat=int(now), exp=int(now+300),
                                            sub='user', email='example@test.invalid', nonce=pending['nonce']),
                                       key, algorithm='RS256', headers={'kid': 'test'})
        return reply

    def revoke(url, **kwargs):
        assert url == chatgpt.ISSUER+'/revoke'
        assert kwargs['form'] == dict(token='new-refresh', token_type_hint='refresh_token', client_id=client_id)
        response = io.BytesIO()
        response.status = 200
        return response

    monkeypatch.setattr(chatgpt, 'connect', revoke)
    auth = chatgpt.ChatGPT(chatgpt.AccountStore(tmp_path/'auth'), request=request, callback=callback)
    account = auth.sign_in()['account']
    assert auth.sign_in(account)['account'] == account
    assert auth.token(account) == 'access'
    assert auth.token(account, rejected='access') == 'new-access'
    assert auth.store.path.stat().st_mode & 0o777 == 0o600
    assert auth.sign_out(account)['remote_revocation_confirmed'] is True
    assert 'new-refresh' not in auth.store.path.read_text()
    assert [grant['grant_type'] for grant in grants] == ['authorization_code', 'authorization_code', 'refresh_token']


def test_oauth_denial_does_not_exchange_code(tmp_path):
    auth = chatgpt.ChatGPT(chatgpt.AccountStore(tmp_path/'auth'),
                           request=lambda *args, **kw: pytest.fail('should not exchange'),
                           callback=lambda params: dict(state=params['state'], error='access_denied'))
    with pytest.raises(chatgpt.ServiceError, match='access_denied'):
        auth.sign_in()
    assert auth.store.list() == []


def test_partial_stream_refused_and_async_cancel_does_not_wait_for_socket():
    stream = io.BytesIO(b'data: {"type":"response.output_item.done","item":{"type":"function_call"}}\n\n')
    with pytest.raises(chatgpt.ServiceError, match='missing_response_completed'):
        completed_turn(sse_events(stream), lambda: False)
    cancel, release, called = (threading.Event() for _ in range(3))

    def slow_stream(*args, **kwargs):
        called.set()
        release.wait(3)
        raise chatgpt.ServiceError('stream', 'fixture_closed')

    def trigger():
        if called.wait(2):
            cancel.set()

    controller = threading.Thread(target=trigger)
    controller.start()
    start = time.monotonic()
    try:
        model = Responses(FixtureGrant(), 'fixture', 'fixture', open_stream=slow_stream)
        with pytest.raises(Interrupted):
            model.complete({}, cancel.is_set)
        assert time.monotonic()-start < 1
    finally:
        release.set()
        controller.join()


@pytest.mark.parametrize('payload,completed', [
    (b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n', True),
    (b'data: {"type":"response.output_item.done","item":{"type":"function_call"}}\n\n', False),
    (b'Plain text with no SSE event.\n', False),
])
def test_text_plain_response_requires_completed_sse(payload, completed):
    from email.message import Message

    def open_stream(*args, **kwargs):
        stream = io.BytesIO(payload)
        stream.headers = Message()
        stream.headers['Content-Type'] = 'text/plain'
        return stream

    model = Responses(FixtureGrant(), 'fixture', 'fixture', open_stream=open_stream)
    if completed:
        assert model.complete({}, lambda: False).output == []
    else:
        with pytest.raises(chatgpt.ServiceError, match='missing_response_completed'):
            model.complete({}, lambda: False)
