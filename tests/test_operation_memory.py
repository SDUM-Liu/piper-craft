import json

import pytest

from astra_vla.operation_memory import MAX_PROBLEMS, MAX_SESSIONS, check, record_problem, record_session


def trace(tmp_path, name, commands=True):
    p=tmp_path/name
    p.mkdir()
    (p/'manifest.json').write_text(json.dumps({'arm':'left'}))
    rows=[{'type':'feedback','state':{'left':{'ee_pose':{'position_m':[.1,0,.2]}}}}]
    if commands:
        rows.append({'type':'command','method':'JointCtrl'})
    (p/'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    return p


def test_static_observation_does_not_create_operation_memory(tmp_path):
    directory=tmp_path/'memory'
    record_session(trace(tmp_path,'static',False),{},directory=directory)
    assert not directory.exists()


def test_failed_stop_is_not_marked_success_and_updates_replace_old_state(tmp_path):
    directory=tmp_path/'memory'
    path=trace(tmp_path,'one')
    record_session(path,{'stop_confirmed':False,'reason':'stale feedback'},directory=directory)
    text=(directory/'current.md').read_text()
    assert 'False' in text and 'stale feedback' in text and '实体任务成功未由此记录证明' in text
    record_session(path,{'stop_confirmed':True,'reason':'confirmed later'},directory=directory)
    state=json.loads((directory/'state.json').read_text())
    assert len(state['sessions'])==1 and state['sessions'][0]['stop_confirmed']


def test_expired_summaries_are_bounded_and_same_issue_is_updated(tmp_path):
    directory=tmp_path/'memory'
    for i in range(20):
        record_session(trace(tmp_path,'session'+str(i)),{},directory=directory)
        record_problem('issue'+str(i),'observed',inference='unproven',directory=directory)
    state=json.loads((directory/'state.json').read_text())
    assert len(state['sessions'])==MAX_SESSIONS and len(state['problems'])==MAX_PROBLEMS
    record_problem('issue19','new fact',change='patch',validation='offline only',status='software_verified',directory=directory)
    state=json.loads((directory/'state.json').read_text())
    assert len(state['problems'])==MAX_PROBLEMS
    assert state['problems'][-1]['observed']=='new fact'
    assert sum(check(directory).values())<=260
    assert '[推断]' in (directory/'problems.md').read_text()


def test_memory_limit_and_evidence_status_are_enforced(tmp_path):
    with pytest.raises(ValueError):
        record_problem('topic','fact',status='assumed_success',directory=tmp_path)
    (tmp_path/'oversize.md').write_text('line\n'*101)
    with pytest.raises(ValueError,match='too long'):
        check(tmp_path)


def test_current_context_replaces_old_status_without_inventing_motion(tmp_path):
    from astra_vla.operation_memory import record_context
    record_context('feedback disconnected','local evidence',directory=tmp_path)
    record_context('new observation only','new evidence',directory=tmp_path)
    state=json.loads((tmp_path/'state.json').read_text())
    assert not state['sessions'] and state['context']['observed']=='new observation only'
    assert 'feedback disconnected' not in (tmp_path/'current.md').read_text()


def test_real_trace_close_automatically_updates_only_temporary_memory(tmp_path,monkeypatch):
    from astra_vla.teleop import Policy
    from astra_vla.workers import teleop
    monkeypatch.setattr(teleop,'ROOT',tmp_path)
    trace=teleop.Trace('test-session','left',Policy())
    trace.activate()
    trace.record('command',method='JointCtrl',arguments=[0]*6)
    trace.close({'stop_confirmed':False,'reason':'simulated disconnected feedback'})
    state=json.loads((tmp_path/'docs/memory/state.json').read_text())
    assert len(state['sessions'])==1 and not state['sessions'][0]['stop_confirmed']
    assert not (trace.path/'memory_error.json').exists()


def test_failed_operation_is_auto_collected_without_reopening_newer_verified_fix(tmp_path):
    import os
    directory=tmp_path/'memory'
    path=trace(tmp_path,'failure')
    result={'stop_confirmed':False,'reason':'feedback stale'}
    (path/'result.json').write_text(json.dumps(result))
    os.utime(path/'result.json',(1,1))
    record_session(path,result,directory=directory)
    state=json.loads((directory/'state.json').read_text())
    assert state['problems'][0]['topic']=='停止未确认' and state['problems'][0]['status']=='open'
    record_problem('停止未确认','new evidence',validation='software checked',status='software_verified',directory=directory)
    record_session(path,result,directory=directory)
    state=json.loads((directory/'state.json').read_text())
    assert len(state['problems'])==1 and state['problems'][0]['status']=='software_verified'
