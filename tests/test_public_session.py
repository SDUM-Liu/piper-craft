from datetime import datetime,timezone
import json

from astra_vla.public_session import PublicSessionFeed, PublicSessionTail, bind_current_session


def event(message, kind='agent_message', phase='commentary', timestamp=100.):
    return {'timestamp':datetime.fromtimestamp(timestamp,timezone.utc).isoformat(),
            'type':'event_msg','payload':{'type':kind,'message':message,'phase':phase}}


def append(path,record):
    with path.open('a') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n')


def test_only_new_public_user_and_assistant_events_are_displayed(tmp_path):
    path=tmp_path/'session.jsonl'
    append(path,event('old'))
    tail=PublicSessionTail(path,wall=lambda:100.)
    assert tail.poll()==[]
    append(path,{'type':'response_item','payload':{'type':'reasoning','text':'private reasoning'}})
    append(path,{'type':'response_item','payload':{'type':'function_call_output','text':'private tool output'}})
    append(path,event('not public','agent_message','analysis'))
    append(path,event('进展：正在检查反馈'))
    append(path,event('# IDE\n## My request:\n继续操作','user_message'))
    append(path,event('已完成','agent_message','final_answer'))
    records=tail.poll()
    assert [x['text'] for x in records]==['进展：正在检查反馈','用户：继续操作','已完成']
    assert tail.poll()==[]


def test_partial_lines_expired_events_and_rotation_do_not_replay(tmp_path):
    path=tmp_path/'session.jsonl'
    path.touch()
    tail=PublicSessionTail(path,wall=lambda:100.)
    tail.poll()
    line=json.dumps(event('partial')).encode()+b'\n'
    with path.open('ab') as f:
        f.write(line[:20])
    assert tail.poll()==[]
    with path.open('ab') as f:
        f.write(line[20:])
    assert tail.poll()[0]['text']=='partial'
    append(path,event('expired',timestamp=1))
    assert tail.poll()==[]
    path.unlink()
    append(path,event('old after rotate'))
    assert tail.poll()==[]
    append(path,event('new after rotate'))
    assert tail.poll()[0]['text']=='new after rotate'


def test_binding_is_explicit_and_missing_source_recovers(tmp_path):
    path=tmp_path/'session.jsonl'
    path.touch()
    binding=tmp_path/'binding.json'
    assert bind_current_session(path,binding=binding)==path
    clock=[0.]
    feed=PublicSessionFeed(binding,clock=lambda:clock[0])
    assert feed.poll()==[]
    append(path,event('public',timestamp=datetime.now(timezone.utc).timestamp()))
    clock[0]=1.
    assert feed.poll()[0]['text']=='public'
    path.unlink()
    clock[0]=2.
    assert feed.poll()==[] and feed.error


def test_oversized_nonpublic_partial_record_does_not_hide_next_public_message(tmp_path):
    path=tmp_path/'session.jsonl'
    path.touch()
    tail=PublicSessionTail(path,wall=lambda:100.)
    tail.poll()
    with path.open('ab') as stream:
        stream.write(b'x'*(1048576+20))
    for _ in range(6):
        assert tail.poll()==[]
    with path.open('ab') as stream:
        stream.write(b'\n')
    append(path,event('new public message'))
    assert tail.poll()[0]['text']=='new public message'
