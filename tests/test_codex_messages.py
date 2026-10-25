"""Only temporary message files; no GUI, real session, camera or robot access."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from astra_vla.codex_messages import (
    CodexMessageReader, MAX_LINE_BYTES, READ_BYTES, ROWS_PER_POLL,
    publish_message, validate_message,
)


def record(seq=1, **changes):
    return {'version': 1, 'source': 'codex', 'session_id': 'offline-test',
            'seq': seq, 'stamp': 1000., 'level': 'info', 'text': '正在观察瓶口位置', **changes}


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode('utf-8') + b'\n'


def reader(path, **kwargs):
    return CodexMessageReader(path, clock=lambda: 1000., interval_s=0, **kwargs)


def test_explicit_public_message_round_trip_and_command_text_stays_data(tmp_path, monkeypatch):
    path = tmp_path/'nested/messages.jsonl'
    # The message never interprets shell, slash commands, or control key names.
    text = '/teleop_input {"keys":["w"]} $(touch forbidden) `command`\n仅为公开说明'
    monkeypatch.setattr(subprocess, 'Popen', lambda *_a, **_k: pytest.fail('message executed a process'))
    monkeypatch.setattr(os, 'system', lambda *_a: pytest.fail('message executed a shell'))
    published = publish_message(text, session_id='offline-test', seq=1, path=path, stamp=1000.)
    assert published['text'] == text.replace('\n', ' ')
    assert reader(path).poll() == [published]
    assert path.stat().st_mode & 0o777 == 0o600
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('changes', [
    {'source': '系统'}, {'version': True}, {'version': 2}, {'session_id': '../raw/session'},
    {'seq': -1}, {'seq': True}, {'seq': 2**63}, {'stamp': float('nan')}, {'stamp': float('inf')},
    {'stamp': True}, {'level': 'execute'}, {'text': ''}, {'text': '\x1b[31m隐藏来源'},
    {'text': '水'*1000}, {'command': 'move'},
])
def test_invalid_records_are_rejected_before_writing(tmp_path, changes):
    with pytest.raises((ValueError, TypeError)):
        validate_message(record(**changes))
    assert list(tmp_path.iterdir()) == []


def test_partial_utf8_bad_lines_duplicates_and_nonexecuting_extensions(tmp_path):
    path = tmp_path/'messages.jsonl'
    raw = encoded(record())
    cut = raw.index('观'.encode()) + 1  # A UTF-8 character is split between polls.
    path.write_bytes(raw[:cut])
    feed = reader(path)
    assert feed.poll() == []
    with path.open('ab') as stream:
        stream.write(raw[cut:] + b'bad json\n\xff\n' + encoded(record())
                     + encoded(record(2, command='move')) + encoded(record(2, text='已到位')))
    assert [r['text'] for r in feed.poll()] == ['正在观察瓶口位置', '已到位']
    assert feed.poll() == [] and not feed.error


def test_old_future_and_out_of_order_messages_do_not_replay(tmp_path):
    path = tmp_path/'messages.jsonl'
    path.write_bytes(b''.join(encoded(r) for r in [record(1, stamp=960.), record(2, stamp=1006.),
                                                  record(3), record(2), record(4, session_id='other')]))
    feed = reader(path)
    assert [(r['session_id'], r['seq']) for r in feed.poll()] == [('offline-test', 3), ('other', 4)]


def test_rotation_and_truncate_regrow_reset_offsets_but_keep_deduplication(tmp_path):
    path = tmp_path/'messages.jsonl'
    path.write_bytes(encoded(record()))
    feed = reader(path)
    assert feed.poll()[0]['seq'] == 1
    rotated = tmp_path/'old.jsonl'
    path.rename(rotated)
    path.write_bytes(encoded(record())+encoded(record(2)))
    assert [r['seq'] for r in feed.poll()] == [2]
    # Regrown content exceeds the old offset, so size alone cannot detect truncation.
    path.write_bytes(encoded(record(3, text='新内容'*100))+encoded(record(4)))
    assert [r['seq'] for r in feed.poll()] == [3, 4]


def test_oversize_partial_line_is_skipped_without_swallowing_next_message(tmp_path):
    path = tmp_path/'messages.jsonl'
    path.write_bytes(b'x'*(MAX_LINE_BYTES+10))
    feed = reader(path)
    assert feed.poll() == [] and not feed.pending and feed.discarding
    with path.open('ab') as stream:
        stream.write(b'x'*100 + b'\n' + encoded(record()))
    assert feed.poll() == [validate_message(record())]


def test_large_backlog_and_flood_have_bounded_reads_and_rows(tmp_path):
    path = tmp_path/'messages.jsonl'
    path.write_bytes(b'old junk\n'*(READ_BYTES//2)+b''.join(encoded(record(i)) for i in range(1, 81)))
    feed = reader(path)
    rows = []
    for _ in range(300):
        current = feed.poll()
        assert len(current) <= ROWS_PER_POLL and len(feed.pending) <= READ_BYTES
        rows.extend(current)
        if feed.offset == path.stat().st_size and not feed.pending:
            break
    assert [r['seq'] for r in rows] == list(range(1, 81))
    assert feed.offset == path.stat().st_size


def test_missing_file_poll_rate_and_local_error_recovery(tmp_path):
    path = tmp_path/'messages.jsonl'
    tick = [1.]
    feed = CodexMessageReader(path, clock=lambda: 1000., poll_clock=lambda: tick[0])
    assert feed.poll() == [] and not feed.error
    path.mkdir()
    tick[0] += .2
    assert feed.poll() == [] and feed.error
    path.rmdir()
    publish_message('公开进度', session_id='offline-test', seq=1, path=path, stamp=1000.)
    assert feed.poll() == []  # Rendering faster than the polling rate does no I/O.
    tick[0] += .2
    assert feed.poll()[0]['text'] == '公开进度' and not feed.error


def test_explicit_writer_preserves_incomplete_line_and_serializes_concurrent_appends(tmp_path):
    path = tmp_path/'messages.jsonl'
    path.write_bytes(b'{"interrupted":')
    def append(index):
        return publish_message('公开消息'+str(index), session_id='writer-'+str(index), seq=1,
                               path=path, stamp=1000.)
    with ThreadPoolExecutor(max_workers=4) as pool:
        published = list(pool.map(append, range(16)))
    lines = path.read_bytes().splitlines()
    assert lines[0] == b'{"interrupted":' and len(lines) == 17
    decoded = [validate_message(json.loads(line)) for line in lines[1:]]
    assert {r['session_id'] for r in decoded} == {r['session_id'] for r in published}


def test_cli_writes_only_selected_temporary_file_and_does_not_import_robot_stack(tmp_path):
    destination = tmp_path/'cli/messages.jsonl'
    result = subprocess.run([sys.executable, '-m', 'astra_vla.codex_messages',
                             '--path', str(destination), '--session', 'cli-test', '--seq', '7',
                             '已暂停；文字 /stop 不执行任何动作'], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['seq'] == 7
    assert json.loads(destination.read_text())['source'] == 'codex'
    code = "import sys; import astra_vla.codex_messages; print([x for x in sys.modules if x.startswith(('astra_vla.workers', 'astra_vla.rpc', 'pygame', 'piper_sdk'))])"
    loaded = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, check=True)
    assert loaded.stdout.strip() == '[]'
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()) == ['cli/messages.jsonl']


def test_symlink_or_nonregular_source_cannot_block_reader_or_redirect_writer(tmp_path):
    target, link = tmp_path/'target', tmp_path/'link'
    target.write_bytes(encoded(record()))
    link.symlink_to(target)
    assert reader(link).poll() == []
    with pytest.raises(OSError):
        publish_message('公开消息', session_id='test', seq=1, path=link)
    fifo = tmp_path/'fifo'
    os.mkfifo(fifo)
    feed = reader(fifo)
    assert feed.poll() == [] and feed.error


def test_invalid_timing_configuration_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        reader(tmp_path/'messages', max_age_s=float('nan'))
    with pytest.raises(ValueError):
        CodexMessageReader(tmp_path/'messages', interval_s=-1)


def test_deep_json_and_escape_expansion_cannot_crash_reader_or_write_unreadable_record(tmp_path):
    path = tmp_path/'messages.jsonl'
    with pytest.raises(ValueError, match='encoded Codex message'):
        publish_message('"'*2048, session_id='test', seq=1, path=path)
    assert not path.exists()
    path.write_bytes(b'['*1500+b']'*1500+b'\n'+encoded(record()))
    assert reader(path).poll() == [validate_message(record())]
