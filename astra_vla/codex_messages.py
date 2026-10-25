"""Explicit public Codex progress for the desktop; text is never a command.

Publish only messages intended for the user, for example::

    python -m astra_vla.codex_messages --session pour-20261017 --seq 1 \
        '已确认静止，下一步观察瓶口位置。'

This module neither discovers Codex conversations nor accesses robot services.
The desktop may poll ``CodexMessageReader`` and display the returned records in
its own informational queue, separately from persistent controller failures.
"""
from collections import OrderedDict
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import time


DEFAULT_PATH = Path(os.environ.get('ASTRA_ROOT', Path(__file__).resolve().parents[1])) / '.runtime/codex_messages.jsonl'
MAX_TEXT_BYTES = 2048
MAX_LINE_BYTES = 4096
READ_BYTES = 65536
ROWS_PER_POLL = 32
LEVELS = ('info', 'warning', 'error')
FIELDS = {'version', 'source', 'session_id', 'seq', 'stamp', 'level', 'text'}


def validate_message(record):
    """Return a plain, bounded display record, rejecting executable extensions."""
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError('Codex message fields do not match version 1')
    if type(record['version']) is not int or record['version'] != 1 or record['source'] != 'codex':
        raise ValueError('Codex message version/source is invalid')
    session = record['session_id']
    if not isinstance(session, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}', session):
        raise ValueError('session_id must be 1–96 letters, digits, dots, colons, underscores or hyphens')
    if type(record['seq']) is not int or not 0 <= record['seq'] < 2**63:
        raise ValueError('seq must be a nonnegative 63-bit integer')
    stamp = record['stamp']
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= 0:
        raise ValueError('stamp must be a finite positive wall-clock timestamp')
    if record['level'] not in LEVELS:
        raise ValueError('level must be info, warning or error')
    text = record['text']
    if not isinstance(text, str):
        raise ValueError('text must be a string')
    text = ' '.join(text.split())
    if (not text or len(text.encode('utf-8')) > MAX_TEXT_BYTES
            or any(ord(char) < 32 or ord(char) == 127 for char in text)):
        raise ValueError('text must contain 1–2048 UTF-8 bytes of displayable public text')
    return {**record, 'text': text, 'stamp': float(stamp)}


def publish_message(text, *, session_id, seq, level='info', path=None, stamp=None):
    """Append one explicit public message; returns the record, never sends motion."""
    record = validate_message({'version': 1, 'source': 'codex', 'session_id': session_id,
                               'seq': seq, 'stamp': time.time() if stamp is None else stamp,
                               'level': level, 'text': text})
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n'
    if len(payload)-1 > MAX_LINE_BYTES:
        raise ValueError('encoded Codex message exceeds 4096 bytes')
    destination = Path(path) if path is not None else DEFAULT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('Codex message destination must be a regular file')
        fcntl.flock(fd, fcntl.LOCK_EX)
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size-1) != b'\n':
            payload = b'\n' + payload  # Preserve an interrupted preceding record as a bad line.
        while payload:
            written = os.write(fd, payload)
            if written <= 0:
                raise OSError('Codex message append made no progress')
            payload = payload[written:]
    finally:
        os.close(fd)
    return record


class CodexMessageReader:
    """Bounded local-file tail. Missing/invalid input cannot affect robot control.

    ``poll()`` returns new validated records. ``error`` contains only local file
    access failures; malformed, duplicate, old and future-dated records are
    ignored. Returned error-level messages still represent Codex prose, never
    controller status. New readers only inspect the final 64 KiB and show recent
    messages (30 seconds by default), so launching the GUI cannot replay a task.
    """
    def __init__(self, path=None, *, clock=time.time, poll_clock=time.monotonic,
                 interval_s=.1, max_age_s=30.):
        if not math.isfinite(interval_s) or not 0 <= interval_s <= 60:
            raise ValueError('interval_s must be between 0 and 60 seconds')
        if not math.isfinite(max_age_s) or not 0 < max_age_s <= 3600:
            raise ValueError('max_age_s must be between 0 and 3600 seconds')
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.clock, self.poll_clock = clock, poll_clock
        self.interval_s, self.max_age_s = interval_s, max_age_s
        self.next_poll = float('-inf')
        self.identity, self.offset, self.anchor = None, 0, b''
        self.pending, self.discarding = b'', False
        self.sequences = OrderedDict()
        self.error = ''

    def _reset(self, identity):
        self.identity, self.offset, self.anchor = identity, 0, b''
        self.pending, self.discarding = b'', False

    def _read(self):
        fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError('Codex message source must be a regular file')
            identity = (info.st_dev, info.st_ino)
            changed = self.identity != identity or info.st_size < self.offset
            if self.anchor and not changed:
                changed = os.pread(fd, len(self.anchor), self.offset-len(self.anchor)) != self.anchor
            if changed:
                self._reset(identity)
            if info.st_size-self.offset > READ_BYTES:
                # Catch up without allowing an old/flooded file to monopolize a frame.
                self.offset = info.st_size-READ_BYTES
                self.pending = b''
                self.discarding = os.pread(fd, 1, self.offset-1) != b'\n'
            data = os.pread(fd, max(0, READ_BYTES-len(self.pending)), self.offset)
            self.offset += len(data)
            self.pending += data
            self.anchor = os.pread(fd, min(64, self.offset), max(0, self.offset-64))
        finally:
            os.close(fd)

    def poll(self):
        tick = self.poll_clock()
        if tick < self.next_poll:
            return []
        self.next_poll = tick + self.interval_s
        try:
            self._read()
            self.error = ''
        except FileNotFoundError:
            self._reset(None)
            self.error = ''
            return []
        except OSError as exc:
            self.error = str(exc)
            return []
        result, now = [], self.clock()
        for _ in range(ROWS_PER_POLL):
            newline = self.pending.find(b'\n')
            if newline < 0:
                if len(self.pending) > MAX_LINE_BYTES:
                    self.pending, self.discarding = b'', True
                break
            line, self.pending = self.pending[:newline], self.pending[newline+1:]
            if self.discarding:
                self.discarding = False
                continue
            if len(line) > MAX_LINE_BYTES:
                continue
            try:
                record = validate_message(json.loads(line))
            except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError):
                continue
            if not -5 <= now-record['stamp'] <= self.max_age_s:
                continue
            session, seq = record['session_id'], record['seq']
            if seq <= self.sequences.get(session, -1):
                continue
            self.sequences[session] = seq
            self.sequences.move_to_end(session)
            if len(self.sequences) > 64:
                self.sequences.popitem(last=False)
            result.append(record)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Publish explicit public Codex text to PIPER CRAFT; never execute commands.')
    parser.add_argument('text', help='Public message to display verbatim as prose; no motion is sent')
    parser.add_argument('--session', required=True, help='Public task/session label, not a credential')
    parser.add_argument('--seq', required=True, type=int, help='Increasing message number within the session')
    parser.add_argument('--level', choices=LEVELS, default='info')
    parser.add_argument('--path', type=Path, default=DEFAULT_PATH, help='Local JSONL output path')
    args = parser.parse_args(argv)
    try:
        record = publish_message(args.text, session_id=args.session, seq=args.seq,
                                 level=args.level, path=args.path)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'无法发布 Codex 消息：{exc}\n')
    print(json.dumps(record, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
