"""Tail only public user/agent events from the explicitly bound Codex session."""
from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat
import time

from astra_vla.codex_messages import MAX_TEXT_BYTES
from astra_vla.config import RUNTIME, atomic_json

BINDING = RUNTIME/'codex_session.json'


def bind_current_session(path=None, *, binding=BINDING):
    if path is None:
        thread = os.environ.get('CODEX_THREAD_ID') or os.environ.get('CODEX_SESSION_ID')
        if not thread or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', thread):
            return None
        matches = list((Path.home()/'.codex/sessions').rglob('*'+thread+'*.jsonl'))
        if len(matches) != 1:
            return None
        path = matches[0]
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != '.jsonl':
        raise ValueError('select an existing current session JSONL file')
    atomic_json(binding, {'path': str(path)})
    return path


class PublicSessionTail:
    """Bounded tail, starting at EOF. Tool/developer/reasoning records are ignored."""
    def __init__(self, path, *, wall=time.time, max_age_s=30.):
        self.path, self.wall, self.max_age_s = Path(path), wall, max_age_s
        self.identity, self.offset, self.pending = None, 0, b''
        self.dropping, self.sequence, self.error = False, 0, None

    def poll(self):
        fd = None
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            meta = os.fstat(fd)
            if not stat.S_ISREG(meta.st_mode):
                raise ValueError('current session must be a regular file')
            identity = (meta.st_dev, meta.st_ino)
            if identity != self.identity or meta.st_size < self.offset:
                self.identity, self.offset = identity, meta.st_size
                self.pending, self.dropping = b'', False
                self.error = None
                return []
            os.lseek(fd, self.offset, os.SEEK_SET)
            chunk = os.read(fd, 262144)
            self.offset += len(chunk)
            self.error = None
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            return []
        finally:
            if fd is not None:
                os.close(fd)
        if self.dropping:
            boundary = chunk.find(b'\n')
            if boundary < 0:
                return []
            chunk = chunk[boundary+1:]
            self.dropping = False
        data = self.pending+chunk
        rows = data.split(b'\n')
        self.pending = rows.pop()
        if len(self.pending) > 1048576:
            self.pending, self.dropping = b'', True
        result = []
        for raw in rows:
            # Avoid decoding nonpublic records, including tool and reasoning payloads.
            prefix = raw[:256]
            if (len(raw) > 1048576 or b'"event_msg"' not in prefix or
                    not any(tag in prefix for tag in (b'"agent_message"', b'"user_message"'))):
                continue
            try:
                row = json.loads(raw)
                payload = row.get('payload', {})
                kind = payload.get('type')
                if row.get('type') != 'event_msg' or kind not in ('agent_message','user_message'):
                    continue
                if kind == 'agent_message' and payload.get('phase') not in (None,'commentary','final','final_answer'):
                    continue
                text = payload.get('message')
                if not isinstance(text, str):
                    continue
                if kind == 'user_message':
                    for marker in ('## My request:', '## My request：'):
                        if marker in text:
                            text = text.split(marker,1)[1]
                    text = '用户：'+text.strip()
                text = ' '.join(text.split())
                if not text:
                    continue
                encoded = text.encode()
                if len(encoded) > MAX_TEXT_BYTES:
                    text = encoded[:MAX_TEXT_BYTES-3].decode('utf-8',errors='ignore')+'…'
                stamp = datetime.fromisoformat(row['timestamp'].replace('Z','+00:00')).timestamp()
                if not -5 <= self.wall()-stamp <= self.max_age_s:
                    continue
            except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
                continue
            self.sequence += 1
            result.append({'version':1,'source':'codex','session_id':'public-session',
                           'seq':self.sequence,'stamp':stamp,'level':'info','text':text})
        return result


class PublicSessionFeed:
    def __init__(self, binding=BINDING, *, clock=time.monotonic):
        self.binding, self.clock = Path(binding), clock
        self.next_poll, self.tail, self.path, self.error = 0., None, None, None

    def poll(self):
        tick = self.clock()
        if tick < self.next_poll:
            return []
        self.next_poll = tick+.1
        try:
            if not self.binding.exists():
                return []
            path = json.loads(self.binding.read_text())['path']
            if path != self.path:
                self.path, self.tail = path, PublicSessionTail(path)
            result = self.tail.poll()
            self.error = self.tail.error
            return result
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.error = str(exc)
            return []


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Bind only the current public Codex conversation to PIPER CRAFT')
    parser.add_argument('--session-file', type=Path)
    args = parser.parse_args()
    path = bind_current_session(args.session_file)
    if path is None:
        parser.exit(1, '当前环境未提供可唯一识别的Codex会话；可显式指定--session-file。\n')
    print('已绑定当前会话的公开消息；从当前末尾开始，不重播历史。')


if __name__ == '__main__':
    main()
