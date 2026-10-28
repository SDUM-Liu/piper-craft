"""Task-level recording ownership. File operations only; never sends robot commands."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

from astra_vla.config import ROOT, atomic_json


class TaskArchive:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.runtime = self.root / '.runtime'
        self.active_path = self.runtime / 'active_task.json'

    @contextmanager
    def locked(self):
        self.runtime.mkdir(parents=True, exist_ok=True)
        with (self.runtime / 'task_archive.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def path(self, task_id):
        if not isinstance(task_id, str) or not re.fullmatch(r'task_[0-9]{8}_[0-9]{6}_[a-f0-9]{8}', task_id):
            raise ValueError('invalid task ID')
        return self.root / 'runs' / task_id

    def read(self, task_id):
        return json.loads((self.path(task_id) / 'task.json').read_text())

    def _active(self):
        if not self.active_path.exists():
            return None
        task_id = json.loads(self.active_path.read_text())['task_id']
        data = self.read(task_id)
        return data if data['state'] == 'active' else None

    def status(self):
        with self.locked():
            return self._active()

    def _begin(self, name, instruction):
        task_id = 'task_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        data = {'schema_version': 1, 'kind': 'robot_task_archive', 'task_id': task_id,
                'name': name or '未命名任务', 'instruction': instruction, 'state': 'active',
                'outcome': 'unknown', 'started_at': time.time(), 'artifacts': []}
        atomic_json(self.path(task_id) / 'task.json', data)
        atomic_json(self.active_path, {'task_id': task_id})
        return data

    def begin(self, name='', instruction=''):
        with self.locked():
            if self._active():
                raise ValueError('已有进行中的任务；先结束当前任务，再新建')
            return self._begin(name, instruction)

    def reserve(self, name, kind, **metadata):
        """Reserve before recording starts, so task finish cannot race a writer."""
        if kind not in ('teleop', 'window') or Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('invalid recording name/kind')
        with self.locked():
            data = self._active() or self._begin('', '')
            relative = ('segments/' if kind == 'teleop' else 'videos/') + name
            if any(row['path'] == relative for row in data['artifacts']):
                raise ValueError('recording already registered')
            data['artifacts'].append({'path': relative, 'kind': kind, 'state': 'open',
                                      'owner_pid': os.getpid(), 'started_at': time.time(), **metadata})
            atomic_json(self.path(data['task_id']) / 'task.json', data)
            return self.path(data['task_id']) / relative

    @classmethod
    def for_artifact(cls, path):
        path = Path(path).resolve()
        task_path = path.parent.parent
        if path.parent.name not in ('segments', 'videos') or not (task_path / 'task.json').exists():
            return None
        archive = cls(task_path.parent.parent)
        archive.path(task_path.name)
        return archive

    def update(self, path, *, discard=False, **values):
        path = Path(path).resolve()
        task_id = path.parent.parent.name
        with self.locked():
            data = self.read(task_id)
            if data['state'] != 'active':
                raise ValueError('task is already finished')
            relative = str(path.relative_to(self.path(task_id)))
            entry = next(row for row in data['artifacts'] if row['path'] == relative)
            if discard:
                data['artifacts'].remove(entry)
            else:
                entry.update(values)
            atomic_json(self.path(task_id) / 'task.json', data)
            return data

    @staticmethod
    def summary(data):
        commands = Counter()
        segments = [row for row in data['artifacts'] if row['kind'] == 'teleop' and row.get('commands')]
        for row in segments:
            commands.update(row['commands'])
        return {'task_id': data['task_id'], 'name': data['name'], 'state': data['state'],
                'outcome': data['outcome'], 'segment_count': len(segments),
                'recording_count': sum(row['kind'] == 'window' for row in data['artifacts']),
                'commands': dict(commands), 'arms': sorted({row['arm'] for row in segments}),
                'last_stop_confirmed': segments[-1].get('result', {}).get('stop_confirmed') if segments else None,
                'summary': data.get('summary', ''), 'basis': data.get('basis', ''),
                'physical_success_verified': data['outcome'] == 'success'}

    def finish(self, *, outcome='unknown', summary='', basis='', reviewed=False):
        if outcome not in ('unknown', 'success', 'failure', 'aborted'):
            raise ValueError('invalid task outcome')
        if outcome == 'success' and (not summary.strip() or basis not in ('visual_review', 'user_confirmation')):
            raise ValueError('成功结果需要实物复核说明及 visual_review/user_confirmation 依据')
        if outcome in ('failure', 'aborted') and (not reviewed or not summary.strip()):
            raise ValueError('失败/中止任务须先完成复核与知识提炼，提供 summary 和 reviewed')
        if outcome in ('failure', 'aborted') and len(summary) > 240:
            raise ValueError('失败/中止总结限240字，确保教训完整进入有界Memory后再删除材料')
        with self.locked():
            data = self._active()
            if not data:
                raise ValueError('没有进行中的任务')
            for row in data['artifacts']:
                if row['state'] != 'open':
                    continue
                try:
                    os.kill(row['owner_pid'], 0)
                except ProcessLookupError:
                    row['state'] = 'interrupted'
                else:
                    raise ValueError('控制或录像仍在进行/收尾；请先退出控制并保存录像')
            data.update(state='finished', outcome=outcome, summary=summary, basis=basis,
                        finished_at=time.time())
            # Persist lessons first. If Memory cannot be written, keep the task active
            # and all material available for review/retry. No motion is involved.
            from astra_vla.operation_memory import record_task
            record_task(self, data)
            if outcome in ('failure', 'aborted'):
                shutil.rmtree(self.path(data['task_id']))
            else:
                atomic_json(self.path(data['task_id']) / 'result.json', self.summary(data))
                atomic_json(self.path(data['task_id']) / 'task.json', data)
            self.active_path.unlink()
            return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    begin = sub.add_parser('begin')
    begin.add_argument('--name', required=True)
    begin.add_argument('--instruction', default='')
    sub.add_parser('status')
    finish = sub.add_parser('finish')
    finish.add_argument('--outcome', choices=('unknown', 'success', 'failure', 'aborted'), default='unknown')
    finish.add_argument('--summary', default='')
    finish.add_argument('--basis', choices=('visual_review', 'user_confirmation'), default='')
    finish.add_argument('--reviewed', action='store_true')
    args = parser.parse_args()
    archive = TaskArchive()
    if args.command == 'begin':
        data = archive.begin(args.name, args.instruction)
    elif args.command == 'status':
        data = archive.status()
    else:
        data = archive.finish(outcome=args.outcome, summary=args.summary, basis=args.basis, reviewed=args.reviewed)
    print(json.dumps(data, ensure_ascii=False))


if __name__ == '__main__':
    main()
