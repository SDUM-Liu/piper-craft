"""Bounded, evidence-labelled operation memory. Never issues robot commands."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time

from astra_vla.config import ROOT, atomic_json

MAX_SESSIONS = 12
MAX_PROBLEMS = 8
MAX_TOTAL_LINES = 260
DIRECTORY = ROOT/'docs/memory'


def short(value, limit=280):
    return ' '.join(str(value or '').split())[:limit]


@contextmanager
def edit(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory/'state.json'
        state = json.loads(path.read_text()) if path.exists() else {'sessions': [], 'problems': []}
        yield state
        state['sessions'] = sorted(state['sessions'], key=lambda x: x['stamp'])[-MAX_SESSIONS:]
        state['problems'] = state['problems'][-MAX_PROBLEMS:]
        atomic_json(path, state)
        render(directory, state)


def render(directory, state):
    current = ['# 当前操作记忆', '', '自动摘要只描述控制记录；实物抓持、倒水和保持力须另有实机证据。', '']
    if state.get('context'):
        current += ['- [当前观察] '+state['context']['observed'], '- [证据] '+state['context']['evidence'], '']
    for item in reversed(state['sessions']):
        current += ['## '+item['session'], '', '- [观察] '+item['summary'],
                    '- [证据] '+item['evidence'], '']
    if not state['sessions']:
        current += ['暂无实际操作记录。']
    problems = ['# 问题与改进', '', '标签：观察＝记录事实；推断＝待验证原因；修改＝具体实现；验证＝实际完成的检查。', '']
    for item in reversed(state['problems']):
        problems += ['## '+item['topic']+' · '+item['status'], '']
        for key, label in [('observed','观察'),('inference','推断'),('change','修改'),('validation','验证'),('evidence','证据')]:
            if item.get(key):
                problems += ['- ['+label+'] '+item[key]]
        problems += ['']
    (directory/'current.md').write_text('\n'.join(current)+'\n')
    (directory/'problems.md').write_text('\n'.join(problems)+'\n')
    from astra_vla.task_memory import TaskMemory
    TaskMemory(directory, root=ROOT).compact()
    check(directory)


def check(directory=DIRECTORY):
    files = list(Path(directory).glob('*.md'))
    counts = {p.name: len(p.read_text().splitlines()) for p in files}
    if any(count > 100 for count in counts.values()) or sum(counts.values()) > MAX_TOTAL_LINES:
        raise ValueError('operation memory too long; compact before continuing: '+str(counts))
    return counts


def record_problem(topic, observed, *, inference='', change='', validation='', evidence='',
                   status='open', directory=DIRECTORY):
    if status not in ('open', 'software_verified', 'hardware_verified', 'resolved'):
        raise ValueError('invalid evidence status')
    item = {key: short(value) for key, value in locals().items()
            if key in ('topic','observed','inference','change','validation','evidence','status')}
    if not item['topic'] or not item['observed']:
        raise ValueError('a problem needs a topic and observed evidence')
    item['stamp'] = time.time()
    with edit(directory) as state:
        state['problems'] = [p for p in state['problems'] if p['topic'] != item['topic']] + [item]


def record_session(path, result, *, directory=DIRECTORY):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    commands = Counter()
    feedback = None
    with (path/'events.jsonl').open() as stream:
        for line in stream:
            if '"type": "command"' in line:
                row = json.loads(line)
                commands[row['method']] += 1
            elif '"type": "feedback"' in line:
                feedback = line
    if not commands:
        return  # Static observation is not an operation experience.
    end = json.loads(feedback)['state'].get(manifest['arm']) if feedback else None
    summary = manifest['arm']+'；SDK写入尝试'+str(sum(commands.values()))+'次'
    summary += '；静止确认='+str(result.get('stop_confirmed') is True)
    summary += '；原因='+short(result.get('reason'), 100)
    if end:
        summary += '；最后SDK位置(m)='+str(end['ee_pose']['position_m'])
    summary += '；实体任务成功未由此记录证明'
    try:
        evidence = str(path.resolve().relative_to(ROOT))
    except ValueError:
        evidence = str(path)
    result_path = path/'result.json'
    stamp = result_path.stat().st_mtime if result_path.exists() else time.time()
    item = {'session': path.name, 'stamp': stamp, 'summary': short(summary),
            'evidence': evidence, 'commands': dict(commands),
            'stop_confirmed': result.get('stop_confirmed') is True}
    with edit(directory) as state:
        state['sessions'] = [s for s in state['sessions'] if s['session'] != path.name] + [item]
        reason = short(result.get('reason'))
        fault_words = ('camera','feedback','deadline','limit','collision','ik','fault','gravity','disagrees')
        if result.get('stop_confirmed') is False or any(word in reason.lower() for word in fault_words):
            topic = ('停止未确认' if result.get('stop_confirmed') is False else
                     '运行问题：'+short(reason.split(':', 1)[-1], 65))
            existing = next((p for p in state['problems'] if p['topic'] == topic), None)
            if existing is None or stamp >= existing['stamp']:
                problem = {'topic':topic, 'stamp':stamp, 'status':'open',
                    'observed':'控制记录：'+reason+'；静止确认='+str(item['stop_confirmed']),
                    'inference':'', 'change':'', 'validation':'自动收录，待分析；没有声明修复或实体任务成功。',
                    'evidence':evidence}
                state['problems'] = [p for p in state['problems'] if p['topic'] != topic]+[problem]


def record_context(observed, evidence='', *, directory=DIRECTORY):
    with edit(directory) as state:
        state['context'] = {'observed': short(observed), 'evidence': short(evidence), 'stamp': time.time()}


def main():
    parser = argparse.ArgumentParser(description='Evidence-labelled, bounded PIPER CRAFT memory')
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    sub.add_parser('compact')
    planning = sub.add_parser('planning-context')
    planning.add_argument('--instruction', required=True)
    planning.add_argument('--scene-id', default='')
    planning.add_argument('--demo-offset', type=int, default=0)
    context = sub.add_parser('context')
    context.add_argument('--observed', required=True)
    context.add_argument('--evidence', default='')
    ingest = sub.add_parser('ingest')
    ingest.add_argument('trace', type=Path)
    record = sub.add_parser('problem')
    record.add_argument('topic')
    record.add_argument('--observed', required=True)
    for field in ('inference','change','validation','evidence'):
        record.add_argument('--'+field, default='')
    record.add_argument('--status', default='open', choices=['open','software_verified','hardware_verified','resolved'])
    args = parser.parse_args()
    if args.command == 'planning-context':
        from astra_vla.task_memory import TaskMemory
        print(json.dumps(TaskMemory(args.directory).context(args.instruction, args.scene_id, args.demo_offset), ensure_ascii=False))
    elif args.command == 'check':
        print(json.dumps(check(args.directory), ensure_ascii=False))
    elif args.command == 'compact':
        with edit(args.directory):
            pass
    elif args.command == 'context':
        record_context(args.observed, args.evidence, directory=args.directory)
    elif args.command == 'ingest':
        record_session(args.trace, json.loads((args.trace/'result.json').read_text()), directory=args.directory)
    else:
        record_problem(args.topic, args.observed, inference=args.inference, change=args.change,
            validation=args.validation, evidence=args.evidence, status=args.status, directory=args.directory)


if __name__ == '__main__':
    main()
