"""Optional visual demonstrations; recording/reference never sends robot commands."""
import argparse
from datetime import datetime
from fractions import Fraction
import json
from pathlib import Path
import queue
import threading
import time
import uuid

from astra_vla.config import ROOT, RUNTIME, atomic_json

CAMERAS = ('top', 'left_wrist', 'right_wrist')
MODES = ('human_objects', 'human_teleop')


class DemonstrationRecording:
    fps = 10

    def __init__(self, destination=None):
        self.destination = Path(destination) if destination else ROOT/'runs'
        self.lock = threading.Lock()
        self.thread = None
        self.state, self.error, self.path = 'idle', '', None
        self.mode = MODES[0]
        self.dropped = self.samples = 0
        self.next_frame = 0.

    def status(self):
        with self.lock:
            return dict(state=self.state, error=self.error, path=str(self.path) if self.path else None,
                        mode=self.mode, dropped=self.dropped, samples=self.samples)

    def start(self, mode, *, simulated=False):
        if mode not in MODES:
            raise ValueError('unknown demonstration mode')
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.path = self.destination/('demonstration_'+datetime.now().strftime('%Y%m%d_%H%M%S')
                                          +'_'+uuid.uuid4().hex[:8])
            self.path.mkdir(parents=True)
            self.mode, self.simulated = mode, simulated
            self.started = time.time()
            self.next_frame = 0.
            self.dropped = self.samples = 0
            self.error, self.state = '', 'recording'
            self.pending, self.stopping = queue.Queue(maxsize=2), threading.Event()
            atomic_json(self.path/'demonstration.json', self.metadata(False, 'recording not finalized'))
            self.thread = threading.Thread(target=self.encode, name='demonstration-recording', daemon=True)
            self.thread.start()

    def metadata(self, closed, error):
        return dict(kind='human_demonstration', version=1, mode=self.mode, simulated=self.simulated,
                    start_stamp=self.started, finish_stamp=time.time(), fps_limit=self.fps,
                    samples=self.samples, dropped_samples=self.dropped, closed=closed, error=error,
                    cameras={name: name+'.mp4' for name in CAMERAS}, trace='samples.jsonl',
                    task='', notes='', robot_motion_verified=False, audio=False,
                    title='', description='', task_tags=[], outcome='unknown', annotation_status='pending_analysis',
                    timing='source wall timestamps; independent cameras, no hardware synchronization',
                    use='optional visual reference; no executable trajectory or automatic replay')

    def due(self):
        return self.state == 'recording' and time.monotonic() >= self.next_frame

    def submit(self, cameras, context):
        if not self.due():
            return
        self.next_frame = time.monotonic()+1/self.fps
        sample = (time.time(), cameras, context)
        try:
            self.pending.put_nowait(sample)
        except queue.Full:
            try:
                self.pending.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.pending.put_nowait(sample)

    def stop(self, reason=''):
        with self.lock:
            if self.state == 'recording':
                self.error, self.state = reason, 'finalizing'
                self.stopping.set()

    def close(self):
        self.stop()
        if self.thread:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError('示范录像仍在收尾，请检查磁盘与编码器')

    def encode(self):
        import av
        import numpy as np
        streams, containers, last_stamp, counts = {}, {}, {}, dict.fromkeys(CAMERAS, 0)
        failure = ''
        try:
            with (self.path/'samples.jsonl').open('w') as trace:
                while not self.stopping.is_set() or not self.pending.empty():
                    try:
                        stamp, cameras, context = self.pending.get(timeout=.05)
                    except queue.Empty:
                        continue
                    observed = {}
                    for name, sample in cameras.items():
                        source_stamp, size, rgb = sample
                        if source_stamp <= last_stamp.get(name, float('-inf')):
                            continue
                        if name not in streams:
                            container = av.open(str(self.path/(name+'.mp4')), 'w')
                            containers[name] = container
                            stream = container.add_stream('libx264', rate=self.fps)
                            stream.width, stream.height = (v//2*2 for v in size)
                            stream.pix_fmt = 'yuv420p'
                            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
                            stream.codec_context.thread_count = 1
                            stream.options = {'preset': 'ultrafast', 'crf': '23'}
                            streams[name] = stream
                        stream = streams[name]
                        pixels = np.frombuffer(rgb, dtype=np.uint8).reshape(size[1], size[0], 3)
                        frame = av.VideoFrame.from_ndarray(pixels[:stream.height, :stream.width], format='rgb24')
                        frame.pts = max(0, round((source_stamp-self.started)*1000))
                        frame.time_base = Fraction(1, 1000)
                        # Two source stamps may quantize to the same millisecond.
                        if frame.pts <= last_stamp.get(name+'_pts', -1):
                            continue
                        for packet in stream.encode(frame):
                            containers[name].mux(packet)
                        last_stamp[name], last_stamp[name+'_pts'] = source_stamp, frame.pts
                        counts[name] += 1
                        observed[name] = dict(source_stamp=source_stamp, video_s=frame.pts/1000,
                                              age_s=stamp-source_stamp, frame=counts[name]-1)
                    trace.write(json.dumps(dict(stamp=stamp, elapsed_s=stamp-self.started,
                                                cameras=observed, context=context),
                                           ensure_ascii=False, allow_nan=False)+'\n')
                    trace.flush()
                    self.samples += 1
            for name, stream in streams.items():
                for packet in stream.encode():
                    containers[name].mux(packet)
            if not all(counts.values()):
                failure = '未收到全部三路新鲜 RGB；示范不完整'
        except Exception as exc:
            failure = str(exc)
        finally:
            for container in containers.values():
                try:
                    container.close()
                except Exception as exc:
                    failure = failure or str(exc)
            with self.lock:
                failure = self.error or failure
                metadata = self.metadata(not failure, failure)
                metadata['camera_frames'] = counts
                try:
                    atomic_json(self.path/'demonstration.json', metadata)
                    if not self.simulated and not failure and self.path.is_relative_to(ROOT/'runs'):
                        try:
                            refresh_memory()
                        except Exception as exc:
                            metadata['memory_index_error'] = str(exc)
                            atomic_json(self.path/'demonstration.json', metadata)
                except Exception as exc:
                    failure = failure or str(exc)
                self.error, self.state = failure, 'error' if failure else 'saved'


def read_demonstration(directory):
    directory = Path(directory).resolve()
    data = json.loads((directory/'demonstration.json').read_text())
    if data.get('kind') != 'human_demonstration' or not data.get('closed') or data.get('error'):
        raise ValueError('示范尚未完整保存或录制失败')
    if data.get('simulated'):
        raise ValueError('离线演示不能选为真实动作参考')
    if not all((directory/(name+'.mp4')).is_file() for name in CAMERAS):
        raise ValueError('示范缺少相机录像')
    return data


def refresh_memory(root=ROOT):
    """Refresh automatic discovery only; no model inference or robot commands."""
    from astra_vla.task_memory import TaskMemory
    TaskMemory(root=root).compact()


def available(destination=None):
    result = []
    for path in sorted((Path(destination) if destination else ROOT/'runs').glob('demonstration_*'), reverse=True):
        try:
            read_demonstration(path)
            result.append(path)
        except (OSError, ValueError, TypeError):
            continue
    return result


def demonstration_label(directory):
    data = read_demonstration(directory)
    name = data.get('title') or data.get('task') or Path(directory).name
    outcome = {'success': '成功已复核', 'failure': '失败', 'unknown': '结果待复核'}.get(data.get('outcome'), '结果待复核')
    return name+' · '+outcome


def annotate(directory, *, task, notes='', title=None, description=None, tags=None,
             outcome=None, basis=None, evidence=None, limitations=None, author='codex', root=ROOT):
    """Persist Codex/human visual review without renaming stable recording paths."""
    directory = Path(directory).resolve()
    data = read_demonstration(directory)
    before = {k: data.get(k) for k in ('title', 'description', 'task', 'outcome')}
    if outcome is None and (task != data.get('task') or
                            (description is not None and description != data.get('description'))):
        outcome = 'unknown'  # Changed task interpretation needs a new success review.
    if not isinstance(task, str) or not task.strip() or len(task) > 1200:
        raise ValueError('示范任务说明不能为空或超过1200字符')
    if outcome is not None and outcome not in ('unknown', 'success', 'failure'):
        raise ValueError('未知示范结果')
    if outcome == 'success':
        if basis not in ('visual_review', 'user_confirmation') or not evidence:
            raise ValueError('示范成功需要视觉/用户复核及证据，保存成功不等于任务成功')
        for item in evidence:
            path = (Path(root)/item).resolve()
            if Path(item).is_absolute() or not path.is_relative_to(Path(root).resolve()) or not path.is_file():
                raise ValueError('成功证据必须是仓库内已有文件')
    for value in (title, description, notes, limitations):
        if value is not None and (not isinstance(value, str) or len(value) > 2000):
            raise ValueError('示范命名/描述应为有界文本')
    if tags is not None and (not isinstance(tags, list) or len(tags) > 12 or
                             any(not isinstance(t, str) or not t.strip() or len(t) > 80 for t in tags)):
        raise ValueError('任务标签最多12个，每个不超过80字符')
    data.update(task=task, notes=notes)
    for field, value in (('title', title), ('description', description), ('task_tags', tags),
                         ('outcome', outcome), ('outcome_basis', basis), ('outcome_evidence', evidence),
                         ('limitations', limitations)):
        if value is not None:
            data[field] = value
    if outcome != 'success' and outcome is not None:
        data.pop('outcome_basis', None)
        data.pop('outcome_evidence', None)
    data['annotation_status'] = 'reviewed' if title and description else 'partial'
    data['annotation'] = {'author': author, 'stamp': time.time(), 'visual_analysis_external': True}
    data['annotation_history'] = (data.get('annotation_history', [])+[before])[-5:]
    atomic_json(directory/'demonstration.json', data)
    refresh_memory(root)
    return data


def select_reference(directory=None, *, destination=None):
    target = Path(destination) if destination else RUNTIME/'demonstration_reference.json'
    if directory is None:
        atomic_json(target, dict(enabled=False))
    else:
        directory = Path(directory).resolve()
        data = read_demonstration(directory)
        if data.get('outcome') == 'failure':
            raise ValueError('失败示范不能选作成功动作参考')
        atomic_json(target, dict(enabled=True, directory=str(directory), mode=data['mode'],
                                 metadata=str(directory/'demonstration.json'),
                                 instruction='Read as visual guidance; observe current RGB and both arms before motion. '
                                             'Do not execute or replay recorded inputs/poses.'))


def export_reference(directory, count=6):
    """Export actual decoded frames with their own PTS; retain gaps and source timing."""
    import av
    import cv2
    data = read_demonstration(directory)
    directory = Path(directory).resolve()
    output = directory/'reference'
    output.mkdir(exist_ok=True)
    lines = ['# 人类动作示范参考', '', f"名称：{data.get('title', '')}", f"类型：{data['mode']}",
             f"结果：{data.get('outcome', 'unknown')}（仅保存完成不能判成功）", f"任务：{data.get('task', '')}",
             f"描述：{data.get('description', '')}", f"适用边界：{data.get('limitations', '')}",
             f"备注：{data.get('notes', '')}", '',
             '仅作可选参考。动作前重新观察现场三路 RGB 和双臂反馈；不重放按键、关节或末端轨迹。',
             '相机独立采样；缺帧和时间差请结合 ../samples.jsonl 复核。', '']
    for name in CAMERAS:
        with av.open(str(directory/(name+'.mp4'))) as container:
            stream = container.streams.video[0]
            duration = float((stream.duration or 0)*stream.time_base)
            start = float((stream.start_time or 0)*stream.time_base)
            targets = [start+(max(0, duration-.2))*i/max(1, count-1) for i in range(count)]
            last_pts = None
            for index, target in enumerate(targets):
                container.seek(int(target/stream.time_base), stream=stream)
                for frame in container.decode(stream):
                    seconds = float(frame.pts*frame.time_base)
                    if seconds+.001 < target:
                        continue
                    if frame.pts != last_pts:
                        filename = f'{name}_{index:02}.jpg'
                        if not cv2.imwrite(str(output/filename), frame.to_ndarray(format='bgr24')):
                            raise OSError('写入示范关键帧失败')
                        lines.append(f'- {name} · 录像时间 {seconds:.3f}s：[关键帧]({filename})')
                        last_pts = frame.pts
                    break
    (output/'index.md').write_text('\n'.join(lines)+'\n')
    return output/'index.md'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('list')
    sub.add_parser('clear')
    for name in ('select', 'export', 'annotate'):
        command = sub.add_parser(name)
        command.add_argument('directory', type=Path)
        if name == 'export':
            command.add_argument('--frames', type=int, choices=range(1, 25), default=6)
        if name == 'annotate':
            command.add_argument('--task', required=True)
            command.add_argument('--notes', default='')
            command.add_argument('--title')
            command.add_argument('--description')
            command.add_argument('--tags', nargs='*')
            command.add_argument('--outcome', choices=('unknown', 'success', 'failure'))
            command.add_argument('--basis', choices=('visual_review', 'user_confirmation'))
            command.add_argument('--evidence', nargs='*')
            command.add_argument('--limitations')
            command.add_argument('--author', default='codex')
    args = parser.parse_args()
    if args.action == 'list':
        print('\n'.join(json.dumps({'directory': str(p), 'label': demonstration_label(p)}, ensure_ascii=False)
                        for p in available()))
    elif args.action == 'clear':
        select_reference()
    elif args.action == 'select':
        select_reference(args.directory)
    elif args.action == 'export':
        print(export_reference(args.directory, args.frames))
    else:
        annotate(args.directory, task=args.task, notes=args.notes, title=args.title, description=args.description,
                 tags=args.tags, outcome=args.outcome, basis=args.basis, evidence=args.evidence,
                 limitations=args.limitations, author=args.author)


if __name__ == '__main__':
    main()
