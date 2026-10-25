"""Rebuild the public accelerated first-pour demo from the original window video."""
import argparse
import json
from pathlib import Path
import subprocess

# Source seconds, speed and visible label. Chronological cuts retain the spill outcome.
SEGMENTS = [(620, 820, 16, 'Approach / grasp'), (820, 1295, 32, 'Lift / carry / tilt'),
            (1295, 1320, 12, 'Align'), (1320, 1350, 3, 'Pour - some water spilled'),
            (1350, 1440, 20, 'Cut off / retract'), (1580, 1700, 20, 'Return / release')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    filters = []
    for index, (start, end, speed, label) in enumerate(SEGMENTS):
        filters.append(f'[0:v]trim=start={start}:end={end},setpts=(PTS-STARTPTS)/{speed},'
                       # Explicit output trim also bounds EOF padding on older ffmpeg builds.
                       f'crop=1280:560:0:0,scale=960:420,fps=24,trim=duration={(end-start)/speed},'
                       f"drawtext=text='{speed}x | {label}':x=12:y=12:fontsize=21:"
                       f'fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6[v{index}]')
    filters.append(''.join(f'[v{i}]' for i in range(len(SEGMENTS)))
                   +f'concat=n={len(SEGMENTS)}:v=1:a=0[out]')
    video = args.destination/'demo-01-first-pour.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(args.source),
                    '-filter_complex', ';'.join(filters), '-map', '[out]', '-an', '-c:v', 'libx264',
                    '-threads', '2', '-r', '24', '-preset', 'fast', '-crf', '27',
                    '-maxrate', '1500k', '-bufsize', '3000k', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(video)], check=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
                    '-filter_complex', 'fps=8,scale=480:-1:flags=lanczos,split[a][b];'
                    '[a]palettegen=max_colors=96:stats_mode=diff[p];'
                    '[b][p]paletteuse=dither=bayer:bayer_scale=4',
                    str(args.destination/'demo-01-first-pour.gif')], check=True)
    (args.destination/'demo-01-edit.json').write_text(json.dumps(
        dict(source='first pour complete window recording, 2026-10-25',
             source_size=[1280, 800], crop=[0, 0, 1280, 560], output_size=[960, 420],
             segments=[dict(start_s=s, end_s=e, speed=x, label=label) for s, e, x, label in SEGMENTS],
             audio=False, outcome='water in cup with a small spill; not spill-free'), indent=2)+'\n')


if __name__ == '__main__':
    main()
