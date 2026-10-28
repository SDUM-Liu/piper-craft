"""Desktop-only preferences; never changes robot motion or calibration policy."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from astra_vla.config import atomic_json

BINDING_HELP = {
    'w': '前进（所选坐标）', 's': '后退', 'a': '左移', 'd': '右移',
    'space': '上移（所选坐标）', 'lshift': '下移（所选坐标）',
    'i': '抬头', 'k': '低头', 'j': '左转', 'l': '右转', 'o': '逆时针侧倾', 'p': '顺时针侧倾',
    'q': '闭合夹爪', 'e': '张开夹爪', 'tab': '循环切档', '1': '选择左臂', '2': '选择右臂',
    'f5': '切换主相机', 'f3': '调试信息', 'f1': '隐藏/显示界面',
    'f8': '窗口置顶', 'f11': '全屏', 'escape': '菜单 / 返回', 'f9': '请求停止',
    'f6': '切换基坐标 / 运动坐标（先停止）',
}
DEFAULT_BINDINGS = {key: key for key in BINDING_HELP}


@dataclass
class DesktopPreferences:
    startup_arm: str = 'left'
    startup_speed: str = 'normal'
    startup_view: str = 'wrist'
    always_on_top: bool = True
    fullscreen: bool = False
    show_minimap: bool = True
    startup_coordinate_mode: str = 'base'
    bindings: dict = field(default_factory=lambda: dict(DEFAULT_BINDINGS))

    def __post_init__(self):
        if self.startup_coordinate_mode not in ('base', 'motion'):
            raise ValueError('invalid desktop coordinate mode')
        if self.startup_arm not in ('left', 'right') or self.startup_speed not in ('slow', 'normal', 'fast') or self.startup_view not in ('wrist', 'top'):
            raise ValueError('invalid desktop startup selection')
        if any(not isinstance(getattr(self, key), bool) for key in ('always_on_top', 'fullscreen', 'show_minimap')):
            raise ValueError('desktop switches must be boolean')
        if not isinstance(self.bindings, dict) or set(self.bindings) != set(DEFAULT_BINDINGS):
            raise ValueError('desktop key bindings must include every action')
        keys = list(self.bindings.values())
        if any(not isinstance(key, str) or not key.strip() or key != key.lower() or len(key) > 32 for key in keys):
            raise ValueError('invalid desktop key name')
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate desktop key binding')
        for action, key in self.bindings.items():
            if key in ('f9', 'escape') and action != key:
                raise ValueError('F9 and Esc are reserved fallback keys')

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        bindings = data.get('bindings', {})
        # Migrate older complete keymaps without silently overwriting a custom
        # F6 binding: choose a free fallback when F6 was already assigned.
        if 'f6' not in bindings:
            free = next(key for key in ('f6', 'f7', 'f10', 'c') if key not in bindings.values())
            data['bindings'] = dict(bindings, f6=free)
        return cls(**data)

    def save(self, path):
        self.__post_init__()
        atomic_json(path, asdict(self))

    def action(self, physical_key):
        if physical_key in ('f9', 'escape'):
            return physical_key  # Always available, including during key capture.
        return next((action for action, key in self.bindings.items() if key == physical_key), None)
