"""Evidence prompts for desktop grasping; no motion or automatic visual inference."""

import copy
import math
import time


FACTS = {
    'auxiliary_clearance': '另一臂/线缆不侵入接近、旋转及试提空间；辅助腕在保持间隙、避开遮挡的前提下尽可能低位侧看，实际看清指尖、目标和桌面高度关系，不要求朝下俯视',
    'tcp_down': '仅抓取臂SDK工具+Z垂直桌面向下，源RGB确认双指向下；不约束辅助腕朝下，不把基座Z或RPY单值当工具Z',
    'target_part': '辨认物体轴线、可夹部位和后续操作：羽毛球球头/羽毛分开，优先核查球头可夹区',
    'above_target': '张爪在目标上方，双指与邻物有间隙，尚未压到目标',
    'height': '辅助腕以尽可能低且有间隙、无遮挡的侧视看清指尖、物体及桌面的相对高度，写明遮挡；朝下俯视或平面投影不替代高度关系，未知不继续下降',
    'closure_direction': '明确进夹方向与双指闭合方向相对物体轴线的关系、抓点及旋转效果',
    'entry_depth': '辅助腕交叉确认目标进入双指内部的程度及夹持高度，排除仅指尖浅夹',
    'bilateral_contact': '闭合路径能接触目标两侧，未夹羽毛/筒沿/邻物；不是由开度或effort推断',
    'tip_clearance': '双指尖到桌面/邻物及后续提起空间可见，不盲推到底',
    'lift_follow': '短提后物体实际离桌且随爪移动，复核相对滑移/转动，非SDK到位',
}

STAGES = (
    dict(id='prepare_views', goal='先安排另一臂避让并建立尽可能低的辅助侧视',
         required=['auxiliary_clearance', 'target_part'],
         action='先看三路源帧和双臂SDK；按普通真机流程分步调整辅助腕位置及朝向，保持间隙并尽可能低位侧看目标，不套用抓取臂垂直向下要求'),
    dict(id='overhead', goal='仅抓取臂TCP垂直向下，张爪到物体上方',
         required=['auxiliary_clearance', 'tcp_down', 'target_part', 'above_target'],
         action='在有间隙高度调整抓取臂工具+Z向下，再基坐标平移；辅助腕保持低位侧视，不带着低位夹爪横扫'),
    dict(id='approach', goal='基坐标小步靠近桌面，辅助腕确认高度',
         required=['auxiliary_clearance', 'tcp_down', 'above_target', 'height', 'tip_clearance'],
         action='基坐标Shift为−Z下降、Space为+Z退离；每段后查看新三路及SDK'),
    dict(id='align', goal='闭爪前确定进夹方向、闭合方向及抓点',
         required=['auxiliary_clearance', 'target_part', 'height', 'closure_direction', 'tip_clearance'],
         action='保留旋转间隙，按SDK与源帧核对基坐标O/P的倾斜效果；调整后重新确认工具Z向下'),
    dict(id='preclose', goal='继续小步下降让物体进入双指空间，再判断是否可闭爪',
         required=['auxiliary_clearance', 'tcp_down', 'target_part', 'height', 'closure_direction',
                   'entry_depth', 'bilateral_contact', 'tip_clearance'],
         action='辅助腕确认高度/入夹，顶视与操作腕确认抓点/方向；任一未知先补视角，不闭爪试运气'),
    dict(id='lift_check', goal='闭合后低位短提并复核夹持',
         required=['auxiliary_clearance', 'entry_depth', 'bilateral_contact', 'tip_clearance', 'lift_follow'],
         action='未带起或滑转则在当次承托可确认后重评方向/抓点/接触/夹深，不反复收爪试提'),
)


def guide():
    return dict(
        id='desktop_grasp', basis='2026-10-30用户指定的桌面抓取流程，实机效果待逐次复核',
        reference_only=True, motion_authorized=False,
        order=[s['id'] for s in STAGES], stages=copy.deepcopy(STAGES), facts=copy.deepcopy(FACTS),
        view_role_note='10-30用户纠正：垂直桌面向下只约束抓取臂进夹；辅助腕在保持双臂/桌面间隙、'
                       '避开遮挡的前提下尽可能低位侧看，以直接辨认指尖、物体、桌面高度及入夹关系。'
                       '顶视看平面布局，不把辅助腕也调成朝下俯视；低位侧视不等于仰视。',
        axis_note='当前基坐标O/P绕∓X，属于倾斜；J/L绕±Z。工具Z朝下时，桌面内改变闭合方向'
                  '需按实际几何核对J/L，不把O/P误称绕桌面法线，也不暗改既有按键。',
        evidence_rule='每阶段实际查看新鲜顶视/双腕及双臂SDK，分别写清看见什么与遮挡。'
                      '仰视短期不纳入；旧案例、图像居中、开度/effort及已提起均不证明夹深。'
                      '一次review只核查该阶段，不证明其他阶段已完成；动作、遮挡或场景变化后重新核查。',
    )


def review_stage(stage_id, review, *, clock=time.time):
    """Check human/Codex authored observations, never certify their visual truth."""
    stage = next((s for s in STAGES if s['id'] == stage_id), None)
    if stage is None:
        raise ValueError('unknown desktop grasp stage')
    if review.get('arm') not in ('left', 'right'):
        raise ValueError('review requires operating arm')
    stamp = review.get('observed_at')
    if (type(stamp) not in (int, float) or not math.isfinite(stamp) or
            not 0 <= clock()-stamp <= 60):
        raise ValueError('stage review requires observations within 60 seconds')
    facts = review.get('facts', {})
    if not isinstance(facts, dict):
        raise ValueError('facts must be an object')
    blockers, accepted = [], {}
    auxiliary = 'right_wrist' if review['arm'] == 'left' else 'left_wrist'
    for name in stage['required']:
        item = facts.get(name, {})
        if not isinstance(item, dict):
            item = {}
        observation = item.get('observation', '')
        views = item.get('views', [])
        if (item.get('status') != 'confirmed' or not isinstance(observation, str) or
                not observation.strip() or len(observation) > 1200 or not isinstance(views, list) or
                not views or any(v not in ('top', 'left_wrist', 'right_wrist', 'sdk') for v in views)):
            blockers.append(dict(fact=name, reason=FACTS[name]))
        elif name in ('height', 'entry_depth') and auxiliary not in views:
            blockers.append(dict(fact=name, reason='需要另一臂腕相机的可见关系描述；SDK/操作腕不替代辅助视角'))
        elif name == 'tcp_down' and ('sdk' not in views or not set(views) & {'top', 'left_wrist', 'right_wrist'}):
            blockers.append(dict(fact=name, reason='需要SDK工具方向及源RGB交叉核查'))
        else:
            accepted[name] = copy.deepcopy(item)
    return dict(stage=stage_id, arm=review['arm'], observed_at=stamp,
                ready_for_next_local_action=not blockers, blockers=blockers, accepted=accepted,
                next_observation=stage['action'], reference_only=True, motion_authorized=False,
                limitation='仅检查所填观察的完整性；不自动识图、不证明事实真实、不控制机器人或代表实机成功')
