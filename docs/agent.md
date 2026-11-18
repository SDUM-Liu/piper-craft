# Agent 第一阶段：证据、可执行技能与离线配对闭环

本阶段交付一个可运行、可测试的窄闭环：原始三路 PNG → 配置颜色区域/二维特征 → 本地技能选择与误差反馈 → 隔离的图像平面实验 → 独立评价 → System 2 单因素假设 → 候选冻结 → 配对选择。它尚不是通用视觉抓取 Agent。

下文“本阶段/没有修改”描述 2026-11-03 Agent 独立实现的范围；随后遥操作维护增加了绝对关节角度输入并临时禁用快档等，见 [操作说明](teleop.md) 与 [公开同步验证](maintenance_20261118.md)。

真实机器人仍通过原有 SDK/ROS/CAN、IK、普通 MOVE_J、桌面遥操作和 TaskArchive 工作。新增真机适配器只读 `observe.capture` 与 `robot.status`。没有新增运动 RPC、伪造焦点/输入心跳、接管已有会话、启用 MOVE_P、修改速度/恢复/绕行/停止或其他控制约束。此次没有真机动作、模型训练或人类示范学习流程。

## 仓库梳理与设计依据

原有模块的职责边界决定了接入位置：

| 现有代码 | 原有职责 | 本阶段接法 |
| --- | --- | --- |
| `observe.py`、`rgb_frame.py`、`workers/cameras.py`、`workers/rgb.py`、`clock_mapping.py` | 源 RGB、身份/时间/原分辨率 | 复用 PNG 与原元数据，另建不可混淆的 observation ID 和 SHA256 |
| `grasp_guidance.py` | 检查人/Codex 填写的阶段观察 | 保持原义，不包装成自动识别或运动守护 |
| `teleop_desktop.Client`、`workers/teleop.py`、`workers/robot.py` | 控制所有权、真实 UI 心跳、SDK 健康、停止 | 保留；第一阶段不将 headless Agent 冒充桌面客户端 |
| `teleop.py`、`teleop_ik.py`、`kinematics.py`、`teleop_detour.py`、恢复/回 Sleep | 普通控制几何与守护 | 没有修改 |
| `operation_memory.py`、`task_memory.py`、`docs/memory/` | 经历摘要、经验与成功地图 | 新建只读选择性检索，不导入全部历史或示范，不自动写成功地图 |
| `task_archive.py` | 当前任务绑定、工件归属、锁、收尾和清理 | 增加 `agent` 工件种类，位于同一任务的 `episodes/`；要求明确匹配活动 task_id |
| 录像、示范、公开消息、启动器与 UI | 原有观察/接管/诊断入口 | 保持接口和行为 |

参考的是 PhysicalRSI 提交 `2574f72bebf9b357140ebe376db71e782575fe65`，没有复制、安装其框架。借鉴点对应原始来源：

- [Skill Registry](https://github.com/PhysicalRSI/PhysicalRSI/blob/2574f72bebf9b357140ebe376db71e782575fe65/src/physicalrsi/skills/registry.py)：声明与可信可执行实现分别绑定，版本可回溯。
- [Self-Harness selection](https://github.com/PhysicalRSI/PhysicalRSI/blob/2574f72bebf9b357140ebe376db71e782575fe65/PhysicalRSI_core/self_harness/selection.py)：冻结比较、完整配对、证据校验及统计选择。
- [Research protocol](https://github.com/PhysicalRSI/PhysicalRSI/blob/2574f72bebf9b357140ebe376db71e782575fe65/docs/research-protocol.md)：开发/验证隔离，复用验证集不能冒充新验证。
- [RGB region provider](https://github.com/PhysicalRSI/PhysicalRSI/blob/2574f72bebf9b357140ebe376db71e782575fe65/src/physicalrsi/perception/grounding_dino.py)：二维区域不直接升级为物体姿态、抓点或接触事实。

该提交中未找到请求提到的 `visual_perception.py`、`visual_servo.py` 同名文件；上述是实际检查到的替代参考。这里实现本地二维几何/反馈拟合，没有照抄假定存在的 API。

## 运行

使用仓库 `.venv/bin/python`；系统 Python 在本次检查中有 NumPy/SciPy ABI 不兼容且缺少 av，不能充当项目测试解释器。

```bash
# 全自动离线演练；输出目录须为空；不连接任何硬件
.venv/bin/python -m astra_vla.agent experiment \
  --output /tmp/piper-agent-experiment --pairs 8

# 读取现有源帧，profiles 是 JSON：标签 -> [HSV下界, HSV上界]
.venv/bin/python -m astra_vla.agent inspect \
  --profiles /tmp/marker_profiles.json \
  --observation /tmp/piper_observation/observation.json \
  --output /tmp/piper-agent-scene

# 省略 --observation 时只读当前三路 RGB 和双臂 SDK，依然不会运动
.venv/bin/python -m astra_vla.agent inspect \
  --profiles /tmp/marker_profiles.json --output /tmp/piper-agent-live-scene
```

示例配置 `{"red_marker": [[0, 180, 180], [8, 255, 255]]}` 表示人为配置的红色标记类别，不能将标签改成 bottle 就宣称实现通用瓶子识别。HSV 的 H 为 OpenCV 的 0–179。

Python Task Interface 是 `astra_vla.agent.runtime.Agent.execute_task`。第一阶段支持 `inspect_scene` 和离线 `align_image`；不支持的 `pick_and_place` 会在创建 Episode 前明确拒绝，不静默降级成“已完成”。图像对齐 destination 必须是 `[u, v]` 像素坐标。调用者先建立/选择正确 TaskArchive，提供可信适配器与注册表。没有导入时设备连接。

## 已实现的数据与执行合同

`SceneState` 保存 timestamp、observation_id、各相机 camera_id/时间戳/序列号/serial/源图绝对路径/SHA256/尺寸、objects、object_features、robot_state、gripper_state、visibility、confidence、uncertainty、spatial_relations 和 provider_identity。三个相机独立记录，不假定同步、不做未标定的跨视角三维融合。

颜色基线执行连通域检测、类别配置匹配、包围框、质心、PCA 轴线及轴端点。多候选不按列表顺序猜目标；出画区域标 partial，遮挡类型未知，近圆区域轴线未知。confidence 是掩膜实心度，不是校准后的类别概率。物理可抓区域未确认时 `actionable_regions=[]`，不会把包围框中心自动叫作抓点。

已给出标记的左右指尖/palm 可以计算二维闭合/接近方向、指间投影和相对位置；连续可比帧可以计算以指间宽度归一化的表观滑移。视角透视或错误关联也可能产生该信号，因此 physical_slippage、entry_depth、bilateral_contact、tip_table_clearance、接触力与物理抓持结果保持 unknown。`inspect_grasp` 技能不会用二维投影放行真实抓持。

局部视觉伺服根据实际反馈增量和图像特征增量拟合 2×2 Jacobian，拒绝零位移、秩不足、病态或残差过大的样本；每次局部动作后重新观测，不读取模型推理来完成每个周期。样本只在当前 Episode 内使用。第一阶段动作和初始标定样本只来自明确标注的图像平面模拟器；proposal 本身 `motion_authorized=false`，不能交给真机绕过既有控制。

Skill Registry 保存明确的 skill_id/name/version/description/input_schema/output_schema/preconditions/completion_conditions/failure_conditions/parameters/implementation/evidence/applicability/dependencies。可执行内置技能为 observe_scene、detect_object、estimate_object_orientation、estimate_visual_error、inspect_grasp、align_image。

注册表绑定可信 Python callable 与源文件闭包摘要；JSON 不执行任意导入或代码。输入/输出按有限且严格的 schema 验证，schema 不支持的字段拒绝。实现中的条件判断决定完成/失败，描述文字不授予权限。重启后必须重新注册可信 callable；恢复历史 manifest 不会自动恢复已删除的历史 Python 源码。源文件改动后旧绑定拒绝执行，需要恢复源码或冻结新版本。

组合是有预算和取消检查的结果分支图，未知状态默认终止；slippage_detected/needs_observation 可显式进入恢复边，不能默认继续 transport。图像对齐要连续两张新帧满足误差门槛才结束技能，完成技能仍不等于任务成功。

## Memory、Episode 与 System 2

- Episodic：同一 TaskArchive 下的 Episode 保存唯一 ID、场景/策略/技能/Memory/模型身份、初始与连续观察、动作尝试及反馈、技能调用、视觉误差、阶段/原因、耗时/调用/观察/恢复/介入计数、源图引用和独立评价。
- Semantic：检索既有成功地图的事实、完成范围、场景假设与不可迁移边界。
- Procedural：检索既有问题中的观察/推断/修改/验证；知识不是可执行技能。默认最多 4 项，可限定 1–8 项，不读取示范视频。检索结果均要求现场适用性评估。

执行中只更新 Episode 短期状态；没有把离线结果写入现有真实成功地图。任务结果默认 unknown，`model_judgment`、`execution_status`、`evaluation_result` 分开。SDK 到位、正常停止、闭合文件或技能返回 succeeded 都不能填物理 success。动作失败也记录已尝试调用，不能因未收到反馈就漏计。

System 2 读取完整 Episode 的阶段、错误序列、动作计数和证据引用，提出待验证假设：识别不可用优先指向感知；对齐预算耗尽且连续已知误差下降才可提出 gain 候选；其他执行/局部模型问题不会一律修改抓取。输出原始证据、问题、原因假设、修改对象、预期收益、验证方法、负面影响和因果未证实标记。

本阶段候选只修改一个 gain 参数；自动生成任意 Python Skill 实现、复杂语义任务规划与长期知识自动审批不在此阶段。候选保留 parent、修改原因、源代码闭包、技能版本和 scope，可恢复父版本；无验证候选不会写默认别名。

## Self-Harness 与可观察指标

实验先运行独立的开发场景产生假设，再冻结 parent/candidate 和比较合同，最后运行新验证场景。同一组条件固定模型、感知、任务分布、控制接口、动作预算和独立评价器；配对初始图像、任务输入与模拟 reset 身份必须一致，父/候选顺序交替。

每个验证结果引用不可覆写的唯一 trial、Episode、全程源图和独立评价凭据。注册与选择时都重检哈希、实际运行参数、版本、预算、模型/感知身份、计数和结果。缺失配对、unknown、样本不足均 inconclusive；重复 trial、改写证据、开发/验证重叠、同一实验目录复用验证集、版本或初始场景不一致直接拒绝。哈希检查完整性，不认证外部评价者的诚实；跨目录人为换 ID 的数据泄漏仍需外部数据管理。

指标齐全保留：task_success、completion_time、number_of_actions、number_of_observations、model_calls、recovery_count、intervention_count、execution_failure、grasp_slippage、placement_error。无法测得的指标为 JSON null。失败任务的 completion_time 为 null，执行耗时仍在 Episode；动作数是尝试数。二维像素误差单独在评价凭据中，不能写成三维 placement_error。

使用预先固定样本量的单侧配对置换检验（≤16 个非零配对精确枚举，否则固定种子 Monte Carlo），alpha 按声明的比较预算校正，同时报告均值差、保守的双侧配对 Hoeffding 区间、样本量和回归。至少 8 对；不使用观察中途的显著性来提前结束。候选需达到预定实际收益门槛、通过检验且无已观测成功回归/候选执行错误。

仅 `promote_offline` 可写 `offline_default.json`，会重新验证全部证据并检查父版本未被替换；不能提升任何真机默认技能。失败/未知子版本和 parent 都留在实验目录中。

离线实验工件使用独立 TaskArchive 根目录，保留全体实验评价证据，符合本次“保留所有评价证据”的要求；不更改原有真实失败任务复核后删除材料的约定。真实操作归档的既有清理行为仍通过回归测试。

## A–E 消融协议和当前边界

附件只明确了 E，A–D 在本阶段作以下显式定义，写入实验目录的 `ablation_protocol.json`：

| 组 | Memory | 固定 Skill | 自主探索 | 修改 Skill |
| --- | --- | --- | --- | --- |
| A | 无 | 无 | 无 | 禁止 |
| B | 有 | 无 | 无 | 禁止 |
| C | 有 | 有 | 无 | 禁止 |
| D | 有 | 有 | 有 | 允许经冻结与配对验证 |
| E | 有 | 有 | 有 | 禁止 |

E 的候选修改拒绝规则有可执行测试。当前只执行了 D 内的 gain 单因素 parent/candidate 实验；A–E 表是后续实验合同，**不是五组已跑完的性能结论**。没有高层模型与真实物体分布，不能回答 Memory 是否提高完成率、Skill 是否减少模型成本或探索与实现修改各贡献多少；这些结论均待验证。颜色基线的模型调用数恒为 0，不能据此推断减少了多少模型调用。

## 分阶段交付及停止条件

| 阶段 | 交付/验收 | 停止条件 |
| --- | --- | --- |
| 1，本次 | 可执行感知/技能/记录/假设/冻结/配对选择的离线链路；新测试及原有回归；只读真机接口 | 没有可靠特征、重复/过期帧、局部模型不可信、停止/动作预算/时限触发即结束相关分支；未知保留未知 |
| 2 | 标注好的真实三路 RGB 回归集；可替换语义识别/关键点后端；测检测/关联/遮挡/方向误差；将运动意图接入真实桌面所有权与焦点/心跳，不改控制守护 | 模型/身份/序列/反馈不新鲜、抓点/接近方向/高度/夹深/承托或双臂间隙未知，不推进物理动作；接管与停止全路径未验证，不接通自主运动 |
| 3 | 按 desktop_grasp_guide 次序实现并验证抓前/接近/闭爪/短提/搬运/释放/结果检查；可控真机配对实验，区分自主与人工辅助 | SDK 到位不得替代实物验收；滑移回到抓点/方向/接触/夹深核查；预算不足或结果不明确不继承 |
| 4 | 相同模型/感知/任务分布/预算的 A–E 完整消融；Skill 实现变更需新源闭包和独立验证集 | 数据泄漏、条件不匹配、样本不足、统计无改善或出现回归均保留 parent，不宣称提升 |

## 验证方法

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check astra_vla tests scripts
.venv/bin/python -m astra_vla.operation_memory compact
.venv/bin/python scripts/check_robot_memory.py
```

新增测试覆盖完整离线实验、真实 TaskArchive 接入、版本/源文件篡改、输入验证、未知结果、证据重检、初始场景匹配、泄漏/重复、样本不足、回归拒绝、E 组限制、二维几何限制、反例滑移分支、异常记录、动作预算和只读真机隔离。具体本次运行结果见 [验收记录](agent_validation.md)。
