# 自动任务记忆、元操作节点与地图

Codex 开始每个任务或继续中断任务时，先读 Memory 三页和 `task_maps.md`，再以新输入指令运行 `operation_memory planning-context`。这会自动发现本机所有完整真实示范和成功案例地图，不依赖示范面板的手动选择。候选每页最多32条；返回 `next_demo_offset` 时由Codex用 `--demo-offset` 继续查看相关旧示范，不能把第一页当成全部历史。保存新示范会刷新索引；名称或描述缺失、结果未复核的示范自动列为待分析。分析和推理由当前 Codex 完成，录制器不在后台启动模型，也不把录像正常保存当成任务成功。

## 主动维护流程

1. 自动查看相关候选、历史完成条件和适用边界。对待分析示范导出并查看三路关键帧；均匀抽帧不足时查看连续关键阶段。生成清楚的标题、任务、动作顺序、描述、标签、观察范围及遮挡/失败边界，标明 Codex 分析来源。只改显示名称，保留目录 ID 和旧证据路径。
2. 复核成功：以真实 RGB/SDK 或当次用户确认验证任务的完成条件。人工遥控成功案例同样可用。未知、未完成、失败、模拟或仅停止/文件收尾正常，均不能支持地图。人手示范中闲置机械臂的关节反馈不是人的动作轨迹，不为此建立机器人关节节点。
3. 将成功案例抽象为元操作节点：操作意图、前置条件、完成判据、观察证据、动态调整条件及案例来源。按需要配备文本、三路视觉/视频时段和 SDK 关节/夹爪/末端状态；这些数值只能作为参考，不生成可执行航点或按键脚本。节点之间的连接也必须有同一成功案例支持，不能拼出未成功验证的路线。
4. 针对新指令及现场，逐项评估目标、物体、场景、双臂分工、起姿、控制后端、抓持、完成条件、旧案例限制，写明通过/调整/拒绝及原因。使用最近60秒的三路图像和双臂反馈，检查时间戳。每次有效场景/指令变化重新评估；执行前仍观察最新 RGB 和反馈，短期复用评估不是运动许可。
5. 执行中按实际可见状态推进节点，动态调整抓点、视角、路径和节奏，不照抄旧坐标、力或时长。新调整在当前任务评估中记录；只有新的真实成功证据才能提升为长期地图版本。失败进入问题记忆，不将失败节点/路线存入成功地图。
6. 旧参考导致失败时立即 `invalidate --reason reference_failure`，删除地图和复用评估，只保留简短失效原因。证据丢失/更改、成功标签撤销和到期由 `compact` 自动移除；过时但证据仍在的地图由 Codex 主动复核、修正或移除。修正必须重新关联成功证据，不能把失败补丁包装为成功经验。

这是有界经验维护，不是模型权重训练；不宣称规划提速或成功率提升已经测得。

## 本机数据与边界

- `docs/memory/task_maps.json` 保存最多8组地图，每组最多24节点、48连接和8个成功案例。只存必要信息；每次修正递增版本并清除旧复用评估。
- `task_maps.md` 是自动生成的短索引，也列出待分析和成功人类示范。Memory 每页100行、合计260行的限制继续执行；完整图和素材不复制进摘要页。
- 最多保留24份短期复用评估和16条失效原因，不保留被移除的失败图。新指令、场景ID、地图版本、证据指纹或有效期不一致，返回 `assessment_required`。不同场景拒绝某图不会把该图本身当成失败，仍可能适用于原场景。
- 数据和原始示范仅存本机；公开仓库发布维护逻辑、文档和测试，不发布私有图、相机素材或控制日志。
- 该模块没有运动发送入口。原生/本地后端的既有停止、输入、健康、反馈及各自检查范围继续生效。

## Codex 使用的维护接口

```bash
python -m astra_vla.operation_memory planning-context --instruction '本次完整指令'
python -m astra_vla.task_memory put runs/当前维护/成功地图.json
python -m astra_vla.task_memory assess 地图ID runs/当前任务/复用评估.json
python -m astra_vla.operation_memory planning-context --instruction '本次完整指令' --scene-id 本次现场ID
python -m astra_vla.task_memory invalidate 地图ID --reason reference_failure --explanation '参考导致失败的观察和范围'
python -m astra_vla.operation_memory compact
python scripts/check_robot_memory.py
```

地图 JSON 的必需字段：`id/name/task/description`，`scene_assumptions/success_criteria/non_transferable` 文本列表，`cases/nodes/edges`。路径均相对仓库根，必须存在于仓库内。可用 `valid_until` 设置复核期限。

案例：`id/kind/source/review`。`kind` 为 `robot_task/human_objects/human_teleop`；`source` 指结果 JSON 或 `demonstration.json`。`review` 需要 `outcome=success`、`basis=visual_review/user_confirmation`、`statement/evidence/result_checks`；检查字段必须与源结果一致且不能只有停止/收尾。人类示范源本身也须已复核成功，三路录像存在。案例声明仍需 Codex 实际看证据，结构校验不是对视觉结论的独立验证。

节点：`id/operation/preconditions/completion/adaptation/case_ids/evidence`；按需增加角色和文本/视觉描述。`reference_state` 可含 `source/arm/joints_deg/gripper/sdk_ee_pose` 等参考信息，但必须标记 `reference_only=true`。连接：`from/to/condition/case_ids`，来源案例须同时支持两个端点。

复用评估：`instruction/scene_id/observed_at/scene_snapshot/scene_evidence/checks`。`scene_snapshot` 使用现有观察 JSON 的 `cameras` 和 `robot.arms`；相机三路都新鲜、双臂反馈新鲜。`scene_evidence` 为三路名称到对应图像的映射。`checks` 的9项对应上面的评估维度，每项包含 `verdict=pass/adapt/reject` 和 `reason`；有调整必须提供 `adjustments` 文本列表，可用 `node_id` 表示当前节点。评估结果取最严格一项，且只对该指令、现场和地图版本短期有效。

## 每段长按后的实读反馈

窗口消息区自动显示每个接受的按键组合结束后的 SDK 六关节角度、夹爪开度、effort参数与使能状态，附按键、持续时间和步骤编号。所有桌面控制都适用，包括 Codex 驱动的按键、人工遥控和关节点动。

先显示松键时的新鲜实读，等待既有静止窗口后更新同一条记录。若输入组合改变并继续运动，不把后一步的最终状态冒充前一步的静止状态。缺失、非有限或过期反馈显示不可用；没有发送动作目标也会注明。SDK effort 不是标定后的指端力。状态 RPC 和控制记录保留最近32步，窗口沿用消息区显示最近记录，故障消息优先；这不是成功判定，也不触发运动。
