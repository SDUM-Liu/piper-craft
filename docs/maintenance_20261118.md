# 公开版维护同步（2026-11-18）

本次将 11-02 公开版本之后的本地软件维护同步到公开仓库。设备占位配置、许可证、第三方声明和已发布倒水 Demo 保留；现场记忆、设备身份、会话、原始录像及控制日志不随代码发布。

## 改动

- J1–J6 绝对角度输入、`teleop_joint_target` RPC 和普通 MOVE_J 执行，共用原有会话、停止、反馈、加速度及跟随守护。目标绑定所选臂和输入序号，到位 0.1° 内及随后连续 0.5 秒静止才完成。
- 快档临时禁用并迁移旧启动偏好；移除固定 TCP 纯旋转 2 mm 漂移异常检查及配置，保留普通目标跟随预算。普通点动不再因 URDF 名义边界建议绕行，低 IK 进展搜索仍保留。
- Agent 第一阶段的二维标记感知、可信可执行技能、Episode、单因素假设、版本冻结和离线配对评价；真机适配器只读，隔离实验不写真实成功地图。
- 文档更新辅助腕退避、完整任务目标与清理边界；保留公开版示范文档对“没有本地自碰撞检查”的准确说明。

## 实际验证

使用本机已有 Python 3.11 项目环境执行，未启动设备、机械臂控制或实时取景：

| 检查范围 | 结果 |
| --- | --- |
| 本地维护源树全量 pytest（含已有离线模型环境） | 780 passed、3 skipped |
| 公开仓库副本全量 pytest（设备占位配置） | 763 passed、20 skipped |
| 两份代码的 Ruff | 通过 |
| 两份代码的 Memory compact/check | 通过，每页 ≤100 行，总计 ≤260 行 |
| 公开版 Agent CLI，8 对离线实验 | 正常结束，仅离线继承候选 |

跳过项依赖未随公开仓库提供的机器人模型/解释器环境或本机历史运行记录。离线模型测试也不连接 CAN，不代表实机运动、控制计算超时根因或历史硬件故障已经验收/修复。

本次 CLI 合成实验使用 parent gain=0.18、candidate gain=0.65；8 对相同初始条件中独立评价成功数为 0/8 与 8/8，平均动作尝试数为 8 与 4.625，单侧配对置换 p=0.00390625。结果仅适用于合成二维图像对齐，不能推广为真实识别、抓取、放置或提速；未完成 A–E 五组完整消融。工件在本机忽略目录保留，不以缺失的公开文件链接充当证据。

## 重现

安装依赖后，在仓库根目录执行；实验输出须使用新的空目录：

```bash
uv sync --locked --python 3.11 --group dev
uv run pytest -q
uv run python -m ruff check astra_vla tests scripts
uv run python -m astra_vla.operation_memory compact
uv run python scripts/check_robot_memory.py
uv run python -m astra_vla.agent experiment --output /tmp/piper-agent-new-run --pairs 8
```

随机种子、统计规则和动作结果可重现，UUID、绝对路径和墙钟时间会不同。首次配置真实设备及后续自主抓放仍按 [Agent 分阶段说明](agent.md) 和 [遥操作流程](teleop.md) 单独验收。
