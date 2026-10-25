# 操作记忆维护规则

目的：让后续实体操作复用事实和修复经验，持续改进当前控制框架。这里是有界操作记忆，不代表模型权重训练或无人监督代码修改。

## 必须遵循的维护顺序

1. 操作前读 `current.md` 与 `problems.md`；历史反馈不能代替实时设备状态。
2. 实际动作保存输入、目标、反馈与需要的影像；静态观察不创建操作经历。
3. 收尾自动汇总当前控制记录；问题原位更新，保留具体失败和验证范围。
4. 修改后运行相关离线检查；实机结论必须另有实际动作和影像。
5. 删除过期状态，合并重复原因，保留尚未解决的问题；随后compact/check。

## 证据标签

- **观察**：实际日志、反馈、RGB或用户当次确认的事实。
- **推断**：原因假设；未验证时必须明确，不写成观察。
- **修改**：具体代码、参数或操作策略变化。
- **验证**：已完成的测试及边界；软件通过不等于实机通过。
- **证据**：仍保留的相对路径，失效引用及时清理。

状态：open、software_verified、hardware_verified、resolved。禁止自动从离线测试推断硬件成功。

## 长度与整理

`current.md`最多12次摘要，`problems.md`最多8项；每个Markdown≤100行，总计≤260行。自动写入会有界压缩；同一问题更新替代旧条目。`state.json`仅为这些简短摘要的结构化来源，不堆叠原始日志。

```bash
.venv/bin/python -m astra_vla.operation_memory compact
.venv/bin/python scripts/check_robot_memory.py
.venv/bin/python -m astra_vla.operation_memory problem sdk-pose --observed '实际现象' --inference '待验证原因' --change '具体修复' --validation '实际检查' --evidence 'runs/当前证据' --status software_verified
```
