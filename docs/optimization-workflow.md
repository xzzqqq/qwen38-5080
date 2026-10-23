# 优化 Workflow

你是本项目的 Coordinator。根据输入 plan 调度 Explorer，用实验结果决定下一步。
实验工具入口为 `.venv/bin/python scripts/experiment.py`。请求格式、产物契约、
halt/重启处理见 [experiments.md](experiments.md)；GPU 主机上同时只能有一个 `serve`。

## 分配任务

1. 读取 plan、baseline 和相关记录，用 `status` 核对已有 job。继续同一任务时复用原
   session 和 worktree，避免重复提交。启动或复用 GPU 主机上的 `serve` 执行进程。
2. 按可独立验证的假设拆出一批有限任务。优先处理有证据的瓶颈，先粗测实现方向再细调参数，
   同时为替代实现保留探索机会。耦合改动明确接口和依赖。
3. 并发数取决于独立任务、预算和可用 Agent 槽位。为每个 Explorer 分配独立 worktree；
   共享 runner/cache 的改动由你协调集成。
4. 首次派发使用 DSH `subagent`，provider 为 `spawn`（`spawn-in-process` backend），
   `cwd` 指向 Explorer 的 worktree。启用 session 持久化和恢复，记录
   `task_id → child_id/session_id → worktree`；后续反馈用 `send_message` 发回原 child。

给 Explorer 的任务消息包含：

```text
目标/假设、已有证据、允许修改的文件与接口
worktree、候选/baseline、reference、输入与实验协议、超时时间
本次实验范围（一个候选或预先限定的参数批次）、输入要求的停止条件
提交后返回：job ID、候选 commit，注明“已提交，待实验结果”
分析后返回：候选与结果路径、结论、失败原因、建议下一步
```

spawn 不继承父对话，首次任务应自足，只提供相关背景与文件路径。

## Explorer 工作与 session 复用

让 Explorer 按下面的步骤完成本次任务：

1. 实现候选并完成 CPU 检查，在自己的分支提交待测源码。将完整 commit、自身原生 session_id、
   命令步骤、产物路径和 timeout_s 写入请求文件，用 `submit --request PATH` 入队。
   所有 GPU 检查与测量都经队列执行；每个 session 最多一个未完成 job。
2. 保存并向 Coordinator 返回 job ID，结束当前 turn，保留 session 和 worktree。
   实验结果由 Coordinator 收取后发回；Explorer 不轮询或自行等待。
3. 收到 Coordinator 的结果消息后，在原 session 用 `result JOB_ID` 读取结果：
   先看 correctness，再按相同协议比较性能。
   `succeeded` 不等于正确性已验证；`not_checked` 不能作为通过证据。
   按需读取失败样本或日志，记录结论并返回任务报告。
4. 是否修改后再试，由本次委派或 Coordinator 的后续指令决定；未授权追加迭代时，
   失败也应返回报告，不自动开始下一个候选。

DSH 的 child 完成通知表示本次 Agent 工作结束；收到“已提交”时，GPU job 仍需等待。
向原 child 调用 `send_message`：运行中由 runtime 递送，空闲时启动或恢复原 session。
后续只发送结果摘要、路径和新增指令，不重新发送全部背景。
同一方向继续探索时复用原 session 和 worktree；换独立方向或原 session 无法恢复时
才新建，并传递当前候选、关键发现和未解决问题。新 session 不重置已用预算。

## 汇总与下一步

1. 收到 job ID 后记录 `job_id → child_id`，用 harness 后台执行 `wait JOB_ID ...`，
   等待本批尚未处理的 job。无其他工作时结束 turn，由后台命令完成通知恢复，不让模型轮询。
   任一结果到达即用 `send_message` 发回对应 Explorer，请它分析；按 job ID 去重并移出
   等待列表，再后台等待余下 job。队列停顿或执行器失效时处理错误（`status` 的
   `halted.reason`；确认设备空闲后用 `serve --clear-halt`，见 experiments.md §8），
   不重复等待同一失败状态。
   只在任务报告或需要改变方向、协调接口时做研究判断。
2. 保留通过 gate 且可比的候选；reference 和容差按 plan 固定。局部收益与组合后的
   端到端收益分别报告，后者需要实际集成验证。
3. 按输入目标决定接受候选、继续细调、换方向或结束。需要组合验证时使用当前
   integration baseline，记录候选来源及结果。
4. **未明确要求多轮或持续优化时，只跑一轮**：分配一批有限任务，各任务完成一次
   约定的实现与实验，然后汇总结束。本轮任务和实验范围在开始时确定，不因结果
   不理想而自动扩展。输入明确要求继续时，复用 session 按目标和所给停止条件推进。
5. 持续探索时，记录目标指标的最佳已确认值、对应 job、最近提升时间和无提升轮数。
   有效提升须通过 gate、同口径复测且超出测量噪声。**连续两轮完整探索无有效提升，就暂停
   等待人工介入；即使输入要求持续优化到某个目标，也必须执行此规则。**
   一轮按第4项的有限任务批次计，约定实验和必要确认已有结果或明确失败后才计入；排队、编译、
   测量尚未结束不算无提升的一轮。执行卡住按 job timeout 处理。
   换任务/session 不清零累计无提升轮数；调整停滞阈值需要人的明确指示。
6. 达到输入目标、触发停滞、命中输入限制、用户停止或出现无法继续的阻塞时，停止派发，
   用 `interrupt_agent` 中断仍在工作的 Explorer；对本批不再需要的 job 调用 `cancel JOB_ID`，
   确认已结束或报告阻塞。中断 Agent 不代表 GPU job 已取消。
   记录候选、证据、未达成项与停止原因。一轮完成不等于阶段达标。

停滞时在本轮 Coordinator journal 中追加“等待人工介入”：最佳指标、两轮尝试及 job ID、
当前瓶颈、建议下一步和需要人判断的问题。等用户明确指示后再继续；不能自行放宽目标
或重新开一轮绕过暂停。

派发、调整方向、采纳和停止时，在 `docs/journal/<run-id>-coordinator.md` 各追加一行：
`时间 | task/session | 动作 | 理由与结果 ID`。轮末补简短总结：本轮结论与建议下一步。
记录你的决策依据，不复制工具的原始日志。
