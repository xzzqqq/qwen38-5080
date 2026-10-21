# 数值与容量验收：经验与参考

这里记录这一阶段踩过的坑和可复用的思路；具体以对应 plan 和脚本为准。

## 1. 先审 reference

常见问题不是模型，而是 reference 本身写错。建议先逐行对照上游实现：

- 公式、归一化、`eps`、dtype；
- 舍入/tie 规则、scale 语义；
- shape、padding、layout。

用小而确定的输入验证 reference，再去看模型误差。如果 reference 不等价，先修
reference，不要用放宽容差掩盖。

## 2. gate 和 diagnostic 分开

- gate：决定任务能否关闭；
- diagnostic：只观察和解释，允许 failure。

建议在 journal 里明确标注，不要把 diagnostic failure 写成“通过”。哪些项算 gate
由具体 plan 决定，不需要所有指标都进 gate。

## 3. 退出判定可以检查结构条件

如果脚本负责 gate，除了平均误差，可以顺便检查：

- reference count 是否等于期望数；
- 输出是否 finite；
- dtype 是否符合预期；
- shape / cache length 是否正确；
- gate 的 failures 是否为 0。

这不是要求所有验收都写死这些字段；如果写了 gate，最好让它能复现。

## 4. 每条 request / prompt 隔离状态

容易出错的地方：

- 生成下一条输入前忘了关闭 capture；
- record 直接引用会继续被写的 store；
- worst 的 name 实际来自下一条 prompt。

按需使用独立快照（例如 `deepcopy`），并检查 worst 的归属。

## 5. worst 聚合注意不要丢指标

如果用 worst 做汇总，分别维护 max nrmse、max max_abs、max max_rel、min cosine
及对应位置，通常比“nrmse 更大就整体替换”更可靠。

## 6. 比较前先判断可比性

同一路径、同一 shape/长度、同一 kernel 的差异才适合严格比较。不同路径或不同
KV/序列长度的差异，先说明是否可比，再决定是否当回归依据。

例如 full 256 vs `128+128` 可以严格比较；full 300 vs `128+128+44` 的 attention
KV 长度不同，不适合直接做 bitwise 判据。

## 7. 峰值显存按 phase 看

如果关心分阶段峰值：

- load 前 reset，记录 load 结束和 load peak；
- prefill 前 reset；
- decode 前再次 reset。

记录时区分 `max_memory_allocated()`、`max_memory_reserved()` 和
`mem_get_info()` free；必要时补 `nvidia-smi`。

## 8. 失败样本尽量可解释

如果某项失败，除了计数，尽量保存：

- tensor/module 名；
- nrmse / max_abs / max_rel / cosine；
- scale 差异、code mismatch、输入极值；
- 第一个或前几个失败样本。

如果差异来自 rounding/tie/边界，写清楚规则和不能完全复现的原因。

## 9. 记录与提交

每次实验建议记录：

- 命令、commit、checkpoint revision；
- 依赖版本和 results 目录；
- gate/diagnostic 结论；
- 已知限制。

提交前按任务需要检查测试、`py_compile`、`git diff --check`、文档链接/表格，
并清理中间 results。失败实验也值得写 journal。
