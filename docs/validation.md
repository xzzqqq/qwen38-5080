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

## 10. bitwise 只在同 shape 有意义（M3）

M3 把“执行方式改造”和“attention 长度变化”分开，才能同时通过 gate 并解释差异：

- 同 shape 的三组（eager→static-exact、padded reference→static-cap、
  static-cap→graph）全部 bitwise；这是可以严格验收的部分。
- 不同 shape 的两组（eager→static-cap、eager→padded）差到 O(1)，但原因是
  `repeat_kv` 展开后的 matmul 在更长的 K 维上用了不同的 float32 累加分组：
  用 microbench 隔离单个 attention op 后，两条路径与 float64 参考的距离相同，
  差异只有 0.22 ulp。**不能用“模式名”判断 bitwise，也不要跨 shape 设容差。**

## 11. free-running 生成不能当 bitwise gate（M3）

即使逐 step（teacher forcing）bitwise，自由贪心生成也可能在几十步后分叉：
sub-ulp 的注意力差异经过 64 层 BF16 残差和 401 个 W4A4 GEMM 会被放大。
gate 应该用“EOS + 长度 + 内容”，与参考 trace 的差异只作 diagnostic，并记录
首处分叉的 top-2 margin，用来区分“实现差异”和“near-tie 翻转”。

## 12. launch-bound 与带宽-bound 的区分（M3）

只看 tok/s 会把两种完全不同的瓶颈混为一谈：

- 先算 `B_mean / tok/s` 得到逻辑带宽；17% SOL 说明不是带宽问题；
- 再看 nsys 的每步 kernel 数与 kernel 累计时间：86 ms 墙钟 vs 30 ms kernel
  累计、7242 个 kernel、93.9% 小于 5 µs → 提交开销占主导。
- graph 之后 kernel 累计几乎不变、墙钟掉到 kernel 累计附近，才进入真正的
  kernel 效率问题。**先用“每步 kernel 数 + 墙钟/kernel 比值”判断，再决定
  优化方向，不要直接按源码 op 数量猜。**

## 13. allocator 策略要固定并写进 provenance（M3）

`expandable_segments` 会改变 8K 的可行性和 peak/reserved 观测值：默认 allocator
在 7680+128 prefill 因碎片 OOM，开启后与 M2 记录逐位一致。它不改变数值，但
**不写进 provenance 就没法解释“同一脚本两次峰值不同”**。

## 14. 渐进式验证：把 loop 做小（M3）

原则：**改一点就跑最小闭环，全量只在里程碑关闭前跑。** 目标是最快拿到反馈，
不是每次都证明整条流水线。

| 层次 | 命令 | 覆盖 | 耗时 |
|---|---|---|---|
| 纯 CPU | `pytest tests/ops tests/model`、`python -c` 调 report/CheckReport | op 数值、报告语义、必需项覆盖 | 秒级 |
| 快档 | `make m3-quick` | graph capture、capacity guard、分阶段显存、两套计时 | ~40 秒 |
| 单场景 | `--only numeric-512` / `--only graph-1024`、bench `--repeats 1 --steps 32` | 改到哪个路径就跑哪个 | 1–3 分钟 |
| 全量 | `make m3-check` / `m3-memory` / `m3-bench` / `m3-profile` / `m3-sol` | 里程碑关闭 | 15 + 12 + 95 + 12 + 1 分钟 |

注：`make m3-quick` 的 40 s 实测拆分为 check 13.3 s / memory 12.9 s / bench 14.7 s，
每个子进程约 5–6 s 是固定开销（import + load）。只改一条路径时优先用
`make verify` 给出的单档，而不是每次都跑完整 quick。

M3 里实际踩到的三次返工，都是 loop 没做小：

1. 改完记录字段后直接跑全量 check（15 分钟），结果是产物缺 `provenance.json`
   才失败。**“产物是否齐全”用一个场景 3 分钟就能确认**，不需要五个。
2. fixture 生成连跑两次全量（每次 3 分钟）才定位到 8K prefill 的 allocator OOM；
   应该先 `--depths 7680` 单独跑。
3. 用脚本批量替换一个大 slice，`old` 恰好为空导致文件被写坏，只能整份重写。
   大范围编辑前先 `cp` 备份，并对“替换目标非空 / 命中次数”做断言。

对应到本仓库的做法：快档产物写 `.tmp/`，不进 `results/`；全量结果才进
`results/m3/<run-id>-*/` 并带 provenance。
