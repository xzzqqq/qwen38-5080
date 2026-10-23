# 2026-10-23 Phase 2 任务 F：per-shape NVFP4 GEMV backend 选型（cute-dsl）——采纳

- 日期：2026-10-23
- 对应：[phase2 plan](../plans/phase2.md) §3 任务 F、§5 gate、§6 反馈阶梯；
  [decisions](../decisions.md) 的 M2 NVFP4 容差（`nrmse ≤ 5e-3`、`cosine ≥ 0.99999`、
  `max_rel ≤ 1e-2`、finite）
- 分支：`exp/phase2-tactics`，base = 集成 commit `9e35873`
- commit：v1 `75301c0`（op，基线缺失 → failed）；v2 `959b667`（op，通过）；
  接入 `3c3ac27`
- job：`job-000032-08b17afa`（v1，failed，139.9 s）、`job-000035-461c30c4`
  （v2 op，succeeded，107.7 s）、`job-000037-150586b0`（check+bench，succeeded，217.8 s）
- checkpoint：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` revision
  `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`
- 环境：torch 2.14.1+cu130、FlashInfer 0.7.2、cutlass-dsl 4.8.1、cuDNN 9.24、
  CUDA 13、RTX 5080 16 GiB（sm_120a）
- 产物：`results/experiments/job-000035-461c30c4/output/op/op.json`、
  `results/experiments/job-000037-150586b0/output/{check/check.json,bench/bench.json}`

## 0. 结论

**采纳**：SM120 上 `mm_fp4` 的 backend 在不同 shape 上差 1.5×以上；把 M=1 decode
的 backend 按 `(N, K)` 固定成一张静态表（11 个 shape 全部选 `cute-dsl`），
L2-free 聚合带宽 **cutlass 760 → cute-dsl 870 GB/s**（18.94 → 16.55 ms），
端到端 **graph-8192 27.781 → 25.084 ms（−2.697 ms / −9.7%）**、
**graph-512-1024 26.779 → 24.090 ms（−2.689 ms / −10.0%）**；
系统吞吐 35.99 → **39.81** / 37.29 → **41.42 tok/s**。

- 数值：cute-dsl 与 cutlass 在 11/11 个 shape 上**逐位一致**
  （`nrmse = 0.0`、`max_abs = 0.0`）；`numeric-512` 的 `eager→static-exact`、
  `padded-ref→static-cap` 与 `graph-1024` 的 `static-cap→graph` 在端到端仍是
  bitwise（check 139/139 全过）。
- `auto` 在 SM120 上等于 b12x（更慢）；cudnn 全面更慢；tactic 级不可接入
  （见 §2），只做 backend 级。

## 1. 要回答的问题（plan §3 任务 F）

1. 每个 shape 上，比当前 cutlass 更快的 backend 是否存在？差多少？
2. per-shape 选型能不能把整步平均从 747 GB/s 拉向 850+（−1.0 ~ −2.5 ms）？

答案：存在；cute-dsl 11/11 全胜，整步 870 GB/s，实测 −2.70 ms（比 op 级估计还多
0.31 ms）。cutlass 在 q_proj/o_proj/out_proj/in_proj_z/k/v 上只有 648–742 GB/s，
是主要差距来源。

## 2. 两个失败记录（都发生在测量脚本，不是 kernel）

### 2.1 v1：cutlass 基线一条都不产出（job-000032，commit `75301c0`）

v1 的候选列表是「官方 backend + tactic 子实验」：cutlass 的 32 档 tactic 通过
`get_gemm_sm120_module_cutlass_fp4()` 返回的 module 调 `module.fp4_gemm(..., tactic)`。
实际返回的是

```python
SimpleNamespace(cutlass_fp4_gemm_runner=...)   # 只有 runner 工厂
```

没有 `fp4_gemm_tactic_num` / `fp4_gemm`。`_cutlass_configs()` 在 stage builder 里
抛 `AttributeError`，而 stage 是「先构建、后测量」，于是**整个 cutlass stage
（包括本来就该走官方入口的 `cutlass:default` 基线）一条都没跑**。

产物表现：每个 case 的 gate `coverage: backend cutlass ...` 失败（
`produced_configs = 0`），`wiring: cutlass 默认 tactic == 显式 tactic 21` 失败
（`bitwise_equal_tactics = []`），agreement 基准退化成 b12x（所以 b12x/cute-dsl/cudnn
之间的 agreement 全 0.0 是"对 b12x 的 0.0"，不能当 cutlass 的证据）。
`proposed_table` 因此全填 cute-dsl 而没有基准，不可用。

**修法（v2）**：cutlass 基线永远只走官方入口 `mm_fp4(..., backend="cutlass")`；
每个 backend 一个独立 stage，构造/编译失败只记 `unavailable`，不影响其它 backend。

### 2.2 为什么收敛到 backend 级（tactic 级不可接入）

- b12x / cudnn 的 tactic **可以**经私有 `TunableRunner.get_valid_tactics()` +
  `forward(tactic=...)` 到达（v1 里确实跑出了数据，例如 cudnn 默认在 v_proj 上
  160 GB/s、某个 engine knob 到 455 GB/s，但仍低于 cute-dsl 默认的 505）；
- cutlass 的 tactic 要拿到底层 tvm module 才能调（`gen_...().build_and_load()`），
  属于私有路径；
- 关键问题不是"能不能测"，而是**能不能在 `linear.py` 里复现**：官方入口
  `mm_fp4(backend=...)` 没有 tactic 参数，静态表只能选 backend。把 tactic 写死需要
  在模型 forward 里直接调私有 module/runner，超出本任务允许的改动面。

所以 v2 删掉全部 tactic 子实验，收敛为 **backend 级**（cutlass / b12x / cute-dsl /
cudnn / auto）。代价是放弃了「某一档 tactic 可能再快几点」的探索；收益是结论
**可经官方入口 1:1 复现**，接入零私有 API。

## 3. backend 级结果（job-000035，11 shape / 4 backend）

计时口径（[bench/micro/nvfp4_tactics.py](../../bench/micro/nvfp4_tactics.py)）：

- 每个 case 扫 checkpoint 里该 shape 的**全部真实实例**（401 个 NVFP4 Linear
  一一对应），不是单实例重复；
- 工作集 ≥4×L2：k_proj/v_proj 的真实实例只有 ~47 MB，复制 **6 份不同地址副本**
  补到 ~4.4×L2，计时结果再除以 6；其余 shape 真实实例本身就 ≥4×L2；
- 一个计时单元 = 该 case 全部（实例×副本）逐层 launch，capture 成 CUDA graph 后
  replay，CUDA event 取 20 次中位数；候选与 cutlass 基线共用同一批实例、同一份
  A/W、同一个 capture/event 边界；
- 数值 gate：与「同一份量化后的 A + checkpoint W 的 128x4 swizzled scale 反量化」
  的 FP32 matmul 比，M2 冻结容差；另加 backend 间 `nrmse ≤ 1e-3` 一致性 gate。

带宽单位 GB/s，括号内为聚合 ms（该 shape 全部真实实例）。`auto` 在 SM120 上走
heuristic 的第一个候选 b12x（非 autotune 模式只跑第一个），所以数值与 b12x 基本一致。

| case | N×K | inst | cutlass | b12x | **cute-dsl** | cudnn | auto |
|---|---|---|---|---|---|---|---|
| in_proj_qkv | 10240×5120 | 48 | 799.5 (1.771) | 850.9 | **871.2 (1.625)** | 803.4 | 842.6 |
| in_proj_z | 6144×5120 | 48 | 732.5 (1.160) | 804.0 | **836.4 (1.016)** | 730.3 | 804.0 |
| q_proj | 12288×5120 | 16 | 648.1 (0.874) | 796.0 | **843.3 (0.671)** | 656.2 | 796.0 |
| k_proj | 1024×5120 | 16 | 164.6 (0.287) | 341.8 | **504.8 (0.094)** | 160.2 | 341.7 |
| v_proj | 1024×5120 | 16 | 164.6 (0.287) | 341.7 | **504.8 (0.094)** | 160.1 | 341.5 |
| o_proj | 5120×6144 | 16 | 681.6 (0.415) | 773.9 | **828.6 (0.342)** | 684.1 | 773.0 |
| out_proj | 5120×6144 | 48 | 684.0 (1.242) | 771.4 | **832.2 (1.021)** | 684.5 | 782.7 |
| gate_proj | 17408×5120 | 64 | 827.3 (3.879) | 876.1 | **882.4 (3.636)** | 828.5 | 871.2 |
| up_proj | 17408×5120 | 64 | 822.5 (3.901) | 871.0 | **882.5 (3.636)** | 828.6 | 871.2 |
| down_proj | 5120×17408 | 64 | 742.0 (4.325) | 840.7 | **883.7 (3.631)** | 744.3 | 840.6 |
| lm_head | 248320×5120 | 1 | 888.9 (0.805) | 902.7 | **911.0 (0.785)** | 897.5 | 902.7 |
| **加权** | 14.399 GB | 401 | **760.1 (18.943)** | — | **870.1 (16.549)** | — | — |

- **cute-dsl 11/11 全胜**，加权 −2.394 ms；分 shape 提升 2.5%（lm_head）到 3.1×
  （k/v：164.6 → 504.8）。
- **cute-dsl 与 cutlass 输出逐位一致**（11/11 `nrmse = 0.0`、`max_abs = 0.0`），
  这是能直接换 backend 的前提。
- k/v 的 504.8 GB/s 仍然不高（N=1024 时输出网格太短，DRAM 流铺不满），但已经是
  cutlass（128 列 tile → 8 个 CTA、164.6 GB/s）的 3×；绝对量只有 0.19 ms，不是
  下一步重点。
- lm_head 911 GB/s = 峰值 960 的 94.9%，已经在 DRAM 效率上限附近。

## 4. 接入设计（commit `3c3ac27`）

### 4.1 静态表 + 只在 M==1 查表

`runner/ops/nvfp4/linear.py`：

```python
DECODE_BACKEND_TABLE = {
    (10240, 5120): "cute-dsl",   # in_proj_qkv x48
    (6144, 5120): "cute-dsl",    # in_proj_z x48
    (12288, 5120): "cute-dsl",   # q_proj x16
    (1024, 5120): "cute-dsl",    # k_proj / v_proj x32
    (5120, 6144): "cute-dsl",    # o_proj x16 + out_proj x48
    (17408, 5120): "cute-dsl",   # gate_proj / up_proj x128
    (5120, 17408): "cute-dsl",   # down_proj x64
    (248320, 5120): "cute-dsl",  # lm_head x1
}
```

- 只有 `x.reshape(-1, K)` 后 **M==1** 才查表；prefill（M>1，chunk=128）继续走
  `self.backend`（checkpoint 传入的 `cutlass`），M2 冻结口径的 prefill 数值语义不变；
- `self.backend` 没有改语义，新增 `decode_backend` 覆盖：`None`（默认）= 查表，
  显式 `"cutlass"` = 完全回到 M2 路径；
- 表未命中（例如将来新增 shape）→ cutlass；
- backend 抛错 → `warnings.warn` + 进程内永久禁用该 backend + 回退 `self.backend`，
  只发生一次；本次运行 `fallbacks=0`、`disabled=[]`；
- `mm_fp4` 其它参数（`block_size=16`、`use_8x4_sf_layout=False`、`use_nvfp4=True`、
  `out=None`）没有动。

### 4.2 JIT 只发生在 warmup（可判定）

- cute-dsl 的 SM12x kernel 按 (N, K) 编译；`GraphDecodeStep.warmup()` 在 capture
  之前跑 `graph_warmup=3` 步 eager decode，首次 M=1 调用就在这里完成 JIT；
- linear.py 记录每次进程内**每个 shape 的首次调用 wall ms**（`first_call_ms`，
  只在非 capture 路径计时），并写进两处产物：
  1. `runner/loader/nvfp4_model.py` 的 `info["nvfp4_decode"]`（表 + 依据 +
     counters；counters 是同一个 dict 的引用，check/bench 产物最后序列化时是全程
     计数）→ `m3_check` 的 `load` diag；
  2. `runner/model/decode_runner.py` 的 `harness_info()["nvfp4_decode"]`
     （`calls` / `first_call_ms` / `selected` / `fallbacks` / `disabled`）→
     `bench.json` 的 `harness`。
- `job-000037` 的实测 harness（graph-8192）：

  ```json
  {"calls": {"cute-dsl": 1844},
   "first_call_ms": {"248320x5120": 19.95, "10240x5120": 1.33, "6144x5120": 1.06,
                     "5120x6144": 1.30, "17408x5120": 1.11, "5120x17408": 1.03,
                     "12288x5120": 0.97, "1024x5120": 0.96},
   "selected": {…8/8 cute-dsl…}, "fallbacks": 0, "disabled": []}
  ```

  8 个 shape 各只有**一条** `first_call_ms`，说明编译只发生一次；`calls` 是 harness
  快照前所有 eager M=1 forward 的调用数（graph 模式的 32 步计时走 replay，不经过
  `forward`，所以不是 32×401）；`load` diag 里同一 dict 在最后序列化时是 1964 次。
- 冷启动成本：8 个 shape 合计约 **28 ms**（lm_head 19.95 ms 占七成），发生在首次
  decode；JIT 产物落在共享 `FLASHINFER_WORKSPACE_BASE`（`.tmp/flashinfer`），
  同一机器后续进程只做 load。清空该目录后的首次运行会重新编译，见 §7。

## 5. 三段验证

### 5.1 op（`job-000035`，107.7 s，198/198 gate）

- 11 case、401 实例、14.3991 GB（与 profile 的 14.399 GB 同口径）；
- 每个 case 的 5 个 backend 数值 gate 全过（nrmse ≈ 1.6e-3，M2 冻结容差）；
  backend 间一致性 11/11 `nrmse = 0.0`；
- 基线 sanity gate：cutlass 整步估计 760.1 GB/s ∈ [600, 960]，与 profile 的
  19.299 ms / 747 GB/s 差 1.9%（图内干扰），测量协议可信。

### 5.2 check（`numeric-512,graph-1024`，16 步，139/139 gate 全过）

- `numeric-512`（115 gate）：`eager→static-exact`（logits/hidden/kv/conv/recurrent）
  与 `padded-ref→static-cap` 全部 **bitwise**（`max_abs = 0.0`）；
- `graph-1024`（24 gate）：`static-cap→graph` bitwise、逐步 logits bitwise、
  logits 地址稳定、reset 后一致；
- `eager→static-cap` 的差异仍是 M3 已归类为 diagnostic 的 attention 长度效应
  （free-running 分叉），不是 gate。

### 5.3 bench（32 步 × 1 次，对照 `job-000029` 同协议 baseline）

| 场景 | baseline job-000029 | 候选 | 变化 |
|---|---|---|---|
| graph-8192 step p50 | 27.781 ms | **25.084 ms** | **−2.697 ms / −9.7%** |
| graph-8192 系统 | 35.99 tok/s | **39.81** | +10.6% |
| graph-512-1024 step p50 | 26.779 ms | **24.090 ms** | −2.689 ms / −10.0% |
| graph-512-1024 系统 | 37.29 tok/s | **41.42** | +11.1% |
| free（8K） | 0.329 GB | **0.335 GB** | 略好 |

- 8K peak allocated 15.288 GB、reserved 15.848 GB；512-1024 free 1.197 GB；
- 32 步的 step p50/p95 离散度 <0.03 ms（8K std 0.013 ms），收益远大于噪声。

## 6. 收益分解：e2e −2.70 vs op −2.39

op 级加权是 18.943 → 16.549（**−2.394 ms**），端到端却是 **−2.70 ms**，多 0.31 ms。
两个可解释的部分：

1. **op 协议低估了 cutlass 在整图里的代价**：op 每个 case 单独跑、shape 之间不
   interleave；profile 里 mm_fp4 是 19.299 ms（cutlass 747 GB/s），而 op 的 cutlass
   是 18.943 ms（760 GB/s），差 0.36 ms 就是图内相邻 kernel 的干扰。cute-dsl 的
   kernel 更短（尤其 k/v 从 8 CTA/0.287 ms 变成 0.094 ms），被干扰放大的部分更小。
2. **k/v_proj 的 164.6 → 504.8 GB/s 在整图里可能还有额外收益**：cutlass 的
   k/v kernel 每层只有 8 个 CTA，在整图里和其它 kernel 竞争 SM 时尾部延迟更明显；
   op 的单 case 聚合没有这种竞争。

这是相关性解释，没有单独做 per-kernel 时间线（8K profile 只统计 kernel 名与总
时间，cutlass 的 401 次都记成同一个 `cutlass::device_kernel`；要拆开需要按
`correlation` 关联到 layer，成本高）。**口径上不要把 −2.70 当成"GEMV 单项收益"，
它包含图内干扰的重新分配**。

## 7. 限制与下一轮建议

限制：

- **只对本 checkpoint / SM120 成立**：表按 (N, K) 写死，换模型结构（fused gate/up、
  改变 hidden/FFN）或换 GPU 架构都要重测；未命中自动回退 cutlass，不会算错但会
  失去收益。
- **cute-dsl 的 JIT 冷启动**：8 个 shape 合计约 28 ms（lm_head 19.95 ms），只发生在
  首次 decode（graph warmup 内）。共享 JIT 缓存存在时后续进程只 load；如果
  `.tmp/flashinfer` 被清空（换机器 / 换 CUDA 版本），首个请求的 TTFT 会多这 ~28 ms
  （更慢的冷编译可能到秒级）。本轮没有测冷缓存场景，也没有把它算进 TTFT 口径。
- **没有做 per-input 的逐位对拍**：op gate 只对每个 shape 的第一个实例、一个固定 A
  验了 cute-dsl vs cutlass；端到端 `numeric-512` 的 bitwise 是同一实现内
  eager/static/graph 的自洽，不证明"cute-dsl 与 cutlass 在任意输入上逐位相同"。
  不过两者在 11 个真实 shape × 真实权重上 `max_abs = 0.0`，且 kernel 都是
  K 顺序 fp32 累加，风险很低。
- **tactic 级没有落地**：见 §2.2。若要再挖，需要私有 API + 在 linear.py 里写死
  tactic，超出本轮改动面。
- 计时是 per-case 扫描，不含图内 kernel 干扰；e2e 的 −2.70 才是正式口径。

下一轮建议：

1. **GEMV 本体的剩余空间有限**：现在 870 GB/s（14.399 GB → 16.55 ms）。物理峰值
   960 GB/s 对应 15.0 ms，即理论上限 −1.55 ms；按可实现的 92–95% DRAM 效率
   （lm_head 已到 911 = 94.9%）只有 **−0.5 ~ −1.0 ms**，且需要新 kernel：
   - in_proj_z / q_proj / o_proj / out_proj 都在 830–843（合计 ~2.5 GB，到 900 可省
     ~0.2 ms）；gate/up/down 882（9.6 GB，到 920 可省 ~0.4 ms）；k/v 504（0.19 ms，
    split-K 或合并 k+v launch 最多省 0.08 ms）。
   - 这些都不是 backend 选型能拿的，要靠 kernel/布局改动；优先级应低于第 2 项。
2. **重点转向非 GEMV 的 8.5 ms**：集成后 profile 里 25.08 ms 中有 16.55 ms 是
   mm_fp4，其余 ~8.5 ms 是 5000+ 个小 kernel（elementwise 4.4、ShapeStableLinear
   1.5、quantize 0.53、norm 0.31、GDN 0.26 …）。本轮 G 和 I 已经在做；下一个
   结构性杠杆是 **norm + activation quant 融合**（0.8 ms 级）和 **gate/up + SiLU·mul
   融合**（把 401 次 quantize 与 elementwise 的一部分消掉）。
3. **FP8 KV 之后这张表需要重测吗**：FP8 KV 只改 attention 的 KV 读写，不改
   NVFP4 Linear 的 (N, K) 与权重，所以 backend 选择不需要重测；但如果引入
   gate/up 融合或改变 chunk，M=1 的 shape 集合会变，命中不到表就自动回退 cutlass
   —— 接入新 shape 前跑一次 `bench/micro/nvfp4_tactics.py --only <新 shape>` 即可。
4. **冷启动**：如果要报 TTFT，建议单独测一次「清空 `.tmp/flashinfer` + 首个请求」，
   把 cute-dsl 的 8 个 shape 编译时间计入，再决定是否在 `load` 阶段做一次
   dummy decode 预热（当前靠 graph warmup 吸收）。

## 8. 复现

```bash
# 1) op：per-shape backend 选型 + 数值 gate（op-only）
cd .tmp/worktrees/explorer-tactics
cat .tmp/req-tactics.json        # session phase2-tactics, timeout 1800, 单 op step
/home/xzq/workspace/qwen/.venv/bin/python /home/xzq/workspace/qwen/scripts/experiment.py \
    submit --request .tmp/req-tactics.json
# 产物：results/experiments/job-000035-461c30c4/output/op/op.json

# 2) 接入后 check + bench
cat .tmp/req-tactics-e2e.json    # timeout 2400, check(numeric-512,graph-1024) + bench
/home/xzq/workspace/qwen/.venv/bin/python /home/xzq/workspace/qwen/scripts/experiment.py \
    submit --request .tmp/req-tactics-e2e.json
# 产物：results/experiments/job-000037-150586b0/output/{check,bench}/

# 3) 纯 CPU 清单自检（不建 CUDA context）
.venv/bin/python bench/micro/nvfp4_tactics.py --check-shapes
```

提交 commit：`959b667`（op 脚本）、`3c3ac27`（接入）；基线 `9e35873`；
对照 job `job-000029-16bb35a9`（`graph-8192` 27.781 ms / 35.99 tok/s、
`graph-512-1024` 26.779 ms / 37.29）。
