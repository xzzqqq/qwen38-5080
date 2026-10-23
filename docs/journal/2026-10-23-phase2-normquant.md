# 2026-10-23 Phase 2 任务 H：RMSNorm + activation quant 融合（decode）

## 状态

完成并采纳。一个 Triton kernel 取代「128 个 decoder-layer RMSNorm（每个约 8 个
elementwise/reduce kernel）+ 272 次 per-consumer `quantize_with_block_size` + 每步
673 个 per-call 标量 kernel」，**语义 bitwise 不变**（决定性参考 = 模型实际执行的
`Qwen3_5RMSNorm.forward` + FlashInfer `nvfp4_quantize`，260/260 gate 全部逐字节相同）。

| 项 | 值 |
|---|---|
| 候选 commit（job 实际运行） | `8b21db484030702d4e0cbcee1711ddde8474c13b` |
| job | `job-000040-baee7dc7`（`succeeded`，216.3 s，`correctness=passed`） |
| op gate | **260/260**（6 个 case），`real-layers` 128 个真实 norm / 272 个消费者 / **1,392,640 个 code 逐字节相同** |
| check | **139/139**（`numeric-512` 115 + `graph-1024` 24），四种执行方式全过 |
| 端到端（32 步 × 1 次） | `graph-8192` 27.781 → **25.404 ms**（−2.377 / −8.6%），系统 35.95 → **39.32 tok/s**（+9.4%）<br>`graph-512-1024` 26.779 → **24.411 ms**（−2.368 / −8.8%），系统 37.29 → **40.91 tok/s**（+9.7%） |
| 接线计数（check 的 `load.norm_quant`） | `enabled=true`、`norms=128`、`quantized_consumers=272`、`fused_norms=2560`（=128×20）、`fused_consumers=5440`（=272×20）、`unconsumed=0`、`fallbacks={"seq_len!=1": 1536}` |
| 产物 | `results/experiments/job-000040-baee7dc7/output/{op,check,bench}/` |
| 失败尝试 | `job-000036-e448042d`（op gate 口径错，1 字节）、`job-000038-ad133bc8`（接线 bug，eager 崩） |

- plan：[phase2.md §3 任务 E 的后续、§4 第 2 行、§5、§6](../plans/phase2.md)
- quant 契约：[任务 E 的 journal](2026-10-23-phase2-quant.md) §6（规则 R2）
- 对照 baseline：`job-000029-16bb35a9`（commit `9e35873`，同协议 32 步 × 1 次）
- checkpoint：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`，revision
  `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`
- 环境：RTX 5080（sm_120a）、torch 2.14.1+cu130、transformers 5.20.2、flashinfer 0.7.2、
  triton 3.8.0、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 1. 结论摘要

- **收益 −2.377 / −2.368 ms（−8.6% / −8.8%）**，比派发时预估的 −0.5 ~ −1.2 ms 大一倍。
  归因（§5.3）：graph 口径的局部加速约 **1.26 ms**（norm+quant 融合），另外约
  **1.1 ms** 来自同一方向顺带删掉的 **673 个 per-call 标量 kernel**
  （401 个 `alpha = input_scale * weight_scale_2`、272 个 `1 / input_scale`）。
- **语义零变化**：`out_bf16` + 每个消费者的 `(a_fp4, a_sf)` 与模块两步路径逐字节相同；
  `real-layers` 一个 case 就有 1,392,640 个 E2M1 code 逐字节一致，`a_sf` 的 128x4
  swizzled 字节（含 127 行 padding 必须为 0）也全部一致。
- 每个消费者有自己的 per-tensor `input_scale`，所以融合 kernel 按消费者**各产一份**
  `(a_fp4, a_sf)`（不共享），48 个 GDN 层 2 消费者、16 个 full-attention 层 3 消费者、
  64 个 post norm 2 消费者，合计 272 份。
- 只走 **decode**（`seq_len == 1`）；prefill 回退原 forward，语义完全等价。
- 与任务 F（decode GEMV backend 选型）有**耦合**：融合路径必须和 F 共用同一条
  backend dispatch，否则 272 个 GEMV 会退回 cutlass，两个候选的收益互相抵消（§7.2）。

## 2. 设计与数据流

### 2.1 融合边界

现有 decode 路径（每层两次 norm，每个消费者一次 quantize）：

```
input_layernorm  --+--> linear_attention 层: in_proj_qkv, in_proj_z      (2 消费者)
                   +--> full_attention  层: q_proj, k_proj, v_proj      (3 消费者)
post_attention_layernorm --+--> gate_proj, up_proj                       (2 消费者)
```

融合后（每层 2 个 kernel，一次 launch 同时产出 BF16 归一化值与所有消费者的 NVFP4 A）：

```
x[5120] BF16 + w1[5120] FP32 (= 1 + weight.float()) + eps + gs_c = fl32(1/input_scale)
  -> 第一遍: 复刻 PyTorch mean(-1) 的 reduction 顺序 -> variance -> rsqrt.approx -> rstd
  -> 第二遍: 5 个 64-block chunk，每个 chunk
       load x/w1 -> y_f32 -> y_bf16（写回 out_bf16）-> y_f32 -> block amax
       -> 每个消费者: sf -> e4m3 -> out scale -> 16 个 e2m1 code -> 打包写 a_fp4/a_sf
```

关键设计点（细节见 `runner/ops/nvfp4/norm_quant.py` 的模块 docstring）：

1. **reduction 顺序必须逐位复刻**。`Qwen3_5RMSNorm._norm` 是
   `x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)`，`mean` 走
   `at::native::mean_kernel_cuda` → `gpu_reduce_kernel` + `MeanOps`。对
   `[1,1,5120]` contiguous FP32，launch 配置是 `block=(512,1)`、`vt0=input_vec_size=4`、
   `step_input=512`、`values_per_thread=10 < warp_split_threshold(16)`（不跨 warp split、
   无 global reduce）、`factor = fl32(1/5120)`。于是每个线程的值是
   `acc_j(t) = (x[4t+j] + x[4(t+512)+j]) + x[4(t+1024)+j]`（第三项仅 `t<256`），
   再 `((acc0+acc1)+acc2)+acc3`，最后 9 级「一半加一半」树（offset 256,128,…,1 =
   block_x_reduce 的 shared memory 树 + warp shfl_down）。kernel 里就是连续的
   `tl.sum(tl.reshape(v, (2, n)), axis=0)`；**不能用一次 `tl.sum` 代替**，Triton 的
   warp 归约是 xor 1,2,4,…,16 的顺序（相邻先加），括号化不同、FP32 结果不同。
2. **平方要钉住**：`x*x` 用 inline PTX `mul.rn.f32`，防止被后面的 reduction 加法收缩成
   fma（`fma(a,a,b)` 与 `mul(a,a)+b` 差一次舍入）。`variance = mul_rn(sum, factor)` 再
   `add.rn(., eps)` 也显式二步，`rstd = rsqrt.approx.f32`。
3. **归一化值必须先 round 回 BF16 再量化**（`cvt.rn.bf16.f32`），否则量化输入 dtype 变了，
   amax 与 code 都会变——这是语义变化，不允许。
4. **量化按任务 E 冻结的规则 R2**：`sf_pre = gs * (amax * c6)`（`c6 = rcp.approx.ftz(6.0)`
   = `0x3e2aaaab`）、`cvt.rn.satfinite.e4m3x2.f32`、`sff = float32(sf_u8)`（位构造精确解码）、
   `out = amax==0 ? 0 : rcp.approx.ftz(mul_rn(sff, rcp.approx.ftz(gs)))`、
   `code = cvt.rn.satfinite.e2m1x2.f32(mul_rn(y, out))`。**两个倒数都是 approx 指令**。
5. **布局**：`a_fp4` U8 `[1, K/2]`（低 nibble 偶数下标）；`a_sf` 128x4 swizzled
   `[128, 320]`，M=1 时只有第 0 行有数据，落点 `off = (b//4)*512 + (b%4)`，其余 padding
   由 `torch.zeros` 一次性清零后不再写（CPU 上已与 `reference.swizzle_scale_128x4` 对
   320 个 block 逐一核对）。
6. **`num_warps`**：eager 扫描 2/4/8 = 0.015968 / 0.015648 / 0.015520 ms/次，差 <3%，
   接入固定 **4**（未细调）。

### 2.2 接线（`runner/model/norm_quant_patch.py`）

- 接入点是**模块级 `Qwen3_5DecoderLayer.forward`**：HF 的 forward 只有六行
  「norm → mixer → residual → norm → mlp → residual」，复制这段就能在 norm 与消费者之间
  插入融合 kernel，不改 mixer / MLP / `NVFP4Linear` 源码，也不影响 prefill。
- 命中条件（任一不满足即整层回退并计数）：`seq_len==1`、BF16、CUDA、`no_grad`、
  `hidden==5120` 且 contiguous；linear_attention 层要求 `attention_mask is None`
  （GDN 会在 `in_proj_qkv` 之前 `apply_mask_to_padding_states`，虽然 seq_len==1 时
  `create_recurrent_attention_mask` 恒返回 None，这里只是把「万一」挡在门外）。
- **`_pending_quantized` 单次槽位**：patch 在调 mixer 前把 `(a_fp4, a_sf)` 放到对应
  `NVFP4Linear` 上，`forward` 取走即清空——所以一个模块在一个 step 里只会命中一次，
  多出来的调用自动回退到普通量化路径；`finally` 里清掉没被消费的槽位并计入
  `counters.unconsumed`。graph capture/replay 依赖「launch 顺序固定」这一点，而 Python
  侧的设置/消费只发生在 capture 时（replay 不跑 Python）。
- **scratch 地址固定**：install 时分配 BF16 归一化输出 1 份 + `a_fp4`/`a_sf` 各 3 槽
  （`a_fp4`/`a_sf` 的 dtype/shape 由 FlashInfer 自己的 probe 输出决定，不猜），
  共 **140800 B**；128 个 norm 复用同一份，靠 stream 顺序保证「写→读→下次覆盖」。
  `a_fp4 = 1*2560`、`a_sf = 128*320`（每槽），3 槽。
- **`alpha` 预计算**：`NVFP4Linear` 构造时算一次 `alpha = input_scale * weight_scale_2`
  （f32 乘法，与逐次计算逐位相同），每步省 401 个标量 kernel；融合路径另外省掉 272 个
  `1/input_scale`（`gs` 在 install 时算成 Python float 传进 kernel）。

## 3. gate 口径与 bitwise 证据

### 3.1 决定性参考（**记录一次口径变更**）

- **上一轮（job-000036）** 用 `F.rms_norm(x.float(), (K,), 1+w, eps) → bf16` 当 gate，
  24 个样本里差 **1 个 bf16 字节**。
- **本轮起改为**：`Qwen3_5RMSNorm.forward`（模型实际执行，bf16 权重、zero-centered
  `1 + w`、fp32 里算完再**一次** cast）+ 真实 `nvfp4_quantize`。`F.rms_norm` 降为
  diagnostic，**bitwise 仍是硬 gate、不降级**。
- 变更理由（本轮用诊断把三种假设都测了，见 §4.1）：`aten::rms_norm` 是 composite，
  与 eager 模块的 aten op 序列完全同序；本轮实测 `pow(int 2) == torch.pow(x, 2.0) == x*x`
  （15 个真实样本、76,800 个元素，0 差异），`mean(-1)` 在 `[1,1,K]`/`[1,K]`/`[K]` 三种
  shape 下也逐位相同，所以差异既不是 pow 语义也不是 reduce 配置。**这 1 个字节的机理
  至今未完全解释**，但它是「两个 PyTorch 参考实现之间」的差异，与候选 kernel 无关：
  候选在与**模块路径**的 260 条 gate、1,392,640 个 code 上全部逐字节相同（含
  layer 0 + saturation 那一组）。

### 3.2 op gate（`scripts/phase2_normquant_check.py`，6 个 case / 260 条 gate）

| case | 内容 | 结果 |
|---|---|---|
| `mean-tree` | 用 torch op 逐级复刻 `mean(-1)` 的 reduction 顺序，与 torch 自己比（5 层 × 4 分布，输入是模型真实 `[1,1,K]`） | 20/20 逐位相同 |
| `rsqrt-approx` | inline `rsqrt.approx.f32` vs `torch.rsqrt`：6 层 × 6 种输入的真实 variance + `logspace(1e-6, 1e9)` 8192 点 | **8228 个样本 0 差异**（最大 variance 1e9，含 saturation 的 ≈1.75e5） |
| `real-layers` | 128 个真实 norm × 各自真实消费者 `input_scale`（112 个 2 消费者 + 16 个 3 消费者），每消费者比 `out_bf16`/`a_fp4`/`a_sf` | **128/128**，1,392,640 code 全同 |
| `scales-extremes` | 全局最小 `input_scale` 0.0012091（gs 827.08，`layers.0.mlp.gate_proj`）与最大 0.18155（gs 5.508，`layers.63.mlp.down_proj`）× 4 种组合 × 3 层 × 2 norm × 3 种输入 | **72/72** |
| `distribution-edge` | 4 层 × 2 norm × {zeros, saturation, embed, gauss-0.5/2/8}，含 `amax==0` 的 padding 行与 `sf_pre` 冲 `satfinite` | **48/48** + `amax==0` 分支确认（out/a_fp4/a_sf 全零） |
| `micro` | 2/3 消费者 × {eager, graph}，`num_warps` 2/4/8，另加 bitwise 复核 | 4/4 |

### 3.3 check（`numeric-512` + `graph-1024`）

139/139 gate 全过：`numeric-512` 的 `eager→static-exact`、`padded-ref→static-cap` 以及
边界长度 1/127/128/129/300 的逐位一致都在里面；`graph-1024` 的 static-cap→graph 逐位一致
也在。四种执行方式**都走融合路径**（`fused_norms=32768`），prefill 只剩
`fallbacks.seq_len!=1=6400`。

## 4. 三次失败与修复（含一次 gate 口径修正）

### 4.1 job-000036：1 字节差异 → **gate 口径错**

- 现象：op 260 条 gate 里只挂 2 条、都是 1 个 bf16 字节：
  `reference-agreement`（`F.rms_norm` vs `Qwen3_5RMSNorm`，**不涉及候选 kernel**）与
  `bitwise[layers.0.input_layernorm/saturation]`；而且后者的 `a_fp4`/`a_sf` **全部
  逐字节相同**——差异落在一个离 E2M1 边界很远的元素上。
- 定位：两个失败的样本是同一组 (layer 0, saturation, input_layernorm)。因为
  `reference-agreement` 完全不碰候选，结论只能是「两个参考实现之间就有 1 字节差异」，
  真实模型跑的是模块路径，`F.rms_norm` 只是另一个实现。
- 处理：决定性参考改为模块路径，`F.rms_norm` 降 diagnostic，并保留三组探针
  （`square-semantics`、`mean(-1)` 的 shape 敏感性、逐样本 3-D/2-D 对比）。
- 教训：**op gate 的参考必须是模型实际执行的代码路径**，不能用「公式相同的另一个 API」
  代替；参考自身的 1 ulp 差异会被误判成候选失败。

### 4.2 job-000038：eager 路径 `a_fp4 / a_sf length must match gs_list` → **接线签名不匹配**

- 现象：op 已 260/260 全过，check 在 `numeric-512` 的第一个执行方式（eager）第 0 层崩。
- 根因：`_NormQuant` 把 **3 槽 scratch 列表整个**传给契约是
  `len(a_fp4) == len(a_sf) == len(gs_list)` 的 kernel；GDN 的 input norm 与所有 post norm
  都是 2 消费者 → `3 != 2` 立刻抛错。与「哪种执行方式/哪一层」无关：每种方式的第 0 层
  都会崩，eager 只是第一个跑的。
- 上一轮没暴露的原因：job-000036 挂在 op 步骤，模型路径根本没被执行过；而 op 脚本不加载
  模型，看不到接线层。
- 修复：launch 前按消费者数量切片；`_NormQuant.try_run` 任何前置条件不满足都返回 None
  **整层回退**（pre-norm 回退原 forward；post-norm 回退「原 norm + 非融合 MLP」），
  并计入 `counters.fallbacks`（`consumer-count-mismatch` / `scratch-slots` / `fused-error`）；
  install 时校验消费者数 ∈ {2,3}。
- 守门：写了一个 **CPU-only 接线自测**（stub `flashinfer` + 假层 + 打桩 kernel，不建
  CUDA context）：2 消费者/3 消费者两条路径的长度与槽位消费、消费者数不匹配回退、
  kernel 抛错回退，4 个场景全过。这类 bug 用它是能在投递前抓住的。

### 4.3 实现期的两次离线拦截（记录方法，供后续复用）

1. **Triton 3.8 不允许 jit body 访问非 constexpr 全局量**（`REDUCE_LANES` 等会直接
   `NameError`）。用 `triton.compile + ASTSource(target=sm_120)` 在**没有 GPU 的机器上**
   离线编译 kernel，把这类错误挡在投递之前。
2. 同一次离线编译读 PTX 得到：`fma.rn.f32 = 0`、`rsqrt.approx.f32 = 1`、
   `mul.rn.f32 = 49`（48 个平方 + 1 个 mean-project）、`add.f32 = 47`（11 累加 + 9 级树
   × 4）、`cvt.rn.satfinite.{e4m3x2,e2m1x2}.f32 = 120/120`、`cvt.rn.bf16.f32 = 40` ——
   与设计逐项对得上，`fma = 0` 直接证明了 bitwise 契约里最脆的一条。
3. 实现期还自己抓到过一次「reduction 忘了平方」（review 时发现），说明这类 kernel 必须
   有「结构对账」（PTX 计数）而不只是端到端数字。

## 5. 结果

### 5.1 端到端（32 步 × 1 次，与 `job-000029` 同协议）

| 场景 | baseline step p50 | 候选 | 变化 | 系统 tok/s | 变化 |
|---|---|---|---|---|---|
| `graph-8192` | 27.781 ms | **25.404 ms** | −2.377 ms / −8.6% | 35.95 → **39.32** | +9.4% |
| `graph-512-1024` | 26.779 ms | **24.411 ms** | −2.368 ms / −8.8% | 37.29 → **40.91** | +9.7% |

离散度：这次是 1 次重复（契约允许），`step p50/p95` = 25.40/25.42 与 24.41/24.43 ms；
正式结论仍以 511 步 × 5 次为准。显存：`graph-8192` peak 15.291 GB / free 0.335 GB，
`graph-512-1024` peak 14.821 / free 1.195 GB；新增 scratch 只有 **140800 B**
（BF16 归一化输出 10240 + 3×a_fp4 7680 + 3×a_sf 122880），对峰值没有可测影响。

### 5.2 局部 microbench（op 的 `micro` case，layer 3）

| 形态 | 两步路径 | 融合 | 倍率 |
|---|---|---|---|
| 2 消费者，eager | 0.06976 ms | **0.015712 ms** | 4.44× |
| 3 消费者，eager | 0.082112 ms | **0.015648 ms** | 5.25× |
| 2 消费者，graph replay | 0.012896 ms | **0.003176 ms** | 4.06× |
| 3 消费者，graph replay | 0.013888 ms | **0.003472 ms** | 4.00× |

按真实分布（112 次 2 消费者 + 16 次 3 消费者）折算：graph 口径 **−1.255 ms/step**，
eager 口径 −7.12 ms/step（eager 的数只作诊断，模型跑在 graph 里）。融合后这两个 op 的
成本只剩两步路径的 **24.7%（graph）/ 22.0%（eager）**。

### 5.3 为什么实测（−2.38 ms）比 graph micro 折算（−1.26 ms）大一倍

两处同方向的「小 kernel 消除」与融合一起生效、但不在 micro 的两步路径里：

- `alpha = input_scale * weight_scale_2`：401 个 NVFP4Linear 各一次 1-element 乘法 kernel
  → 构造时预计算（值逐位相同）；
- `1 / input_scale`（量化前的 global divisor）：融合路径的 272 个消费者不再逐次计算，
  `gs` 在 install 时算成 Python float。

673 个 per-call 标量 kernel，按 profile 的 ~1.5–2 µs/个小 kernel 计 ≈ 1.0–1.3 ms，
与 2.38 − 1.26 ≈ 1.1 ms 对得上。**这笔账必须单列**：它是「kernel 数量」收益，不是
「融合」收益。

## 6. 覆盖与可复现

```bash
# 队列（GPU 主机，单 job）
.venv/bin/python scripts/experiment.py submit --request .tmp/req-normquant.json  # commit 8b21db4
.venv/bin/python scripts/experiment.py result job-000040-baee7dc7
# 产物
results/experiments/job-000040-baee7dc7/output/op/op.json      # 260 gate
results/experiments/job-000040-baee7dc7/output/check/check.json # 139 gate + load.norm_quant
results/experiments/job-000040-baee7dc7/output/bench/bench.json # 2 场景 x 32 步
```

请求的三步：op（`phase2_normquant_check.py`，约 100 s，含 Triton/FlashInfer 首次编译）
→ check（`m3_check.py --only numeric-512,graph-1024 --steps 16`）→ bench
（`bench/e2e/decode.py --only graph-8192,graph-512-1024 --steps 32 --repeats 1`）。

## 7. 限制与已知问题

### 7.1 功能边界

- **只走 decode**（`hidden_states.shape == (1,1,5120)`）。prefill / chunked prefill 回退原
  forward；`fallbacks.seq_len!=1` 就是它的计数（`graph-1024` 里 1536 = 64 层 × 24 个
  prefill chunk）。prefill 的 norm+quant 没有优化。
- **K 固定 5120**（kernel 里 chunk 数 5、树 512 都是静态的）。hidden 不是 5120 时 install
  直接失败并回退原路径（`install_from_env` 只记录不抛）。
- **消费者数量硬编码 2/3**；install 时校验，不满足就拒绝安装。
- `per_token_activation=True`（per-token scale）路径未覆盖；只有 per-tensor `input_scale`。
- 环境开关：**`NORM_QUANT_FUSED=0`** 关闭融合（回退原路径），用于对照。
- `num_warps=4` 未细调（2/4/8 差 <3%）。
- micro 只测了 layer 3 的一组输入/权重；128 个 norm 的 per-call 时间分布未逐层测。

### 7.2 与任务 F（decode GEMV backend 选型）的交互 —— **必须记住**

`NVFP4Linear` 原来有两条到 `mm_fp4` 的路：普通路径与融合路径的 `_pending_quantized`
快路径。融合快路径最初直接调 `self._mm(...)`，用的是构造时的 `self.backend`（cutlass）——
**这会绕过任务 F 的 decode backend 选型**（F 把 401 个 GEMV 从 cutlass 760 GB/s 换到
cute-dsl 870 GB/s）。融合路径覆盖 272 个消费者，不处理的话合并后这 272 个 GEMV 会退回
cutlass，白丢约 1.5–2 ms，两个候选的收益互相抵消。

Coordinator 在合并时把两条路统一成一个 `_gemm(a_fp4, a_sf, m)`：`pending` 路径传 `m=1`
走选型、普通路径传 `x.shape[0]`。**这条不变量要写进后续任何融合/快路径改动**：融合路径
必须与 decode backend 选型共用同一条 dispatch。本 journal 记录的是 Coordinator 的集成
改动，任务 H 分支本身不含它。

### 7.3 未解释的 1 字节

`F.rms_norm` 与 `Qwen3_5RMSNorm` 在 (layer 0, saturation, input_layernorm) 上差 1 个 bf16
字节，两种 shape 下都差，而 `pow` 语义与 `mean` 的 shape 敏感性都已排除（§3.1）。它是
两个参考实现之间的事，不影响候选（候选与模块路径 260/260 逐字节相同），但如果以后有人
想把 `F.rms_norm` 当参考，必须知道这一点。

## 8. 下一步建议

1. **gated RMSNorm（`Qwen3_5RMSNormGated`）+ out_proj 的 quant** 可以套同一套 per-consumer
   契约：48 个 GDN 层的 gated norm 输出（`[1,1,48,128]` → 6144）直接喂 `out_proj`
   （NVFP4Linear, K=6144）。它多一个 silu(gate) 乘法、norm 在 head 维（128）上做，
   reduction 顺序同样要复刻；收益上限 = 48 次 norm（内部多 kernel）+ 48 次 quantize
   → 48 个 fused kernel，且它落在 profile 里 GDN elementwise 4.41 ms 的大头里。
2. **剩余 per-call 标量 kernel 清点**：本轮删掉 673 个（alpha 401 + divisor 272）。
   还能查的：GDN 的 `A_log/dt_bias`→`g/beta` 每层两次小 kernel（48×2）、attention 的
   `scaling`、`sigmoid(gate)` 前的 reshape/contiguous（16 层）。建议下一轮先用 profile
   把「<5 µs kernel」重新归类，再决定是否值得。
3. **op 脚本的 CPU 接线自测应该固化**：本轮是 ad-hoc（stub `flashinfer` + 假层 + 打桩
   kernel，不建 CUDA context）。它抓住的正是「op gate 看不见接线层」这个盲区，
   建议下一轮做成 `tests/model/test_norm_quant_patch.py`（用 `pytest.importorskip` +
   monkeypatch stub，确保不初始化 CUDA）。
4. 若继续压 norm 这条线：融合 kernel 目前是 1 个 program、**3.2 µs/次（graph）**，
   128 次/step ≈ **0.41 ms**；两步路径同一段是 1.65 ms。这条边界已经压到很低，
   再优化它的绝对空间 <0.4 ms，下一轮应该看 GEMV（19.3 ms）与 GDN 侧。
