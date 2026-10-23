# Phase 2 任务 B：CUDA core NVFP4 GEMV（M=1）

- 日期：2026-10-23
- 对应：[phase2 plan](../plans/phase2.md) §3 任务 B、§5 gate、§6 反馈阶梯；
  [decisions](../decisions.md) 的 M2 NVFP4 容差（`nrmse ≤ 5e-3`、`cosine ≥ 0.99999`、
  `max_rel ≤ 1e-2`、finite）
- 代码 commit：v1 `1848b4d`；v2（reference 修复）`b4d6a15`，branch
  `exp/phase2-gemv-cuda`（base = main `cc1c9aa`）
- job：`job-000022-f821eb71`（v1，failed：reference 错）、`job-000025-bae867f6`
  （v2，failed：只剩一条 gate 规格错，见 §5）
- checkpoint revision `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`
- 产物：`results/experiments/job-000025-bae867f6/output/op/op.json`

## 1. 实现

新增三个文件（未改 `linear.py`、模型加载、`runner/experiment/*`）：

| 文件 | 内容 |
|---|---|
| `runner/ops/nvfp4/gemv_kernel.cu` | CUDA core（SIMT，不用 tensor core）GEMV kernel；E2M1 位运算解成 fp32，FP32 累加 |
| `runner/ops/nvfp4/gemv.py` | `Nvfp4GemvWeights`：绑定 packed 权重 + 128x4 swizzled scale + alpha，缓存 split-K workspace，`torch.utils.cpp_extension` 首次调用编译 |
| `bench/micro/nvfp4_gemv.py` | op gate + microbench（已在 `schema.OP_SCRIPTS` 注册，`SCENARIOS` 6 个 case） |

数据流（一层，M=1）：

```
a_fp4 [K/2] U8（FlashInfer nvfp4_quantize，低 nibble = 偶数 k）
a_sf  [128, K/16] 128x4 swizzled（M=1，只读 tile row 0）
w     [N, K/2] U8                       ──┐
w_sf  [N_pad, K/16] 128x4 swizzled      ──┴→ GEMV kernel → out [N] BF16
alpha = input_scale * weight_scale_2（kernel 最后乘一次）
```

并行划分与访存（细节见 kernel 头部注释）：

- `grid = (N/128, k_split)`。**一个 block = 一个 128 行 tile**，正好等于 128x4 scale
  swizzle 的行 tile，于是每个 c-tile（64 个 k）在 swizzled buffer 里的 512B 被整个
  block 用满 —— scale 的 DRAM 访问没有放大（scale 占总字节的 1/9）。
- M=1 时单层 block 数常常不足（`k_proj` 只有 8 个 tile），用 `blockIdx.y` 做 split-K
  把 block 数拉高；部分和在**同一个 kernel 内**用 counter + `__threadfence` 归约
  （末尾 block 读 `partials[split, N]` 求和后写 BF16 并复位 counter），每层仍然只有
  1 个 launch，不增加每步 kernel 数。
- 16 warp × 8 行：每个 lane 负责 k-group（512 个 k）内第 lane 个 16-value block =
  8 个 packed byte；这 16 个 k 只对应 1 个 scale byte，且 a 的 16 个值解成 fp32 后
  在寄存器里被 8 行复用 —— 这是把「每字节权的 ALU/共享内存开销」压下去的关键。
- 权重：warp 每行读 **256B 连续**（32 lane × 8B），8 行一组，地址按 K/2 跨行；
  scale：每个 k-group 把 8×512B 连续拷进 shared，按 **528B 分块**（16B 对齐 +
  每 c-tile 错 4 个 bank），读时同一 c-tile 的 4 个 lane 读同一个 32-bit 字（broadcast）。
- 预期瓶颈：权重带宽（每步 14.4 GB，M=1 无复用）。E2M1 用 ~7 条指令/值解成 fp32，
  设计上让 ALU 时间 < 访存时间；实测资源：**57 registers、0 spill、4228B smem**
  （512 线程/block → 2 block/SM）。

## 2. swizzled scale 的原地消费方式（本轮必须回答的问题）

**结论：kernel 原地直接消费 `NVFP4Linear.weight_scale_swizzled`（128x4），不生成
unsquizzled 副本、不做 repack。** 寻址在 kernel 内现算：

```
off(tile, row, c, d) = tile*(K/16*128) + c*512 + (row%32)*16 + ((row/32)%4)*4 + d
                       （c = k/64，d = (k%64)/16，row 为 tile 内行号）
```

理由：

- 8K + graph 常驻后只剩约 **0.12 GB** 显存，全模型再留一份 linear `[N, K/16]` scale
  要 **1.6 GB**，不可行（decisions 里 `NVFP4Linear` 构造后释放原始 linear scale 就是
  为了这个）；
- 原地 repack 虽然同尺寸可行（一次性成本），但会让权重 layout 与现有 `NVFP4Linear`
  分叉、需要接入轮同步改加载路径，收益只是省掉偏移计算（本来就在寄存器里算）；
- swizzle 偏移的代价可以忽略：block 覆盖 128 行 tile，scale 的全局读仍是
  **每 c-tile 512B 连续**、完美合并。

验证方式（不只看最终输出）：`layout_probe` case 用合成权重 + 手算期望，单独验
(a) E2M1 nibble 顺序（低 nibble = 偶数 k，期望 256；单侧 nibble 不一致会得到 512，
判别力写成 diagnostic；两侧同时反 nibble 是 k 轴重标号、输出不变，语义等价）；
(b) 128x4 swizzled scale 映射（scale 值取 `[1,1.25,1.5,2][(n+b)%4]`，只有 block
0..3 非零，期望 `16·s0+8·s1+0+24·s3`，错位假设最小差 2.0）。两个探针在
`job-000025` 里 **max_abs_err = 0.0**。

## 3. 正确性

### 3.1 v1（`job-000022-f821eb71`）：reference 错，不是 kernel

6 个 case 里候选与 `mm_fp4` 对「reference」的 `nrmse` **逐位相同**（in_proj_qkv
0.9632354974746704、gate_proj 0.7127349972724915、lm_head 0.9586970806121826、
k_proj 0.9755951166152954、down_proj 0.9157723188400269），cosine 0.95 量级。
两个独立实现对同一个 reference 偏差到同一个数，只能是 reference 错。

根因：`bench/micro/nvfp4_gemv.py` 的 reference 把 block scale 当 **uint8** 用了：

```python
sf = unswizzle_scale_128x4(_u8(a_sf), ...)[0].float()      # 0x38 -> 56.0，不是 1.0
wsf_linear = unswizzle_scale_128x4(_u8(wsf), ...).float()  # 同上
```

`_u8()` 是 `view(torch.uint8)`，`.float()` 拿到的是**字节整数值**而不是 UE4M3 数值。
量化证据（CPU 读真实 checkpoint）：`in_proj_qkv.weight_scale` 的 E4M3 数值范围
**[5, 448]、median 32**（这个 checkpoint 的 block scale 吃满 E4M3 范围），
byte/E4M3 比值 **0.3–14.8、median 3.0**。两条 scale 各被放大 3–9× 且逐 block 变化
→ ref 被放大 → `nrmse = |out-ref|/|ref|` **归一化后饱和到 ~1**（0.96），
`cosine` 因比值有方差降到 0.95（若只是常数倍，cosine 会是 1.0）。
CPU 复现同型：合成比值≈60 时 buggy-vs-correct 得 nrmse 66.8 / cosine 0.989。

### 3.2 v2（`job-000025-bae867f6`）：修复 + 自检

修复：新增 `_scale_f8()`（按 `float8_e4m3fn` 解释，绝不走 uint8），
`dequant_activation` / `dequant_weight_scale` 都改用它。同时新增三条自检
（防止 reference 再错时被静默放过）：

| 检查 | 内容 | v2 结果 |
|---|---|---|
| 双 reference 交叉 | 主路径解**同一份** `a_fp4`/`a_sf` + swizzled W scale；第二条走 M2 冻结语义（`quantize_activation_reference(x)` + checkpoint **线性** W scale + `weight_scale_2`） | 见 §3.3 与 §5 |
| `baseline: mm_fp4 与 reference 在冻结容差内` | reference 自检：baseline 必须自己也过容差 | 5/5 通过（nrmse ≈ 1.66e-3） |
| `baseline: 候选与 mm_fp4 输出一致` | 不依赖 reference 的独立一致性 | 5/5 通过，**nrmse = 0.0 / cosine = 1.0** |

### 3.3 最终数值（`job-000025`，`k_split` = auto 策略）

| case | n | k | 实例数 | `gemv_nrmse` | `gemv_cosine` | `gemv_max_rel` | `cand_vs_mm_fp4` |
|---|---|---|---|---|---|---|---|
| in_proj_qkv | 10240 | 5120 | 48 | 1.672e-3 | 0.9999990 | 2.71e-3 | **0.0 / 1.0** |
| gate_proj | 17408 | 5120 | 64 | 1.670e-3 | 0.9999990 | 2.96e-3 | **0.0 / 1.0** |
| lm_head | 248320 | 5120 | 1 | 1.659e-3 | 0.9999990 | 3.29e-3 | **0.0 / 1.0** |
| k_proj(+v_proj) | 1024 | 5120 | 32 | 1.684e-3 | 0.9999990 | 2.80e-3 | **0.0 / 1.0** |
| down_proj | 5120 | 17408 | 64 | 1.719e-3 | 0.9999990 | 2.87e-3 | **0.0 / 1.0** |

`mm_fp4_nrmse` 与 `gemv_nrmse` 逐位相同，`cand_vs_mm_fp4_nrmse = 0.0`：**候选与 M2
冻结的正确路径（`mm_fp4`）逐位一致**。这比过容差更强 —— 容差本身留了约 3× 余量
（decisions），而这里连 bf16 舍入都没差。

### 3.4 布局分别验

- `layout_probe`：nibble 顺序与 swizzled scale 映射两个 gate 全过，
  `nibble_probe_max_abs_err = 0.0`、`scale_probe_max_abs_err = 0.0`。
- 候选与 `mm_fp4` 逐位一致，也说明 a 侧 nibble、a_sf 寻址、W scale 寻址与
  FlashInfer/cutlass 完全同解。

### 3.5 k_split 一致性

`k_split ∈ {1, 2, 4}` 三档在**所有 shape 上输出完全相同**（`k_split_maxdiff_vs_1`
= 0.0，5/5），即只差 fp32 求和顺序且 bf16 下不可见。数值上三档都正确（新 gate
`numeric: 各 k_split 档结果一致` 固化这一点）。

### 3.6 CPU 自检（本轮没有跑任何 CUDA 命令）

| 检查 | 结果 |
|---|---|
| kernel 的 swizzle 寻址公式 vs `swizzle_scale_128x4`（1024×320 / 128×64 / 5120×1088 逐元素） | 0 mismatch |
| M=1 的 `a_sf` row0 寻址 `c*512+d` | 0 mismatch |
| E2M1 16 个码字解码（含 `-0` 与 0.5 次正规） | 16/16 |
| 按 kernel 循环结构逐 lane/逐 warp 模拟 vs reference matmul（`k_split` 1/2/4，含末组 `nc<8` 的 masking） | **bit-exact** |
| reference bug 复现（uint8 vs E4M3） | 同型（nrmse ≫ 1、cosine < 1） |
| `nvcc -c -arch=sm_120a` 编译 | 通过（57 regs / 0 spill / 4228B smem） |
| `SCENARIOS` AST 字面量求值 | 通过（6 个 case，含 4 个必测 shape） |

## 4. 性能（同口径真实实例聚合）

计时口径：一次计时迭代扫过该 shape 在模型里的**全部真实实例**（逐层 launch），
工作集 0.09–3.2 GB，远大于 L2（GB203 64 MB）；候选与 `mm_fp4` 用同一批实例、
同一份 A/W、同一个 CUDA event 边界，warmup 5 次后取 20 次中位数。
有效带宽 = (packed weight + scale 字节) / ms。单实例数字只作 diagnostic（L2-hot，
本表不用）。

| case | 聚合字节 | 实例 | `k_split` | 候选 ms | 候选 GB/s | `mm_fp4` ms | `mm_fp4` GB/s | 比值 |
|---|---|---|---|---|---|---|---|---|
| in_proj_qkv | 1.416 GB | 48 | 4 | 2.265 | 625.1 | 2.421 | 584.7 | **1.069×** |
| gate_proj | 3.209 GB | 64 | 2 | 4.733 | 678.0 | 3.906 | 821.5 | **0.825×** |
| lm_head | 0.715 GB | 1 | 1 | 0.844 | 846.9 | 0.805 | 888.2 | 0.953× |
| down_proj | 3.209 GB | 64 | 4 | 4.442 | 722.4 | 4.307 | 745.1 | 0.970× |
| k_proj(+v) | 0.094 GB | 32 | 4 | 0.592 | 159.6 | 1.540 | 61.3 | 2.604×（`l2_multiple=1.41`，不计） |

**合计（4 个 L2-clean shape，8.548 GB）：候选 12.283 ms = 695.9 GB/s，
`mm_fp4` 11.438 ms = 747.3 GB/s → 0.931×。** 只有把 L2 被抬高的 `k_proj` 也算进去
才是 1.008×（8.642 GB、671.3 vs 665.9 GB/s，Coordinator 汇总里的 1.01× 就是这一口径）。
按每 shape **事后最优** `k_split`（in_proj_qkv 2、gate_proj 1、lm_head 1、down_proj 4）
是 11.772 ms = 726.2 GB/s → **0.972×**。也就是说：**候选没有胜过 cutlass，L2-clean
口径下慢 3–7%**；`k_proj` 的 2.6× 只反映它 0.094 GB 的工作集塞得进 L2，不是真实收益。

`k_split` 扫描（聚合 ms / GB/s）：

| case | ks=1 | ks=2 | ks=4 | 最优 |
|---|---|---|---|---|
| in_proj_qkv | 2.627 / 538.8 | **1.966 / 719.9** | 2.265 / 625.1 | ks=2 |
| gate_proj | **4.519 / 710.0** | 4.733 / 678.0 | 5.215 / 615.3 | ks=1 |
| lm_head | **0.844 / 846.9** | 0.848 / 843.8 | 0.967 / 739.3 | ks=1 |
| down_proj | 11.040 / 290.6 | 5.865 / 547.1 | **4.442 / 722.4** | ks=4 |
| k_proj(+v) | 1.705 / 55.4 | 0.919 / 102.7 | **0.592 / 159.6** | ks=4 |

解释（有界的假设，未再花预算验证）：

1. **block 数/占用是主因。** kernel 是 512 线程、57 regs → 2 block/SM = 168 个并发
   block。`down_proj`（N=5120 → 40 个 tile）必须靠 split-K 才能填满 SM（ks=1 只有
   290 GB/s，ks=4 到 722 GB/s）；`lm_head`（1940 个 tile、11.5 波）不 split 就接近
   峰值（847 GB/s）。而 `gate_proj`（136 个 tile）已经接近填满，再 split 反而更差
   （710 → 678 → 615 GB/s），因为 split-K 把每行的连续读段从 2560B 切成 1280B/640B，
   行内流式被切碎。auto 策略（“block 数 ≥ 2×SM”）在这两个方向之间取了折中，
   对 `gate_proj` 偏保守（用了 ks=2，比最优慢 4.5%）、对 `in_proj_qkv` 偏激进
   （用了 ks=4，比最优慢 15%）。
2. **`gate_proj` 是本轮最大单点损失（−17.5%）**，而 gate+up 两个 tensor 占每步
   6.4 GB / 14.4 GB，所以它直接决定总账。差距不在解码 ALU（CPU 模拟与逐位一致说明
   算法没问题），更可能在「每行 256B 一段、8 行一组」的请求形态（32B sector 粒度、
   128 行 tile 的 40KB/ warp-iteration 工作集）对 DRAM/L2 预取不如 cutlass 的大 tile。

**与模型的关系**：模型 8K 单步 GEMV 是 20.7 ms / 14.399 GB = 696 GB/s，而 `mm_fp4`
在同口径聚合扫描下是 747 GB/s（14.399 GB → 19.27 ms）。也就是说 GEMV 这一项在模型
里只因为 kernel 之间的干扰/尾部损失了约 **1.4 ms/step（占 44.38 ms 的 3.2%）**，
换 GEMV kernel（哪怕换成一个 747 GB/s 的）最多也就拿回这部分。

## 5. 记录：一条 gate 的规格是错的

`job-000025` 唯一失败项是 `reference: 两条 A 反量化路径一致（nrmse <= 1e-2）`，
只有 `gate_proj` 超阈值：**nrmse = 1.38e-2，a_code_mismatch = 13**（其余 shape
是 3.1e-7 / 2.2e-7 / 3.8e-3 / 3.2e-3，a_code_mismatch 0/0/1/4，`a_sf_mismatch`
全 0）。

**这条 gate 的规格是错的**：它比的是**两个不同的量化器**——主路径解 FlashInfer 的
`a_fp4/a_sf`，第二条走 M2 冻结的 `quantize_activation_reference`。E 的
`job-000024-78f923c5`（commit `6a5a1a2`，51/51 gate）已经把 FlashInfer activation
quant 的算术定位到 `non-TMA quantize_with_block_size`（`ELTS_PER_THREAD=16`、
fast-math `rcp`）：M2 reference 规则与它的 codewise 差异率是 **9.84e-4**
（15392/15638528），且与 global divisor 大小相关。`gate_proj` 的 `input_scale`
是四个 shape 里最大的（827.1），差异率最高，正好对应这里的 13 个 code 差异和
1.38e-2 的 cross nrmse —— 这是**量化器语义差异的预期值，不是布局错误**。
把它设成 `nrmse ≤ 1e-2` 的 gate 等于要求两个不同量化器逐位一致，应改为
diagnostic（或改成“只对同一量化器”的比较）。

候选的正确性证据不受这条影响：`cand_vs_mm_fp4_nrmse = 0.0`（5/5），
`gemv_nrmse ≈ 1.67e-3` 在冻结容差内（余量 3×），`cosine ≈ 0.999999`，
`max_rel ≈ 2.7e-3`。

附带一个产物缺陷：`a_decoded_rel_maxdiff` 这个 diagnostic 的归一化写错了
（`a_scaled` 不含 `input_scale`，`a_ref_deq` 含，所以它读出来约等于
`|input_scale − 1|`，例如 gate_proj 826.1 ↔ input_scale 827.1），**不能当差异指标用**；
判断 A 一致性应看 `a_code_mismatch` / `a_sf_mismatch` / `ref_cross_nrmse`。

## 6. 结论

1. **不接入**：L2-clean 同口径聚合下候选 695.9 GB/s，`mm_fp4` 747.3 GB/s
   （0.93×；事后最优 `k_split` 也只有 0.97×）。cutlass 在 M=1 已经接近可达的
   DRAM 效率，换 SIMT kernel 不是这一项的杠杆；模型里 GEMV 只剩约 1.4 ms/step
   的干扰损失可拿。
2. **一个已验证正确的 SIMT GEMV 仍然有独立价值**：它能直接吃 128x4 swizzled
   scale（不需要 linear 副本、不需要 repack），内部自己完成 `a4·a_sf` 解码、
   `w4·w_sf` 解码与 FP32 归约，输出与 `mm_fp4` **逐位一致**。这使它适合作为
   「norm + activation quant + GEMV」融合的底座（cutlass 做不到）。
3. 工程问题已回答：swizzled scale **原地消费**（§2 公式），nibble 顺序与 scale
   布局**分别验证**（§3.4），`k_split` 三档数值一致（§3.5）。

## 7. 限制与下一轮建议

限制：

- 数值 reference 只对每个 shape 的**第一个实例**算（同 shape 同一条代码路径）；
  其余实例只参与计时。逐实例权重映射没有独立对拍。
- `gate_proj` 的 −17.5% 只有相关性解释（block 数/占用 + 行内流式被切碎），
  没有做对照实验（例如改 block 形状、按列切 128 行 tile、`cp.async`/TMA 预取）。
- 计时用 torch caching allocator 的输出缓冲复用；含分配的完整调用没有单独测。
- `a_decoded_rel_maxdiff` diagnostic 归一化错（§5），下一版要修或删。

下一轮建议（若继续这个方向）：

1. **融合**：把 norm + activation quant 并成一个小 kernel（E 的规则已可 bitwise
   复现 FlashInfer，`job-000024`），再接本 GEMV；收益是省掉每层独立的
   `quantize_with_block_size`（401 次、0.52 ms）与 norm 的一部分（209 次、0.32 ms），
   合计上限约 0.8 ms/step。注意不要让每个 GEMV block 重复算整段 K 的 amax/round
   （in_proj_qkv 会有 ~80× 冗余）。
2. 若还想在 GEMV 本体上追回 `gate_proj` 的 17%：优先试**按列切 128 行 tile**
   （2 个 block 各管 64 行，scale 读 2× sector 放大 ≈ +11% 流量，但 block 数翻倍）
   或**每 block 4 warp / 8 行 × 更多 block**，并用 `cp.async` 预取行段；这属于新一轮
   参数/结构探索，不要在本轮结论里当成已实现。
3. `k_split` 策略改成按 shape 实测表（本 journal §4）而不是“block ≥ 2×SM”规则；
   但那只是 3% 量级的微调。

## 8. 复现

```bash
# 提交（GPU 队列；不要在本地跑 CUDA）
cd .tmp/worktrees/explorer-gemv-cuda
.venv/bin/python /home/xzq/workspace/qwen/scripts/experiment.py submit \
  --request .tmp/req-gemv-cuda.json          # commit b4d6a15, timeout 1200
.venv/bin/python scripts/experiment.py result job-000025-bae867f6
# 产物：results/experiments/job-000025-bae867f6/output/op/op.json

# CPU 自检（不建 CUDA context）
.venv/bin/python -m py_compile bench/micro/nvfp4_gemv.py runner/ops/nvfp4/gemv.py
```

CPU 自检脚本本轮放在 worktree 的 `.tmp/`（不入库，按需重建），内容是：
`check_layout.py`（§3.6 前三行）、`check_kernel_sim.py`（逐 lane 模拟）、
`check_ref_fix.py`（reference bug 复现）。其中最关键的一条可以 5 行复现：

```python
import torch
from runner.ops.nvfp4.reference import swizzle_scale_128x4
lin = torch.arange(1024 * 320, dtype=torch.float32).reshape(1024, 320)
sw = swizzle_scale_128x4(lin).view(-1)
idx = lambda tile, row, c, d: (tile * (320 * 128) + c * 512
                               + (row % 32) * 16 + ((row // 32) % 4) * 4 + d)
assert float(sw[idx(0, 5, 3, 2)]) == float(lin[5, 3 * 4 + 2])   # kernel 寻址 == swizzle
```

下一轮若继续这个方向，建议把 `check_layout.py` / `check_kernel_sim.py` 整理成
`tests/ops/test_nvfp4_gemv.py`（纯 CPU）并入 `make verify` 的 S1 档。
