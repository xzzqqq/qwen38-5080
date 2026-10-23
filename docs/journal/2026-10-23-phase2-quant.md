# 2026-10-23 Phase 2 任务 E：activation quant 差异归因

## 状态

完成（一次归因 + 一次留出验证）。结论：PyTorch reference 与 FlashInfer 的 activation
NVFP4 量化差异来自 kernel 的 fast-math 算术序列——FP8 block scale 用
`gs * (amax * rcp_approx(6))`，out scale 用两次 `rcp.approx.ftz.f32`，元素用
`cvt.rn.satfinite.e2m1x2.f32`；把这三处按 PTX 语义复刻后（规则 R2），在
**15,638,528 个 code / 977,408 个 block** 上逐位复现 FlashInfer 的 code 与 scale，
残余 0。**可以冻结融合 kernel 所需的 quant reference**；但 bitwise reference 必须
使用 `rcp.approx.ftz` 指令（或同卡 inline PTX 探针），用数学上更精确的 `1/x`
会留下 ~1.5e-4 的 code 差异。

- plan：[phase2.md §3 任务 E](../plans/phase2.md)
- 脚本 commit（job 实际运行）：`6a5a1a255761242ea798f791ff28c984367755be`
- job：`job-000024-78f923c5`（`succeeded`，113.2 s，`correctness=passed`，**51/51 gate**）
- 产物：`results/experiments/job-000024-78f923c5/output/op/op.json`、
  `.../mismatch_blocks.jsonl`（5,584 个 mismatch block 明细，5.36 MB）
- checkpoint：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`，revision
  `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`
- 环境：RTX 5080（sm_120a）、torch 2.14.1+cu130、FlashInfer 0.7.2、triton 3.8.0、
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- kernel 源：`flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh`
  （`cvt_warp_fp16_to_fp4`、`fp32_vec_to_e2m1`、`reciprocal_approximate_ftz`）和
  `.../kernels/quantization.cuh`（`quantize_with_block_size`）
- 本 journal commit 另修正了 `_hex_bf16` 的编码（见 §8 限制），不改变任何 gate/metric。

## 1. 结论摘要

| 项 | 值 |
|---|---|
| 覆盖 | 977,408 blocks / 15,638,528 codes；K∈{5120,6144,17408} |
| 规则 R2 残余 | **0 codes / 0 blocks**（全部 8 个 case，fit 与 heldout 都是 0） |
| 现有 R0（M2 reference）差异 | 11,450 blocks / 15,392 codes（code 比例 **9.84e-4**） |
| offline R1（正确舍入 `1/x`）残余 | 2,337 codes（**1.49e-4**） |
| M2 失败样本 replay | 记录 10 个，复现 **10/10**（`scale_max_abs` 0/8/32、`code_mismatch` 6–31 全部一致） |
| `rcp.approx.ftz(6.0)` | `0x3e2aaaab`，与 `fl(1/6)` **0 ulp 差** |
| E2M1 tie 规则 | 硬件 `cvt.rn.satfinite.e2m1x2` 是 ties-to-even，7 个中点全部实测确认 |
| 残余类别 | 0 个无法解释的 block（block 级 `residual = 0`） |

R0 差异的 block 级分类（全部 11,450 个 mismatch block 都有分类）：

| 类别 | blocks |
|---|---|
| `global-divisor-mul-recip`（sf 相同，code 在 E2M1 边界 1 ulp 内翻转） | 7,122 |
| `e2m1-midpoint-tie`（sf 相同，code 恰好落在/跨过中点） | 2,489 |
| `fp8-scale-rounding`（FP8 scale code 差 1 个 e4m3 ulp） | **1,836** |
| `signed-zero`（输入 −0.0，R0 给 +0，硬件给 −0 code） | 3 |
| `subnormal-underflow` / `saturation` / `residual` | 0 / 0 / **0** |

## 2. 规则 R2（逐条 PTX 语义）

```
输入：x BF16 [M, K]，K % 16 == 0；input_scale 为 per-tensor F32（反量化乘数）
gs = fl32(1.0 / input_scale)                     # linear.py 的 global divisor

对每个 16-element block（沿 K，k0 = 16*b）：
  amax = max_i |float32(x[k0+i])|                # bf16 -> f32 精确；取绝对值后求 max
  sf_pre = mul_rn(gs, mul_rn(amax, c6))          # c6 = rcp.approx.ftz.f32(6.0f)
  sf_u8  = cvt.rn.satfinite.e4m3x2.f32(sf_pre)   # round-nearest-even + satfinite(±448)
  sff    = float32(sf_u8)
  if amax == 0.0:
      out = 0.0f                                 # kernel 显式短路，不调用 rcp
  else:
      out = rcp.approx.ftz.f32(mul_rn(sff, rcp.approx.ftz.f32(gs)))
  code_i = cvt.rn.satfinite.e2m1x2.f32(mul_rn(float32(x[k0+i]), out))
  # ties-to-even；饱和到 ±6；-0.0 保留符号位（code 8）
```

打包与布局（与 `runner/ops/nvfp4/linear.py` 一致，本次未改）：

- 每字节两个 E2M1 code，**低 nibble 是偶数下标**，高 nibble 是奇数下标；
- block scale 按 128x4 swizzled 布局写出（rows 补到 128、cols 补到 4 的倍数，padding 为 0），
  对应 `nvfp4_quantize(..., sfLayout=layout_128x4, do_shuffle=False, backend="cuda")`；
- 反量化仍用 `alpha = input_scale * weight_scale_2`，block 乘子 `sff * input_scale`；
  `input_scale` 不折进 e4m3 scale（会饱和）。

### `rcp.approx.ftz` 实测

用 Triton `tl.inline_asm_elementwise` 在同卡上执行 `rcp.approx.ftz.f32`：

- 15 个输入（`6.0` + 14 个真实 `gs`）：**只有 1 个**与 `fl(1/x)` 不同；
- `rcp.approx.ftz(6.0) = 0x3e2aaaab`，与 `fl(1/6)` 完全相同 → **c6 的 1-ulp 歧义不存在**；
- `gs = 73.1428604`（`layers.38.mlp.gate_proj/up_proj`，`input_scale=14/1024`）时
  `rcp.approx` 返回 `0.013671875 = 14/1024`，比正确舍入的 `1/gs` **高 1 ulp**
  （`0.013671874068677425`）。这一条解释了 real-M2 两个 case 里 841/875 个 R1 残余 code；
- 外层 rcp（`sff * rcp(gs)`，814 个不同输入）没有单独统计差异数；但 heldout-gaussian
  的 `gs` 的 rcp 与 `fl(1/x)` 相同、R1 仍有 130 个 code 差异，说明**外层 rcp 也有
  1-ulp 偏差**。结论：两个倒数都必须用 approx 指令。

E2M1 探针（0 mismatch，软件实现与硬件一致）：

| 输入 | 0.0 | −0.0 | 0.25 | 0.75 | 1.25 | 1.75 | 2.5 | 3.5 | 5.0 | 6.0 | 6.5 | −6.5 | ±1e10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code | 0 | 8 | 0 | 2 | 2 | 4 | 4 | 6 | 6 | 7 | 7 | 15 | 7/15 |

## 3. 为什么 R0 / R1 不够

**R0（`quantize_activation_reference`）**：`sf = e4m3((x*gs).amax()/6)`，code 用
`(x*gs)/sf` 除法。两个差异点：

1. FP8 scale 的乘法顺序/倒数不同：kernel 是 `gs*(amax*rcp6)`，R0 是
   `fl(amax*gs)/6`。两者差 1–2 个 fp32 ulp；当 `sf_pre` 正好落在 e4m3 相邻值中点时
   （checkpoint 的 `input_scale ≈ calib_amax/2688`，很多 bf16 amax 会精确命中整数中点，
   例如 `gs=279.2727, amax=2.921875 → sf_pre=136.0 = (128+144)/2`），两边会 round 到
   **不同的 e4m3 code**（`0x70` vs `0x71`，绝对差 = 1 个 e4m3 ulp；在 256–448 区间
   就是 M2 记录的 `scale_max_abs=32`）。一个 block 的 scale 差 1 code，会让该 block
   16 个 code 里若干/全部改变，dequant 误差 ~12.5%，这正是 M2 `numeric.json` 里
   `nrmse=0.0587 / max_rel=0.18` 的来源。
2. out scale 用除法而不是 `rcp.approx`：`t = (x*gs)/sf` 与 `x*rcp(sf*rcp(gs))` 差
   ~1–2 fp32 ulp；只有 `t` 恰好落在 E2M1 边界（0.25/0.75/1.25/1.75/2.5/3.5/5.0）附近
   1 ulp 时 code 才翻转，表现为 `global-divisor-mul-recip` / `e2m1-midpoint-tie`。
3. `-0.0` 符号：R0 用 `values < 0`，把 −0.0 变成 +0 code；硬件 `cvt` 保留符号位
   （code 8）。实测 3 个 block（其中 1 个在 `layers.57.mlp.down_proj` 的 decode 抓到了
   完整样本：`x = -0.0 (0x80000000)`，R0 code 0，FlashInfer code 8）。

代表样本（`mismatch_blocks.jsonl`）：

| 类别 | 数据 | 关键值 |
|---|---|---|
| `fp8-scale-rounding` | synthetic-gaussian-k5120 | `gs=279.2727, amax=2.921875, sf_pre=136.0`（中点，距离 0 ulp），`sf_ref=0x70` vs `sf_flash=0x71`，5 个 code 改变，这些 code 到 E2M1 边界 9.5e4–1.2e6 ulp → 差异完全由 scale 解释 |
| `global-divisor-mul-recip` | synthetic-scale-tie-k6144 | `gs=122.88, amax=9.75, sf` 相同（0x74），1 个 code 改变（14→15），`t` 到边界 **1.0 ulp** |
| `e2m1-midpoint-tie` | synthetic-gaussian-k5120 | `gs=279.2727, amax=5.0625, sf` 相同（0x77），`t` 到边界 **0.0 ulp**（恰好 0.25），R0 tie→0，FlashInfer 1 ulp 偏上→0.5（code 1） |

R0 的差异不是随机噪声：`mismatch_blocks.jsonl` 里 1,836 个 scale 差异全部是
**1 个 e4m3 code**，绝对差分布 `{0.0625, 0.125, …, 32}`（就是各量级的 1 ulp）。

**R1（offline，正确舍入 `1/x`）**：把 R0 的两个结构性问题都修好后仍有 2,337 个 code
差异（1.49e-4），全部来自 `rcp.approx` 与 `fl(1/x)` 的 1-ulp 差；纯 software 变体在
fit 集上的 code 差异：`exact_rcp=2207`、`div=7013`、`div_then_rcp=6274`，而用
inline-PTX 实测 rcp 的 R2 是 **0**。所以「数学上更精确的倒数」反而对不上。

## 4. 覆盖与残余

| case | K | blocks | codes | R0 blocks | R0 codes | R1 codes |
|---|---|---|---|---|---|---|
| synthetic-gaussian-k5120 | 5120 | 40,960 | 655,360 | 260 | 1,070 | 0 |
| synthetic-outlier-k17408 | 17408 | 139,264 | 2,228,224 | 1,145 | 1,201 | 87 |
| synthetic-scale-tie-k6144 | 6144 | 49,152 | 786,432 | 1,264 | 1,316 | 84 |
| real-m2-k5120 | 5120 | 138,240 | 2,211,840 | 1,639 | 2,724 | 841 |
| real-m2-k17408 | 17408 | 391,680 | 6,266,880 | 6,423 | 8,355 | 875 |
| real-m2-k6144 | 6144 | 82,944 | 1,327,104 | 400 | 405 | 320 |
| heldout-gaussian-k6144 | 6144 | 73,728 | 1,179,648 | 0 | 0 | 130 |
| heldout-outlier-k5120 | 5120 | 61,440 | 983,040 | 319 | 321 | 0 |
| **合计** | | **977,408** | **15,638,528** | **11,450** | **15,392** | **2,337** |

- 样本来源：3 组 synthetic（真实 `input_scale` 尺度的高斯、长尾离群值、把 bf16 amax
  固定到 e4m3 中点上的 stress）+ 3 个 real-M2 replay（14 个 M2 失败模块 × 3 个 K，
  同 prompt + numeric.json 的 fixed_tokens 重放 prefill + 16 步 teacher forcing）+
  2 个 heldout（另一 seed/scale，不参与规则选择）；
- 非 TMA kernel：所有量化调用的 M ≤ 128（decode M=1、prefill chunk ≤128，replay 把
  token 拼成 M=72），实际生产路径就是 `quantize_with_block_size`（ELTS_PER_THREAD=16）；
- M2 replay：10/10 记录的 `(module, scale_max_abs, code_mismatch)` 复现，包括
  math 的 `layers.57.mlp.gate_proj (32, 10)`、chat 的
  `layers.48.mlp.down_proj (8, 31)`、`layers.10.mlp.gate_proj (32, 17)`；
- 残余：**R2 0 code**；11,450 个 R0 mismatch block 全部有分类；`residual=0`。
  code 级直方图里另有 5,310 个 `residual` code，它们全部位于 `fp8-scale-rounding`
  block 内（由 1-code scale 差解释，block 级已归类为 scale），不是未解释差异。
- 稳定性：同一输入连续两次 FlashInfer quantize、两次 reference 均 bitwise 一致
  （每个 case 的 `determinism:*` gate）。

## 5. 能否冻结融合 kernel 的 quant reference

**能，冻结规则 R2**。两种等价的冻结方式：

1. **GPU reference（推荐做 bitwise gate）**：用 Triton `tl.inline_asm_elementwise`
   直接执行 `rcp.approx.ftz.f32` 与 `cvt.rn.satfinite.e2m1x2.f32`，其余用 FP32
   `mul.rn`；本 job 的 `_probe_rcp_gpu` 就是这套做法，在全部 fit+heldout 上 0 残余。
2. **离线 CPU reference**：结构照 R2，但第二个倒数只能写 `1/x`（R1）。它**不可能
   bitwise**：本数据上残余 1.49e-4 个 code，集中在 E2M1 边界；用作数值 gate 时必须
   显式接受该残差，或改用方式 1。

不建议把 R0 继续当 reference：它的差异率 9.84e-4，且会随 `input_scale` 的取值在
0（heldout-gaussian 的 1.18M code 全对）到 1.6e-3（scale-tie case）之间跳变，
无法用固定容差区分「reference 自身差异」和真实实现错误。

## 6. 融合 kernel 的 quant 契约（给下一轮）

- **输入**：BF16 activation `[M,K]`（M=1 decode / ≤128 prefill chunk），K%16==0；
  若融合 norm，必须在 kernel 内**先 round 回 BF16** 再量化，否则量化输入 dtype 变了，
  code 会变（这是语义变化，要单列）。
- **global divisor**：`gs = fl32(1/input_scale)`，per-tensor F32；不能改成乘
  `input_scale`。
- **block**：16 个连续元素；`amax` 用 FP32 从 BF16 精确转换后取绝对值 max。
- **SF**：`mul_rn(gs, mul_rn(amax, rcp.approx.ftz(6.0f)))`，再
  `cvt.rn.satfinite.e4m3x2.f32`；写 128x4 swizzled、rows pad 128、cols pad 4、padding=0。
- **元素**：`out = rcp.approx.ftz(mul_rn(sff, rcp.approx.ftz(gs)))`（amax==0 时 0），
  `code = cvt.rn.satfinite.e2m1x2.f32(mul_rn(float32(x), out))`；
  **必须用 approx 指令**，`__frcp_rn`/`1/x`/软件 RNE 都会破坏 bitwise。
- **nibble**：低 nibble 偶数下标；**alpha**：`input_scale * weight_scale_2` 单独传。
- **已验证 shape**：K∈{5120,6144,17408}；`input_scale` 是 per-tensor 常数，per-token
  scale（`per_token_activation=True`）路径未覆盖。

**fusion 边界建议**：先做 **RMSNorm → activation quant 一个小 kernel，直接产出
packed A + swizzled SF，喂给现成 GEMV**。理由：norm 需要整行 reduction，quant 需要
per-16 block amax，两者都在「一行 activation」上，融合后省掉 BF16 normalized 的
写回+读入和一次 quantize launch（M3 profile 里 <5 µs kernel 占 93.6%，launch/空隙
是主要收益，字节是次要的）；GEMV 的输入契约完全不变。`gate/up + SiLU·mul` 建议放在
**down_proj 的 quant kernel 侧**（读两路 BF16、silu·mul、再按同一契约量化），不要先
动 gate/up GEMM 的 epilogue：这样 GEMM 不用改，语义变化只在「silu·mul 在 FP32 还是
BF16」这一点上，可以单列 gate。等 norm+quant 与现有 FlashInfer 路径 bitwise 对齐后，
再评估把 silu·mul 挪进 GEMM epilogue 的收益。

## 7. B/C 观察的交叉验证（divisor 相关性）

B/C 看到「同 seed 下 `in_proj_qkv`/`k_proj` 的 packed A 与 R0 逐字节相等，而
`gate_proj`/`lm_head` 只有 `input_scale` 不同，divisor 53.5/36.8/54.0/827.1 对应
byte_diff 0/0/1/10」。R2 与这个现象一致，但**不预测「divisor 越大差得越多」**：

- 差异率由三件确定性的事决定：(a) `gs*(amax*rcp6)` 是否正好落在 e4m3 中点（amax 是
  bf16、会跨 token 重复，所以一旦命中就整批翻转）；(b) `x*out` 是否落在 E2M1 边界
  1 ulp 内；(c) 输入里有没有 −0.0。三者都是 `gs` 的确定性函数，但**不是单调函数**。
- 本数据里 14 个真实 `gs` 只有 1 个（73.1428604）的 `rcp.approx` 与 `fl(1/x)` 差 1 ulp，
  其余 13 个完全相同；而 R0 的差异率在各 case 间从 0（`gs=117.5`，1.18M code 全对）到
  1.6e-3，`gs=279.3`（大 divisor）是 0.63% block、`gs=45.9` 是 0.52% block——没有随
  divisor 增大而增大的趋势。
- B/C 的 4 个点样本量小：按本数据 0–1.6e-3 的 code 差异率，几千个 code 的期望差异是
  0–8 个 code（≤4 byte），0/0/1/10 byte 落在同一量级的自然散布内。所以那条现象是
  `gs` 特定对齐（以及 827.1 那个 `gs` 的 rational 关系）造成的，不是规模规律；
  用 R2 重算 reference 即可解释，不需要新实验。

## 8. 限制

- 只覆盖 K∈{5120,6144,17408}、非 TMA 路径（M<1024）与上述分布；TMA kernel
  （`quantize_with_block_size_tma`，M≥1024 且 K%512==0）未覆盖——生产 decode/prefill
  的 M≤128，不在该路径上。
- `input_scale` 只覆盖 per-tensor 常数；`nvfp4_quantize(per_token_activation=True)` /
  `out_scale` 路径未覆盖。
- 边界：实测到了 `signed-zero`（3 block）、E2M1 中点（7 个中点全部实测）、饱和
  （探针 6.5→7、−6.5→15；outlier 数据 amax=17536 时两侧都饱和，0 差异）；**没有实测到**
  `amax==0` 的 mismatch（两侧都输出 sf=0/code=0，padding 行的 SF=0 只验证布局）和
  `sf_pre` 下溢到 e4m3 subnormal/0 的 block（最小 `sf_pre=0.12`，远高于 2^-9）；
  规则里 `amax==0 → out=0` 与 `sf=0 → out=rcp(0)=inf → 饱和` 的分支按源码实现，
  但未被真实数据触发。
- `mismatch_blocks.jsonl` 是按 group 截断的（每 group 最多 512 个 block，共 5,584 个），
  block 级直方图来自未截断的聚合计数；job 产物里的 `x_hex` 是 float32 位型拆成两个
  16-bit 半字（`[2i]` 低半字、`[2i+1]` 高半字），这是 `_hex_bf16` 漏了 `.to(bf16)` 的
  编码 bug；本 journal commit 已修，重跑会写 4-hex 的 bf16 位型；gate/metric 不受影响。
- 真实 hidden capture 花了 107 s（模型 load + 3 prompt × (prefill + 16 decode)），
  是本 job 的主要耗时；后续同类实验按 ~110 s 计预算。

## 9. 复现

```bash
# 队列（GPU 主机，单 job）
.venv/bin/python scripts/experiment.py submit --request .tmp/req-quant.json   # 脚本 commit 6a5a1a2
.venv/bin/python scripts/experiment.py result job-000024-78f923c5
# 产物
results/experiments/job-000024-78f923c5/output/op/op.json
results/experiments/job-000024-78f923c5/output/op/mismatch_blocks.jsonl
```

## 10. 下一步

1. 把 R2 的 quant 契约写进融合 kernel 的 gate（GPU inline-PTX reference 做 bitwise，
   CPU reference 只做 diagnostic）。
2. 实现 RMSNorm → quant 融合（保持 BF16 rounding 点不变），与现路径 bitwise 对拍，
   覆盖 K∈{5120,6144,17408} 和 M=1/128。
3. `silu·mul + quant` 作为第二个融合边界，单列语义变化（silu·mul 的精度位置）。
4. 若后续 shape/`input_scale` 超出本次覆盖，直接用
   `scripts/phase2_quant_attrib.py` 的同一套 case 扩一项重跑即可（≤5 min：CPU 部分
   ~3 s，模型 capture ~110 s）。
