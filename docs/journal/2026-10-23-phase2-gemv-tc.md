# 2026-10-23 Phase 2 任务 C：tensor core NVFP4 GEMV（swap-AB）——方向判负

## 状态

任务 C（tensor core NVFP4 GEMV）**判负，不接入主线**。native FP4 tensor core +
swap-AB 在避开 L2 复用的计时下，4 个代表 shape 按真实实例数加权的聚合吞吐是
**664 GB/s，只有 cutlass `mm_fp4`（789 GB/s）的 0.84×**；外推整步 21.7 ms vs
baseline profile 20.7 ms。唯一打平的是 lm_head（887 vs 890 GB/s，0.4% 差距）。
换算成整步收益是负的，因此不做参数细调、不接入。

第二版测量还有一次**数值 gate 失败**（4/4 case），但证据指向测量脚本自身的
reference/接线回归而不是 kernel（见 §6）；方向已由性能否决，没有为它再花一个 job。

- 分支：`exp/phase2-gemv-tc`，最终 commit
  `20d75d144d7d8a8d4a0e0e93009a54b20edbe947`
- 第一版（v1，L2 命中污染）：commit `50205e7`，job `job-000018-0d4d3cb6`，
  100 s，36/36 gate 通过，`results/experiments/job-000018-0d4d3cb6/output/op/op.json`
- 第二版（v2，L2-free + graph 边界）：commit `20d75d1`，job `job-000021-87bd146b`，
  115.5 s，`gate_failed`，`results/experiments/job-000021-87bd146b/output/op/op.json`
- checkpoint：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`
  revision `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`
- 环境：torch 2.14.1+cu130、triton 3.8.0、FlashInfer 0.7.2、driver 580.105.08、
  CUDA 13、RTX 5080 16 GiB（sm_120a）
- 对照：短协议 baseline `job-000012`（commit `1ac7460`，32 步 × 1 次）
  graph-8192 44.33 ms / 22.54 sys tok/s；历史 511×5 是 44.38 ms / 22.52。
  FlashInfer `mm_fp4`（cutlass）401 次/step = **20.7 ms**，14.399 GB → 696 GB/s

复现：

```bash
cd /home/xzq/workspace/qwen/.tmp/worktrees/explorer-gemv-tc
cat .tmp/req-gemv-tc.json      # session phase2-gemv-tc, timeout_s 1200, 单 op step
/home/xzq/workspace/qwen/.venv/bin/python /home/xzq/workspace/qwen/scripts/experiment.py \
    submit --request .tmp/req-gemv-tc.json
```

## 1. 要回答的问题（plan §3 任务 C）

1. native FP4 MMA + swap-AB 是否胜过任务 B / baseline？
2. repack、padding、workspace 的额外成本是否抵消收益？

答案：**不胜过**（0.84×）；成本侧反而全为零（repack 0、padding 0、workspace 2.8 KB），
即"成本不是问题，收益本身就是负的"。

## 2. 实现与数据流

`runner/ops/nvfp4/gemm_tc.py`（Triton，`tl.dot_scaled`，e2m1 × e2m1 + per-16 ue4m3 scale）。

```
w U8 [N, K/2]（checkpoint 原样，K 连续）      w_scale F8 [N, K/16]（线性，无 swizzle）
a U8 [1, K/2]                                a_scale F8 [1, K/16]
  -> 每个 CTA 负责 BM 行权重，沿 K 以 BK=256 分块：
       tl.dot_scaled(W[BM,BK/2], Ws[BM,BK/16], A[BK/2,16], As[16,BK/16])
  -> mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
     .f32.e2m1.e2m1.f32.ue4m3（per-16 block scale，FP32 累加）
  -> 取第 0 列 * alpha(= input_scale * weight_scale_2) -> BF16 [1, N]
```

**swap-AB 是必须的**：不 swap 时 W 是 MMA 的 B 操作数（逻辑 `[K, N]`），而 W 在显存里
N 是外层 stride、K 连续，Triton 会退化成跨步 load（每行只取 1 B），DRAM 效率崩掉。
swap 后 W 成为 A 操作数，沿 K 完全 coalesced；代价是 B 的 16 列里只有 1 列有用
（算力浪费 1/16，但 M=1 是纯带宽受限，MMA 吞吐远未饱和）。

三个实现细节（都是离线可验证的坑）：

1. **PTX 确认**：`ptxas` 离线编译（`GPU Target("cuda",120,32)`，无 CUDA context）得到
   `mma.sync...kind::mxf4nvf4...scale_vec::4X...ue4m3`、`target sm_120a`；
   `tl.dot_scaled` 在这个 shape 上真的走原生 FP4 MMA，不是 bf16 软件模拟。
2. **u8 load 只有 1 B/线程**：直接按 `u8 [BM, BK/2]` 载入时 Triton 发 `ld.global.b8` +
   `st.shared.b8`、`cp.async=0`（完全不流水）。原因是 fp4 MMA 操作数的 shared 布局
   粒度是 4 B/线程。改成把权重按 **i32 视图载入**（`w.view(torch.int32)` →
   `[BM, BK/8]`）再在 kernel 里拆成 4 个 byte lane，才得到 4 B `cp.async` 流水
   （shared 17 KB / 2 stage，`ldmatrix` 4 条）。
3. **byte lane 顺序**：`tl.join` 的嵌套顺序决定字节顺序，
   `join(join(b0,b2), join(b1,b3))` 展开才是 `b0,b1,b2,b3`；直觉写法
   `join(join(b0,b1), join(b2,b3))` 得到 `b0,b2,b1,b3`。用 Triton interpreter
   （纯 CPU）逐字节验证过：后者 `match=False`，前者 `match=True`。
   同一条 CPU 路径还把**整个 kernel** 对照 dequant reference 跑通（8/8，误差 = interpreter
   的 bf16 截断量级，所有 diff 与 ref 反号 = 截断而非布局错）。

**成本口径**（不能藏进计时）：

| 项 | 值 | 说明 |
|---|---|---|
| repack | **0 B / 0 ms** | kernel 直接吃 checkpoint 的 `U8 [N, K/2]` + **线性** `[N, K/16]`；不需要 128x4 swizzle、不需要转置 |
| padding | **0 B** | 所有模型 shape 的 N 都能被 64/128 整除、K 都能被 256 整除（`padding_bytes()` 上报）；非整除 N 本轮不支持（显式报错，无 masked 尾块） |
| workspace | **K/2 + K/16 = 2880 B**（K=5120） | 只有 activation 的 packed bytes + scale；无全局 scratch；shared memory 是 CTA 内资源（8.7–34.8 KB/CTA） |

反向成本：baseline 需要 128x4 swizzled scale，实测一次性 swizzle（v2 计时）
in_proj_qkv 0.088 ms / gate_proj 0.153 ms / k_proj 0.158 ms / **lm_head 1.377 ms**
（79 MB 的 copy）。
方向若胜出，这一项是净省；但方向本身没胜出。

## 3. 第一版测量为什么会出现 >peak 的 GB/s

v1（job-000018）报出 in_proj_qkv **1425 GB/s**、gate_proj **2423 GB/s**，都超过
960 GB/s 物理峰值。两个独立偏差：

1. **L2 复用**：GB203 的 L2 约 64 MB。`in_proj_qkv` 29.5 MB、`gate_proj` 50.2 MB、
   `k_proj` 2.6 MB 的权重 + scale 都能整块住进 L2，`timed()` 反复跑同一个 kernel 时
   从 L2 命中，不是 DRAM 流量。真实 decode 每步只读一遍权重，没有这种复用。
2. **把 host 开销算进 event 计时**：v1 的 `ms_mm_fp4` 在小 shape 上恒为 60–68 µs，
   与 shape 大小无关（k_proj 2.6 MB 也要 60 µs）——这是 FlashInfer `mm_fp4`
   每调用的 Python/dispatch 开销被 `start.record(); fn(); end.record()` 包进了区间，
   不是 GPU 时间。生产里 8K decode 是 graph replay，没有这段开销。

v2 的修正（`bench/micro/nvfp4_gemv.py`）：

- **工作集 > L2**：每个 case 准备 `copies` 份**不同地址**的权重副本，一个计时单元 =
  依次跑完所有副本（工作集 ≥ 256 MiB）：lm_head 1 份（715 MB 本身 11× L2）、
  gate_proj 6 份（287 MiB）、in_proj_qkv 10 份（281 MiB）、k_proj 80 份（225 MiB）。
  **候选与 `mm_fp4` 用完全相同的一套副本协议**（baseline 也吃不同地址的 b 副本），
  所以基线不再被 L2 抬高。
- **graph replay 边界**：计时单元 capture 成 CUDA graph 后 replay，区间里只剩 GPU
  kernel + graph node 开销，与真实 8K decode 的边界一致。
- **同一份量化后的 A**：候选和 baseline 都吃 FlashInfer `nvfp4_quantize` 的 packed
  bytes，候选的 block scale 由 128x4 swizzled 反变换（`_unswizzle_a_scale`）得到；
  `same_quantized_a_as_baseline` 升为 gate。
- **加权聚合**：按 checkpoint 里数出来的真实实例数加权
  （`in_proj_qkv 48`、`gate_proj 64`、`k_proj 16`、`lm_head 1`；其他 shape：
  `up_proj 64`、`down_proj 64`、`out_proj 48`、`in_proj_z 48`、`q/k/v/o_proj` 各 16，
  401 个 NVFP4 Linear 合计与 manifest 一致），写进 `op.json.aggregate`。

## 4. 结果（v2，L2-free + graph replay）

单位：`ms_inst` = 每实例；`GB/s` 按 `forward_bytes(1, K, N)`（权重 + scale + A + 输出）。

| shape | N | instances | copies | 工作集 | 候选 best | 候选 µs/inst | 候选 GB/s | mm_fp4 µs/inst | mm_fp4 GB/s | 聚合候选 | 聚合 mm_fp4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| in_proj_qkv | 10240 | 48 | 10 | 281 MiB | c64x256 | 38.5 | 767 | 37.2 | **793** | 1.846 ms | 1.787 ms |
| gate_proj | 17408 | 64 | 6 | 287 MiB | c128x256 | 80.3 | 625 | 61.7 | **813** | 5.141 ms | 3.950 ms |
| lm_head | 248320 | 1 | 1 | 682 MiB | c64x256 | 806.6 | 887 | 804.5 | **890** | 0.807 ms | 0.805 ms |
| k_proj | 1024 | 16 | 80 | 225 MiB | c32x256 | 20.1 | 147 | 17.9 | **165** | 0.321 ms | 0.287 ms |
| **合计** | | | | | | | **664** | | **789** | **8.116 ms** | **6.829 ms** |

- `aggregate_speedup = 0.8415`；覆盖 step 权重字节 **37.4%**（5.387 GB / 14.399 GB）。
- `est_step_ms` = 14.399 GB ÷ 聚合 GB/s：**候选 21.69 ms vs cutlass 18.26 ms**
  （baseline profile 实测 20.7 ms）。未覆盖的 63% 主要是 up_proj/down_proj
  （64+64 实例、各 ~50 MB，与 gate_proj 同类），按同档外推；这是估计不是实测整步。
- 每配置都试过：gate_proj 上 c32/c64/c128 = 100.3 / 84.6 / 80.3 µs，都明显慢于
  cutlass 的 61.7 µs；lm_head 上 c32/c64/c128 = 1259 / 807 / 955 µs（c32 明显缺 CTA），
  cutlass 805 µs。**没有任何一组粗粒度参数能把聚合拉过 cutlass。**

### 为什么慢

- lm_head 打平说明"纯 DRAM 流 + 大 N"这条路上两者等价（都到 0.93 峰值带宽）。
- 差距来自小 N / 中 N：
  - `gate_proj`（N=17408，136 CTA@128）：候选 625 GB/s。多出来的成本是
    global → shared → MMA 的 round trip（每字节至少 2 次 on-chip 搬运）、
    byte lane 重组的 ~2.5 个整数 op/字节，以及 BN=16 列里 15 列空转带来的操作数带宽浪费。
  - `k_proj`（N=1024，BM=32/64 → 只有 32/16 个 CTA）：147 GB/s，被 occupancy 锁死；
    核函数时间 20 µs 里绝大部分是几十个 SM 各读 ~80–175 KB 的串行延迟，不是带宽。
    cutlass 同样受这个限制（165 GB/s），所以两者都远低于峰值——这一档只有 split-K
    或合并 k/v_proj 才有救。
- 结论：**task C 的假设（native FP4 + swap-AB 能比 cutlass 更快）不成立**。
  cutlass 在 M=1 上已经贴着带宽跑，而 swap-AB 只是把"布局问题"解决掉，并没有
  比"GEMM 当 GEMV 用"的 cutlass 更省流量。

## 5. 与任务 B 的关系

B（CUDA core GEMV）与 C 共用同一份量化 A/W 和计时边界；本方向没有形成对 B 的
可比数据（B 在另一条分支）。C 的结论是"TC 路线不划算"，不构成对 B 的判断；
按 plan §3，B/C 首轮选一个接入，C 已退出。

## 6. 第二版数值 gate 失败（4/4）与根因调查

v2 的 `op.json` 里 4 个 case 全部 numeric gate 失败：

| case | 候选 nrmse | 候选 cosine | baseline(mm_fp4) nrmse | baseline cosine |
|---|---|---|---|---|
| in_proj_qkv | 1.0 | 2.3e-5 | 0.889 | 0.983 |
| gate_proj | 1.0 | 0.410 | 0.321 | 0.981 |
| lm_head | 1.0 | 9.8e-5 | 0.889 | 0.982 |
| k_proj | 1.0 | 5.5e-7 | 0.921 | 0.985 |

**这不是"shape-dependent 的 `_unswizzle_a_scale` 取错 scale"**，证据：

1. `_unswizzle_a_scale` 的结果与 reference quantizer 的 scale **逐字节相同**：
   4 个 case 的 `reference_quantizer_scale_diff = 0`（0/320），
   `byte_diff = 0 / 10 / 1 / 0`（packed A 也基本相同）。
2. 失败是 **4/4 全部**，而 4 个 case 的 K 都是 5120、A scale 的形状和反变换路径完全相同；
   如果是反变换错，不可能 k_proj/lm_head 用同一段代码、同一个 K 却和 in_proj_qkv 不同。
3. **baseline `mm_fp4`（M2 冻结的正确路径）与同一个 reference 也不一致**
   （nrmse 0.32–0.92）。说明是 **reference / 接线** 出了问题，不是候选 kernel。
4. `runner/ops/nvfp4/gemm_tc.py` 在两个 commit 之间 **没有任何改动**
   （`git diff 50205e7 20d75d1 -- runner/ops/nvfp4/gemm_tc.py` 为空），
   同一 commit 上用 Triton interpreter 跑真实 kernel 对照 dequant reference 仍然
   8/8 通过。
5. 布局探针（nibble / scale）在 v2 里仍然全过，scale 探针 nrmse = 0.0。

现象本身：以 in_proj_qkv 为例，由 nrmse=1.0、cosine≈0、max_rel=1.0 反解出候选输出
相对 reference 的模长比 ≈ 4.6e-5，即输出被压到接近 0；三个 tile 配置的指标逐位相同
（输入确定，不是归约抖动），而 layout 探针仍然全过。

**仍未定位**：这是 v2 脚本（`bench/micro/nvfp4_gemv.py`）的回归，v1 的同一 kernel 在
同一 4 个 shape 上是 nrmse 0.0016–0.0017 / cosine ≥ 0.999999（36/36 gate 通过）。
定位它只需要一个 job 的一行 bisect：把 A 换回 reference quantizer 的
（`a_packed, a_scale = a_ref_packed, a_ref_scale`），或者去掉 `copies`/`out=` 预分配，
看 numeric gate 是否恢复。**没有做**——方向已被性能否决，修它不会改变结论。

记录口径（重要）：

- 候选 kernel 的数值正确性证据来自 **v1（job-000018）+ CPU interpreter 测试**；
  v2 的 numeric gate 因 reference 接线问题**无效**，不能当作"候选算错"的证据，
  也不能当作"候选正确"的复证。
- `bench/micro/nvfp4_gemv.py` 当前 HEAD 的 numeric gate **不可信**，谁要继续用它
  必须先做上面那次 bisect；它的 timing 部分（多副本 + graph）是可用的。

## 7. 交叉证据：activation quantizer 差异（给任务 E）

v2 顺手把 v1 的 `same_quantized_a_as_flashinfer=False` 查清楚了（这是 decisions.md
2026-10-21 留的已知差异，任务 E 的前置项）：

同一份 `x`（`torch.manual_seed(0)`、bf16 `randn(1,5120)*0.5`）、同一个 K=5120、
同一条量化代码路径，只有 tensor 的 `input_scale` 不同：

| shape | input_scale | 1/input_scale | E2M1 packed byte 差异 | F8 block scale 差异 |
|---|---|---|---|---|
| in_proj_qkv | 0.018694 | 53.49 | **0** / 2560 | 0 / 320 |
| gate_proj | 0.0012091 | **827.08** | **10** / 2560 | 0 / 320 |
| lm_head | 0.018508 | 54.03 | **1** / 2560 | 0 / 320 |
| k_proj | 0.027158 | 36.82 | **0** / 2560 | 0 / 320 |

观察：

- **block scale（F8_E4M3）在所有 shape 上都完全一致**，差异 100% 落在 E2M1 code 上
  ——不是 scale 舍入，是 code 舍入。
- 差异量级与 global divisor 的大小相关：divisor 最大的 gate_proj（827，是其他
  3 个的 15–22 倍）有 10 个字节不同；divisor 最小的两个（36.8 / 53.5）字节完全相同。
  合理机制：reference 用 fp32 除法 `block / sf` 再和 E2M1 中点比较，FlashInfer 用
  `cvt.rn.satfinite.e2m1x2.f32` 乘倒数；除数越大，两边在最后一位上的分歧越容易
  把落在中点附近的值分到相邻 code。
- 只有 4 个样本，是相关性不是证明。**任务 E 若要归因，建议固定 `x` 扫
  `1/input_scale`（例如 32/64/128/256/512/1024），统计 code mismatch 与
  "距中点 < 1 ulp 的元素数"**，就能把这条从相关性做成因果。
- 对 gate 的影响：v1 里 gate_proj 的 `baseline_nrmse` 因此是 0.0122（baseline 用
  FlashInfer 的 A，而 reference 用 reference 的 A），其余 3 个 shape 是 0.0017；
  差异只有 10 个字节，却把 nrmse 抬了 7 倍，说明这个对比对 A 的 code 一致性非常敏感
  ——**以后凡是要和 reference 对比的 baseline，必须先锁死"同一份 A"**。

## 8. 结论与下一步建议

1. **C 不接入**；GEMV 方向回到 B（CUDA core GEMV）作为唯一候选。
2. 若还想从 GEMV 拿收益，从 v2 的数据看只有三个具体可控项：
   - **小 N 的 occupancy**：`k_proj`/`v_proj`（N=1024）在候选和 cutlass 上都是
     147/165 GB/s（16 CTA 锁死）。可做 split-K（K 方向切 4–8 段 + fp32 原子或
     两段归约），或把 `k_proj`+`v_proj` 合成一次 N=2048 的 launch。收益上限
     约 0.25 ms/step（k_proj 0.321 → ~0.06 ms，v_proj 同理）。
   - **per-shape 选型**：cutlass 自己在 4 个 shape 上是 793 / 813 / 890 / 165 GB/s，
     gate_proj 明显低于 lm_head（813 vs 890）。`mm_fp4` 支持
     `b12x / cutlass / cudnn` 多 backend（SM120 上 `auto` 偏好 `b12x`），
     按 shape 选 tactic 是纯配置收益、零风险，值得单独测一轮。
   - **统一大 N tile**：cutlass 在 M=1 时按 128 行 tile 干活，N 小的时候
     tile 数不足；如果存在 per-shape tile 选择空间，也是配置层面的收益。
3. **测量口径沉淀**（这轮最大的可复用产出）：
   - 任何 M=1 GEMV microbench 只要权重能放进 64 MB L2，就必须用多副本工作集
     （≥256 MiB）或按真实实例数聚合，否则会报出 >960 GB/s 的假数据；
   - 跨实现比较必须用 graph replay 计时，否则 FlashInfer 类库的 host 开销会污染基线；
   - 同一 A/W 对比时，A 必须是**同一份** packed bytes（reference quantizer 与
     FlashInfer 有 per-code 差异）。
