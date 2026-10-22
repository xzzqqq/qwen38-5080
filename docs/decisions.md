# 决策记录

格式：日期、决定、理由、被否决的选项。

## 2026-10-19 llama.cpp 作为外部工具，不入库

- 决定：`llama.cpp/` 不纳入版本管理；用 `scripts/setup_llama_cpp.sh` 按固定
  commit `f174553` 拉取并编译，commit 和编译参数记录在 `results/`。
- 理由：它只用于 M1 参照，不修改。固定 commit + 脚本已经能保证可复现。
- 被否决：
  - git submodule：会强制递归 clone 一个历史很大的仓库，收益只有钉版本。
  - 把源码 copy 进仓库：体积大、更新麻烦。
- 以后如果必须打 patch，放 `scripts/patches/`，仍然不引入 submodule。

## 2026-10-19 CUDA 工具链用 micromamba env，不动系统

- 决定：编译使用 `/home/xzq/micromamba/envs/cuda13`（nvcc 13.0.88 +
  cudart + cublas 13.0.2.14）；系统的 `/usr/local/cuda` 仍是 12.4，不改。
- 理由：12.4 不支持 `sm_120`；装系统级 CUDA 需要 root，而且会动到驱动相关的
  环境。conda env 可以精确固定版本，也方便复现。
- 被否决：apt/runfile 装系统级 CUDA 13；用 12.4 编 `compute_90` PTX 兜底
  （不符合 `sm_120a` 目标，性能会偏低）。

## 2026-10-20 M1 后修订 baseline 与 spec 的验收口径

- 决定：
  - M1（IQ4_XS）只作外部参照，不参与 baseline 验收；Phase 1 为 M1–M3。
  - 显存预算改用实测可用 16.61 GB（15841 MiB）；DFlash2 是否装得下待 Phase 3 验证。
  - Phase 2 以约 80% 逻辑带宽（FP8 KV@8K 约 51 tok/s）为进入 spec 的门槛，
    55 tok/s 留作后续目标。
  - Spec 先测容量、τ 和相对同一 target AR 的净加速，按有效输出速度选择
    MTP / DFlash2；量化后 τ 低于 BF16 的 75% 只作诊断。
  - 8K 下先不做层 offload；M2 实测峰值后再定。
- 理由：M1 的 55.9 tok/s 来自 13.35 GB 的 IQ4_XS，不能外推到 14.41 GB 的
  NVFP4；AR 从 80% 提到 85% 带宽只多约 6.25% 吞吐，先验证 spec 的容量和净加速
  收益更高。
- 被否决：
  - 把 M1 的 tok/s 当 baseline 或实现效率参照。
  - 按 16303 MiB 总量而不是 15841 MiB 可用量算显存余量。
  - Phase 2 硬性追 55 tok/s 后再进 spec。

## 2026-10-20 M2 checkpoint 固定与分片策略

- 决定：
  - 固定 `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` 到 revision
    `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`；下载后校验 `crc32.txt`。
  - 下载两个 shard。vision 和文本混在 shard 1，不能靠“跳过 vision”跳掉整个文件；
    加载时按 `model.visual.` 前缀过滤。
  - embedding 保持 host BF16；文本权重和 `lm_head` 常驻 GPU。
- 理由：文件树和 index 实测 shard 1 有 979 个文本 + 333 个 vision tensor；只下
  shard 2 会缺一半文本权重。固定 revision + CRC 才能保证 M2 和后续阶段读的是同一份权重。
- 被否决：
  - 从 `main` 拉取：不可复现。
  - 用 HTTP Range 拼一个 text-only safetensors：M2 不需要，增加实现和校验复杂度；
    完整下载 17.92 GB 在 371 GB 可用磁盘内可接受。

## 2026-10-20 M2 NVFP4Linear 固定 cutlass 与对拍容差

- 决定：
  - `NVFP4Linear` 的 backend 固定 `cutlass`，不用未记录的 `auto`；后续比较
    `b12x`/`cudnn` 时单独跑。
  - 权重传 `weight_packed.T`，scale 传 128x4 swizzled 后的转置；`alpha` 用
    `input_scale × weight_scale_2`。
  - 模块只保留 swizzled scale，构造后释放原始 linear scale；否则全模型会多占约 1.6 GB。
  - 对拍容差固定为 `nrmse ≤ 5e-3`、`cosine ≥ 0.99999`、`max_rel ≤ 1e-2`、输出 finite。
  - FlashInfer JIT 的 `CPATH` 必须同时包含 `cuda13/targets/x86_64-linux/include`
    和 pip 的 `nvidia/cu13/include`，用 `scripts/m2_env.*` 固化。
- 理由：真实权重 M=1/8/128/3/17/129 的 nrmse ≈ 0.00166、cosine ≈ 0.9999986；
  容差留了约 3× 余量，又能抓住 layout/scale 错误。
- 被否决：
  - 先用 `backend="auto"`：不同 shape 可能选不同 backend，M2 需要单一可复现路径。
  - 用「输出看起来差不多」代替固定容差：无法区分 BF16 舍入和 layout 错误。

## 2026-10-21 M2 chunked prefill 数值口径与 shape-stable a/b

- 决定：
  - 96 个 BF16 `in_proj_a/b` 包成 `ShapeStableLinear(block_m=128)`，固定 M=128
    分块、尾块补零；避免 M=300 vs 128 的 GEMM reduction 差异进入 GDN gate。
  - chunked prefill 的数值 reference 只用**边界对齐**的 full 前缀，例如
    `full 256 vs 128+128`；`make m2-chunk` 固化这个检查。
  - 不再把 `full 300 vs 128+128+44` 当作 bitwise 失败判据。
- 理由：a/b 修复后前缀和 full 256 都 exact；full 300 的第 2 段 attention 看到
  KV 长度 300（含 44 个被 mask 的未来 key），而 chunked 第 2 段只看 256 个 key。
  有效 key 相同但 attention reduction/softmax 长度不同，FP 不会 bitwise 相等；
  这是 reference 口径问题，不是 cache/state 错误。
- 被否决：
  - 保持原生 `nn.Linear`：M=300 vs 128 的 a/b 差异可达 0.0625，会经 g/beta、
    GDN、norm 和 W4A4 MLP 放大。
  - 用任意 full 长度做 chunked 的 bitwise reference：会把 attention KV 长度差异
    误判成实现 bug。
  - 用“短 prompt 单段、长上下文只 chunked”绕过：长上下文路径仍未验证。

## 2026-10-21 M2 关闭与验收口径

- 决定：
  - M2 以 `docs/journal/2026-10-21-m2-close.md` 的记录关闭；ShapeStableLinear
    保留，当前实现冻结给 M3 做 correctness reference。
  - runner 数值容差固定为：NVFP4Linear `nrmse ≤ 5e-3`、`cosine ≥ 0.99999`、
    `max_rel ≤ 1e-2`；GDN `nrmse ≤ 1e-3`、`cosine ≥ 0.99999`、`max_rel ≤ 5e-3`。
  - activation quant 的 PyTorch reference 与 FlashInfer 在 FP8 scale/E2M1 midpoint
    上有少量差异，只作诊断，不参与退出判定；失败样本（scale_max_abs、code_mismatch）
    存入 numeric.json。参考容差 `nrmse ≤ 3e-2`、`cosine ≥ 0.9995`、
    `max_rel ≤ 2e-1`。
  - 退出判定必须同时检查 reference count、hidden/cache finite、dtype、cache length；
    count=0 或 cache 非有限/长度错误不能通过。
  - 8K 容量以 prefill peak 15.620 GB、decode peak 15.423 GB、reserved 15.859 GB、
    free 0.413 GB 作为 baseline；M3 的 static cache/graph 必须和 eager 输出对拍后再报性能。
- 理由：3 prompt ×16 步、真实生成 2048 上限、512/7680 容量都通过；NVFP4
  的同一 A/W dequant reference 每 prompt 0/6800 failure，GDN recurrence 每 prompt
  0/816 failure，lm_head 每步 16/16。
- 被否决：
  - 把 activation quant rounding 差异直接算成 NVFP4Linear 失败。
  - 用任意 full 长度做 chunked prefill 的 bitwise reference。
  - 在 M3 性能 baseline 之前继续改 padding/GDN/conv kernel。

## 2026-10-21 M3 固定 allocator、四模式验收口径与 graph 陷阱

- 决定：
  - M3 所有进程统一 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
    （`runner/__init__.py` 里设置，可用同名环境变量覆盖）。默认 allocator 在
    depth 7680 + chunk 128 的 prefill 会在 attention softmax 处因 segment 碎片
    OOM；开启后 load 14.458/14.556 GB、prefill peak 15.620 GB、free 0.413 GB，
    与 M2 关闭记录逐位一致，说明 M2 的数字也是在同等 allocator 策略下测的。
    这是分配器策略，不改变任何数值，但会影响 peak/reserved 的观测值。
  - 对拍 gate 只看**同 shape** 的三组：eager→static-exact、padded DynamicCache
    reference→static-cap、static-cap→graph，全部要求 bitwise。
    `unpadded→padded`（eager→static-cap / eager→padded-ref）只报 attention
    长度效应和首处分叉，不作门槛。
  - attention 长度效应单独用 `bench/micro/attn_length_effect.py` 隔离：同一个
    attention op，K/V 长度 513 vs 补到 1024（mask 后有效 key 相同），BF16 输出
    最大差 9.77e-4（中位 0.22 ulp，只有 0.5% 元素变化），两者与 float64 参考的
    距离相同（3.87e-3）。所以它是 FP reduction 顺序，不是实现错误。
  - free-running greedy（三条 smoke）**不要求**与冻结 eager trace bitwise 一致：
    sub-ulp 的注意力差异经过 64 层 BF16 残差和 401 个 W4A4 GEMM 会被放大，
    在 logits 上到 O(1)。smoke 的 gate 是「EOS + 长度合理 + 内容合理」，
    与 trace 的差异只作 diagnostic，并记录首处分叉的 top-2 margin。
    逐 step bitwise 由 graph-1024/8192 的 16 步检查负责。
  - `GraphDecodeStep` 必须显式把 static cache 作为 `past_key_values` 传进
    `model(...)`。漏传时 transformers 会自己建一个 `DynamicCache`，graph replay
    写进 graph-private pool，外部 cache 长度不变、KV 槽不变，但 logits 看起来
    “在变”，很难发现。已加运行时 guard：warmup 必须让 static cache 前进
    `warmup_iters`，否则直接报错；check 里另有「replay 写入 static KV 槽」gate。
  - M3 同时报模型速度和系统速度：模型速度用冻结 trace + 逐步 CUDA events
    （不含 host/sampling），系统速度是真实贪心闭环墙钟。
- 理由：8K 的余量只有 0.413 GB，allocator 策略直接决定能不能跑；bitwise 只在
  同 shape 有意义，跨 shape 的差异不能当回归依据（validation §6）。graph 的
  cache 传参错误会静默走另一条路径，必须用结构性 gate 拦住。
- 被否决：
  - 只用默认 allocator：8K prefill 直接 OOM，且与 M2 记录不可比。
  - 用 `torch.compile(mode="reduce-overhead")` 代替手写 CUDA Graph：会把
    Inductor 融合、kernel 选择和 graph 收益混在一起（M3 要单独观察 graph）。
  - 把 free-running greedy trace 当 gate：数值上不可达，只会掩盖真正的问题。
  - 用 nsys 2023.4（CUDA 12.4 自带）采集：它没有 Blackwell 的 GPU target info，
    报告里没有 CUDA kernel data；改用 `/opt/nvidia/nsight-systems-cli/2026.2.1`。

## 2026-10-21 M3 review 修复：入口 guard、必测覆盖、capture 峰值、SOL 分母、provenance

- 决定：
  - **capacity guard 进公开入口**：`prefill` 同时检查 `max_context` 和所选 capacity
    （含 `reserve_steps`）；`system_decode` / `model_timed_decode` 在计时区间开始前
    一次性 `require_capacity(steps)`；`ensure_graph` 为 warmup 预留 headroom。
    计时循环内不增加逐 step 同步。
  - **必测 reference 不允许降级**：padded reference、fill=3 隔离检查失败时写
    gate failure（不是 diagnostic）；每个场景带 `required` 前缀清单，`result()` 里
    有 coverage gate，缺项即整场景失败。`check_snapshot` 另外核对 forward 数。
  - **warmup/capture 峰值独立记录**：`ensure_graph` 里 prefill→warmup→capture→prefill
    各自 reset/读 peak，`CaptureInfo` 保存 warmup 和 capture 峰值；外层 stage 的
    peak 只代表最后一个子阶段，不能当 capture 峰值。memory 的 `ok` 增加 reset 后
    logits/state 与干净初态 bitwise 对比。
  - **SOL 分母统一用有效 `L_t`**：plan §6 的口径对所有执行方式一致；static-cap /
    graph 扫满 capacity 的额外字节单列，不进 `B_ideal`。`docs/sol.md` §7 的
    “紧凑 KV 读”改成全模型口径（16 层 K+V），与展开流量可比。
  - **provenance 可恢复**：dirty tree 时把 `git diff`+staged 写成 `git_diff.patch`，
    并保存源码/文档/输入的 `source_manifest.json`（逐文件 sha256），results/models/
    .tmp 不进 manifest。clean commit 时由 commit id 唯一确定源码。
- 理由：review 指出的五处都是"以后可能错误通过"或"记录不可恢复"，不是已提交数值
  失败；修完后在干净 commit 上复跑 check/memory/sol/profile/bench，让记录的
  provenance 直接指向 commit。
- 被否决：
  - 只在 helper 层保留 `require_capacity`：公开入口不检查等于没有保护。
  - 把 padded reference 的 OOM 当 diagnostic：必测 reference 没跑成不能算通过。
  - 用 capacity 当 `B_ideal` 的 KV 字节：同一 workload 的 SOL 会随实现变化。
  - 只记录 `commit + dirty`：无法恢复实际运行的源码。

## 2026-10-21 bench 迭代成本：三档验证与重复次数的口径

- 决定（三条，适用于 M3 之后的所有 bench 复跑）：
  1. **里程碑关闭时的全量 bench，非 graph 场景重复次数从 5 降到 3**（graph 场景
     保持 5）。报告仍给中位数、min/max，并标注 `n=3`；如果某个非 graph 场景的
     run 间极差超过中位数的 3%，该场景回到 5 次并在 journal 里说明。
     依据：M3 实测非 graph 的 run 间波动约 1%（eager-512 11.47–11.59、
     exact-512-1024 11.60–11.72），5 次对这类 launch-bound 模式没有额外信息量；
     单场景每少一次测量省约 90 秒，8 个非 graph 场景合计省约 30 分钟。
  2. **迭代期只重跑受影响的场景**：用 `--only <scenario>`，未改代码路径的场景
     沿用旧结果。但同一张表里必须逐场景标注 commit/provenance，不能把不同
     commit 的数字混报成一次结果；里程碑关闭仍必须全量。
     （2026-10-22 更新：`--resume` 现在自己核对源码 + 运行配置 + 环境指纹，
     见下一条决定。）
  3. **开发期一律用 `make m3-quick`**（单场景 `--only` + `--repeats 1 --steps 32`），
     产物写 `.tmp/m3-quick/`，不进 `results/`；只有全量结果进
     `results/m3/<run-id>-*/` 并带 provenance。
- 范围：M3 已提交的全量矩阵（11 场景 × 5 次）**不改**，仍按 plan §2.4 的 5 次口径；
  以上三条从下一次 bench 起生效。
- 理由：bench 的 89 分钟测量时间来自 511 步 × 2 趟 × 6 遍 × 11 场景。步数是
  §2.2 的计时协议不能动；能压缩的只有"每场景重复几次"和"哪些场景必须重跑"。
  M3 的实测波动数据支持非 graph 降到 3 次，同时保留 graph 的 5 次，因为它更接近
  后续要追踪的目标指标。
- 被否决：
  - 全局把 `--repeats` 默认改成 3：会让 graph 的波动统计变弱，且与 plan §2.4
    的默认值混淆。
  - 迭代期混用不同 commit 的结果却不标注：无法复现，也违反 validation §6 的
    可比性要求。
  - 缩短 `--steps` 到 256 来省时间：改变了 §2.2 的计时区间，数字不可比。

## 2026-10-22 开发/关闭循环的实测口径：保留 5 次、暂不做自适应重复

- 决定：
  1. **M3 收尾的全量 bench 保持 5 次重复**（plan §2.4）。2026-10-21 那条
     “非 graph 降到 3 次”的决定只适用于**后续里程碑**，并且要写进那个里程碑的
     plan 后再生效；M3 不追溯修改。
  2. **自适应重复（1→3→5，极差 <1% 就停）暂不实现**，只作为 Phase 2 的候选：
     它会改结果 schema、停止规则和 provenance 字段，不适合在 M3 收尾时动。
     记录它要解决的问题：bench 89 分钟里约 40 分钟花在“第 4、5 次重复几乎没有
     新信息”上（M3 实测 run 间极差 ~1%）。
  3. **warmup 保持“完整一次”**，M3 不缩到 64 步。理由见下。后续里程碑如果要用
     短 warmup，必须在新 plan 里写明并给出证据。
  4. 开发档实测：`make m3-quick` = **41 s**（check 13.3 s / memory 12.9 s /
     bench 14.7 s），不是最初估的 5 分钟。剩余优化空间主要是 3 个子进程各约
     5–6 s 的固定开销；默认用法是只跑**改动对应的那一档**
     （`make verify` 会给出命令），而不是每次都跑完整 quick。
- warmup 的三种含义（容易混淆，写清楚）：
  1. **capture warmup**（`GraphDecodeStep.warmup`，3 次 side-stream forward）：
     PyTorch 对 CUDA Graph capture 的硬性要求，用来固定 FlashInfer JIT/workspace/
     tactic 和 cuBLAS handle；约 0.3 s，必须保留。
  2. **场景 warmup**（bench 里每次 warmup + 5 次测量中的那一遍完整 pass）：
     让 GPU 时钟/功耗进入稳态、allocator 长到稳态 block size、避免第一次测量是
     冷样本；这是 plan §2.4 的“完整 warmup 一次”。
  3. 没有第三种。数值对拍不做 warmup（每次都是干净初态）。
- 缩场景 warmup 到 64 步的代价（所以 M3 不改）：
  - **时钟/功耗爬坡**：eager 每步 86 ms 但 GPU 利用率只有 ~40%，64 步只有 5.5 s，
    不足以让 SM clock 稳定；graph 32 ms × 64 = 2 s 更短。第 1 次测量会吸收爬坡，
    中位数不受影响，但 min/max 里会出现一个冷样本。
  - **allocator 预增长**：eager/static-exact 的 KV 随 L 增长，最大的 block 出现在
    L≈8191；64 步 warmup 只到 7744（8K 档），第一次测量仍会扩 allocator。
  - **与 plan §2.4 的字面口径不一致**，要么改 plan 要么改记录，容易让后续对比
    失去可比性。
  - 收益：bench 89 分钟里省约 16 分钟（1/6）。
- 理由：M3 的数值已经冻结、bench 是收尾项，优先保证口径和 plan 一致；把“缩短
  warmup / 自适应重复”留给 Phase 2，在改之前先把代价写清楚。
- 被否决：
  - 现在就上 64 步 warmup：省 16 分钟换一个冷样本和一处 plan 偏差，不划算。
  - 现在就做自适应重复：改 schema 的收益远低于在 Phase 2 和 microbench 流程一起做。

## 2026-10-22 `--resume` 必须核对运行配置与环境指纹

- 决定：bench 的 `--resume` 复用条件从"源码一致"扩展为**三者都一致**：
  1. **源码**：commit、dirty 状态、`source_manifest.patch_sha256`；
  2. **运行配置**：`steps`、`repeats`（`--repeats-non-graph` 生效后的值）、
     `chunk_size`、`graph_warmup`、fixtures manifest 的内容 hash；
  3. **环境指纹**：python / torch / transformers / FlashInfer / CUDA / GPU / driver。
  每个 `result.json` 记录 `run_config` 与 `environment_fingerprint`；目录级在
  `provenance.json` 记录整个矩阵的 `run_config`。任一不一致就重跑该场景并打印
  差异；矩阵配置变了则写新目录，不覆盖旧记录。旧结果没有这两个字段时**不复用**
  （无法确认就不复用）。
- 理由：之前只比源码，换 `--steps` / `--repeats` / `--chunk-size` / fixtures 后
  `--resume` 会静默跳过旧场景，同一个 `bench.json` 里混进不同协议，中位数和
  %SOL 都无法解释；driver/torch/FlashInfer 版本变化同理。源码一致只保证"代码
  一样"，不保证"实验一样"。
- 被否决：
  - 只比 `steps`/`repeats`：漏掉 `chunk_size`、`graph_warmup` 和 fixtures 内容，
    而这些都会改变数值或计时；
  - 只比配置不比环境：换驱动/库版本后的数字不可与之前合并；
  - 对旧结果"尽力复用"：无法确认配置时复用等于引入不可见的混合口径。
- 相关：2026-10-21 "bench 迭代成本" 那条的第 2 点（迭代期只重跑受影响场景）
  现在由这三层指纹自动判定，不再依赖人工记录。
