# M3 Nsight Systems 时间线摘要
run_dir: /home/xzq/workspace/qwen/results/m3/20261022T015940-m3-profile

## eager-512（eager depth=512 capacity=None steps=8）

- kernel 实例：57936，每步 7242 个
- 每步 kernel 累计时间：30320.7 µs，每步墙钟 112216.7 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 93.9%，累计时间 27.9%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 30046 | 51.9% | 23782.4 | 9.8% |
| 1-2us | 17883 | 30.9% | 24398.2 | 10.1% |
| 2-5us | 6463 | 11.2% | 19477.4 | 8.0% |
| 5-10us | 230 | 0.4% | 1704.4 | 0.7% |
| 10-50us | 1242 | 2.1% | 34518.7 | 14.2% |
| 50-100us | 2064 | 3.6% | 132261.7 | 54.5% |
| >=100us | 8 | 0.0% | 6423.0 | 2.6% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 171980.1 | 70.9% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 9253.3 | 3.8% |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native:` | 4624 | 8035.0 | 3.3% |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bin` | 5392 | 4403.5 | 1.8% |
| `void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::Bl` | 3208 | 4391.9 | 1.8% |

## eager-7680（eager depth=7680 capacity=None steps=8）

- kernel 实例：57936，每步 7242 个
- 每步 kernel 累计时间：44083.7 µs，每步墙钟 114805.4 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 92.6%，累计时间 18.6%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 29725 | 51.3% | 23627.6 | 6.7% |
| 1-2us | 18096 | 31.2% | 24582.4 | 7.0% |
| 2-5us | 5834 | 10.1% | 17463.4 | 5.0% |
| 5-10us | 304 | 0.5% | 1934.7 | 0.5% |
| 10-50us | 1475 | 2.5% | 43535.5 | 12.3% |
| 50-100us | 1982 | 3.4% | 126457.6 | 35.9% |
| >=100us | 520 | 0.9% | 115068.2 | 32.6% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 164841.6 | 46.7% |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native:` | 1536 | 76711.3 | 21.8% |
| `std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, __nv_` | 112 | 14681.3 | 4.2% |
| `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native:` | 256 | 11561.0 | 3.3% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 9288.2 | 2.6% |

## graph-512（graph depth=512 capacity=1024 steps=8）

- kernel 实例：58320，每步 7290 个（graph 展开后 node 数）
- 每步 kernel 累计时间：29741.0 µs，每步墙钟 32511.3 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 93.6%，累计时间 25.9%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 34872 | 59.8% | 25429.7 | 10.7% |
| 1-2us | 14392 | 24.7% | 20222.2 | 8.5% |
| 2-5us | 5302 | 9.1% | 16087.1 | 6.8% |
| 5-10us | 162 | 0.3% | 962.8 | 0.4% |
| 10-50us | 1840 | 3.2% | 54721.6 | 23.0% |
| 50-100us | 1744 | 3.0% | 114082.5 | 47.9% |
| >=100us | 8 | 0.0% | 6422.2 | 2.7% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 165407.3 | 69.5% |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native:` | 1536 | 9625.0 | 4.0% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 8751.7 | 3.7% |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native:` | 4624 | 7539.1 | 3.2% |
| `void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::Bl` | 3208 | 4172.8 | 1.8% |

## graph-7680（graph depth=7680 capacity=8192 steps=8）

- kernel 实例：58320，每步 7290 个（graph 展开后 node 数）
- 每步 kernel 累计时间：42353.3 µs，每步墙钟 44909.8 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 93.6%，累计时间 18.2%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 34475 | 59.1% | 25131.7 | 7.4% |
| 1-2us | 14480 | 24.8% | 20129.3 | 5.9% |
| 2-5us | 5622 | 9.6% | 16529.5 | 4.9% |
| 5-10us | 23 | 0.0% | 140.2 | 0.0% |
| 10-50us | 1176 | 2.0% | 32532.9 | 9.6% |
| 50-100us | 2024 | 3.5% | 130131.0 | 38.4% |
| >=100us | 520 | 0.9% | 114232.1 | 33.7% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 169080.2 | 49.9% |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native:` | 1536 | 75129.6 | 22.2% |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_1` | 128 | 17740.9 | 5.2% |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_1` | 128 | 16486.1 | 4.9% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 8785.8 | 2.6% |

## static-cap-512（static-cap depth=512 capacity=1024 steps=8）

- kernel 实例：58320，每步 7290 个
- 每步 kernel 累计时间：31028.7 µs，每步墙钟 109975.5 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 93.6%，累计时间 27.1%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 30161 | 51.7% | 23880.3 | 9.6% |
| 1-2us | 18413 | 31.6% | 25393.0 | 10.2% |
| 2-5us | 5997 | 10.3% | 17937.9 | 7.2% |
| 5-10us | 157 | 0.3% | 956.9 | 0.4% |
| 10-50us | 1510 | 2.6% | 40607.3 | 16.4% |
| 50-100us | 2074 | 3.6% | 133032.4 | 53.6% |
| >=100us | 8 | 0.0% | 6421.4 | 2.6% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 169837.0 | 68.4% |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native:` | 1536 | 10170.0 | 4.1% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 9302.2 | 3.7% |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native:` | 4624 | 7916.2 | 3.2% |
| `void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::Bl` | 3208 | 4425.6 | 1.8% |

## static-exact-512（static-exact depth=512 capacity=1024 steps=8）

- kernel 实例：58320，每步 7290 个
- 每步 kernel 累计时间：30385.8 µs，每步墙钟 111009.1 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 93.9%，累计时间 28.0%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 29866 | 51.2% | 23634.9 | 9.7% |
| 1-2us | 18853 | 32.3% | 26001.2 | 10.7% |
| 2-5us | 6044 | 10.4% | 18334.9 | 7.5% |
| 5-10us | 307 | 0.5% | 2498.0 | 1.0% |
| 10-50us | 1075 | 1.8% | 28875.1 | 11.9% |
| 50-100us | 2167 | 3.7% | 137321.2 | 56.5% |
| >=100us | 8 | 0.0% | 6421.5 | 2.6% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 172144.3 | 70.8% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 9330.8 | 3.8% |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native:` | 4624 | 7921.9 | 3.3% |
| `void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::Bl` | 3208 | 4422.0 | 1.8% |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bin` | 5392 | 4397.3 | 1.8% |


## 复现

- 采集：`STEPS=8 make m3-profile`（nsys 2026.2.1，`--capture-range=cudaProfilerApi`，`--cuda-graph-trace=node`）。
- 原始 per-kernel CSV 以 `gpu_trace.csv.gz` 保存（36 MB → 1.2 MB）；`trace.sqlite` 可由 `nsys stats` 从 `.nsys-rep` 重新生成。
- 统计：`.venv/bin/python scripts/m3_profile_stats.py --dir <run_dir>`。

