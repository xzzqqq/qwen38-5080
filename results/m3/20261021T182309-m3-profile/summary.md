# M3 Nsight Systems 时间线摘要
run_dir: /home/xzq/workspace/qwen/results/m3/20261021T182309-m3-profile

## eager-512（eager depth=512 capacity=None steps=8）

- kernel 实例：57936，每步 7242 个
- 每步 kernel 累计时间：30229.6 µs，每步墙钟 110051.6 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 93.9%，累计时间 27.8%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 30423 | 52.5% | 24015.6 | 9.9% |
| 1-2us | 17644 | 30.5% | 24121.8 | 10.0% |
| 2-5us | 6333 | 10.9% | 19122.3 | 7.9% |
| 5-10us | 222 | 0.4% | 1655.8 | 0.7% |
| 10-50us | 1245 | 2.1% | 34573.4 | 14.3% |
| 50-100us | 2061 | 3.6% | 131932.8 | 54.6% |
| >=100us | 8 | 0.0% | 6415.4 | 2.7% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 171705.9 | 71.0% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 9181.5 | 3.8% |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native:` | 4624 | 7997.4 | 3.3% |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bin` | 5392 | 4373.5 | 1.8% |
| `void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::Bl` | 3208 | 4358.6 | 1.8% |

## eager-7680（eager depth=7680 capacity=None steps=8）

- kernel 实例：57936，每步 7242 个
- 每步 kernel 累计时间：44072.5 µs，每步墙钟 110468.7 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 92.6%，累计时间 18.6%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 29702 | 51.3% | 23601.1 | 6.7% |
| 1-2us | 18135 | 31.3% | 24631.2 | 7.0% |
| 2-5us | 5817 | 10.0% | 17428.6 | 4.9% |
| 5-10us | 306 | 0.5% | 1948.9 | 0.6% |
| 10-50us | 1477 | 2.5% | 43629.2 | 12.4% |
| 50-100us | 1979 | 3.4% | 126295.4 | 35.8% |
| >=100us | 520 | 0.9% | 115045.7 | 32.6% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 164777.3 | 46.7% |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native:` | 1536 | 76714.7 | 21.8% |
| `std::enable_if<T7, void>::type internal::gemvx::kernel<int, int, __nv_` | 112 | 14675.5 | 4.2% |
| `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native:` | 256 | 11567.6 | 3.3% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 9283.6 | 2.6% |

## graph-512（graph depth=512 capacity=1024 steps=8）

- kernel 实例：58320，每步 7290 个（graph 展开后 node 数）
- 每步 kernel 累计时间：29717.8 µs，每步墙钟 32632.1 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 93.6%，累计时间 25.9%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 34857 | 59.8% | 25370.0 | 10.7% |
| 1-2us | 14418 | 24.7% | 20222.3 | 8.5% |
| 2-5us | 5291 | 9.1% | 16042.6 | 6.7% |
| 5-10us | 162 | 0.3% | 965.1 | 0.4% |
| 10-50us | 1839 | 3.2% | 54627.4 | 23.0% |
| 50-100us | 1745 | 3.0% | 114088.4 | 48.0% |
| >=100us | 8 | 0.0% | 6426.6 | 2.7% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 165328.7 | 69.5% |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native:` | 1536 | 9617.4 | 4.0% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 8735.1 | 3.7% |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native:` | 4624 | 7522.1 | 3.2% |
| `void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::Bl` | 3208 | 4164.5 | 1.8% |

## graph-7680（graph depth=7680 capacity=8192 steps=8）

- kernel 实例：58320，每步 7290 个（graph 展开后 node 数）
- 每步 kernel 累计时间：42341.4 µs，每步墙钟 44955.8 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 93.6%，累计时间 18.2%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 34485 | 59.1% | 25130.6 | 7.4% |
| 1-2us | 14463 | 24.8% | 20097.3 | 5.9% |
| 2-5us | 5624 | 9.6% | 16510.4 | 4.9% |
| 5-10us | 28 | 0.0% | 166.4 | 0.0% |
| 10-50us | 1172 | 2.0% | 32326.7 | 9.5% |
| 50-100us | 2028 | 3.5% | 130272.7 | 38.5% |
| >=100us | 520 | 0.9% | 114227.7 | 33.7% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 169020.3 | 49.9% |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native:` | 1536 | 75128.6 | 22.2% |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_1` | 128 | 17729.0 | 5.2% |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_1` | 128 | 16488.9 | 4.9% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 8784.6 | 2.6% |

## static-cap-512（static-cap depth=512 capacity=1024 steps=8）

- kernel 实例：58320，每步 7290 个
- 每步 kernel 累计时间：31002.2 µs，每步墙钟 111726.6 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 93.6%，累计时间 27.0%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 30182 | 51.8% | 23844.5 | 9.6% |
| 1-2us | 18456 | 31.6% | 25428.7 | 10.3% |
| 2-5us | 5936 | 10.2% | 17797.6 | 7.2% |
| 5-10us | 154 | 0.3% | 940.1 | 0.4% |
| 10-50us | 1519 | 2.6% | 41057.7 | 16.6% |
| 50-100us | 2065 | 3.5% | 132522.5 | 53.4% |
| >=100us | 8 | 0.0% | 6426.8 | 2.6% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 169760.9 | 68.4% |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native:` | 1536 | 10193.4 | 4.1% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 9289.4 | 3.7% |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native:` | 4624 | 7890.4 | 3.2% |
| `void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::Bl` | 3208 | 4417.1 | 1.8% |

## static-exact-512（static-exact depth=512 capacity=1024 steps=8）

- kernel 实例：58320，每步 7290 个
- 每步 kernel 累计时间：30324.9 µs，每步墙钟 111076.4 µs（含 host 提交/同步/空隙）
- <5 µs kernel：数量 93.9%，累计时间 27.9%

| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |
|---|---|---|---|---|
| <1us | 30019 | 51.5% | 23704.3 | 9.8% |
| 1-2us | 18786 | 32.2% | 25899.2 | 10.7% |
| 2-5us | 5966 | 10.2% | 18151.0 | 7.5% |
| 5-10us | 300 | 0.5% | 2460.6 | 1.0% |
| 10-50us | 1077 | 1.8% | 28977.2 | 11.9% |
| 50-100us | 2164 | 3.7% | 136987.8 | 56.5% |
| >=100us | 8 | 0.0% | 6418.9 | 2.6% |

| top kernel | 次数 | 总时长 µs | 时间占比 |
|---|---|---|---|
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute:` | 3208 | 171924.4 | 70.9% |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_k` | 5648 | 9291.6 | 3.8% |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native:` | 4624 | 7892.1 | 3.3% |
| `void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::Bl` | 3208 | 4409.2 | 1.8% |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Bin` | 5392 | 4381.5 | 1.8% |

