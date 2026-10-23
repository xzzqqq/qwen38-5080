// NVFP4 M=1 GEMV（CUDA core / SIMT），直接消费 FlashInfer 128x4 swizzled scale。
//
// 数据流（一次 forward = 一层）
// -----------------------------
//   a_fp4 [K/2] U8     FlashInfer nvfp4_quantize 的输出，低 nibble = 偶数 k
//   a_sf  [128, K/16]  128x4 swizzled；M=1，本 kernel 只读 tile row 0
//   w     [N, K/2] U8  checkpoint 的 packed weight
//   w_sf  [N_pad, K/16] 128x4 swizzled —— 就是 NVFP4Linear 里保留的那一份；
//        本 kernel 直接按 swizzle 偏移读取：不生成 unsquizzled 副本、不做 repack
//   -> out [N] BF16
//
// 为什么必须能直接吃 swizzled 布局：8K + graph 常驻后只剩约 0.12 GB，全模型
// 再留一份 linear [N, K/16] scale 要 1.6 GB，不可行。
//
// 并行划分
// --------
//   grid = (N / 128, k_split)。一个 block 覆盖一个 128 行 tile —— 正好等于 128x4
//   scale swizzle 的行 tile，于是每个 c-tile（64 个 k）在 swizzled buffer 里的 512B
//   被整个 block 用满，scale 的 DRAM 访问没有放大（scale 占总字节的 1/9）。
//   M=1 时单层 block 数常常不足（k_proj 只有 8 个 tile），用 blockIdx.y 做 split-K
//   把 block 数拉到 >2×SM；部分和在同一个 kernel 内用 counter + threadfence 归约，
//   每层仍然只有 1 个 kernel launch。
//
// 线程映射（16 warp × 8 行 = 128 行）
// -----------------------------------
//   每个 lane 负责 k-group（512 个 k）内第 lane 个 16-value block = 8 个 packed byte；
//   这 16 个 k 只对应 1 个 scale byte，且 a 的 16 个值放进寄存器后被 8 行复用 ——
//   这是把“每字节权的 ALU/共享内存开销”压下去的关键。
//
// 访存
// ----
//   权重：warp 每行读 256B 连续（32 lane × 8B），合并良好；16 行一组的地址相隔
//         K/2 字节，各自都是整段，DRAM sector 全满。
//   scale：每个 k-group 把 8×512B 连续拷进 shared，按 528B 分块（16B 对齐 + 每
//          c-tile 错 4 个 bank）；读时同一 c-tile 的 4 个 lane 读同一个 32-bit 字
//          （broadcast，无 bank conflict）。
//   a：每 lane 8B 连续 + 1 个 scale byte（L2 常驻），解码后留在寄存器。
//
// 预期瓶颈：权重带宽（每步 14.4 GB，M=1 无数据复用）。E2M1 用纯位运算解成 fp32
// （约 7 条指令/值），目标 ALU 时间 < 访存时间；不用 LUT，避免 shared 随机访问。

#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kRowsPerTile = 128;   // = scale swizzle 的行 tile
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;
constexpr int kRowsPerWarp = kRowsPerTile / kWarps;  // 8
constexpr int kKPerLane = 16;                        // 一个 16-value block
constexpr int kBytesPerLane = kKPerLane / 2;          // 8
constexpr int kKPerGroup = 32 * kKPerLane;            // 512
constexpr int kCTilesPerGroup = kKPerGroup / 64;      // 8
constexpr int kScaleTileBytes = 512;                  // 128 行 × 4 列 scale / c-tile
constexpr int kScaleTileStride = kScaleTileBytes + 16;
constexpr int kScaleStageBytes = kCTilesPerGroup * kScaleTileStride;

// E2M1 -> fp32：bit3 = sign，bit[2:1] = exp(bias 1)，bit0 = mantissa。
// code 2..7 的 fp32 位型 = 0x3F000000 + mag<<22；code 0/1 是 (0, 0.5)，单独修正。
__device__ __forceinline__ float e2m1_to_float(unsigned code) {
  const unsigned mag = code & 7u;
  unsigned bits = (mag < 2u) ? (mag * 0x3F000000u)
                             : (0x3F000000u + (mag << 22));
  bits |= (code & 8u) << 28;
  return __uint_as_float(bits);
}

// UE4M3（scale 的符号位恒为 0）-> fp32，交给硬件 cvt。
__device__ __forceinline__ float ue4m3_to_float(unsigned byte) {
  return __half2float(__nv_cvt_fp8_to_halfraw(
      static_cast<__nv_fp8_storage_t>(byte & 0x7Fu), __NV_E4M3));
}

__global__ __launch_bounds__(kThreads) void gemv_kernel(
    const uint8_t* __restrict__ w,       // [N, K/2]
    const uint8_t* __restrict__ wsf,     // [N_pad, K/16] 128x4 swizzled
    const uint8_t* __restrict__ a,       // [K/2]
    const uint8_t* __restrict__ asf,     // [128, K/16] 128x4 swizzled
    const float* __restrict__ alpha,     // [1]
    __nv_bfloat16* __restrict__ out,     // [N]
    float* __restrict__ partials,        // [k_split, N]，k_split > 1 时才用
    int* __restrict__ counters,          // [N/128]，k_split > 1 时才用
    int n,
    int k,
    int k_split) {
  const int tile = blockIdx.x;
  const int split = blockIdx.y;
  const int row0 = tile * kRowsPerTile;
  const int half_k = k >> 1;
  const int k16 = k >> 4;
  const int n_ctiles = k >> 6;

  __shared__ __align__(16) uint8_t s_sf[kScaleStageBytes];
  __shared__ int s_last;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int lane_ctile = lane >> 2;   // group 内 c-tile
  const int lane_col = lane & 3;      // c-tile 内 scale 列 d

  // split-K：按 c-tile 均分 [c_begin, c_end)
  const int c_begin = static_cast<int>((static_cast<long long>(n_ctiles) * split) / k_split);
  const int c_end =
      static_cast<int>((static_cast<long long>(n_ctiles) * (split + 1)) / k_split);
  const size_t sf_tile_base = static_cast<size_t>(tile) * k16 * kRowsPerTile;

  float acc[kRowsPerWarp];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) acc[r] = 0.0f;

  const int tile_row0 = warp << 3;  // 本 warp 的第一行在 128 行 tile 内的下标
  const size_t row_base = static_cast<size_t>(row0) + tile_row0;

  for (int c0 = c_begin; c0 < c_end; c0 += kCTilesPerGroup) {
    const int nc = (c_end - c0 < kCTilesPerGroup) ? (c_end - c0) : kCTilesPerGroup;
    __syncthreads();
    {  // 本 group 的 scale：nc×512B 连续 -> shared（528B 分块）
      const uint8_t* src = wsf + sf_tile_base + static_cast<size_t>(c0) * kScaleTileBytes;
      const int chunks = nc * (kScaleTileBytes / 16);
      for (int i = threadIdx.x; i < chunks; i += kThreads) {
        const uint4 v = *reinterpret_cast<const uint4*>(src + static_cast<size_t>(i) * 16);
        *reinterpret_cast<uint4*>(s_sf + (i >> 5) * kScaleTileStride + (i & 31) * 16) = v;
      }
    }
    __syncthreads();

    const int nblocks = nc * 4;
    const bool active = lane < nblocks;
    const int kbyte = (c0 << 5) + lane * kBytesPerLane;  // group 内第 lane 个 block 的字节偏移

    // a：16 个 k 值（一个 block）解成 fp32 留在寄存器，8 行共用
    float aval[kKPerLane];
    if (active) {
      const uint2 av = *reinterpret_cast<const uint2*>(a + kbyte);
      const float as = ue4m3_to_float(
          asf[static_cast<size_t>(c0 + lane_ctile) * kScaleTileBytes + lane_col]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const unsigned b0 = (av.x >> (8 * j)) & 0xFFu;
        const unsigned b1 = (av.y >> (8 * j)) & 0xFFu;
        aval[2 * j] = e2m1_to_float(b0 & 0xFu) * as;
        aval[2 * j + 1] = e2m1_to_float(b0 >> 4) * as;
        aval[8 + 2 * j] = e2m1_to_float(b1 & 0xFu) * as;
        aval[8 + 2 * j + 1] = e2m1_to_float(b1 >> 4) * as;
      }
    } else {
#pragma unroll
      for (int j = 0; j < kKPerLane; ++j) aval[j] = 0.0f;
    }

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int tile_row = tile_row0 + r;         // 行在 128 行 tile 内的下标
      const int b_row = tile_row & 31;            // swizzle: b
      const int a_q = (tile_row >> 5) & 3;        // swizzle: a
      if (active) {
        const float s = ue4m3_to_float(
            s_sf[lane_ctile * kScaleTileStride + b_row * 16 + a_q * 4 + lane_col]);
        const size_t row = row_base + r;
        const uint2 wv = *reinterpret_cast<const uint2*>(w + row * half_k + kbyte);
        float p = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const unsigned b0 = (wv.x >> (8 * j)) & 0xFFu;
          const unsigned b1 = (wv.y >> (8 * j)) & 0xFFu;
          p = fmaf(aval[2 * j], e2m1_to_float(b0 & 0xFu), p);
          p = fmaf(aval[2 * j + 1], e2m1_to_float(b0 >> 4), p);
          p = fmaf(aval[8 + 2 * j], e2m1_to_float(b1 & 0xFu), p);
          p = fmaf(aval[8 + 2 * j + 1], e2m1_to_float(b1 >> 4), p);
        }
        acc[r] = fmaf(s, p, acc[r]);
      }
    }
  }

  // 每个 lane 只覆盖 group 内第 lane 个 block：跨 32 个 lane 归约到整行
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    float v = acc[r];
    v += __shfl_down_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 1);
    acc[r] = v;
  }

  const float al = alpha[0];
  if (k_split == 1) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r)
        out[row_base + r] = __float2bfloat16(acc[r] * al);
    }
    return;
  }

  if (lane == 0) {
    float* dst = partials + static_cast<size_t>(split) * n + row_base;
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) dst[r] = acc[r];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    s_last = (atomicAdd(counters + tile, 1) == k_split - 1) ? 1 : 0;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int i = threadIdx.x; i < kRowsPerTile; i += kThreads) {
    const size_t row = static_cast<size_t>(row0) + i;
    float sum = 0.0f;
    for (int sp = 0; sp < k_split; ++sp) sum += partials[static_cast<size_t>(sp) * n + row];
    out[row] = __float2bfloat16(sum * al);
  }
  if (threadIdx.x == 0) counters[tile] = 0;  // 复位，下一次 launch 直接用
}

void nvfp4_gemv_launch(
    torch::Tensor a_fp4,
    torch::Tensor a_sf,
    torch::Tensor w,
    torch::Tensor wsf,
    torch::Tensor alpha,
    torch::Tensor out,
    torch::Tensor partials,
    torch::Tensor counters,
    int64_t k_split) {
  const int n = static_cast<int>(w.size(0));
  const int k = static_cast<int>(w.size(1)) * 2;
  const dim3 grid(static_cast<unsigned>(n / kRowsPerTile),
                  static_cast<unsigned>(k_split));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gemv_kernel<<<grid, kThreads, 0, stream>>>(
      w.data_ptr<uint8_t>(),
      wsf.data_ptr<uint8_t>(),
      a_fp4.data_ptr<uint8_t>(),
      a_sf.data_ptr<uint8_t>(),
      alpha.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      partials.data_ptr<float>(),
      counters.data_ptr<int>(),
      n,
      k,
      static_cast<int>(k_split));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_gemv", &nvfp4_gemv_launch,
        "NVFP4 M=1 GEMV (CUDA core, swizzled 128x4 scale)");
}
