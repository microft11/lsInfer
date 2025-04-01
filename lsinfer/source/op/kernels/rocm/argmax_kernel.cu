#include "../kernels_interface.h"
#include "argmax_kernel.cuh"
#include "tensor/tensor.h"
#include <hip/hip_runtime.h>

namespace kernel {
// 保留原有设备函数（HIP 完全兼容）
__forceinline__ __device__ void warp_reduce_argmax(float& val, size_t& ptr) {
  float tmp_val;
  size_t tmp_ptr;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = __shfl_down_sync(mask, val, k, warpSize);
    tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);
    if (ptr == SIZE_MAX || tmp_ptr == SIZE_MAX) continue;
    if (tmp_val > val) {
      val = tmp_val;
      ptr = tmp_ptr;
    } else if (tmp_val == val && tmp_ptr < ptr) {
      ptr = tmp_ptr;
    }
  }
}

__forceinline__ __device__ void block_reduce_argmax(float& val, size_t& ptr, float* shared_value,
                                                  size_t* shared_ptr) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  warp_reduce_argmax(val, ptr);

  __syncthreads();
  if (lane_id == 0) {
    shared_value[warp_id] = val;
    shared_ptr[warp_id] = ptr;
  }

  __syncthreads();
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared_value[lane_id];
    ptr = shared_ptr[lane_id];
  } else {
    val = 0;
    ptr = SIZE_MAX;
  }

  if (warp_id == 0) {
    warp_reduce_argmax(val, ptr);
  }
}

// 内核函数保持不变
__global__ void argmax_kernel_fp32(const float* input_ptr, size_t size, size_t* output_idx) {
  __shared__ size_t shared_max_ptr[32];
  __shared__ float shared_max_value[32];
  uint32_t tid = threadIdx.x;
  if (tid >= size) {
    return;
  }

  size_t max_index = threadIdx.x;
  float max_value = input_ptr[max_index];
  for (size_t i = tid; i < size; i += blockDim.x) {
    if (input_ptr[i] > max_value) {
      max_index = i;
      max_value = input_ptr[i];
    }
  }

  block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
  __syncthreads();
  if (threadIdx.x == 0) {
    *output_idx = max_index;
  }
}

// 替换 CUDA API 为 HIP API
size_t argmax_kernel_hip(const float* input_ptr, size_t size, void* stream) {
  // 替换为 HIP 的内存分配器（需根据项目实际实现调整）
  std::shared_ptr<base::DeviceAllocator> alloc_hip =
      base::HIPDeviceAllocatorFactory::get_instance();  // 假设存在 HIP 分配器
  size_t* index = static_cast<size_t*>(alloc_hip->allocate(sizeof(size_t)));
  size_t output_index = 0;

  if (!stream) {
    argmax_kernel_fp32<<<1, 512>>>(input_ptr, size, index);
    hipMemcpy(&output_index, index, sizeof(size_t), hipMemcpyDeviceToHost);
  } else {
    hipStream_t stream_ = static_cast<hipStream_t>(stream);
    argmax_kernel_fp32<<<1, 512, 0, stream_>>>(input_ptr, size, index);
    hipMemcpyAsync(&output_index, index, sizeof(size_t), hipMemcpyDeviceToHost, stream_);
  }
  return output_index;
}

}  // namespace kernel