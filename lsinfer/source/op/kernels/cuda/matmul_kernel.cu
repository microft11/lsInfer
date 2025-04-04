#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "matmul_kernel.cuh"
namespace kernel {
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK, int UNROLL_FACTOR = 4>
__global__ void matmul_kernel_cu_fp32(const float* __restrict__ input, 
                                              const float* __restrict__ weight, 
                                              float* __restrict__ output, 
                                              int M, int K) {
  // 共享内存缓存输入数据
  __shared__ float input_smem[THREAD_PER_BLOCK];
  __shared__ float sdata[THREAD_PER_BLOCK];
  
  const unsigned int tid = threadIdx.x;
  const int start_row = blockIdx.x * ROW_PER_BLOCK;
  const int end_row = min(start_row + ROW_PER_BLOCK, K);
  
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

  // 预加载输入数据到共享内存
  for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
    input_smem[i] = input[i];
  }
  __syncthreads();

  for (int p = start_row; p < end_row; ++p) {
    float sum = 0.0f;
    const int row_offset = p * M;
    
    // 向量化加载和计算
    const float4* weight_vec = reinterpret_cast<const float4*>(weight + row_offset);

    #pragma unroll UNROLL_FACTOR
    for (int i = tid; i < pack_num; i += THREAD_PER_BLOCK) {
      float4 wt = weight_vec[i];
      float in0 = input_smem[i * pack_size];
      float in1 = input_smem[i * pack_size + 1];
      float in2 = input_smem[i * pack_size + 2];
      float in3 = input_smem[i * pack_size + 3];
      
      sum += in0 * wt.x + in1 * wt.y + in2 * wt.z + in3 * wt.w;
    }

    for (int i = pack_off + tid; i < M; i += THREAD_PER_BLOCK) {
      sum += input_smem[i] * weight[row_offset + i];
    }

    sdata[tid] = sum;
    for (int stride = THREAD_PER_BLOCK / 2; stride > 0; stride >>= 1) {
      __syncthreads();
      if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
      }
    }

    if (tid == 0) {
      output[p] = sdata[0];
    }
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK, int UNROLL_FACTOR = 4>
__global__ void matmul_kernel_cu_fp32int8(const float* __restrict__ input, 
                                                   const int8_t* __restrict__ weight,
                                                   const float* __restrict__ scales, 
                                                   const int32_t group_size,
                                                   float* __restrict__ output, 
                                                   int M, int K) {
  // 共享内存缓存输入数据和缩放因子
  __shared__ float input_smem[THREAD_PER_BLOCK];
  __shared__ float scale_smem[THREAD_PER_BLOCK];
  __shared__ float sdata[THREAD_PER_BLOCK];
  
  const unsigned int tid = threadIdx.x;
  const int start_row = blockIdx.x * ROW_PER_BLOCK;
  const int end_row = min(start_row + ROW_PER_BLOCK, K);

  // 预加载输入数据到共享内存
  for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
    input_smem[i] = input[i];
  }
  __syncthreads();

  for (int p = start_row; p < end_row; ++p) {
    float sum = 0.0f;
    const int row_offset = p * M;
    
    // 预加载缩放因子
    const int group_idx = (p * M) / group_size;
    if (tid == 0) {
      scale_smem[0] = scales[group_idx];
    }
    __syncthreads();
    const float scale = scale_smem[0];

    #pragma unroll UNROLL_FACTOR
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = row_offset + i;
      sum += input_smem[i] * scale * static_cast<float>(weight[weight_idx]);
    }

    sdata[tid] = sum;
    
    // 优化的归约操作，减少同步点
    for (int stride = THREAD_PER_BLOCK / 2; stride > 0; stride >>= 1) {
      __syncthreads();
      if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
      }
    }

    if (tid == 0) {
      output[p] = sdata[0];
    }
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
  const tensor::Tensor& output, const float scale, 
  const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  CHECK_EQ(M, input.get_dim(0));

  constexpr int THREADS_PER_BLOCK = 256;
  constexpr int ROW_PER_BLOCK = 1;

  if (config && config->stream) {
    matmul_kernel_cu_fp32<THREADS_PER_BLOCK, ROW_PER_BLOCK><<<K, THREADS_PER_BLOCK, 0, config->stream>>>(
    input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32<THREADS_PER_BLOCK, ROW_PER_BLOCK><<<K, THREADS_PER_BLOCK>>>(
    input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
        const tensor::Tensor& output, int32_t group_size,
        const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  CHECK_EQ(M, input.get_dim(0));

  constexpr int THREADS_PER_BLOCK = 256;
  constexpr int ROW_PER_BLOCK = 1;
  if (config->stream) {
    matmul_kernel_cu_fp32int8<THREADS_PER_BLOCK, ROW_PER_BLOCK><<<K, 
                              THREADS_PER_BLOCK, 0, config->stream>>>(
    input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
    const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<THREADS_PER_BLOCK, ROW_PER_BLOCK><<<K, THREADS_PER_BLOCK>>>(
    input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
    const_cast<float*>(output.ptr<float>()), M, K);
  }
}
}  // namespace kernel