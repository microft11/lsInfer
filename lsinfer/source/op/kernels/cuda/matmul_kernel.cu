#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <cassert>
#include "matmul_kernel.cuh"
namespace kernel {

template <int THREAD_PER_BLOCK=128, int ROW_PER_BLOCK=1>
__global__ void matmul_kernel_cu_fp32(const float* __restrict__ input, 
                                    const float* __restrict__ weight, 
                                    float* __restrict__ output, 
                                    int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  const unsigned int tid = threadIdx.x;
  const int start_row = blockIdx.x * ROW_PER_BLOCK;
  
  if (start_row >= K) return;

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

  for (int p = start_row; p < min(start_row + ROW_PER_BLOCK, K); ++p) {
    float sum = 0.0f;
    const size_t row_offset = static_cast<size_t>(p) * M;
    const float4* weight_float4_ptr = (float4*)(weight + row_offset);
    const float4* input_float4_ptr = (float4*)input;

    // Process packed data
    for (int i = tid; i < pack_num; i += THREAD_PER_BLOCK) {
      float4 input_float4 = input_float4_ptr[i];
      float4 weight_float4 = weight_float4_ptr[i];
      sum += input_float4.x * weight_float4.x + 
             input_float4.y * weight_float4.y + 
             input_float4.z * weight_float4.z + 
             input_float4.w * weight_float4.w;
    }

    // Process remaining elements
    for (int i = pack_off + tid; i < M; i += THREAD_PER_BLOCK) {
      sum += input[i] * weight[row_offset + i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction using CUB
    typedef cub::BlockReduce<float, THREAD_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(sdata[tid]);
    
    if (tid == 0) {
      output[p] = block_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK=128, int ROW_PER_BLOCK=1>
__global__ void matmul_kernel_cu_fp32int8(const float* __restrict__ input, 
                                        const int8_t* __restrict__ weight,
                                        const float* __restrict__ scales, 
                                        const int32_t group_size,
                                        float* __restrict__ output, 
                                        int M, int K) {
  extern __shared__ float shared_mem[];
  float* input_smem = shared_mem;
  float* sdata = shared_mem + M;  // Separate space for input and reduction
  
  const unsigned int tid = threadIdx.x;
  const int start_row = blockIdx.x * ROW_PER_BLOCK;
  
  if (start_row >= K) return;

  // Load input into shared memory (coalesced access)
  for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
    input_smem[i] = input[i];
  }
  __syncthreads();

  for (int p = start_row; p < min(start_row + ROW_PER_BLOCK, K); ++p) {
    float sum = 0.0f;
    const size_t row_offset = static_cast<size_t>(p) * M;

    // Process in 4-element chunks for better memory efficiency
    constexpr int chunk_size = 4;
    for (int i = tid * chunk_size; i < M; i += THREAD_PER_BLOCK * chunk_size) {
      for (int j = 0; j < chunk_size && (i + j) < M; ++j) {
        const size_t weight_idx = row_offset + i + j;
        const int group_idx = weight_idx / group_size;
        sum += input_smem[i + j] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
      }
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction using CUB
    typedef cub::BlockReduce<float, THREAD_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(sum);
    
    if (tid == 0) {
      output[p] = block_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, const float scale, 
                    const CudaConfig* config) {
  // Input validation
  CHECK(!input.is_empty() && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(!weight.is_empty() && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  CHECK_EQ(M, input.get_dim(0));

  // Ensure memory alignment
  assert(reinterpret_cast<uintptr_t>(input.ptr<float>()) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(weight.ptr<float>()) % 16 == 0);

  // Kernel configuration
  constexpr int threads = 128;  // Optimal for RTX 3060
  constexpr int rows_per_block = 2;
  const dim3 grid((K + rows_per_block - 1) / rows_per_block);
  const dim3 block(threads);

  // Launch kernel
  if (config && config->stream) {
    matmul_kernel_cu_fp32<threads, rows_per_block>
        <<<grid, block, 0, config->stream>>>(input.ptr<float>(), 
                                            weight.ptr<float>(), 
                                            const_cast<float*>(output.ptr<float>()), 
                                            M, K);
  } else {
    matmul_kernel_cu_fp32<threads, rows_per_block>
        <<<grid, block>>>(input.ptr<float>(), 
                        weight.ptr<float>(), 
                        const_cast<float*>(output.ptr<float>()), 
                        M, K);
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, int32_t group_size,
                          const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(!input.is_empty() && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(!weight.is_empty() && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  CHECK_EQ(M, input.get_dim(0));

  // Kernel configuration
  constexpr int threads = 128;
  constexpr int rows_per_block = 2;
  const dim3 grid((K + rows_per_block - 1) / rows_per_block);
  const dim3 block(threads);
  
  // Calculate shared memory size
  const size_t shared_mem_size = (threads + threads) * sizeof(float);

  // Launch kernel
  if (config->stream) {
    matmul_kernel_cu_fp32int8<threads, rows_per_block>
        <<<grid, block, shared_mem_size, config->stream>>>(input.ptr<float>(), 
                                                          weight.ptr<int8_t>(), 
                                                          scale.ptr<float>(), 
                                                          group_size,
                                                          const_cast<float*>(output.ptr<float>()), 
                                                          M, K);
  } else {
    matmul_kernel_cu_fp32int8<threads, rows_per_block>
        <<<grid, block, shared_mem_size>>>(input.ptr<float>(), 
                                          weight.ptr<int8_t>(), 
                                          scale.ptr<float>(), 
                                          group_size,
                                          const_cast<float*>(output.ptr<float>()), 
                                          M, K);
  }
}

}  // namespace kernel