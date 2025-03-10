#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "base/buffer.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#endif

#ifdef USE_ROCM
#include <hip/hip_runtime_api.h>
#endif

TEST(test_buffer, use_external1) {
  using namespace base;

#ifdef USE_CUDA
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  float* ptr = nullptr;
  cudaMalloc(&ptr, 32 * sizeof(float));  // CUDA 分配
#elif defined(USE_ROCM)
  auto alloc = base::ROCmDeviceAllocatorFactory::get_instance();
  float* ptr = nullptr;
  hipMalloc(&ptr, 32 * sizeof(float));  // ROCm 分配
#else
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];  // CPU 分配
#endif

  Buffer buffer(32, nullptr, ptr, true);
  CHECK_EQ(buffer.is_external(), true);

#ifdef USE_CUDA
  cudaFree(buffer.ptr());  // 释放 CUDA 资源
#elif defined(USE_ROCM)
  hipFree(buffer.ptr());  // 释放 ROCm 资源
#else
  delete[] buffer.ptr();  // 释放 CPU 资源
#endif
}
