#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/cpu/matmul_kernel.h"
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"

#if defined(USE_CUDA)
    #include <cuda_runtime_api.h>
    #include <cublas_v2.h>
#elif defined(USE_ROCM)
    #include <hip/hip_runtime.h>
    #include <hipblas.h>
#endif

using namespace kernel;

TEST(test_matmul_cu, matmul_linear_stream5) {
  auto alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 4, 4, true, alloc_cpu);

  for (int i = 0; i < 4; ++i) {
    input.index<float>(i) = float(i);
  }

  for (int i = 0; i < 16; ++i) {
    weight.index<float>(i) = float(i);
  }

  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_gpu(base::DataType::kDataTypeFp32, 4, true, alloc_gpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);

  Config* config = new Config;
  
#if defined(USE_CUDA)
  cudaStreamCreate(&config->stream);
#elif defined(USE_ROCM)
  hipStreamCreate(&config->stream);
#endif

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_gpu, 1.f, config);
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f, config);

  out_gpu.to_cpu();
  for (int i = 0; i < out_gpu.size(); ++i) {
    ASSERT_EQ(out_gpu.index<float>(i), out_cpu.index<float>(i));
  }
}

TEST(test_matmul_cu, matmul_linear_course) {
  auto alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }

  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f, nullptr);

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}

TEST(test_matmul_cu, matmul_linear_course_gpu) {
  auto alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }

  input.to_cuda();
  weight.to_cuda();

  tensor::Tensor out_gpu(base::DataType::kDataTypeFp32, 3, true, alloc_gpu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_gpu, 1.f, nullptr);

  tensor::Tensor out_cpu = out_gpu.clone();
  out_cpu.to_cpu();

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}
