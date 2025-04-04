#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/cpu/matmul_kernel.h"
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh" 
#include "base/buffer.h"

using namespace kernel;

TEST(test_matmul_hip, matmul_linear_stream5) {
  auto alloc_hip = base::HIPDeviceAllocatorFactory::get_instance();  // 替换 CUDA
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // 初始化输入和权重
  tensor::Tensor input(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 4, 4, true, alloc_cpu);

  for (int i = 0; i < 4; ++i) input.index<float>(i) = float(i);
  for (int i = 0; i < 16; ++i) weight.index<float>(i) = float(i);

  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  // 数据传输到设备
  input.to_hip(nullptr);  // 替换 to_cuda
  weight.to_hip(nullptr);

  tensor::Tensor out_hip(base::DataType::kDataTypeFp32, 4, true, alloc_hip);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);

  // 创建 HIP 流
  HipConfig* config = new HipConfig;  // 替换 CudaConfig
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));  // 替换 cudaStreamCreate
  config->stream = stream;

  // 调用 HIP 矩阵乘法内核
  kernel::get_matmul_kernel(base::DeviceType::kDeviceHIP)(  // 替换 kDeviceCUDA
      input, weight, out_hip, 1.f, config);

  // CPU 参考计算
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(
      input_cpu, weight_cpu, out_cpu, 1.f, nullptr);

  // 验证结果
  out_hip.to_cpu();
  for (int i = 0; i < out_hip.size(); ++i) {
    ASSERT_EQ(out_hip.index<float>(i), out_cpu.index<float>(i));
  }

  HIP_CHECK(hipStreamDestroy(stream));  // 替换 cudaStreamDestroy
  delete config;
}

TEST(test_matmul_hip, matmul_linear_course) {
  auto alloc_hip = base::HIPDeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // 初始化测试数据
  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = 1.f;
  input.index<float>(1) = 1.f;
  input.index<float>(2) = -1.f;

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }

  // CPU 计算参考结果
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(
      input, weight, out_cpu, 1.f, nullptr);

  // 验证标准结果
  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}

TEST(test_matmul_hip, matmul_linear_course_hip) {
  auto alloc_hip = base::HIPDeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // 初始化数据
  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = 1.f;
  input.index<float>(1) = 1.f;
  input.index<float>(2) = -1.f;

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }

  // 传输到设备
  input.to_hip();
  weight.to_hip();

  // HIP 计算
  tensor::Tensor out_hip(base::DataType::kDataTypeFp32, 3, true, alloc_hip);
  kernel::get_matmul_kernel(base::DeviceType::kDeviceHIP)(
      input, weight, out_hip, 1.f, nullptr);

  // 验证结果
  tensor::Tensor out_cpu = out_hip.clone();
  out_cpu.to_cpu();

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}