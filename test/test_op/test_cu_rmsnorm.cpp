#include <hip/hip_runtime.h> 
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"  // 保留 .cuh 后缀
#include "base/buffer.h"
#include <random>

TEST(test_rmsnorm_hip, rmsnorm_nostream) {
  auto alloc_hip = base::HIPDeviceAllocatorFactory::get_instance();  // 替换 CUDA -> HIP
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 15;

  // 初始化随机数据
  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  // 数据传输到设备
  tensor::Tensor in_hip = in_cpu.clone();
  tensor::Tensor wei_hip = wei_cpu.clone();
  tensor::Tensor out_hip = out_cpu.clone();
  in_hip.to_hip(nullptr);  // 替换 to_hip
  wei_hip.to_hip(nullptr);
  out_hip.to_hip(nullptr);

  // 调用 HIP 内核
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceHIP)(  // 替换 kDeviceCUDA
      in_hip, wei_hip, out_hip, nullptr);

  // 验证结果
  out_hip.to_cpu();
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(
      in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_hip.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
}


TEST(test_rmsnorm_cu, rmsnorm_stream) {
  auto alloc_cu = base::HIPDeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 ;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_hip(nullptr);
  wei_cu.to_hip(nullptr);
  out_cu.to_hip(nullptr);
  hipStream_t stream;
  hipStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceHIP)(in_cu, wei_cu, out_cu,
                                                            stream);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu,
                                                           nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  hipStreamDestroy(stream);
}

TEST(test_rmsnorm_cu, rmsnorm_stream2) {
  auto alloc_cu = base::HIPDeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 15;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_hip(nullptr);
  wei_cu.to_hip(nullptr);
  out_cu.to_hip(nullptr);
  hipStream_t stream;
  hipStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceHIP)(in_cu, wei_cu, out_cu,
                                                            stream);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu,
                                                           nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  hipStreamDestroy(stream);
}