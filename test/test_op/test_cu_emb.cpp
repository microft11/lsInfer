#include <hip/hip_runtime.h>  // 替换 CUDA 头文件
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"
#include "base/base.h"

TEST(test_emb_hip, emb1_nostream) {
  auto alloc_hip = base::HIPDeviceAllocatorFactory::get_instance();  // 替换：CUDA -> HIP
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input(base::DataType::kDataTypeFp32, 1, true, alloc_cpu);
  input.index<int32_t>(0) = 1;

  tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_hip);

  // 初始化权重
  for (int i = 0; i < size; ++i) {
    weight.index<float>(i) = static_cast<float>(i);
  }
  weight.to_hip();  // 替换：to_cuda -> to_hip

  // 调用 HIP 版本的 embedding kernel
  kernel::get_emb_kernel(base::DeviceType::kDeviceHIP)(  // 替换：kDeviceCUDA -> kDeviceHIP
      input, weight, output, token, nullptr);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.index<float>(i), 512 + i);  // 检查结果
  }
}

TEST(test_emb_hip, emb2_nostream) {
  auto alloc_hip = base::HIPDeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  input.index<int32_t>(0) = 2;

  tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_hip);

  for (int i = 0; i < size; ++i) {
    weight.index<float>(i) = static_cast<float>(i);
  }
  weight.to_hip();

  kernel::get_emb_kernel(base::DeviceType::kDeviceHIP)(
      input, weight, output, token, nullptr);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.index<float>(i), 1024 + i);
  }
}

TEST(test_emb_hip, emb1_stream) {
  auto alloc_hip = base::HIPDeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  input.index<int32_t>(0) = 1;

  tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_hip);

  for (int i = 0; i < size; ++i) {
    weight.index<float>(i) = static_cast<float>(i);
  }
  weight.to_hip();

  hipStream_t stream;  // 替换：cudaStream_t -> hipStream_t
  HIP_CHECK(hipStreamCreate(&stream));  // 替换：cudaStreamCreate

  kernel::get_emb_kernel(base::DeviceType::kDeviceHIP)(
      input, weight, output, token, stream);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.index<float>(i), 512 + i);
  }

  HIP_CHECK(hipStreamDestroy(stream));  // 替换：cudaStreamDestroy
}