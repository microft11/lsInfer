#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "base/buffer.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#define DEVICE_ALLOCATOR CUDADeviceAllocatorFactory::get_instance()
#define COPY_TO_HOST(dst, src, size) cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)
#define SET_VALUE_CU(ptr, size, value) set_value_cu(ptr, size, value)
#elif defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include "../utils_hip.cuh"
#define DEVICE_ALLOCATOR ROCmDeviceAllocatorFactory::get_instance()
#define COPY_TO_HOST(dst, src, size) hipMemcpy(dst, src, size, hipMemcpyDeviceToHost)
#define SET_VALUE_CU(ptr, size, value) set_value_hip(ptr, size, value)
#else
#define DEVICE_ALLOCATOR CPUDeviceAllocatorFactory::get_instance()
#define COPY_TO_HOST(dst, src, size) std::memcpy(dst, src, size)
#define SET_VALUE_CU(ptr, size, value) \
    for (int i = 0; i < size; ++i) { *(ptr + i) = value; }
#endif

TEST(test_tensor, to_cpu) {
  using namespace base;
  auto alloc = DEVICE_ALLOCATOR;
  tensor::Tensor t1(DataType::kDataTypeFp32, 32, 32, true, alloc);
  ASSERT_EQ(t1.is_empty(), false);
  SET_VALUE_CU(t1.ptr<float>(), 32 * 32, 1.f);

  t1.to_cpu();
  ASSERT_EQ(t1.device_type(), base::DeviceType::kDeviceCPU);
  float* cpu_ptr = t1.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(cpu_ptr + i), 1.f);
  }
}

TEST(test_tensor, clone) {
  using namespace base;
  auto alloc = DEVICE_ALLOCATOR;
  tensor::Tensor t1(DataType::kDataTypeFp32, 32, 32, true, alloc);
  ASSERT_EQ(t1.is_empty(), false);
  SET_VALUE_CU(t1.ptr<float>(), 32 * 32, 1.f);

  tensor::Tensor t2 = t1.clone();
  float* p2 = new float[32 * 32];
  COPY_TO_HOST(p2, t2.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  COPY_TO_HOST(p2, t1.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  ASSERT_EQ(t2.data_type(), base::DataType::kDataTypeFp32);
  ASSERT_EQ(t2.size(), 32 * 32);

  t2.to_cpu();
  std::memcpy(p2, t2.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }
  delete[] p2;
}

TEST(test_tensor, init1) {
  using namespace base;
  auto alloc = CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;
  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc);
  ASSERT_EQ(t1.is_empty(), false);
}

TEST(test_tensor, assign) {
  using namespace base;
  auto alloc = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1(DataType::kDataTypeFp32, 32, 32, true, alloc);
  ASSERT_EQ(t1.is_empty(), false);

  int32_t size = 32 * 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }
  std::shared_ptr<Buffer> buffer =
      std::make_shared<Buffer>(size * sizeof(float), nullptr, ptr, true);
  buffer->set_device_type(DeviceType::kDeviceCPU);

  ASSERT_EQ(t1.assign(buffer), true);
  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_NE(t1.ptr<float>(), nullptr);
  delete[] ptr;
}
