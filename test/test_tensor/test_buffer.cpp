#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "../utils.cuh"
#define DEVICE_ALLOCATOR base::CUDADeviceAllocatorFactory::get_instance()
#define MEM_ALLOC(ptr, size) cudaMalloc(&(ptr), (size))
#define MEM_FREE(ptr) cudaFree(ptr)
#define MEM_COPY(dst, src, size, direction) cudaMemcpy((dst), (src), (size), (direction))
#endif

#ifdef USE_ROCM
#include <hip/hip_runtime_api.h>
#define DEVICE_ALLOCATOR base::ROCmDeviceAllocatorFactory::get_instance()
#define MEM_ALLOC(ptr, size) hipMalloc(&(ptr), (size))
#define MEM_FREE(ptr) hipFree(ptr)
#define MEM_COPY(dst, src, size, direction) hipMemcpy((dst), (src), (size), (direction))
#endif

#ifndef DEVICE_ALLOCATOR  // 默认使用 CPU
#define DEVICE_ALLOCATOR base::CPUDeviceAllocatorFactory::get_instance()
#define MEM_ALLOC(ptr, size) (ptr) = new float[(size) / sizeof(float)]
#define MEM_FREE(ptr) delete[](ptr)
#define MEM_COPY(dst, src, size, direction) memcpy((dst), (src), (size))
#endif

TEST(test_buffer, allocate) {
  using namespace base;
  auto alloc = DEVICE_ALLOCATOR;
  Buffer buffer(32, alloc);
  ASSERT_NE(buffer.ptr(), nullptr);
}

TEST(test_buffer, use_external) {
  using namespace base;
  auto alloc = DEVICE_ALLOCATOR;
  float* ptr = nullptr;
  MEM_ALLOC(ptr, 32 * sizeof(float));
  Buffer buffer(32, nullptr, ptr, true);
  ASSERT_EQ(buffer.is_external(), true);
  MEM_FREE(ptr);
}

TEST(test_buffer, device_memcpy1) {
  using namespace base;
  auto alloc = DEVICE_ALLOCATOR;
  auto alloc_cu = DEVICE_ALLOCATOR;

  int32_t size = 32;
  float* ptr = nullptr;
  MEM_ALLOC(ptr, size * sizeof(float));
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }
  
  Buffer buffer(size * sizeof(float), nullptr, ptr, true);
  buffer.set_device_type(DeviceType::kDeviceCPU);
  ASSERT_EQ(buffer.is_external(), true);

  Buffer device_buffer(size * sizeof(float), alloc_cu);
  device_buffer.copy_from(buffer);

  float* ptr2 = new float[size];
  MEM_COPY(ptr2, device_buffer.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], float(i));
  }

  MEM_FREE(ptr);
  delete[] ptr2;
}

TEST(test_buffer, device_memcpy2) {
  using namespace base;
  auto alloc = DEVICE_ALLOCATOR;
  auto alloc_cu = DEVICE_ALLOCATOR;

  int32_t size = 32;
  float* ptr = nullptr;
  MEM_ALLOC(ptr, size * sizeof(float));
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }
  Buffer buffer(size * sizeof(float), nullptr, ptr, true);
  buffer.set_device_type(DeviceType::kDeviceCPU);
  ASSERT_EQ(buffer.is_external(), true);

  // cpu to device
  Buffer device_buffer(size * sizeof(float), alloc_cu);
  device_buffer.copy_from(buffer);

  float* ptr2 = new float[size];
  MEM_COPY(ptr2, device_buffer.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], float(i));
  }

  MEM_FREE(ptr);
  delete[] ptr2;
}

// TEST(test_buffer, device_memcpy3) {
//   using namespace base;
//   auto alloc_cu = DEVICE_ALLOCATOR;

//   int32_t size = 32;
//   Buffer device_buffer1(size * sizeof(float), alloc_cu);
//   Buffer device_buffer2(size * sizeof(float), alloc_cu);

//   set_value_cu((float*)device_buffer2.ptr(), size);

//   ASSERT_EQ(device_buffer1.device_type(), DeviceType::kDeviceCUDA);
//   ASSERT_EQ(device_buffer2.device_type(), DeviceType::kDeviceCUDA);

//   device_buffer1.copy_from(device_buffer2);

//   float* ptr2 = new float[size];
//   MEM_COPY(ptr2, device_buffer1.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
//   for (int i = 0; i < size; ++i) {
//     ASSERT_EQ(ptr2[i], 1.f);
//   }
//   delete[] ptr2;
// }

// TEST(test_buffer, device_memcpy4) {
//   using namespace base;
//   auto alloc = DEVICE_ALLOCATOR;
//   auto alloc_cu = DEVICE_ALLOCATOR;

//   int32_t size = 32;
//   Buffer device_buffer1(size * sizeof(float), alloc_cu);
//   Buffer device_buffer2(size * sizeof(float), alloc);

//   ASSERT_EQ(device_buffer1.device_type(), DeviceType::kDeviceCUDA);
//   ASSERT_EQ(device_buffer2.device_type(), DeviceType::kDeviceCPU);

//   set_value_cu((float*)device_buffer1.ptr(), size);
//   device_buffer2.copy_from(device_buffer1);

//   float* ptr2 = (float*)device_buffer2.ptr();
//   for (int i = 0; i < size; ++i) {
//     ASSERT_EQ(ptr2[i], 1.f);
//   }
// }
