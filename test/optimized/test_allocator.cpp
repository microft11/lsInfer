#include <tensor/tensor.h>
#include "../utils.cuh"
#include "base/buffer.h"
#include <hip/hip_runtime.h>  // 替换 CUDA 头文件
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/base.h"

TEST(test_buffer, use_external1) {
  using namespace base;
  auto alloc = base::HIPDeviceAllocatorFactory::get_instance();  // 替换：CUDADevice -> HIPDevice
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  CHECK_EQ(buffer.is_external(), true);
  
  // 替换 cudaFree -> hipFree
  hipError_t err = hipFree(buffer.ptr());  // 注意：此处逻辑有问题，ptr 是主机内存！
  EXPECT_EQ(err, hipSuccess);  // 实际应删除此操作，见注意事项
}

// 新增：测试 HIP 内存分配
TEST(test_buffer, hip_allocation) {
  auto alloc = base::HIPDeviceAllocatorFactory::get_instance();
  void* device_ptr = alloc->allocate(32 * sizeof(float));
  EXPECT_NE(device_ptr, nullptr);
  alloc->release(device_ptr);
}