#include <hip/hip_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"

TEST(test_add_hip, add1_nostream) {
  auto alloc_hip = base::HIPDeviceAllocatorFactory::get_instance();  // 替换：CUDA -> HIP

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_hip);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_hip);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_hip);

  set_value_hip(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);  // 替换：set_value_cu
  set_value_hip(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

  kernel::get_add_kernel(base::DeviceType::kDeviceHIP)(t1, t2, out, nullptr);  // 替换：kDeviceCUDA -> kDeviceHIP
  hipDeviceSynchronize();  // 替换：cudaDeviceSynchronize

  float* output = new float[size];
  HIP_CHECK(hipMemcpy(output, out.ptr<float>(), size * sizeof(float), hipMemcpyDeviceToHost));  // 替换：cudaMemcpy

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }
  delete[] output;
}

TEST(test_add_hip, add1_stream) {
  auto alloc_hip = base::HIPDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_hip);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_hip);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_hip);

  set_value_hip(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_hip(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));  // 替换：cudaStreamCreate
  kernel::get_add_kernel(base::DeviceType::kDeviceHIP)(t1, t2, out, stream);
  HIP_CHECK(hipDeviceSynchronize());

  float* output = new float[size];
  HIP_CHECK(hipMemcpy(output, out.ptr<float>(), size * sizeof(float), hipMemcpyDeviceToHost));

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }

  HIP_CHECK(hipStreamDestroy(stream));  // 替换：cudaStreamDestroy
  delete[] output;
}

TEST(test_add_hip, add_align1) {
  auto alloc_hip = base::HIPDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 13;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_hip);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_hip);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_hip);

  set_value_hip(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.1f);
  set_value_hip(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.3f);

  kernel::get_add_kernel(base::DeviceType::kDeviceHIP)(t1, t2, out, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  float* output = new float[size];
  HIP_CHECK(hipMemcpy(output, out.ptr<float>(), size * sizeof(float), hipMemcpyDeviceToHost));

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(output[i], 5.4f, 0.1f);
  }
  delete[] output;
}