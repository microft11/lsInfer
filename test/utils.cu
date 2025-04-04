#include <glog/logging.h>
#include <hip/hip_runtime.h>
#include "utils.cuh" 

// HIP 版本的内核函数（命名改为 _hip）
__global__ void test_function_hip(float* hip_arr, int32_t size, float value) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size) {
    return;
  }
  hip_arr[tid] = value;
}

// 包装函数（HIP 实现）
void test_function(float* arr, int32_t size, float value) {
  if (!arr) {
    return;
  }
  float* hip_arr = nullptr;
  HIP_CHECK(hipMalloc(&hip_arr, sizeof(float) * size));  // 替换 cudaMalloc
  HIP_CHECK(hipDeviceSynchronize());

  // 调用 HIP 内核
  test_function_hip<<<1, size>>>(hip_arr, size, value);  // 内核名改为 _hip
  HIP_CHECK(hipDeviceSynchronize());

  // 拷贝回主机
  HIP_CHECK(hipMemcpy(arr, hip_arr, size * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(hip_arr));  // 替换 cudaFree
}

// HIP 版本的 set_value
void set_value_hip(float* arr_hip, int32_t size, float value) {
  int32_t threads_num = 512;
  int32_t block_num = (size + threads_num - 1) / threads_num;
  HIP_CHECK(hipDeviceSynchronize());

  test_function_hip<<<block_num, threads_num>>>(arr_hip, size, value);  // 使用 HIP 内核
  HIP_CHECK(hipDeviceSynchronize());
}
