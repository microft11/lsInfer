#include "emb_kernel.cuh"
#include <hip/hip_runtime.h>  // 直接使用 HIP 头文件

namespace kernel {

// 内核函数命名改为 _hip 以区分（可选）
__global__ void emb_kernel_hip_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

void emb_kernel_hip(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  tensor::Tensor input_hip;
  if (input.device_type() != base::DeviceType::kDeviceHIP) {  // 修改：kDeviceCUDA -> kDeviceHIP
    input_hip = input.clone();
    input_hip.to_hip();  // 假设有 to_hip() 方法，类似 to_cuda()
  }

  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  
  // 检查设备类型是否为 HIP
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceHIP);  // 修改：kDeviceCUDA -> kDeviceHIP

  constexpr int32_t max_seq_len = 512;
  constexpr int32_t thread_num = 128;
  int32_t* in_ptr = input_hip.ptr<int32_t>();
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());

  if (stream) {
    hipStream_t stream_ = static_cast<hipStream_t>(stream);  // 修改：cudaStream_t -> hipStream_t
    emb_kernel_hip_fp32<<<max_seq_len, thread_num, 0, stream_>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
  } else {
    emb_kernel_hip_fp32<<<max_seq_len, thread_num>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
  }
}
}  // namespace kernel