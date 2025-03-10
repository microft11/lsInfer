#pragma once

#if defined(USE_CUDA)
    #include <cuda_runtime.h>
    #define DEVICE_API __device__
#elif defined(USE_ROCM)
    #include <hip/hip_runtime.h>
    #define DEVICE_API __device__
#else
    #define DEVICE_API
#endif

namespace kernel {

// 根据编译平台选择不同的配置结构体

#if defined(USE_CUDA)
struct CudaConfig {
   cudaStream_t stream = nullptr;
   ~CudaConfig() {
     if (stream) {
       cudaStreamDestroy(stream);
     }
   }
};
using Config = CudaConfig;  // 使用统一的Config名称

#elif defined(USE_ROCM)
struct RocmConfig {
   hipStream_t stream = nullptr;
   ~RocmConfig() {
     if (stream) {
       hipStreamDestroy(stream);
     }
   }
};
using Config = RocmConfig;  // 使用统一的Config名称

#else
struct CpuConfig {
   // CPU-specific configuration can go here
   // CPU doesn't need a stream, but you could add other settings if needed
   void *stream = nullptr;
   CpuConfig() = default;
   ~CpuConfig() = default;
};
using Config = CpuConfig;  // 使用统一的Config名称

#endif

}  // namespace kernel
