#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H

#include <hipblas/hipblas.h>  // 替换：cublas_v2.h -> hipblas.h
#include <hip/hip_runtime.h>  // 替换：cuda_runtime_api.h

namespace kernel {

// 修改结构体名：CudaConfig -> HipConfig
struct HipConfig {
  hipStream_t stream = nullptr;  // 替换：cudaStream_t -> hipStream_t
  ~HipConfig() {
    if (stream) {
      hipStreamDestroy(stream);  // 替换：cudaStreamDestroy -> hipStreamDestroy
    }
  }
};

}  // namespace kernel
#endif  // BLAS_HELPER_H