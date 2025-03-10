#ifndef MATMUL_KERNEL_CU_CUH
#define MATMUL_KERNEL_CU_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      const Config* config = nullptr);

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const Config* config = nullptr);
}  // namespace kernel

#endif  // MATMUL_KERNEL_CU_CUH
