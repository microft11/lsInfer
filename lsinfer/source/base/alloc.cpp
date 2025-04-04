#include "base/alloc.h"
#include "base/base.h"
#include <hip/hip_runtime.h>  // 替换为 HIP 头文件

namespace base {

void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);
  if (!byte_size) {
    return;
  }

  hipStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<hipStream_t>(stream);  // 替换：CUstream_st -> hipStream_t
  }

  switch (memcpy_kind) {
    case MemcpyKind::kMemcpyCPU2CPU:
      std::memcpy(dest_ptr, src_ptr, byte_size);
      break;
    case MemcpyKind::kMemcpyCPU2HIP:  // 修改：kMemcpyCPU2CUDA -> kMemcpyCPU2HIP
      if (!stream_) {
        HIP_CHECK(hipMemcpy(dest_ptr, src_ptr, byte_size, hipMemcpyHostToDevice));  // 替换：cudaMemcpy
      } else {
        HIP_CHECK(hipMemcpyAsync(dest_ptr, src_ptr, byte_size, hipMemcpyHostToDevice, stream_));  // 替换：cudaMemcpyAsync
      }
      break;
    case MemcpyKind::kMemcpyHIP2CPU:  // 修改：kMemcpyCUDA2CPU -> kMemcpyHIP2CPU
      if (!stream_) {
        HIP_CHECK(hipMemcpy(dest_ptr, src_ptr, byte_size, hipMemcpyDeviceToHost));
      } else {
        HIP_CHECK(hipMemcpyAsync(dest_ptr, src_ptr, byte_size, hipMemcpyDeviceToHost, stream_));
      }
      break;
    case MemcpyKind::kMemcpyHIP2HIP:  // 修改：kMemcpyCUDA2CUDA -> kMemcpyHIP2HIP
      if (!stream_) {
        HIP_CHECK(hipMemcpy(dest_ptr, src_ptr, byte_size, hipMemcpyDeviceToDevice));
      } else {
        HIP_CHECK(hipMemcpyAsync(dest_ptr, src_ptr, byte_size, hipMemcpyDeviceToDevice, stream_));
      }
      break;
    default:
      LOG(FATAL) << "Unknown memcpy kind: " << static_cast<int>(memcpy_kind);
  }

  if (need_sync) {
    HIP_CHECK(hipDeviceSynchronize());  // 替换：cudaDeviceSynchronize
  }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
  
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byte_size);
  } else {
    if (stream) {
      hipStream_t stream_ = static_cast<hipStream_t>(stream);  // 替换：cudaStream_t
      HIP_CHECK(hipMemsetAsync(ptr, 0, byte_size, stream_));  // 替换：cudaMemsetAsync
    } else {
      HIP_CHECK(hipMemset(ptr, 0, byte_size));  // 替换：cudaMemset
    }
    if (need_sync) {
      HIP_CHECK(hipDeviceSynchronize());  // 替换：cudaDeviceSynchronize
    }
  }
}

}  // namespace base