#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#elif defined(USE_ROCM)
#include <hip/hip_runtime.h>
#endif
#include "base/alloc.h"
namespace base {

void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);
  if (!byte_size) {
    return;
  }

#if defined(USE_CUDA)
  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<cudaStream_t>(stream);
  }
#elif defined(USE_ROCM)
  hipStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<hipStream_t>(stream);
  }
#endif

  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(dest_ptr, src_ptr, byte_size);
  } 
#if defined(USE_CUDA)
  else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
    }
  }
#elif defined(USE_ROCM)
  else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      hipMemcpy(dest_ptr, src_ptr, byte_size, hipMemcpyHostToDevice);
    } else {
      hipMemcpyAsync(dest_ptr, src_ptr, byte_size, hipMemcpyHostToDevice, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    if (!stream_) {
      hipMemcpy(dest_ptr, src_ptr, byte_size, hipMemcpyDeviceToHost);
    } else {
      hipMemcpyAsync(dest_ptr, src_ptr, byte_size, hipMemcpyDeviceToHost, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    if (!stream_) {
      hipMemcpy(dest_ptr, src_ptr, byte_size, hipMemcpyDeviceToDevice);
    } else {
      hipMemcpyAsync(dest_ptr, src_ptr, byte_size, hipMemcpyDeviceToDevice, stream_);
    }
  }
#endif
  else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
  }

#if defined(USE_CUDA) || defined(USE_ROCM)
  if (need_sync) {
#if defined(USE_CUDA)
    cudaDeviceSynchronize();
#elif defined(USE_ROCM)
    hipDeviceSynchronize();
#endif
  }
#endif
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byte_size);
  } 
#if defined(USE_CUDA)
  else {
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      cudaMemsetAsync(ptr, 0, byte_size, stream_);
    } else {
      cudaMemset(ptr, 0, byte_size);
    }
    if (need_sync) {
      cudaDeviceSynchronize();
    }
  }
#elif defined(USE_ROCM)
  else {
    if (stream) {
      hipStream_t stream_ = static_cast<hipStream_t>(stream);
      hipMemsetAsync(ptr, 0, byte_size, stream_);
    } else {
      hipMemset(ptr, 0, byte_size);
    }
    if (need_sync) {
      hipDeviceSynchronize();
    }
  }
#endif
}

#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif

// CPU Device Allocator
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }

#ifdef KUIPER_HAVE_POSIX_MEMALIGN
  void* data = nullptr;
  const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
  int status = posix_memalign((void**)&data,
                              ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
                              byte_size);
  if (status != 0) {
    return nullptr;
  }
  return data;
#else
  return malloc(byte_size);
#endif
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

// CUDA Device Allocator
#if defined(USE_CUDA)
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess);
  
  if (byte_size > 1024 * 1024) {
    auto& big_buffers = big_buffers_map_[id];
    int sel_id = -1;
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
          big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
        if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          sel_id = i;
        }
      }
    }
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }

    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (cudaSuccess != state) {
      LOG(ERROR) << "Error: CUDA error when allocating " << (byte_size >> 20) << " MB memory!";
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  }

  auto& cuda_buffers = cuda_buffers_map_[id];
  for (int i = 0; i < cuda_buffers.size(); i++) {
    if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
      cuda_buffers[i].busy = true;
      no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
      return cuda_buffers[i].data;
    }
  }

  void* ptr = nullptr;
  state = cudaMalloc(&ptr, byte_size);
  if (cudaSuccess != state) {
    LOG(ERROR) << "Error: CUDA error when allocating " << (byte_size >> 20) << " MB memory!";
    return nullptr;
  }
  cuda_buffers.emplace_back(ptr, byte_size, true);
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  if (!ptr) return;

  if (cuda_buffers_map_.empty()) return;

  cudaError_t state = cudaSuccess;
  for (auto& it : cuda_buffers_map_) {
    if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
      auto& cuda_buffers = it.second;
      std::vector<CudaMemoryBuffer> temp;
      for (int i = 0; i < cuda_buffers.size(); i++) {
        if (!cuda_buffers[i].busy) {
          state = cudaSetDevice(it.first);
          state = cudaFree(cuda_buffers[i].data);
          CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device " << it.first;
        } else {
          temp.push_back(cuda_buffers[i]);
        }
      }
      cuda_buffers.clear();
      it.second = temp;
      no_busy_cnt_[it.first] = 0;
    }
  }

  for (auto& it : cuda_buffers_map_) {
    auto& cuda_buffers = it.second;
    for (int i = 0; i < cuda_buffers.size(); i++) {
      if (cuda_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
        return;
      }
    }
    auto& big_buffers = big_buffers_map_[it.first];
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return;
      }
    }
  }

  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}

#endif  // USE_CUDA

// ROCm Device Allocator
#if defined(USE_ROCM)
ROCMDeviceAllocator::ROCMDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceROCM) {}

void* ROCMDeviceAllocator::allocate(size_t byte_size) const {
  int id = -1;
  hipError_t state = hipGetDevice(&id);
  CHECK(state == hipSuccess);

  if (byte_size > 1024 * 1024) {
    auto& big_buffers = big_buffers_map_[id];
    int sel_id = -1;
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
          big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
        if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          sel_id = i;
        }
      }
    }
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }

    void* ptr = nullptr;
    state = hipMalloc(&ptr, byte_size);
    if (hipSuccess != state) {
      LOG(ERROR) << "Error: ROCm error when allocating " << (byte_size >> 20) << " MB memory!";
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  }

  auto& rocm_buffers = rocm_buffers_map_[id];
  for (int i = 0; i < rocm_buffers.size(); i++) {
    if (rocm_buffers[i].byte_size >= byte_size && !rocm_buffers[i].busy) {
      rocm_buffers[i].busy = true;
      no_busy_cnt_[id] -= rocm_buffers[i].byte_size;
      return rocm_buffers[i].data;
    }
  }

  void* ptr = nullptr;
  state = hipMalloc(&ptr, byte_size);
  if (hipSuccess != state) {
    LOG(ERROR) << "Error: ROCm error when allocating " << (byte_size >> 20) << " MB memory!";
    return nullptr;
  }
  rocm_buffers.emplace_back(ptr, byte_size, true);
  return ptr;
}

void ROCMDeviceAllocator::release(void* ptr) const {
  if (!ptr) return;

  if (rocm_buffers_map_.empty()) return;

  hipError_t state = hipSuccess;
  for (auto& it : rocm_buffers_map_) {
    if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
      auto& rocm_buffers = it.second;
      std::vector<ROCMMemoryBuffer> temp;
      for (int i = 0; i < rocm_buffers.size(); i++) {
        if (!rocm_buffers[i].busy) {
          state = hipSetDevice(it.first);
          state = hipFree(rocm_buffers[i].data);
          CHECK(state == hipSuccess) << "Error: ROCm error when releasing memory on device " << it.first;
        } else {
          temp.push_back(rocm_buffers[i]);
        }
      }
      rocm_buffers.clear();
      it.second = temp;
      no_busy_cnt_[it.first] = 0;
    }
  }

  for (auto& it : rocm_buffers_map_) {
    auto& rocm_buffers = it.second;
    for (int i = 0; i < rocm_buffers.size(); i++) {
      if (rocm_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += rocm_buffers[i].byte_size;
        rocm_buffers[i].busy = false;
        return;
      }
    }
    auto& big_buffers = big_buffers_map_[it.first];
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return;
      }
    }
  }

  state = hipFree(ptr);
  CHECK(state == hipSuccess) << "Error: ROCm error when releasing memory on device";
}
#endif  // USE_ROCM

}  // namespace base
