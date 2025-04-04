#include <hip/hip_runtime.h>  // 替换为 HIP 头文件
#include "base/alloc.h"

namespace base {

// 修改类名：CUDADeviceAllocator -> HIPDeviceAllocator
HIPDeviceAllocator::HIPDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceHIP) {}

void* HIPDeviceAllocator::allocate(size_t byte_size) const {
  int id = -1;
  hipError_t state = hipGetDevice(&id);  // 替换：cudaGetDevice -> hipGetDevice
  CHECK(state == hipSuccess);  // 替换：cudaSuccess -> hipSuccess
  
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
    state = hipMalloc(&ptr, byte_size);  // 替换：cudaMalloc -> hipMalloc
    if (hipSuccess != state) {  // 替换：cudaSuccess -> hipSuccess
      char buf[256];
      snprintf(buf, 256,
               "Error: HIP error when allocating %lu MB memory! Maybe there's no enough memory "
               "left on device.",
               byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  }

  auto& hip_buffers = hip_buffers_map_[id];  // 替换：cuda_buffers -> hip_buffers
  for (int i = 0; i < hip_buffers.size(); i++) {
    if (hip_buffers[i].byte_size >= byte_size && !hip_buffers[i].busy) {
      hip_buffers[i].busy = true;
      no_busy_cnt_[id] -= hip_buffers[i].byte_size;
      return hip_buffers[i].data;
    }
  }
  
  void* ptr = nullptr;
  state = hipMalloc(&ptr, byte_size);  // 替换：cudaMalloc -> hipMalloc
  if (hipSuccess != state) {  // 替换：cudaSuccess -> hipSuccess
    char buf[256];
    snprintf(buf, 256,
             "Error: HIP error when allocating %lu MB memory! Maybe there's no enough memory "
             "left on device.",
             byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  hip_buffers.emplace_back(ptr, byte_size, true);
  return ptr;
}

void HIPDeviceAllocator::release(void* ptr) const {
  if (!ptr) {
    return;
  }
  if (hip_buffers_map_.empty()) {  // 替换：cuda_buffers_map_ -> hip_buffers_map_
    return;
  }
  
  hipError_t state = hipSuccess;  // 替换：cudaSuccess -> hipSuccess
  for (auto& it : hip_buffers_map_) {
    if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
      auto& hip_buffers = it.second;
      std::vector<HipMemoryBuffer> temp;  // 替换：CudaMemoryBuffer -> HipMemoryBuffer
      for (int i = 0; i < hip_buffers.size(); i++) {
        if (!hip_buffers[i].busy) {
          state = hipSetDevice(it.first);  // 替换：cudaSetDevice -> hipSetDevice
          state = hipFree(hip_buffers[i].data);  // 替换：cudaFree -> hipFree
          CHECK(state == hipSuccess)  // 替换：cudaSuccess -> hipSuccess
              << "Error: HIP error when releasing memory on device " << it.first;
        } else {
          temp.push_back(hip_buffers[i]);
        }
      }
      hip_buffers.clear();
      it.second = temp;
      no_busy_cnt_[it.first] = 0;
    }
  }

  for (auto& it : hip_buffers_map_) {
    auto& hip_buffers = it.second;
    for (int i = 0; i < hip_buffers.size(); i++) {
      if (hip_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += hip_buffers[i].byte_size;
        hip_buffers[i].busy = false;
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
  
  state = hipFree(ptr);  // 替换：cudaFree -> hipFree
  CHECK(state == hipSuccess)  // 替换：cudaSuccess -> hipSuccess
      << "Error: HIP error when releasing memory on device";
}

// 修改类名：CUDADeviceAllocatorFactory -> HIPDeviceAllocatorFactory
std::shared_ptr<HIPDeviceAllocator> HIPDeviceAllocatorFactory::instance = nullptr;

}  // namespace base