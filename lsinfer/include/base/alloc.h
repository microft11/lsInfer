#ifndef KUIPER_INCLUDE_BASE_ALLOC_H_
#define KUIPER_INCLUDE_BASE_ALLOC_H_

#include <map>
#include <memory>
#include "base.h"
#include <hip/hip_runtime.h>  // 替换为 HIP 头文件

namespace base {

// 修改枚举：CUDA -> HIP
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2HIP = 1,    // 修改：kMemcpyCPU2CUDA -> kMemcpyCPU2HIP
  kMemcpyHIP2CPU = 2,    // 修改：kMemcpyCUDA2CPU -> kMemcpyHIP2CPU
  kMemcpyHIP2HIP = 3,    // 修改：kMemcpyCUDA2CUDA -> kMemcpyHIP2HIP
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

  virtual DeviceType device_type() const { return device_type_; }

  virtual void release(void* ptr) const = 0;

  virtual void* allocate(size_t byte_size) const = 0;

  virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, 
                      void* stream = nullptr, bool need_sync = false) const;

  virtual void memset_zero(void* ptr, size_t byte_size, 
                          void* stream = nullptr, bool need_sync = false);

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

// CPU 分配器（保持不变）
class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

// 修改：CudaMemoryBuffer -> HipMemoryBuffer
struct HipMemoryBuffer {
  void* data;
  size_t byte_size;
  bool busy;

  HipMemoryBuffer() = default;

  HipMemoryBuffer(void* data, size_t byte_size, bool busy)
      : data(data), byte_size(byte_size), busy(busy) {}
};

// 修改：CUDADeviceAllocator -> HIPDeviceAllocator
class HIPDeviceAllocator : public DeviceAllocator {
 public:
  explicit HIPDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceHIP) {}

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;

 private:
  mutable std::map<int, size_t> no_busy_cnt_;
  mutable std::map<int, std::vector<HipMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<HipMemoryBuffer>> hip_buffers_map_;  // 修改：cuda_buffers_map_ -> hip_buffers_map_
};

// CPU 工厂（保持不变）
class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

// 修改：CUDADeviceAllocatorFactory -> HIPDeviceAllocatorFactory
class HIPDeviceAllocatorFactory {
 public:
  static std::shared_ptr<HIPDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<HIPDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<HIPDeviceAllocator> instance;
};

}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_ALLOC_H_