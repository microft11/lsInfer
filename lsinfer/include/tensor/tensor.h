#ifndef KUIPER_INCLUDE_TENSOR_TENSOR_H_
#define KUIPER_INCLUDE_TENSOR_TENSOR_H_

#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"
#include "base/alloc.h"  // 新增设备分配器头文件

// 根据编译环境定义宏
#ifdef USE_CUDA
#include <driver_types.h>
#elif defined(USE_ROCM)
#include <hip/hip_runtime.h>
#endif

namespace tensor {

// Tensor类，支持在不同设备（CPU、CUDA、ROCm）上的操作
class Tensor {
 public:
  explicit Tensor() = default;

  // 支持通过DataType、维度初始化Tensor，包含设备分配器和ptr
  explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                  bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                  bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);

  explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

  void to_cpu();  // 转换到CPU

#ifdef USE_CUDA
  void to_cuda(cudaStream_t stream = nullptr);  // 转换到CUDA设备
#endif

#ifdef USE_ROCM
  void to_rocm(hipStream_t stream = nullptr);  // 转换到ROCm设备
#endif

  bool is_empty() const;  // 判断Tensor是否为空

  void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                   bool need_alloc, void* ptr);

  template <typename T>
  T* ptr();

  template <typename T>
  const T* ptr() const;

  void reshape(const std::vector<int32_t>& dims);

  std::shared_ptr<base::Buffer> get_buffer() const;

  size_t size() const;

  size_t byte_size() const;

  int32_t dims_size() const;

  base::DataType data_type() const;

  int32_t get_dim(int32_t idx) const;

  const std::vector<int32_t>& dims() const;

  std::vector<size_t> strides() const;

  bool assign(std::shared_ptr<base::Buffer> buffer);

  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  void set_device_type(base::DeviceType device_type) const;

  base::DeviceType device_type() const;

  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

  // 支持设备类型的指针操作
  template <typename T>
  T* ptr(int64_t index);

  template <typename T>
  const T* ptr(int64_t index) const;

  template <typename T>
  T& index(int64_t offset);

  template <typename T>
  const T& index(int64_t offset) const;

  tensor::Tensor clone() const;

 private:
  size_t size_ = 0;
  std::vector<int32_t> dims_;
  std::shared_ptr<base::Buffer> buffer_;
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;

  // 新增设备指针和分配器
  std::shared_ptr<base::DeviceAllocator> device_allocator_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceCPU;
};

template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

}  // namespace tensor

#endif  // KUIPER_INCLUDE_TENSOR_TENSOR_H_
