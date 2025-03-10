# cmake/rocm.cmake
find_package(hip REQUIRED)
set(CMAKE_CXX_COMPILER ${HIP_CXX_COMPILER})
set(CMAKE_CUDA_ARCHITECTURES "")

# 定义 ROCm 相关宏
add_definitions(-DUSE_ROCM)

# 设置 HIP 作为编译器
set(CMAKE_HIP_ARCHITECTURES "gfx90a")  # 这里根据实际 GPU 选择架构
