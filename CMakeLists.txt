cmake_minimum_required(VERSION 3.18)
project(MyInferenceFramework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 选项：手动控制 CUDA 或 ROCm 的编译
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_ROCM "Enable ROCm support" OFF)
option(USE_CPM "Use CPM for dependency management" OFF)
option(LLAMA3_SUPPORT OFF)
option(QWEN2_SUPPORT OFF)

if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "Cannot enable both CUDA and ROCm at the same time!")
endif()

if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
  add_definitions(-DLLAMA3_SUPPORT)
endif()
if (QWEN2_SUPPORT)
  message(STATUS "QWEN2 SUPPORT")
  add_definitions(-DQWEN2_SUPPORT)
endif()

# ---- 依赖项管理 ----
if(USE_CPM)
  include(cmake/CPM.cmake)

  CPMAddPackage(NAME GTest GITHUB_REPOSITORY google/googletest VERSION 1.15.0)
  CPMAddPackage(NAME glog GITHUB_REPOSITORY google/glog VERSION 0.7.1 OPTIONS "BUILD_TESTING Off")
  CPMAddPackage(NAME Armadillo GITLAB_REPOSITORY conradsnicta/armadillo-code GIT_TAG 14.0.1)
  CPMAddPackage(NAME sentencepiece GITHUB_REPOSITORY google/sentencepiece VERSION 0.2.0)
  find_package(sentencepiece REQUIRED)
  find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h)
  if(SENTENCEPIECE_INCLUDE_DIR)
    include_directories(${SENTENCEPIECE_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "SentencePiece headers not found!")
  endif()

  if (LLAMA3_SUPPORT OR QWEN2_SUPPORT)
    CPMAddPackage(NAME absl GITHUB_REPOSITORY abseil/abseil-cpp GIT_TAG 20240722.0 OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON")
    CPMAddPackage(NAME re2 GITHUB_REPOSITORY google/re2 GIT_TAG 2024-07-02)
    CPMAddPackage(NAME nlohmann_json GITHUB_REPOSITORY nlohmann/json VERSION 3.11.3)
  endif()
endif()

find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h)
include_directories(${SENTENCEPIECE_INCLUDE_DIR})


# 依赖库
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
# 自动检测 GPU 支持
if (NOT USE_CUDA AND NOT USE_ROCM)
  find_package(CUDA QUIET)
  if (CUDA_FOUND)
    message(STATUS "CUDA detected, enabling CUDA support")
    set(USE_CUDA ON)
    enable_language(CUDA)
  else()
    message(STATUS "No CUDA found, defaulting to CPU mode")
  endif()
endif()

# 头文件目录
include_directories(lsinfer/include)

# CPU 源文件
file(GLOB_RECURSE CPU_SOURCES 
  lsinfer/source/base/*.cpp
  lsinfer/source/model/*.cpp
  lsinfer/source/op/*.cpp
  lsinfer/source/op/kernels/cpu/*.cpp
  lsinfer/source/sampler/*.cpp
  lsinfer/source/tensor/*.cpp
)

# message(STATUS "CPU_SOURCES: ${CPU_SOURCES}")
add_library(MyInferenceFramework STATIC ${CPU_SOURCES})
target_compile_definitions(MyInferenceFramework PRIVATE USE_CPU)

# **CUDA 部分**
if (USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  file(GLOB_RECURSE CUDA_SOURCES lsinfer/source/op/kernels/cuda/*.cu)
  add_library(MyInferenceFramework_CUDA STATIC ${CUDA_SOURCES})
  target_include_directories(MyInferenceFramework_CUDA PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(MyInferenceFramework_CUDA PRIVATE ${CUDAToolkit_LIBRARIES})
  target_compile_definitions(MyInferenceFramework_CUDA PRIVATE USE_CUDA)
  target_link_libraries(MyInferenceFramework PRIVATE MyInferenceFramework_CUDA)
endif()

# **ROCm 部分**
if (USE_ROCM)
  find_package(hip REQUIRED)

  file(GLOB_RECURSE ROCM_SOURCES lsinfer/source/op/kernels/rocm/*.cpp)
  add_library(MyInferenceFramework_ROCM STATIC ${ROCM_SOURCES})
  target_include_directories(MyInferenceFramework_ROCM PRIVATE ${HIP_INCLUDE_DIRS})
  target_link_libraries(MyInferenceFramework_ROCM PRIVATE hip::device)
  target_compile_definitions(MyInferenceFramework_ROCM PRIVATE USE_ROCM)
  target_link_libraries(MyInferenceFramework PRIVATE MyInferenceFramework_ROCM)
endif()

# 依赖库链接
if(USE_CPM)
  target_include_directories(MyInferenceFramework PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()

target_link_libraries(MyInferenceFramework PRIVATE sentencepiece glog::glog pthread Armadillo)
# 安装目标
install(TARGETS MyInferenceFramework DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
# 测试与示例
# add_subdirectory(test)
add_subdirectory(demo)
