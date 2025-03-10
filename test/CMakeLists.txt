include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
aux_source_directory(../test DIR_TEST)

aux_source_directory(../test/test_cu DIR_TEST_CU)
aux_source_directory(../test/test_op DIR_TEST_OP)
aux_source_directory(../test/test_model DIR_TEST_MODEL)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
aux_source_directory(../test/optimized DIR_TEST_OPTIMIZED)

add_executable(test_llm ${DIR_TEST} ${DIR_TEST_CU} ${DIR_TEST_OP} ${DIR_TEST_OPTIMIZED} ${DIR_TEST_TENSOR} ${DIR_TEST_MODEL})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
target_link_libraries(test_llm ${link_ext_lib})
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ../lsinfer/include)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/build)

link_directories(/usr/local/lib)
# 查找 SentencePiece 库
find_library(SENTENCEPIECE_LIBRARY sentencepiece)
if(SENTENCEPIECE_LIBRARY)
    target_link_libraries(test_llm ${SENTENCEPIECE_LIBRARY})
else()
    message(FATAL_ERROR "SentencePiece library not found!")
endif()

if (LLAMA3_SUPPORT OR QWEN2_SUPPORT)
    message(STATUS "LINK LLAMA3 SUPPORT")
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(MyInferenceFramework absl::base re2::re2 nlohmann_json::nlohmann_json)
endif ()
target_link_libraries(test_llm MyInferenceFramework)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# 只在 CUDA 启用时进行可分离编译
if (USE_CUDA)
    set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

# 传递宏定义
if (USE_CUDA)
    target_compile_definitions(test_llm PUBLIC USE_CUDA)
    find_package(CUDA REQUIRED)
    target_link_libraries(test_llm CUDA::CUDA)
    message(STATUS "Compiling test_llm with CUDA support")
elseif (USE_ROCM)
    target_compile_definitions(test_llm PUBLIC USE_ROCM)
    find_package(hip REQUIRED)
    target_link_libraries(test_llm hip::hipblas)
    message(STATUS "Compiling test_llm with ROCm support")
else ()
    message(STATUS "Compiling test_llm with CPU support")
endif ()
