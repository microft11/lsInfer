#ifndef TEST_CU_CUH
#define TEST_CU_CUH

#pragma once
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd) \
do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error: %s at %s:%d\n", \
               hipGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void test_function(float* arr, int32_t size, float value = 1.f);

void set_value_cu(float* arr_cu, int32_t size, float value = 1.f);
#endif  // TEST_CU_CUH
