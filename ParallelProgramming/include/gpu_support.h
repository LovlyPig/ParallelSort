#pragma once
#include <cuda_runtime.h>
#include <vector>

void cuda_check(cudaError_t status, const char* action=NULL, const char* file=NULL, int32_t line=0) {
    // check for cuda errors

    if (status != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(status));
        if (action) {
            printf("While running %s    (file: %s, line: %d)\n", action, file, line);
            exit(1);
        }
    }
}

#define CUDA_CHECK(status) cuda_check((status), #status, __FILE__, __LINE__)

