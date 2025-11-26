#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>

#include "cpu_support.h"

void radix_sort(uint32_t *a, size_t n) {

    for (size_t i = 0; i < 3; i++) {
        std::vector<std::vector<uint32_t>> bin(256);

        for (size_t j = 0; j < n; j++) {
            bin[(a[j] >> i*8) & 0xff].push_back(a[j]);
        }

        size_t index = 0;
        for (size_t j = 0; j < 256; j++) {
            for (size_t k = 0; k < bin[j].size(); k++) {
                a[index++] = bin[j][k];
            }
        }
    }
    
}

__global__ void gpu_radix_sort(uint32_t *bin, uint32_t *a, size_t n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n)   return;

    int i = a[tid];
    if (bin[a[tid]] == 0) [[likely]] {
        bin[a[tid]] = a[tid];
    } else {
        while (bin[i] != 0) [[likely]] {
            ++i;
        }
        bin[i] = a[tid];
    }

    __syncthreads();
    a[tid] = bin[tid];
}

void call_gpu(Tester *tester) {
    size_t n = tester->n;

    gpu_radix_sort<<<n/512, 512>>>(tester->bin, tester->g_data, n);
    cuda_check(cudaGetLastError());
}

int main(int argc, char* argv[]) {

    size_t n = size_t(1) << 20;

    int count = 1;
    if (argc > 1) count = atoi(argv[1]);

    // Tester tester(n);
    // tester.bench(radix_sort, "radix_sort");

    Tester tester2(n);
    tester2.gpu_bench(call_gpu, "gpu_radix_sort");

    return 0;
}   