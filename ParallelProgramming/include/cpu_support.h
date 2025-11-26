#pragma once
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <functional>

#include "gpu_support.h"

void print_words(const uint32_t *x, size_t size) {
    for (size_t i = 0; i < size; i++) {
        printf("%08x\n", x[i]);
    }
}

void zero_words(uint32_t *x, size_t size) {
    for (size_t i = 0; i < size; i++) {
        x[i] = 0;
    }
}

class Tester {
private:
    uint32_t *data1, *data2;

    double get_time_in_seconds() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

public:
    void bench(std::function<void(uint32_t*, size_t)> func, const std::string s) {
        
        double current_time = get_time_in_seconds();
        func(data2, n);
        double elapsed_time = get_time_in_seconds() - current_time;

        double speedup = bench_cpu_elapsed/elapsed_time;

        for (size_t i = 0; i < n; i++) {
            if (data1[i] != data2[i]) {
                printf("Mismatch at index %zu: %08x != %08x\n", i, data1[i], data2[i]);
                break;
            }
        }

        printf("CPU algorithm sort time: %.5f seconds\n", bench_cpu_elapsed);
        printf("CPU my %s sort time: %.5f seconds\n", s, elapsed_time);
        printf("speedup %.3fx\n\n", speedup);

    }

    void gpu_bench(std::function<void(Tester*)> func, const std::string s) {
        
        double current_time = get_time_in_seconds();
        func(this);
        cudaDeviceSynchronize();
        double elapsed_time = get_time_in_seconds() - current_time;

        cuda_check(cudaMemcpy(data2, g_data, n*sizeof(uint32_t), cudaMemcpyDeviceToHost));

        double speedup = bench_cpu_elapsed/elapsed_time;

        for (size_t i = 0; i < n; i++) {
            if (data1[i] != data2[i]) {
                printf("Mismatch at index %zu: %08x != %08x\n", i, data1[i], data2[i]);
                break;
            }
        }

        printf("CPU algorithm sort time: %.5f seconds\n", bench_cpu_elapsed);
        printf("GPU my %s sort time: %.5f seconds\n", s, elapsed_time);
        printf("speedup %.3fx\n\n", speedup);

    }

    Tester(size_t size) {
        uint32_t random;

        n = size;

        data1 = (uint32_t*)malloc(size*sizeof(uint32_t));
        data2 = (uint32_t*)malloc(size*sizeof(uint32_t));

        for (size_t i = 0; i < size; i++) {
            random = rand() & 0xFFFF;
            random = (random << 16) + (rand() & 0xFFFF);
            data1[i] = random == 0 ? 1 : random;
            data2[i] = data1[i];
        }

        double current_time = get_time_in_seconds();
        std::sort(data1, data1+size);
        bench_cpu_elapsed = get_time_in_seconds() - current_time;

        cuda_check(cudaMalloc((void**)&g_data, size*sizeof(uint32_t)));
        cuda_check(cudaMemcpy(g_data, data1, size*sizeof(uint32_t), cudaMemcpyHostToDevice));
        cuda_check(cudaMalloc((void**)&bin, size*sizeof(uint32_t)));
        cuda_check(cudaMemcpy(bin, 0, size*sizeof(uint32_t), cudaMemcpyHostToDevice));

    }

    ~Tester() {
        cudaFree(g_data);
        cudaFree(bin);
        free(data1);
        free(data2);
    }

    size_t n;
    uint32_t *g_data, *bin;
    double bench_cpu_elapsed;
};

