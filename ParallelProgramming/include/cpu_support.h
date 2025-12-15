#pragma once
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <functional>

#include "gpu_support.h"
#include <tbb/tbb.h>

__device__ __host__ void print(uint32_t *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        printf("%08x ", data[i]);
    }
    printf("\n");
}

struct GpuLayout {
    uint32_t *data;
    uint32_t **bucket;
    uint32_t **bucket_data;
    uint32_t *bucket_row;
    size_t size;

    uint32_t *hist, *offset, *data_out;

    size_t blocksize;

    GpuLayout() : data(nullptr), bucket(nullptr), bucket_row(nullptr), bucket_data(nullptr), size(0), 
                hist(nullptr), offset(nullptr), data_out(nullptr), blocksize(0) {}
    ~GpuLayout() {
        if (data) cudaFree(data);
        if (bucket) cudaFree(bucket);
        if (bucket_row) cudaFree(bucket_row);
        if (bucket_data) cudaFree(bucket_data);

        if (hist) cudaFree(hist);
        if (offset) cudaFree(offset);
        if (data_out) cudaFree(data_out);
    }

    void malloc_2D(size_t rows, size_t cols) {
        CUDA_CHECK(cudaMalloc((void**)&data_out, rows * cols * sizeof(uint32_t)));
        bucket_data = (uint32_t**)malloc(rows * sizeof(uint32_t*));
        if (bucket_data == NULL) {
            printf("Host memory allocation failed. (file: %s, line: %d)\n", __FILE__, __LINE__);
            cudaFree(data_out);
            exit(1);
        }

        for (size_t i = 0; i < rows; i++) {
            bucket_data[i] = data + i * cols;
        }

        CUDA_CHECK(cudaMalloc((void**)&bucket, rows * sizeof(uint32_t*)));
        CUDA_CHECK(cudaMemcpy(bucket, bucket_data, rows * sizeof(uint32_t*), cudaMemcpyHostToDevice));
    }

    void swap() {
        std::swap(data, data_out);
    }

    void clear() {
        if (hist)
            CUDA_CHECK(cudaMemset(hist, 0, 256*256* sizeof(uint32_t)));

        if (offset)
            CUDA_CHECK(cudaMemset(offset, 0, 256*256* sizeof(uint32_t)));
    }

    void clean() {
        if (data) {
            CUDA_CHECK(cudaFree(data));
            data = nullptr;
        }
        if (bucket) {
            CUDA_CHECK(cudaFree(bucket));
            bucket = nullptr;
        }
        if (bucket_row) {
            CUDA_CHECK(cudaFree(bucket_row));
            bucket_row = nullptr;
        }
        if (bucket_data) {
            free(bucket_data);
            bucket_data = nullptr;
        }

        if (hist) {
            CUDA_CHECK(cudaFree(hist));
            hist = nullptr;
        }
        if (offset) {
            CUDA_CHECK(cudaFree(offset));
            offset = nullptr;
        }
        if (data_out) {
            CUDA_CHECK(cudaFree(data_out));
            data_out = nullptr;
        }
    }

};

void setup_1(std::vector<uint32_t> &host_array, GpuLayout &layout) {
    layout.clean();
    
    layout.size = host_array.size();
    CUDA_CHECK(cudaMalloc((void**)&layout.data, layout.size*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(layout.data, host_array.data(), layout.size*sizeof(uint32_t), cudaMemcpyHostToDevice));
    layout.malloc_2D(256, layout.size/128);
    CUDA_CHECK(cudaMalloc((void**)&layout.bucket_row, 256*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(layout.bucket_row, 0, 256*sizeof(uint32_t)));
}

void setup_2(std::vector<uint32_t> &host_array, GpuLayout &layout) {
    layout.clean();

    layout.size = host_array.size();
    layout.blocksize = layout.size / 256;
    CUDA_CHECK(cudaMalloc((void**)&layout.data, host_array.size()*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&layout.data_out, host_array.size()*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(layout.data, host_array.data(), host_array.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&layout.hist, 256*256* sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&layout.offset, 256*256 * sizeof(uint32_t)));

    layout.clear();
}

class Tester {
private:
    std::vector<uint32_t> data1, data2;

    double get_time_in_seconds() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

    bool compare() {
        for (size_t i = 0; i < n; i++) {
            if (data1[i] != data2[i]) {
                printf("Mismatch at index %zu: %08x != %08x\n", i, data1[i], data2[i]);
                // 获取附近内容，方便debug
                size_t lo = (i > 8) ? i - 8 : 0;
                size_t hi = std::min(n, i + 9);
                printf("Context around mismatch (index:value)\n");
                for (size_t k = lo; k < hi; k++) {
                    printf("%6zu: %08x    %08x\n", k, data1[k], data2[k]);
                }
                return false;
            }
        }

        return true;
    }

    void bench() {
        // 100 runs for different buckets vs sort()
        // 16 2.433x
        // 256  4.615x
        // 65536  1.404x  
        for (size_t i = 0; i < 4; i++) {
            std::vector<std::vector<uint32_t>> bucket(256);

            for (size_t j = 0; j < n; j++) {
                bucket[(data1[j] >> i*8) & 0xff].push_back(data1[j]);
            }

            size_t index = 0;
            for (size_t j = 0; j < 256; j++) {
                for (size_t k = 0; k < bucket[j].size(); k++) {
                    data1[index++] = bucket[j][k];
                }
            }
        }
    
    }

public:
    double cpu_bench(std::function<void(uint32_t*, size_t)> func, const char *s) {
        //init();

        double current_time = get_time_in_seconds();
        func(data2.data(), n);
        double elapsed_time = get_time_in_seconds() - current_time;

        double speedup = bench_cpu_elapsed/elapsed_time;

        if (!compare()) {
            exit(1);
        }

        printf("CPU algorithm sort time: %.5f seconds\n", bench_cpu_elapsed);
        printf("CPU my %s sort time: %.5f seconds\n", s, elapsed_time);
        printf("speedup %.3fx\n\n", speedup);

        return speedup;
    }

    template<typename FS, typename FC>
    double gpu_bench(GpuLayout&layout, FS &setup, FC &func, const char *s) {
        // init();
        setup(data2, layout);

        double current_time = get_time_in_seconds();
        func(layout);
        double elapsed_time = get_time_in_seconds() - current_time;

        CUDA_CHECK(cudaMemcpy(data2.data(), layout.data, n*sizeof(uint32_t), cudaMemcpyDeviceToHost));

        double speedup = bench_cpu_elapsed/elapsed_time;

        if (!compare()) {
            exit(1);
        }

        printf("CPU algorithm sort time: %.5f seconds\n", bench_cpu_elapsed);
        printf("GPU my %s sort time: %.5f seconds\n", s, elapsed_time);
        printf("speedup %.3fx\n\n", speedup);

        return speedup;
    }

    // 随机初始化
    void init() {
        uint32_t random;

        data1.resize(n);
        data2.resize(n);
        for (size_t i = 0; i < n; i++) {
            random = rand() & 0xFFFF;
            random = (random << 16) + (rand() & 0xFFFF);
            data1[i] = random;
            data2[i] = data1[i];
        }

        double current_time = get_time_in_seconds();
        bench();
        bench_cpu_elapsed = get_time_in_seconds() - current_time;
    }

    Tester(size_t size) : n(size) {}

    ~Tester() {}

    size_t n;
    double bench_cpu_elapsed;
};

