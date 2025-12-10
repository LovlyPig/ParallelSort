#include <iostream>
#include <string>
#include <iomanip>
#include <memory>

#include "cpu_support.h"

void radix_sort_tbb(uint32_t *a, size_t n) {

    const size_t buckets = 256, num = 4;

    std::unique_ptr<uint32_t[]> temp(new uint32_t[n]);
    uint32_t *current_in = a;
    uint32_t *current_out = temp.get();

    for (size_t i = 0; i < num; i++) {
        const size_t shift = i * 8;

        // 将数组划分为多个块以进行并行处理
        size_t concurrency = tbb::this_task_arena::max_concurrency();
        size_t num_blocks = std::min(n, std::max<size_t>(1, concurrency * 4));
        size_t block_size = (n + num_blocks - 1) / num_blocks;

        std::vector<uint32_t> hist((size_t)num_blocks * buckets, 0);

        // 计算每个块中的数据分配到哪些桶
        tbb::parallel_for((size_t)0, num_blocks, [&](size_t bi) {
            size_t start = bi * block_size;
            size_t end = std::min(n, start + block_size);
            uint32_t *h = &hist[bi * buckets];
            for (size_t i = start; i < end; i++) {
                uint32_t bucket = (current_in[i] >> shift) & 0xff;
                h[bucket]++;
            }
        });

        // 计算每个桶的前缀和获得每个桶的起始位置 
        // 100 runs 3.903x -> 3.957x 相比非并行快一点点
        std::vector<uint32_t> base(buckets, 0);
        tbb::parallel_for((size_t)0, buckets, [&](size_t bi) {
            uint32_t sum = 0;
            for (size_t i = 0; i < num_blocks; i++) {
                sum += hist[i * buckets + bi];
            }
            base[bi] = sum;
        });
        uint32_t pre = 0;
        for (size_t bucket = 0; bucket < buckets; bucket++) {
            uint32_t temp = base[bucket];
            base[bucket] = pre;
            pre += temp;
        }

        // 计算每个块的桶起始偏移
        std::vector<uint32_t> offsets((size_t)num_blocks * buckets);
        tbb::parallel_for((size_t)0, buckets, [&](size_t bi) {
            uint32_t off = base[bi];
            for (size_t i = 0; i < num_blocks; i++) {
                offsets[i*buckets + bi] = off;
                off += hist[i*buckets + bi];
            }
        });

        // 将数据从对应的桶中写到输出数组
        tbb::parallel_for((size_t)0, num_blocks, [&](size_t bi) {
            size_t start = bi * block_size;
            size_t end = std::min(n, start + block_size);
            std::vector<uint32_t> cur_pos(buckets);
            uint32_t *off_pos = &offsets[bi * buckets];
            // 拷贝到局部变量，减少竞争
            for (size_t i = 0; i < buckets; i++) 
                cur_pos[i] = off_pos[i];

            for (size_t i = start; i < end; i++) {
                uint32_t bucket = (current_in[i] >> shift) & 0xff;
                uint32_t write_pos = cur_pos[bucket]++;
                current_out[write_pos] = current_in[i];
            }
        });
        // 交换
        std::swap(current_in, current_out);
    }
    
}

// 无优化串行GPU
__global__ void gpu_radix_sort(uint32_t **bucket, uint32_t *row, uint32_t *a, size_t n) {
    
    #pragma unroll
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 256; j++) {
            row[j] = 0;
        }

        for (size_t j = 0; j < n; j++) {
            uint32_t index = (a[j] >> (i * 8)) & 0xff;
            bucket[index][row[index]++] = a[j];    
        }

        size_t pos = 0;
        #pragma unroll
        for (size_t j = 0; j < 256; j++) {
            for (size_t k = 0; k < row[j]; k++) {
                a[pos++] = bucket[j][k];
            }
        }
    }

}

__global__ void gpu_radix_sort_opt(uint32_t *a_in, uint32_t *a_out, uint32_t *hist, uint32_t *offset, const size_t n, const size_t shift, const size_t blocksize) {
    int tid = threadIdx.x;
    if (tid >= 256) return;

    size_t start = tid*blocksize, end = min(n, start + blocksize);
    #pragma unroll
    for (size_t i = start; i < end; i++) {
        uint32_t index = (a_in[i] >> shift) & 0xff;
        atomicAdd(&hist[tid * 256 + index], 1);
    }
    __syncthreads();

    __shared__ uint32_t temp[256];

    #pragma unroll
    for (size_t i = 0; i < 256; i++) 
        temp[tid] += hist[i * 256 + tid];
    __syncthreads();

    //temp[tid] = __shfl_up_sync(0xffffffff, temp[tid], 1);
    if (tid == 0) [[unlikely]] {
        //temp[tid] = 0;
        uint32_t pre = 0;
        #pragma unroll
        for (size_t i = 0; i < 256; i++) {
            uint32_t val = temp[i];
            temp[i] = pre;
            pre += val;
        }
    }
    __syncthreads();
    // for (int offset = 1; offset <= tid; offset *= 2) {
    //     printf("offset %d  tid %d\n", offset, tid);
    //     uint32_t val = __shfl_up_sync(0xffffffff, temp[tid], offset);
    //     temp[tid] += val;
    // }

    uint32_t off = temp[tid];
    #pragma unroll
    for (size_t i = 0; i < 256; i++) {
        offset[i * 256 + tid] = off;
        off += hist[i * 256 + tid];
    }
    __syncthreads();

    #pragma unroll
    for (size_t i = start; i < end; i++) {
        uint32_t *o = &offset[tid*256];
        uint32_t index = (a_in[i] >> shift) & 0xff;
        uint32_t write_pos = atomicAdd(&o[index], 1);
        a_out[write_pos] = a_in[i];
    }

}

void call_gpu_opt(GpuLayout &layout) {
    size_t n = layout.size;

    for (size_t i = 0; i < 4; i++) {
        gpu_radix_sort_opt<<<1,256>>>(layout.data, layout.data_out, layout.hist, layout.offset, n, i*8, layout.blocksize);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
        layout.swap();
        layout.clear();
    }

}

void call_gpu(GpuLayout &layout) {
    size_t n = layout.size;

    gpu_radix_sort<<<1,1>>>(layout.bucket, layout.bucket_row, layout.data, n);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char* argv[]) {

    size_t n = size_t(1) << 20;

    int count = 1;
    if (argc > 1) count = atoi(argv[1]);

    Tester tester(n);
    GpuLayout layout;
    double speedup_avg1 = 0.0, speedup_avg2 = 0.0;

    for (int i = 0; i < count; i++) {
        tester.init();
        speedup_avg1 += tester.cpu_bench(radix_sort_tbb, "radix_sort_tbb");
        speedup_avg2 += tester.gpu_bench(layout, setup_2, call_gpu_opt, "gpu_radix_sort_opt");
    }
    speedup_avg1 /= count;
    speedup_avg2 /= count;

    printf("radix_sort_tbb average speedup over %d runs: %.3fx\n", count, speedup_avg1);
    printf("gpu_radix_sort average speedup over %d runs: %.3fx\n", count, speedup_avg2);

    return 0;
}   