#include <iostream>
#include <string>
#include <iomanip>
#include <memory>

#include "cpu_support.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

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
    temp[tid] = 0;

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

    // thrust::device_ptr<uint32_t> d_ptr(layout.data);
    // thrust::sort(d_ptr, d_ptr + n);

}

__global__ void hist(uint32_t *a, uint32_t *hist, size_t n, size_t shift, size_t blocksize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int block_id = tid / (int)blocksize;

    uint32_t index = a[tid] >> shift & 0xff;
    atomicAdd(&hist[block_id * 256 + index], 1);
}

__global__ void prefix_sum(uint32_t *hist, uint32_t *offset) {
    int tid = threadIdx.x;
    if (tid >= 256) return;

    __shared__ uint32_t temp[256];
    temp[tid] = 0;

    #pragma unroll
    for (size_t i = 0; i < 256; i++) 
        temp[tid] += hist[i * 256 + tid];
    __syncthreads();

    if (tid == 0) [[unlikely]] {
        uint32_t pre = 0;
        #pragma unroll
        for (size_t i = 0; i < 256; i++) {
            uint32_t val = temp[i];
            temp[i] = pre;
            pre += val;
        }
    }
    __syncthreads();

    #pragma unroll
    for (size_t i = 0; i < 256; i++) {
        offset[i * 256 + tid] = temp[tid];
        temp[tid] += hist[i * 256 + tid];
    }
}

__global__ void scatter(uint32_t *a_in, uint32_t *a_out, uint32_t *offset, size_t n, size_t shift, size_t blocksize) {
    int tid = threadIdx.x;
    if (tid >= 256) return;
    size_t start = tid*blocksize, end = min(n, start + blocksize);

    #pragma unroll
    for (size_t i = start; i < end; i++) {
        uint32_t *o = &offset[tid*256];
        uint32_t index = (a_in[i] >> shift) & 0xff;
        uint32_t write_pos = atomicAdd(&o[index], 1);
        a_out[write_pos] = a_in[i];
    }
}


void call_gpu_opt_2(GpuLayout &layout) {
    size_t n = layout.size;

    for (size_t i = 0; i < 4; i++) {
        hist<<<n/1024, 1024>>>(layout.data, layout.hist, n, i*8, layout.blocksize);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        prefix_sum<<<1,256>>>(layout.hist, layout.offset);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        scatter<<<1, 256>>>(layout.data, layout.data_out, layout.offset, n, i*8, layout.blocksize);
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

__device__ void swap(uint32_t &a, uint32_t &b) {
    uint32_t t = a;
    a = b;
    b = t;
}

__global__ void bitonic_sort_kernel(uint32_t *data, size_t n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = tid + bid * blockDim.x;

    if (global_tid >= n) return;

    __shared__ uint32_t sdata[1024]; // assume blockDim.x = 1024

    sdata[tid] = data[global_tid];
    __syncthreads();

    // Bitonic sort within the block
    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (sdata[tid] > sdata[ixj]) swap(sdata[tid], sdata[ixj]);
                } else {
                    if (sdata[tid] < sdata[ixj]) swap(sdata[tid], sdata[ixj]);
                }
            }
            __syncthreads();
        }
    }

    data[global_tid] = sdata[tid];
}

void call_bitonic_sort(GpuLayout &layout) {
    size_t n = layout.size;
    dim3 block(1024);
    dim3 grid((n + 1023) / 1024);

    bitonic_sort_kernel<<<grid, block>>>(layout.data, n);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char* argv[]) {

    size_t n = size_t(1) << 22; 

    int count = 1;
    if (argc > 1) count = atoi(argv[1]);

    Tester tester(n);
    GpuLayout layout;
    double speedup_avg1 = 0.0, speedup_avg2 = 0.0, speedup_avg3 = 0.0, speedup_avg4 = 0.0, speedup_avg5 = 0.0, speedup_avg6 = 0.0;

    for (int i = 0; i < count; i++) {
        tester.init();
        speedup_avg1 += tester.cpu_bench(radix_sort_tbb, "radix_sort_tbb");
        //speedup_avg2 += tester.gpu_bench(layout, setup_1, call_gpu, "gpu_radix_sort");
        speedup_avg3 += tester.gpu_bench(layout, setup_2, call_gpu_opt, "gpu_radix_sort_opt");
        speedup_avg4 += tester.gpu_bench(layout, setup_2, call_gpu_opt_2, "gpu_radix_sort_opt_2");
        speedup_avg5 += tester.gpu_bench(layout, setup_1, call_bitonic_sort, "bitonic_sort");
    }
    speedup_avg1 /= count;
    //speedup_avg2 /= count;
    speedup_avg3 /= count;
    speedup_avg4 /= count;
    speedup_avg5 /= count;
    speedup_avg6 /= count;

    printf("radix_sort_tbb average speedup over %d runs: %.3fx\n", count, speedup_avg1);
    //printf("gpu_radix_sort average speedup over %d runs: %.3fx\n", count, speedup_avg2);
    printf("gpu_radix_sort_opt average speedup over %d runs: %.3fx\n", count, speedup_avg3);
    printf("gpu_radix_sort_opt_2 average speedup over %d runs: %.3fx\n", count, speedup_avg4);
    printf("bitonic_sort average speedup over %d runs: %.3fx\n", count, speedup_avg5);

    return 0;
}   


