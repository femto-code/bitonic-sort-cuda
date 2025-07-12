#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 1024

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void bitonicSortShared(int* arr) {
    __shared__ int s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    s_data[tid] = arr[gid];
    __syncthreads();

    for (int k = 2; k <= BLOCK_SIZE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool ascending = (tid & k) == 0;
                int val_i = s_data[tid];
                int val_j = s_data[ixj];

                if ((ascending && val_i > val_j) || (!ascending && val_i < val_j)) {
                    s_data[tid] = val_j;
                    s_data[ixj] = val_i;
                }
            }
            __syncthreads();
        }
    }

    arr[gid] = s_data[tid];
}

__global__ void bitonicMergeKernel(int* arr, int j, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int ixj = i ^ j;
    if (ixj > i && ixj < n) {
        bool ascending = (i & k) == 0;
        int val_i = arr[i];
        int val_j = arr[ixj];

        if ((ascending && val_i > val_j) || (!ascending && val_i < val_j)) {
            arr[i] = val_j;
            arr[ixj] = val_i;
        }
    }
}

// Utility function to run the full sort
void bitonicSort(int* h_arr, int n) {

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // printf("GPU: %s, SM count: %d, Shared memory per block: %zu\n", prop.name, prop.multiProcessorCount, prop.sharedMemPerBlock);

    // int minGridSize, blockSize;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bitonicSortShared, 0, 0);
    // printf("Suggested block size: %d\n", blockSize);

    int* d_arr;
    size_t bytes = n * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_arr, bytes));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));
    
    // cudaMalloc(&d_arr, bytes);
    // cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
    // Step 1: Shared memory block-wise sort
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // printf("numBlocks=%d BLOCK_SIZE=%d\n", numBlocks, BLOCK_SIZE);
    bitonicSortShared<<<numBlocks, BLOCK_SIZE>>>(d_arr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // bitonicSortShared<<<numBlocks, BLOCK_SIZE>>>(d_arr);
    // cudaDeviceSynchronize();

    // Step 2: Global merge steps
    int threads = BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;

    for (int k = 2; k <= n; k <<= 1) { // k = 2, 4, 8, ...
        for (int j = k >> 1; j > 0; j >>= 1) { // j = k/2, k/4, ..., 1
            bitonicMergeKernel<<<blocks, threads>>>(d_arr, j, k, n);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}