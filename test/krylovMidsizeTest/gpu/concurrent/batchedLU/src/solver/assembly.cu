// Libraries
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <algorithm>

// cuComplex
#include <cuComplex.h>

// cuBLAS
#include <cublas_v2.h>

// Header Files
#include "assembly.cuh"
#include "../helper/helper.cuh"

__global__ void assembleGlobalMatrixBatched_kernel(cuDoubleComplex ** __restrict__ const d_ptr_A, const cuDoubleComplex * __restrict__ const d_ptr_K,
                                                   const cuDoubleComplex * __restrict__ const d_ptr_M, const int nnz_sub, const int *freq_square,
                                                   const int batchSize, const int num_matrix, const int num_blocks){
    // Get thread index
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_thread_local = threadIdx.x;
    int idx_block = blockIdx.x;
    int freq_shift = 0;

    printf("%i", idx_thread_local);

    if (idx_thread_local < nnz_sub){
        for (size_t i = 0; i < (int)batchSize/num_blocks; ++i){
            const cuDoubleComplex k = d_ptr_K[idx_thread_local];
            cuDoubleComplex A = d_ptr_M[idx_thread_local];
            A.x *= freq_square[idx_block + freq_shift];
            A.y *= freq_square[idx_block + freq_shift];
            A.x += k.x;
            A.y += k.y;
            freq_shift += num_blocks;
            d_ptr_A[idx_block + freq_shift][idx_thread_local] = A;

            printf("hello");
        }
    }
}

// Assembles global matrix for batched execution
void assembly::assembleGlobalMatrixBatched(cudaStream_t stream, cuDoubleComplex **d_ptr_A,
                                           cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M,
                                           int nnz_sub, int *freq_square, const int batchSize, const int num_matrix){
    constexpr int block = 320;                              // Number of threads per block
    //int grid = std::min(32, (nnz_sub*batchSize)/block + 1); // Number of blocks per grid
    int grid = (nnz_sub*batchSize)/block + 1; // Number of blocks per grid
    assembleGlobalMatrixBatched_kernel <<< grid, block >>> (d_ptr_A, d_ptr_K, d_ptr_M, nnz_sub, freq_square, batchSize, num_matrix, grid);
    cudaError_t cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
}
