// Libraries
#include <iostream>
#include <cassert>

// cuComplex
#include <cuComplex.h>

// cuBLAS
#include <cublas_v2.h>

// Header Files
#include "assembly.cuh"
#include "../helper/helper.cuh"

// Namespace
using namespace staccato;

__global__ void assembleGlobalMatrix4Batched_kernel(cuDoubleComplex ** __restrict__ const d_ptr_A_batch, const cuDoubleComplex * __restrict__ const d_ptr_K,
                                                    const cuDoubleComplex * __restrict__ const d_ptr_M, const int nnz_sub, const int * __restrict__ const freq_square,
                                                    const int batchSize, const int subcomponents, int num_blocks)
{
    // Thread indices
    int idx_thread_global = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_thread_local  = threadIdx.x;
    // Block information
    int idx_block = blockIdx.x;
    int block_size = blockDim.x;
    // Total size of array batch
    int nnz_batch = nnz_sub * batchSize;

    //if (idx_thread_global < nnz_batch){
    if (idx_thread_global < 1){
        const cuDoubleComplex k = d_ptr_K[idx_thread_global];
        cuDoubleComplex A = d_ptr_M[idx_thread_global];
        A.x *= freq_square[idx_thread_global];
        A.y *= freq_square[idx_thread_global];
        A.x += k.x;
        A.y += k.y;
        d_ptr_A_batch[idx_thread_global][idx_thread_global] = A;
    }

/*
    if (idx < nnz){
        const cuDoubleComplex k = d_ptr_K[idx_thread_local];
        cuDoubleComplex A = d_ptr_M[idx_thread_local];
        A.x *= freq_square[idx_block];
        A.y *= freq_square[idx_block];
        A.x += k.x;
        A.y += k.y;
    }
*/
}

// Assembles global matrix for batched execution
void assembly::assembleGlobalMatrixBatched(cudaStream_t stream, cuDoubleComplex **d_ptr_A_batch,
                                           cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M,
                                           const int nnz_sub, const int *freq_square, const int batchSize, const int subComponents){
    constexpr int block = 1024;                         // Number of threads per block
    int grid = (int)(nnz_sub*batchSize/block) + 1;      // Number of blocks per grid (sufficient for a grid to cover nnz_sub*batchSize)
    //int grid = batchSize;
    assembleGlobalMatrix4Batched_kernel <<< grid, block, 0, stream >>> (d_ptr_A_batch, d_ptr_K, d_ptr_M, nnz_sub, freq_square, batchSize, subComponents, grid);
    cudaError_t cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
}
