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

__global__ void assembleGlobalMatrixBatched_kernel(cuDoubleComplex ** __restrict__ const d_ptr_A_batch, const cuDoubleComplex * __restrict__ const d_ptr_K,
                                                    const cuDoubleComplex * __restrict__ const d_ptr_M, const int nnz_sub, const int * __restrict__ const freq_square,
                                                    const int batchSize)
{
    // Thread indices
    int idx_thread_global = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_thread_freq   = (int)idx_thread_global/nnz_sub;
    int idx_thread_K_M    = idx_thread_global - idx_thread_freq*nnz_sub;
    // Total size of array batch
    int nnz_batch = nnz_sub * batchSize;
    //extern __shared__ int freq_squared_shared[];

    if (idx_thread_global < nnz_batch){
        //freq_shared[idx_thread_freq] = freq_square[idx_thread_freq];
        const cuDoubleComplex k = d_ptr_K[idx_thread_K_M];
        cuDoubleComplex A = d_ptr_M[idx_thread_K_M];
        A.x *= -freq_square[idx_thread_freq];
        A.y *= -freq_square[idx_thread_freq];
        //A.x *= -freq_squared_shared[idx_thread_freq];
        //A.y *= -freq_squared_shared[idx_thread_freq];
        A.x += k.x;
        A.y += k.y;
        d_ptr_A_batch[idx_thread_freq][idx_thread_K_M] = A;
    }
}

__global__ void constructMatricesBatched_kernel(cuDoubleComplex * __restrict__ d_ptr_B, cuDoubleComplex * __restrict__ d_ptr_C,
                                                cuDoubleComplex ** __restrict__ d_ptr_B_batch, cuDoubleComplex ** __restrict__ d_ptr_C_batch,
                                                const int nnz_sub_B, const int batchSize)
{
    // Thread indices
    int idx_thread_global = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_thread_freq   = (int)idx_thread_global/nnz_sub_B;
    int idx_thread_B_C    = idx_thread_global - idx_thread_freq*nnz_sub_B;
    // Total size of array batch
    int nnz_batch = nnz_sub_B * batchSize;

    if (idx_thread_global < nnz_batch){
        const cuDoubleComplex b = d_ptr_B[idx_thread_B_C];
        const cuDoubleComplex c = d_ptr_C[idx_thread_B_C];
        d_ptr_B_batch[idx_thread_freq][idx_thread_B_C] = b;
        d_ptr_C_batch[idx_thread_freq][idx_thread_B_C] = c;
    }
}

// Assembles global matrix for batched execution
void assembly::assembleGlobalMatrixBatched(cudaStream_t stream, cuDoubleComplex **d_ptr_A_batch,
                                           cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M,
                                           const int nnz_sub, const int *freq_square, const int batchSize)
{
    constexpr int block = 1024;                         // Number of threads per block
    int grid = (int)(nnz_sub*batchSize/block) + 1;      // Number of blocks per grid (sufficient for a grid to cover nnz_sub*batchSize)
    //size_t shared_memory_size = batchSize*sizeof(int);  // Size of shared memory
    //assembleGlobalMatrixBatched_kernel <<< grid, block, shared_memory_size, stream >>> (d_ptr_A_batch, d_ptr_K, d_ptr_M, nnz_sub, freq_square, batchSize);
    assembleGlobalMatrixBatched_kernel <<< grid, block, 0, stream >>> (d_ptr_A_batch, d_ptr_K, d_ptr_M, nnz_sub, freq_square, batchSize);
    cudaError_t cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
}

// Constructs matrices needed for Interface Jacobian in batch
void assembly::constructMatricesBatched(cudaStream_t stream, cuDoubleComplex *d_ptr_B, cuDoubleComplex *d_ptr_C,
                                        cuDoubleComplex **d_ptr_B_batch, cuDoubleComplex **d_ptr_C_batch,
                                        const int nnz_sub_B, const int batchSize)
{
    constexpr int block = 1024;
    int grid = (int)(nnz_sub_B*batchSize/block) + 1;
    constructMatricesBatched_kernel <<< grid, block, 0, stream >>> (d_ptr_B, d_ptr_C, d_ptr_B_batch, d_ptr_C_batch, nnz_sub_B, batchSize);
    cudaError_t cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
}
