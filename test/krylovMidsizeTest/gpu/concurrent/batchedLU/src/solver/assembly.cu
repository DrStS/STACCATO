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

__global__ void assembleGlobalMatrix4Batched_kernel(
    cuDoubleComplex * __restrict__ const d_ptr_A,
    const cuDoubleComplex * __restrict__ const d_ptr_K,
    const cuDoubleComplex * __restrict__ const d_ptr_M,
    const int nnz_sub, const double freq_square)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < nnz_sub )
    {
        const cuDoubleComplex k = d_ptr_K[idx];
        cuDoubleComplex A = d_ptr_M[idx];
        A.x *= freq_square;
        A.y *= freq_square;
        A.x += k.x;
        A.y += k.y;
        d_ptr_A[idx] = A;
    }
}

// Assembles global matrix for batched execution
void assembly::assembleGlobalMatrixBatched(cudaStream_t stream, cuDoubleComplex *d_ptr_A,
                                           cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M,
                                           int nnz_sub, double freq_square){
    constexpr int block_size = 128;
    assembleGlobalMatrix4Batched_kernel<<<((nnz_sub-1)/block_size)+1,block_size,0,stream>>>(d_ptr_A,d_ptr_K,d_ptr_M,nnz_sub,freq_square);
    cudaError_t cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
}
