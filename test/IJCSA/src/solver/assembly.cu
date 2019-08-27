/*  Copyright &copy; 2019, Stefan Sicklinger, Munich
*
*  All rights reserved.
*
*  This file is part of STACCATO.
*
*  STACCATO is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  STACCATO is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with STACCATO.  If not, see http://www.gnu.org/licenses/.
*/
/*************************************************************************************************
* \file assembly.cu
* Written by Ji-Ho Yang
* This file contains CUDA kernels for matrix/vector assembly
* \date 7/12/2019
**************************************************************************************************/

// Libraries
#include <iostream>
#include <cassert>

#include <stdio.h>

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
                                                    const cuDoubleComplex * __restrict__ const d_ptr_M, const cuDoubleComplex * __restrict__ const d_ptr_D,
                                                    const int nnz_A_sub, const int * __restrict__ const freq, const int * __restrict__ const freq_square,
                                                    const int batchSize)
{
    // Thread indices
    int idx_thread_global  = threadIdx.x + blockDim.x * blockIdx.x;			 // Unique thread ID of all the threads in the grid
    int idx_thread_freq    = (int)idx_thread_global/nnz_A_sub;				 // Assigns the frequency which this thread is used for
    int idx_thread_element = idx_thread_global - idx_thread_freq*nnz_A_sub;  // Assigns the array index which this thread is used for

    // Total size of array batch
    int nnz_batch = nnz_A_sub * batchSize;
    //extern __shared__ int freq_squared_shared[];

    if (idx_thread_global < nnz_batch){
        //freq_shared[idx_thread_freq] = freq_square[idx_thread_freq];
        const cuDoubleComplex k = d_ptr_K[idx_thread_element];
        cuDoubleComplex A = d_ptr_M[idx_thread_element];
        cuDoubleComplex D = d_ptr_D[idx_thread_element];
        A.x *= -freq_square[idx_thread_freq];
        A.y *= -freq_square[idx_thread_freq];
        D.x *= freq[idx_thread_freq];
        D.y *= freq[idx_thread_freq];
        //A.x *= -freq_squared_shared[idx_thread_freq];
        //A.y *= -freq_squared_shared[idx_thread_freq];
        A.x += k.x;
        A.y += k.y;
        A.x -= D.y;
        A.y += D.x;
        d_ptr_A_batch[idx_thread_freq][idx_thread_element] = A;
    }
}

// Assembles global matrix for batched execution
void assembly::assembleGlobalMatrixBatched(cudaStream_t stream, cuDoubleComplex **d_ptr_A_batch,
                                           cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M, cuDoubleComplex *d_ptr_D,
                                           const int nnz_A_sub, const int *freq, const int *freq_square, const int batchSize)
{
    constexpr int block = 1024;                           // Number of threads per block
    int grid = (int)(nnz_A_sub*batchSize/block) + 1;      // Number of blocks per grid (sufficient for a grid to cover nnz_sub*batchSize)
    size_t shared_memory_size = batchSize*sizeof(int);    // Size of shared memory
    //assembleGlobalMatrixBatched_kernel <<< grid, block, shared_memory_size, stream >>> (d_ptr_A_batch, d_ptr_K, d_ptr_M, nnz_sub, freq_square, batchSize);
    assembleGlobalMatrixBatched_kernel <<< grid, block, 0, stream >>> (d_ptr_A_batch, d_ptr_K, d_ptr_M, d_ptr_D, nnz_A_sub, freq, freq_square, batchSize);
    cudaError_t cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
}

__global__ void constructMatricesBatched_kernel(cuDoubleComplex * __restrict__ d_ptr_mat,
                                                cuDoubleComplex ** __restrict__ d_ptr_mat_batch,
                                                const int nnz_sub_mat, const int batchSize)
{
    // Thread indices
    int idx_thread_global = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_thread_freq   = (int)idx_thread_global/nnz_sub_mat;
    int idx_thread_mat    = idx_thread_global - idx_thread_freq*nnz_sub_mat;
    // Total size of array batch
    int nnz_batch = nnz_sub_mat * batchSize;

    if (idx_thread_global < nnz_batch){
        const cuDoubleComplex val = d_ptr_mat[idx_thread_mat];
        d_ptr_mat_batch[idx_thread_freq][idx_thread_mat] = val;
    }
}

// Constructs matrices needed for Interface Jacobian in batch
void assembly::constructMatricesBatched(cudaStream_t stream, cuDoubleComplex *d_ptr_mat, cuDoubleComplex **d_ptr_mat_batch, const int nnz_sub_mat, const int batchSize){
    constexpr int block = 1024;
    int grid = (int)(nnz_sub_mat*batchSize/block) + 1;
    constructMatricesBatched_kernel <<< grid, block, 0, stream >>> (d_ptr_mat, d_ptr_mat_batch, nnz_sub_mat, batchSize);
    cudaError_t cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
}

__global__ void updateRhsBatched_kernel(cuDoubleComplex ** __restrict__ d_ptr_deltaC_batch,
                                        cuDoubleComplex ** __restrict__ d_ptr_rhs_batch,
                                        const int nnz_rhs_sub, const int batchSize)
{
    // Thread indices
    int idx_thread_global  = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_thread_freq    = (int)idx_thread_global/nnz_rhs_sub;
    int idx_thread_element = idx_thread_global - idx_thread_freq*nnz_rhs_sub;
    // Total size of array batch
    int nnz_batch = nnz_rhs_sub * batchSize;
    // Update RHS
    if (idx_thread_global < nnz_batch){
        const cuDoubleComplex deltaC = d_ptr_deltaC_batch[idx_thread_freq][idx_thread_element];
        cuDoubleComplex rhs          = d_ptr_rhs_batch[idx_thread_freq][idx_thread_element];
        rhs.x -= deltaC.x;
        rhs.y -= deltaC.y;
        d_ptr_rhs_batch[idx_thread_freq][idx_thread_element] = rhs;
    }
}

// Updates RHS in batch
void assembly::updateRhsBatched(cudaStream_t stream, cuDoubleComplex **d_ptr_deltaC_batch, cuDoubleComplex **d_ptr_rhs_batch, const int nnz_rhs_sub, const int batchSize){
    //std::cout << "TID = " << tid << " executing updateRhsBatched for idx = " << idx << std::endl;
    constexpr int block = 1024;
    int grid = (int)(nnz_rhs_sub*batchSize/block) + 1;
    updateRhsBatched_kernel <<< grid, block, 0, stream >>> (d_ptr_deltaC_batch, d_ptr_rhs_batch, nnz_rhs_sub, batchSize);
    cudaError_t cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
}

__global__ void collocateFinalSolution_kernel(cuDoubleComplex ** __restrict__ d_ptr_Y_separate_batch,
                                              cuDoubleComplex ** __restrict__ d_ptr_Y_combined_batch,
                                              const int nnz_Y_sub, const int batchSize)
{
    // Thread indices
    int idx_thread_global  = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_thread_freq    = (int)idx_thread_global/nnz_Y_sub;
    int idx_thread_element = idx_thread_global - idx_thread_freq*nnz_Y_sub;
    // Total size of array batch
    int nnz_batch = nnz_Y_sub * batchSize;
    // Collocate values
    if (idx_thread_global < nnz_batch){
        const cuDoubleComplex val = d_ptr_Y_separate_batch[idx_thread_freq][idx_thread_element];
        d_ptr_Y_combined_batch[idx_thread_freq][idx_thread_element] = val;
    }
}

// Collocate Y_sim or exp to global Y
void assembly::collocateFinalSolution(cudaStream_t stream, cuDoubleComplex **d_ptr_Y_separate_batch, cuDoubleComplex **d_ptr_Y_combined_batch, const int nnz_Y_sub, const int batchSize){
    constexpr int block = 1024;
    int grid = (int)(nnz_Y_sub*batchSize/block) + 1;
    collocateFinalSolution_kernel <<< grid, block, 0, stream >>> (d_ptr_Y_separate_batch, d_ptr_Y_combined_batch, nnz_Y_sub, batchSize);
    cudaError_t cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
}

__global__ void collocateFinalSolutionPostProcess_kernel(cuDoubleComplex ** __restrict__ d_ptr_Y_separate_batch,
                                                         cuDoubleComplex ** __restrict__ d_ptr_Y_combined_batch,
                                                         int *__restrict__ d_ptr_mapper,
                                                         const int nnz_Y_sub, const int batchSize)
{
    // Thread indices
    int idx_thread_global  = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_thread_freq    = (int)idx_thread_global/nnz_Y_sub;
    int idx_thread_element = idx_thread_global - idx_thread_freq*nnz_Y_sub;
    // Total size of array batch
    int nnz_batch = nnz_Y_sub * batchSize;
    // Collocate values
    if (idx_thread_global < nnz_batch){
        const int idx_mapper = d_ptr_mapper[idx_thread_element];
        const cuDoubleComplex val = d_ptr_Y_separate_batch[idx_thread_freq][idx_thread_element];
        d_ptr_Y_combined_batch[idx_thread_freq][idx_mapper] = val;
    }
}

// Collocate Y_sim or exp to global total Y including internal DOFs
void assembly::collocateFinalSolutionPostProcess(cudaStream_t stream, cuDoubleComplex **d_ptr_Y_separate_batch, cuDoubleComplex **d_ptr_Y_combined_batch,
                                                 int *d_ptr_mapper, const int nnz_Y_sub, const int batchSize){
    constexpr int block = 1024;
    int grid = (int)(nnz_Y_sub*batchSize/block) + 1;
    collocateFinalSolutionPostProcess_kernel <<< grid, block, 0, stream >>> (d_ptr_Y_separate_batch, d_ptr_Y_combined_batch, d_ptr_mapper, nnz_Y_sub, batchSize);
    cudaError_t cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
}
