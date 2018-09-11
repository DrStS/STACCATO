// Libraries
#include <iostream>
#include <cassert>

// cuComplex
#include <cuComplex.h>

// cuBLAS
#include <cublas_v2.h>

// Header Files
#include "assembly.cuh"

// Assembles global matrix ( A = K - f^2*M_tilde )
void assembly::assembleGlobalMatrix(cudaStream_t stream, cublasStatus_t cublasStatus, cublasHandle_t cublasHandle,
                                    cuDoubleComplex *d_ptr_A, cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M,
                                    int nnz, int mat_shift, cuDoubleComplex one, double freq_square){
    // Copy M to A
    cublasSetStream(cublasHandle, stream);
    cublasStatus = cublasZcopy(cublasHandle, nnz, d_ptr_M, 1, d_ptr_A + mat_shift, 1);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    // Scale M by f^2
    cublasSetStream(cublasHandle, stream);
    cublasStatus = cublasZdscal(cublasHandle, nnz, &freq_square, d_ptr_A + mat_shift, 1);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    // Sum A with K
    cublasSetStream(cublasHandle, stream);
    cublasStatus = cublasZaxpy(cublasHandle, nnz, &one, d_ptr_K, 1, d_ptr_A + mat_shift, 1);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
}

// Assembles global matrix for batched execution
void assembly::assembleGlobalMatrix4Batched(cudaStream_t stream, cublasStatus_t cublasStatus, cublasHandle_t cublasHandle,
                                            cuDoubleComplex *d_ptr_A, cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M,
                                            int nnz_sub, cuDoubleComplex one, double freq_square){
    // Copy M to A
    cublasSetStream(cublasHandle, stream);
    cublasStatus = cublasZcopy(cublasHandle, nnz_sub, d_ptr_M, 1, d_ptr_A, 1);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    // Scale M by f^2
    cublasSetStream(cublasHandle, stream);
    cublasStatus = cublasZdscal(cublasHandle, nnz_sub, &freq_square, d_ptr_A, 1);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
    // Sum A with K
    cublasSetStream(cublasHandle, stream);
    cublasStatus = cublasZaxpy(cublasHandle, nnz_sub, &one, d_ptr_K, 1, d_ptr_A, 1);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
}
