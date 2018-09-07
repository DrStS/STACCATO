#pragma once

namespace assembly{
    void assembleGlobalMatrix(int tid, cudaStream_t stream, cublasStatus_t cublasStatus, cublasHandle_t cublasHandle,
                                cuDoubleComplex *d_ptr_A, cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M,
                                int nnz, int mat_shift, cuDoubleComplex one, double freq_square);
}
