#pragma once

namespace assembly{
    void assembleGlobalMatrixBatched(cudaStream_t stream, cuDoubleComplex *d_ptr_A[],
                                     cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M,
                                     int nnz_sub, int *freq_square, const int batchSize, const int num_matrix);
}
