#pragma once

namespace staccato{
    namespace assembly{
        void assembleGlobalMatrixBatched(cudaStream_t stream, cuDoubleComplex **d_ptr_A_batch,
                                         cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M,
                                         const int nnz_sub, const int *freq_square, const int batchSize);
        void constructMatricesBatched(cudaStream_t stream, cuDoubleComplex *d_ptr_B, cuDoubleComplex *d_ptr_C,
                                      cuDoubleComplex **d_ptr_B_batch, cuDoubleComplex **d_ptr_C_batch,
                                      const int nnz_sub_B, const int batchSize);
    } // namespace::staccato
} // namespace::assembly
