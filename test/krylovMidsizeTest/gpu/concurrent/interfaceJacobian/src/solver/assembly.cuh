#pragma once

namespace staccato{
    namespace assembly{
        void assembleGlobalMatrixBatched(cudaStream_t stream, cuDoubleComplex *d_ptr_A,
                                         cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M,
                                         int nnz_sub, double freq_square);
    }
}
