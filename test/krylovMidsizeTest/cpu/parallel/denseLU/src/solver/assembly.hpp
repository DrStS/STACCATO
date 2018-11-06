#pragma once

namespace assembly{
    void assembleGlobalMatrix(MKL_Complex16 *A, MKL_Complex16 *K, MKL_Complex16 *M, int mat_shift, int nnz, MKL_Complex16 one, double freq_square);
}
