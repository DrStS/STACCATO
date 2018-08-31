#pragma once

namespace assembly{
    void assembleGlobalMatrix(void *A, void *K, void *M, int mat_shift, int nnz, MKL_Complex16 one, double freq_square);
}
