// MKL
#include <mkl.h>

// Header Files
#include "assembly.hpp"

// Assembles global matrix ( A = K - f^2*M_tilde )
void assembly::assembleGlobalMatrix(MKL_Complex16 *A, MKL_Complex16 *K, MKL_Complex16 *M, int mat_shift, int nnz, MKL_Complex16 one, double freq_square){
    cblas_zcopy(nnz, M, 1, A + mat_shift, 1);
    cblas_zdscal(nnz, freq_square, A + mat_shift, 1);
    cblas_zaxpy(nnz, &one, K, 1, A + mat_shift, 1);
}
