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
* \file assembly.cuh
* Written by Ji-Ho Yang
* This file contains CUDA kernels for matrix/vector assembly
* \date 7/12/2019
**************************************************************************************************/

#pragma once

namespace staccato{
    namespace assembly{
        void assembleGlobalMatrixBatched(cudaStream_t stream, cuDoubleComplex **d_ptr_A_batch,
                                         cuDoubleComplex *d_ptr_K, cuDoubleComplex *d_ptr_M, cuDoubleComplex *d_ptr_D,
                                         const int nnz_A_sub, const int *freq, const int *freq_square, const int batchSize);
        void constructMatricesBatched(cudaStream_t stream, cuDoubleComplex *d_ptr_mat, cuDoubleComplex **d_ptr_mat_batch, const int nnz_sub_mat, const int batchSize);
        void updateRhsBatched(cudaStream_t stream, cuDoubleComplex **d_ptr_deltaC_batch, cuDoubleComplex **d_ptr_rhs_batch, const int nnz_rhs_sub, const int batchSize);
        void collocateFinalSolution(cudaStream_t stream, cuDoubleComplex **d_ptr_Y_separate_batch, cuDoubleComplex **d_ptr_Y_combined_batch, const int nnz_Y_sub, const int batchSize);
        void collocateFinalSolutionPostProcess(cudaStream_t stream, cuDoubleComplex **d_ptr_Y_separate_batch, cuDoubleComplex **d_ptr_Y_combined_batch,
                                               int *d_ptr_mapper, const int nnz_Y_sub, const int batchSize);

    } // namespace::staccato
} // namespace::assembly
