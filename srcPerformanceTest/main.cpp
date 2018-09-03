/*           This file has been prepared for Doxygen automatic documentation generation.          */
/***********************************************************************************************//**
 * \mainpage
 * \section LICENSE
 *  Copyright &copy; 2018, Dr. Stefan Sicklinger, Munich \n
 *  All rights reserved. \n
 *
 *  This file is part of STACCATO.
 *
 *  STACCATO is free software: you can redistribute it and/or modify \n
 *  it under the terms of the GNU General Public License as published by \n
 *  the Free Software Foundation, either version 3 of the License, or \n
 *  (at your option) any later version. \n
 *
 *  STACCATO is distributed in the hope that it will be useful, \n
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of \n
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the \n
 *  GNU General Public License for more details. \n
 *
 *  You should have received a copy of the GNU General Public License \n
 *  along with STACCATO.  If not, see <a href="http://www.gnu.org/licenses/">http://www.gnu.org/licenses/</a>.
 *
 *  Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it with Intel Math Kernel Libraries(MKL)
 * (or a modified version of that library), containing parts covered by the terms of the license of the MKL,
 * the licensors of this Program grant you additional permission to convey the resulting work.
 *
 * \section DESCRIPTION
 *  This is the main file of STACCATO Performance Tests
 *
 *
 * \section COMPILATION
 *
 * \section HOWTO
 * Please find all further information on
 * <a href="https://github.com/DrStS/STACCATO">STACCATO Project</a>
 *
 *
 * <EM> Note: The Makefile suppresses per default all compile and linking command output to the terminal.
 *       You may enable this information by make VEREBOSE=1</EM>
 *
 *
 *
 **************************************************************************************************/
/***********************************************************************************************//**
 * \file main.cpp
 * This file holds the main function of STACCATO Performance Tests.
 * \author Jiho Yang
 * \date 8/21/2018
 * \version
 **************************************************************************************************/

// Libraries
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <omp.h>

// MKL
#include <mkl.h>

// Header Files
#include "config/config.hpp"
#include "io/io.hpp"
#include "helper/Timer.hpp"
#include "data/dataStructure.hpp"
#include "solver/assembly.hpp"

// Definitions
#define	PI	3.14159265359

//#define MKL_Complex16 std::complex<double>

int main(int argc, char *argv[]) {

    /*--------------------
    COMMAND LINE ARGUMENTS
    --------------------*/
    double freq_max;
    int mat_repetition, num_matrix;
    int tid, nt_mkl, nt;
    std::string parallel_mode, sparse_mode, arg_parallel, arg_sparse;
    // Configure test environment with command line arguments
    config::configureTest(argc, argv, freq_max, mat_repetition, num_matrix, tid, nt_mkl, nt, parallel_mode, sparse_mode, arg_parallel, arg_sparse);

#if defined(_WIN32) || defined(__WIN32__)
    std::string filePathPrefix = "C:/software/examples/";
#endif
#if defined(__linux__)
    std::string filePathPrefix = "/opt/software/examples/";
#endif

    /*----------------------
    Filatpaths and Filenames
    ----------------------*/
    // Vector of filepaths
    std::string filepath[2];
    filepath[0] = filePathPrefix + "MOR/r_approx_180/\0";
    filepath[1] = filePathPrefix + "MOR/r_approx_300/\0";
    // Solution filepath
    std::string filepath_sol = "output/";
    // Solution filename
    std::string filename_sol = "solution.dat";
    // Array of matrix sizes (row)
    int row_baseline[] = {126, 132, 168, 174, 180, 186, 192, 288, 294, 300, 306, 312};
    // Array of filenames
    std::string baseName_K = "KSM_Stiffness_r\0";
    std::string baseName_M = "KSM_Mass_r\0";
    std::string baseName_D = "KSM_Damping_r\0";
    std::string base_format = ".mtx\0";
    std::string filename_K[12];
    std::string filename_M[12];
    std::string filename_D[12];

    /*--------
    Parameters
    --------*/
    bool isComplex = 1;
    double freq, freq_square;
    double freq_min = 1;
    const double alpha = 4*PI*PI;
    MKL_Complex16 one;				// Dummy scaling factor for global matrix assembly
    one.real = 1;
    one.imag = 0;
    MKL_Complex16 zero;               // Dummy scaling factor for PARDISO
    zero.real = 0;
    zero.imag = 0;
    MKL_Complex16 rhs_val;
    rhs_val.real = 1.0;
    rhs_val.imag = 0.0;

    timerTotal.start();

    /*-----------
    Read Matrices
    -----------*/
    // Define Matrices
    std::vector<std::vector<MKL_Complex16>> K_sub(12);
    std::vector<std::vector<MKL_Complex16>> M_sub(12);
    std::vector<std::vector<MKL_Complex16>> D_sub(12);
    // Read MTX files
    for (size_t i = 0; i < 7; i++){
        filename_K[i] = baseName_K + std::to_string(row_baseline[i]) + base_format;
        filename_M[i] = baseName_M + std::to_string(row_baseline[i]) + base_format;
        filename_D[i] = baseName_D + std::to_string(row_baseline[i]) + base_format;
        io::readMtxDense(K_sub[i], filepath[0], filename_K[i], isComplex);
        io::readMtxDense(M_sub[i], filepath[0], filename_M[i], isComplex);
        io::readMtxDense(D_sub[i], filepath[0], filename_D[i], isComplex);
        K_sub[i].pop_back();
        M_sub[i].pop_back();
        D_sub[i].pop_back();
    }
    for (size_t i = 7; i < 12; i++){
        filename_K[i] = baseName_K + std::to_string(row_baseline[i]) + base_format;
        filename_M[i] = baseName_M + std::to_string(row_baseline[i]) + base_format;
        filename_D[i] = baseName_D + std::to_string(row_baseline[i]) + base_format;
        io::readMtxDense(K_sub[i], filepath[1], filename_K[i], isComplex);
        io::readMtxDense(M_sub[i], filepath[1], filename_M[i], isComplex);
        io::readMtxDense(D_sub[i], filepath[1], filename_D[i], isComplex);
        K_sub[i].pop_back();
        M_sub[i].pop_back();
        D_sub[i].pop_back();
    }
    std::cout << ">> Matrices imported" << std::endl;

    // Get matrix sizes
    std::vector<int> row_sub(num_matrix);
    std::vector<int> size_sub(num_matrix);
    std::vector<size_t> ptr_mat_shift(num_matrix);
    std::vector<size_t> ptr_vec_shift(num_matrix);
    int nnz = 0;
    int row = 0;
    size_t idx;

    for (size_t j = 0; j < mat_repetition; j++){
        for (size_t i = 0; i < 12; i++){
            idx = i + 12*j;
            row_sub[idx] = row_baseline[i];
            size_sub[idx] = row_sub[i]*row_sub[i];
            ptr_mat_shift[idx] = nnz;
            ptr_vec_shift[idx] = row;
            nnz += size_sub[idx];
            row  += row_sub[idx];
        }
    }

    // Combine matrices into a single array
    std::vector<MKL_Complex16> K(nnz);
    std::vector<MKL_Complex16> M(nnz);
    std::vector<MKL_Complex16> D(nnz);
    auto K_sub_ptr = &K_sub[0];
    auto M_sub_ptr = &M_sub[0];
    auto D_sub_ptr = &D_sub[0];
    size_t array_shift = 0;
    for (size_t j = 0; j < mat_repetition; j++){
        for (size_t i = 0; i < 12; i++){
            K_sub_ptr = &K_sub[i];
            M_sub_ptr = &M_sub[i];
            D_sub_ptr = &D_sub[i];
            std::copy(K_sub_ptr->begin(), K_sub_ptr->end(), K.begin() + array_shift);
            std::copy(M_sub_ptr->begin(), M_sub_ptr->end(), M.begin() + array_shift);
            std::copy(D_sub_ptr->begin(), D_sub_ptr->end(), D.begin() + array_shift);
            array_shift += size_sub[i];
        }
    }

    std::cout <<">> Matrices combined\n" << std::endl;

    // Generate CSR format
    timerAux.start();
    std::vector<int> csrRowPtr(row+1);
    std::vector<int> csrColInd(nnz);
    dataStructure::generateCSR(csrRowPtr, csrColInd, row_sub, size_sub, row, nnz, num_matrix);
    timerAux.stop();
    if (sparse_mode == "Sparse Block Diagonal System"){
        std::cout <<">> CSR Format Generated" << std::endl;
        std::cout <<">>>> Time taken = " << timerAux.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;
    }

    // Allocate global matrices
    std::vector<MKL_Complex16> A(nnz*nt);

    // Initialise RHS vectors
    std::vector<MKL_Complex16> rhs(row*freq_max, rhs_val);

    // Initialise solution vectors
    std::vector<MKL_Complex16> sol(row*freq_max);

    // M = 4*pi^2*M (Single computation suffices)
    cblas_zdscal(nnz, alpha, M.data(), 1);

    /*--------------------
    PARDISO Initialisation
    --------------------*/
    // Check if sparse matrix is good
    sparse_checker_error_values check_err_val;
    sparse_struct pt;
    int error = 0;
    sparse_matrix_checker_init(&pt);
    pt.n = row;
    pt.csr_ia = csrRowPtr.data();
    pt.csr_ja = csrColInd.data();
    pt.indexing = MKL_ZERO_BASED;
    pt.print_style = MKL_C_STYLE;
    pt.message_level = MKL_PRINT;
    check_err_val = sparse_matrix_checker(&pt);
    if (sparse_mode == "Sparse Block Diagonal System"){
        printf(">>>> Matrix check details: (%d, %d, %d)\n", pt.check_result[0], pt.check_result[1], pt.check_result[2]);
        if (check_err_val == MKL_SPARSE_CHECKER_SUCCESS) { printf(">>>> Matrix check result: MKL_SPARSE_CHECKER_SUCCESS\n"); }
        if (check_err_val == MKL_SPARSE_CHECKER_NON_MONOTONIC) { printf(">>>> Matrix check result: MKL_SPARSE_CHECKER_NON_MONOTONIC\n"); }
        if (check_err_val == MKL_SPARSE_CHECKER_OUT_OF_RANGE) { printf(">>>> Matrix check result: MKL_SPARSE_CHECKER_OUT_OF_RANGE\n"); }
        if (check_err_val == MKL_SPARSE_CHECKER_NONORDERED) { printf(">>>> Matrix check result: MKL_SPARSE_CHECKER_NONORDERED\n"); }
        error = 1;
    }
    // Pardiso variables
    void *pardiso_pt[64] = {};   // Internal solver memory pointer
    MKL_INT pardiso_mtype = 13; // Real Complex Unsymmetric Matrix
    MKL_INT pardiso_nrhs = 1;   // Number of RHS
    MKL_Complex16 pardiso_ddum; // Complex dummy
    MKL_INT pardiso_idum;   // Integer dummy
    // Pardiso control parameters
    MKL_INT pardiso_iparm[64] = {};
    MKL_INT pardiso_maxfct, pardiso_mnum, pardiso_phase, pardiso_error, pardiso_msglvl;
    pardiso_iparm[0] = 1;           // No solver default
    pardiso_iparm[1] = 2;           // Fill-in reordering from METIS
    pardiso_iparm[3] = 0;           // No iterative-direct algorithm
    pardiso_iparm[4] = 0;           // No user fill-in reducing permutation
    pardiso_iparm[5] = 0;           // Write solution into x
    pardiso_iparm[6] = 0;           // Not in use
    pardiso_iparm[7] = 2;           // Max numbers of iterative refinement steps
    pardiso_iparm[8] = 0;           // Not in use
    pardiso_iparm[9] = 13;          // Perturb the pivot elements with 1E-13
    pardiso_iparm[10] = 1;          // Use nonsymmetric permutation and scaling MPS
    pardiso_iparm[11] = 0;          // Conjugate transposed/transpose solve
    pardiso_iparm[12] = 1;          // Maximum weighted matching algorithm is switched-on (default for non-symmetric)
    pardiso_iparm[13] = 0;          // Output: Number of perturbed pivots
    pardiso_iparm[14] = 0;          // Not in use
    pardiso_iparm[15] = 0;          // Not in use
    pardiso_iparm[16] = 0;          // Not in use
    pardiso_iparm[17] = -1;         // Output: Number of nonzeros in the factor LU
    pardiso_iparm[18] = -1;         // Output: Mflops for LU factorization
    pardiso_iparm[19] = 0;          // Output: Numbers of CG Iterations
    pardiso_iparm[23] = 1;          // 2-level factorisation
    pardiso_iparm[36] = -99;        // VBSR format
    pardiso_iparm[34] = 1;          // Zero based indexing
    pardiso_maxfct = 1;             // Maximum number of numerical factorizations
    pardiso_mnum = 1;               // Which factorization to use
    pardiso_msglvl = 0;             // Print statistical information
    pardiso_error = 1;              // Initialize error flag

    // Loop over frequency
    int sol_shift = 0;
    int last_sol_shift = 0;
    int mat_shift = 0;
    std::cout << ">> Frequency loop started (" << sparse_mode << ")" << std::endl;

    /*--------------------------
    Sparse Block Diagonal System
    --------------------------*/
    if (sparse_mode == "Sparse Block Diagonal System"){
        timerLoop.start();
        #pragma omp parallel private(tid, freq, freq_square, mat_shift, pardiso_error)
        {
            // Get thread number
            tid = omp_get_thread_num();
            // Compute matrix shift
            mat_shift = tid*nnz;
            // Compute solution shift
            //sol_shift = last_sol_shift + tid*row;

            #pragma omp for
            for (int it = (int)freq_min; it <= (int)freq_max; it++) {
                // Compute scaling
                freq = (double)it;
                freq_square = -(freq*freq);

                // Assemble global matrix ( A = K - f^2*M_tilde)
                assembly::assembleGlobalMatrix(A.data(), K.data(), M.data(), mat_shift, nnz, one, freq_square);

                /*-----
                PARDISO
                -----*/
                // Symbolic factorization
                pardiso_phase = 11;
                pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase, &row, A.data() + mat_shift,
                        csrRowPtr.data(), csrColInd.data(), &pardiso_idum, &pardiso_nrhs,
                        pardiso_iparm, &pardiso_msglvl, &pardiso_ddum, &pardiso_ddum, &pardiso_error);
                if (pardiso_error != 0) {std::cout << "ERROR during symbolic factorisation: " << pardiso_error; exit(1);}

                // Numerical factorization
                pardiso_phase = 22;
                pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase, &row, A.data() + mat_shift,
                        csrRowPtr.data(), csrColInd.data(), &pardiso_idum, &pardiso_nrhs,
                        pardiso_iparm, &pardiso_msglvl, &pardiso_ddum, &pardiso_ddum, &pardiso_error);
                if (pardiso_error != 0) {std::cout << "ERROR during numerical factorisation: " << pardiso_error; exit(2);}

                // Backward substitution
                pardiso_phase = 33;
                pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase, &row, A.data() + mat_shift,
                        csrRowPtr.data(), csrColInd.data(), &pardiso_idum, &pardiso_nrhs,
                        pardiso_iparm, &pardiso_msglvl, rhs.data(), sol.data()+sol_shift, &pardiso_error);
                if (pardiso_error != 0) {std::cout << "ERROR during backward substitution: " << pardiso_error; exit(3);}

                sol_shift += row;

                //if (tid == omp_get_num_threads()-1) last_sol_shift = sol_shift + nnz;
            } // frequency loop
        } // omp parallel
    } // dense mode

    /*---------------------
    Multiple Dense Matrices
    ---------------------*/
    else if (sparse_mode == "Multiple Dense Matrices"){
        int row_shift, prev_row_shift;
        size_t i, j;
        // Pivots for LU Decomposition
        std::vector<lapack_int> pivot(nnz);
        timerLoop.start();
        #pragma omp parallel private(tid, freq, freq_square, mat_shift, array_shift, row_shift, i, j, sol_shift, prev_row_shift)
        {
            // Get thread number
            tid = omp_get_thread_num();
            // Compute matrix shift
            mat_shift = tid*nnz;
            // Previous row shift
            prev_row_shift = 0;

            #pragma omp for
            for (int it = (int)freq_min; it <= (int)freq_max; it++) {
                // Compute scaling
                freq = (double)it;
                freq_square = -(freq*freq);

                // Assemble global matrix ( A = K - f^2*M_tilde)
                //assembly::assembleGlobalMatrix(A.data(), K.data(), M.data(), mat_shift, nnz, one, freq_square);
                cblas_zcopy(nnz, M.data(), 1, A.data() + mat_shift, 1);
                cblas_zdscal(nnz, freq_square, A.data() + mat_shift, 1);
                cblas_zaxpy(nnz, &one, K.data(), 1, A.data() + mat_shift, 1);

                /*-----
                LAPACKE
                -----*/
                array_shift = 0;
                row_shift = tid*row + prev_row_shift;
                for (j = 0; j < mat_repetition; j++){
                    for (i = 0; i < 12; i++){
                        // LU Decomposition
                        LAPACKE_zgetrf(LAPACK_ROW_MAJOR, row_sub[i], row_sub[i], A.data() + mat_shift + array_shift, row_sub[i], pivot.data());
                        // Solve system
                        LAPACKE_zgetrs(LAPACK_ROW_MAJOR, 'N', row_sub[i], 1, A.data() + mat_shift + array_shift, row_sub[i], pivot.data(), rhs.data() + row_shift, row_sub[i]);
                        // Update array and row shifts
                        array_shift += size_sub[i];
                        row_shift += row_sub[i];
                    }
                }
                // Move onto next batch of frequency arrays
                prev_row_shift = nt*row;
            } // frequency loop
        } // omp parallel
    } // dense mode
    timerLoop.stop();
    timerTotal.stop();

    // Output messages
    std::cout << "\n" << ">> Frequency loop finished" << std::endl;
    std::cout << ">>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6 << "\n" << std::endl;

    // Pardiso termination and release of memory
    if (sparse_mode == "Sparse Block Diagonal System"){
        pardiso_phase = -1;
        pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase, &row, &pardiso_ddum, csrRowPtr.data(), csrColInd.data(), &pardiso_idum, &pardiso_nrhs,
                pardiso_iparm, &pardiso_msglvl, &pardiso_ddum, &pardiso_ddum, &pardiso_error);
        std::cout << ">> PARDISO internal memory freed\n" << std::endl;
    }

    // Output solutions
    io::writeSolVecComplex(rhs, filepath_sol, filename_sol);

    std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}
