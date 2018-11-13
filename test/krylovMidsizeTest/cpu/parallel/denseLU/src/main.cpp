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
    // Configure test environment with command line arguments
    config::configureTest(argc, argv, freq_max, mat_repetition, num_matrix, tid, nt_mkl, nt);

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
    int row_baseline[] = { 126, 132, 168, 174, 180, 186, 192, 288, 294, 300, 306, 312 };
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
    const double alpha = 4 * PI*PI;
    MKL_Complex16 one;				// Dummy scaling factor for global matrix assembly
    one.real = 1;
    one.imag = 0;
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
    for (size_t i = 0; i < 7; i++) {
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
    for (size_t i = 7; i < 12; i++) {
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

    for (size_t j = 0; j < mat_repetition; j++) {
        for (size_t i = 0; i < 12; i++) {
            idx = i + 12 * j;
            row_sub[idx] = row_baseline[i];
            size_sub[idx] = row_sub[i] * row_sub[i];
            ptr_mat_shift[idx] = nnz;
            ptr_vec_shift[idx] = row;
            nnz += size_sub[idx];
            row += row_sub[idx];
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
    for (size_t j = 0; j < mat_repetition; j++) {
        for (size_t i = 0; i < 12; i++) {
            K_sub_ptr = &K_sub[i];
            M_sub_ptr = &M_sub[i];
            D_sub_ptr = &D_sub[i];
            std::copy(K_sub_ptr->begin(), K_sub_ptr->end(), K.begin() + array_shift);
            std::copy(M_sub_ptr->begin(), M_sub_ptr->end(), M.begin() + array_shift);
            std::copy(D_sub_ptr->begin(), D_sub_ptr->end(), D.begin() + array_shift);
            array_shift += size_sub[i];
        }
    }

    std::cout << ">> Matrices combined\n" << std::endl;

    // Allocate global matrices
    std::vector<MKL_Complex16> A(nnz*nt);

    // Initialise RHS vectors
    std::vector<MKL_Complex16> rhs(row*freq_max, rhs_val);

    // Initialise solution vectors
    std::vector<MKL_Complex16> sol(row*freq_max);

    // M = 4*pi^2*M (Single computation suffices)
    cblas_zdscal(nnz, alpha, M.data(), 1);

    // Loop over frequency
    int sol_shift = 0;
    int last_sol_shift = 0;
    int mat_shift = 0;

    /*---------------------
    Multiple Dense Matrices
    ---------------------*/
    int row_shift, prev_row_shift, row_mat;
    size_t i;
    // Pivots for LU Decomposition
    std::vector<lapack_int> pivot(nnz*nt);
    timerLoop.start();
#pragma omp parallel private(tid, freq, freq_square, mat_shift, array_shift, row_shift, i, sol_shift, prev_row_shift, row_mat) num_threads(nt)
    {
        // Get thread number
        tid = omp_get_thread_num();
        // Compute matrix shift
        mat_shift = tid * nnz;
        // Previous row shift
        prev_row_shift = 0;
    //#pragma omp critical
    //            std::cout << "This is thread " << tid + 1 << " out of " << omp_get_num_threads() << std::endl;
#pragma omp for
        // Frequency loop
        for (int it = (int)freq_min; it <= (int)freq_max; it++) {
            // Compute scaling
            freq = (double)it;
            freq_square = -(freq*freq);

            // Assemble global matrix ( A = K - f^2*M_tilde)
            assembly::assembleGlobalMatrix(A.data(), K.data(), M.data(), mat_shift, nnz, one, freq_square);

            /*-----
            LAPACKE
            -----*/
            array_shift = 0;
            row_shift = tid * row;
            for (i = 0; i < num_matrix; i++) {
                row_mat = row_sub[i];
                // LU Decomposition
                LAPACKE_zgetrf(LAPACK_COL_MAJOR, row_mat, row_mat, A.data() + mat_shift + array_shift, row_mat, pivot.data() + mat_shift);
                // Solve system
                LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', row_mat, 1, A.data() + mat_shift + array_shift, row_mat, pivot.data() + mat_shift, rhs.data() + prev_row_shift + row_shift, row_mat);
                // Update array and row shifts
                array_shift += size_sub[i];
                row_shift += row_sub[i];
            }
            // Move onto next batch of frequency arrays
            prev_row_shift += nt*row;
        } // frequency loop
    } // omp parallel
    timerLoop.stop();

    // Output messages
    std::cout << "\n" << ">> Frequency loop finished" << std::endl;
    std::cout << ">>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6 << "\n" << std::endl;

    // Output solutions
    io::writeSolVecComplex(rhs, filepath_sol, filename_sol);

    timerTotal.stop();
    std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}
