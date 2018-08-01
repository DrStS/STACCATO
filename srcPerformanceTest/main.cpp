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
 * \author Stefan Sicklinger
 * \date 8/1/2018
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
#include "io/io.hpp"
#include "helper/Timer.hpp"

// Definitions
#define	PI	3.14159265359

//#define MKL_Complex16 std::complex<double>

int main (int argc, char *argv[]){

	// OpenMP Threads
	int nt = mkl_get_max_threads();
	//int nt = 1;
	mkl_set_num_threads(nt);

	int tid = omp_get_thread_num();
	std::cout << "\n>> Software will use the following number of threads: " << nt << " threads\n" << std::endl;

	// Filepaths
#if defined(_WIN32) || defined(__WIN32__) 
	std::string filePathPrefix = "C:/software/examples/";
#endif
#if defined(__linux__) 
	std::string filePathPrefix = "/opt/software/examples/";
#endif
	std::string filepath_small = "MOR/small/";
	std::string filepath_mid = "MOR/mid/";
	std::string filepath_large = "MOR/large/";
	std::string filepath_sol = "output/";

	// Filenames
	std::string filename_K_small   = "KSM_Stiffness_r21.mtx";
	std::string filename_M_small   = "KSM_Mass_r21.mtx";
	std::string filename_D_small   = "KSM_Damping_r21.mtx";
	std::string filename_K_mid     = "KSM_Stiffness_r189.mtx";
	std::string filename_M_mid     = "KSM_Mass_r189.mtx";
	std::string filename_D_mid     = "KSM_Damping_r189.mtx";
	std::string filename_K_large   = "KSM_Stiffness_r2520.mtx";
	std::string filename_M_large   = "KSM_Mass_r2520.mtx";
	std::string filename_D_large   = "KSM_Damping_r2520.mtx";
	std::string filename_sol_small = "solution_mkl_small.dat";
	std::string filename_sol_mid   = "solution_mkl_mid.dat";
	std::string filename_sol_large = "solution_mkl_large.dat";

	// Parameters
	bool isComplex = true;
	double freq, freq_square;
	double freq_min = 1;
	double freq_max = 1000;
	const double alpha = 4*PI*PI;
	MKL_Complex16 one;	// Dummy scaling factor for global matrix assembly
	one.real = 1;
	one.imag = 0;
	MKL_Complex16 rhs_val;
	rhs_val.real = 1;
	rhs_val.imag = 0;

	// Time measurement
	std::chrono::high_resolution_clock::time_point start_time_M_tilde, start_time_loop, start_time_it, start_time_small, start_time_mid, start_time_large, start_time_total;
	std::chrono::high_resolution_clock::time_point end_time_M_tilde, end_time_loop, end_time_it, end_time_small, end_time_mid, end_time_large, end_time_total;
	std::chrono::seconds time_loop, time_it, time_small, time_mid, time_large, time_total;

	std::vector<std::chrono::seconds> vec_time_small((size_t)freq_max);
	std::vector<std::chrono::seconds> vec_time_mid((size_t)freq_max);
	std::vector<std::chrono::seconds> vec_time_large((size_t)freq_max);

	//start_time_total = std::chrono::high_resolution_clock::now();
	anaysisTimer01.start();

	// Matrices
	std::vector<MKL_Complex16> K_small, M_small, D_small, K_mid, M_mid, D_mid, K_large, M_large, D_large;

	// Read MTX files
	io::readMtxDense(K_small, filePathPrefix+filepath_small, filename_K_small, isComplex);
	io::readMtxDense(M_small, filePathPrefix+filepath_small, filename_M_small, isComplex);
	io::readMtxDense(D_small, filePathPrefix+filepath_small, filename_D_small, isComplex);
	io::readMtxDense(K_mid,   filePathPrefix+filepath_mid,   filename_K_mid,   isComplex);
	io::readMtxDense(M_mid,   filePathPrefix+filepath_mid,   filename_M_mid,   isComplex);
	io::readMtxDense(D_mid,   filePathPrefix+filepath_mid,   filename_D_mid,   isComplex);
	io::readMtxDense(K_large, filePathPrefix+filepath_large, filename_K_large, isComplex);
	io::readMtxDense(M_large, filePathPrefix+filepath_large, filename_M_large, isComplex);
	io::readMtxDense(D_large, filePathPrefix+filepath_large, filename_D_large, isComplex);

	// Readjust matrix size (matrix size initially increased by 1 due to segmentation fault. See also io.cpp)
	K_small.pop_back();
	M_small.pop_back();
	D_small.pop_back();
	K_mid.pop_back();
	M_mid.pop_back();
	D_mid.pop_back();
	K_large.pop_back();
	M_large.pop_back();
	D_large.pop_back();

	// Get matrix size
	int size_small = K_small.size();
	int size_mid   = K_mid.size();
	int size_large = K_large.size();
	int row_small  = sqrt(size_small);
	int row_mid    = sqrt(size_mid);
	int row_large  = sqrt(size_large);

	// Allocate global matrices
	std::vector<MKL_Complex16> A_small(size_small);
	std::vector<MKL_Complex16> A_mid(size_mid);
	std::vector<MKL_Complex16> A_large(size_large);

	// Initialise RHS vectors
	std::vector<MKL_Complex16> rhs_small(row_small, rhs_val);
	std::vector<MKL_Complex16> rhs_mid(row_mid, rhs_val);
	std::vector<MKL_Complex16> rhs_large(row_large, rhs_val);

	// Initialise solution vectors
	std::vector<MKL_Complex16> sol_small(row_small);
	std::vector<MKL_Complex16> sol_mid(row_mid);
	std::vector<MKL_Complex16> sol_large(row_large);

	// M = 4*pi^2*M (Single computation suffices)
	start_time_M_tilde = std::chrono::high_resolution_clock::now();
	cblas_zdscal(size_small, alpha, M_small.data(), 1);
	end_time_M_tilde = std::chrono::high_resolution_clock::now();
	std::cout << ">> M_tilde (small) computed with Intel MKL" << std::endl;
	std::cout << ">>>> Time taken = " << std::chrono::duration_cast<std::chrono::seconds>(end_time_M_tilde - start_time_M_tilde).count() << " (sec)" << "\n" << std::endl;

	start_time_M_tilde = std::chrono::high_resolution_clock::now();
	cblas_zdscal(size_mid, alpha, M_mid.data(), 1);
	end_time_M_tilde = std::chrono::high_resolution_clock::now();
	std::cout << ">> M_tilde (mid) computed with Intel MKL" << std::endl;
	std::cout << ">>>> Time taken = " << std::chrono::duration_cast<std::chrono::seconds>(end_time_M_tilde - start_time_M_tilde).count() << " (sec)" << "\n" << std::endl;

	start_time_M_tilde = std::chrono::high_resolution_clock::now();
	cblas_zdscal(size_large, alpha, M_large.data(), 1);
	end_time_M_tilde = std::chrono::high_resolution_clock::now();
	std::cout << ">> M_tilde (large) computed with Intel MKL" << std::endl;
	std::cout << ">>>> Time taken = " << std::chrono::duration_cast<std::chrono::seconds>(end_time_M_tilde - start_time_M_tilde).count() << " (sec)" << "\n" << std::endl;

	// Pivots for LU Decomposition
	std::vector<lapack_int> pivot_small(size_small);
	std::vector<lapack_int> pivot_mid(size_mid);
	std::vector<lapack_int> pivot_large(size_large);

	int i = 0;
	start_time_loop = std::chrono::high_resolution_clock::now();
	// Loop over frequency
	for (size_t it = (size_t)freq_min; it <= (size_t)freq_max; it++){
		start_time_it = std::chrono::high_resolution_clock::now();
		// Compute scaling
		freq = (double)it;
		freq_square = -(freq*freq);

		/*------------
		Small matrics
		------------*/
		start_time_small = std::chrono::high_resolution_clock::now();
		// Assemble global matrix ( A = K - f^2*M_tilde)
		cblas_zcopy(size_small, M_small.data(), 1, A_small.data(), 1);
		cblas_zdscal(size_small, freq_square, A_small.data(), 1);
		cblas_zaxpy(size_small, &one, K_small.data(), 1, A_small.data(), 1);
		// LU Decomposition
		LAPACKE_zgetrf(LAPACK_COL_MAJOR, row_small, row_small, A_small.data(), row_small, pivot_small.data());
		// Solve system
		LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', row_small, 1, A_small.data(), row_small, pivot_small.data(), rhs_small.data(), row_small);
		end_time_small = std::chrono::high_resolution_clock::now();

		/*------------
		Mid matrics
		------------*/
		start_time_mid = std::chrono::high_resolution_clock::now();
		// Assemble global matrix ( A = K - f^2*M_tilde)
		cblas_zcopy(size_mid, M_mid.data(), 1, A_mid.data(), 1);
		cblas_zdscal(size_mid, freq_square, A_mid.data(), 1);
		cblas_zaxpy(size_mid, &one, K_mid.data(), 1, A_mid.data(), 1);
		// LU Decomposition
		LAPACKE_zgetrf(LAPACK_COL_MAJOR, row_mid, row_mid, A_mid.data(), row_mid, pivot_mid.data());
		// Solve system
		LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', row_mid, 1, A_mid.data(), row_mid, pivot_mid.data(), rhs_mid.data(), row_mid);
		end_time_mid = std::chrono::high_resolution_clock::now();

		/*------------
		Large matrics
		------------*/
		start_time_large = std::chrono::high_resolution_clock::now();
		// Assemble global matrix ( A = K - f^2*M_tilde)
		cblas_zcopy(size_large, M_large.data(), 1, A_large.data(), 1);
		cblas_zdscal(size_large, freq_square, A_large.data(), 1);
		cblas_zaxpy(size_large, &one, K_large.data(), 1, A_large.data(), 1);
		// LU Decomposition
		LAPACKE_zgetrf(LAPACK_COL_MAJOR, row_large, row_large, A_large.data(), row_large, pivot_large.data());
		// Solve system
		LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', row_large, 1, A_large.data(), row_large, pivot_large.data(), rhs_large.data(), row_large);
		end_time_large = std::chrono::high_resolution_clock::now();


		// Copy solution to solution vector
		cblas_zcopy(row_small, rhs_small.data(), 1, sol_small.data(), 1);
		cblas_zcopy(row_mid,   rhs_mid.data(),   1, sol_mid.data(), 1);
		cblas_zcopy(row_large, rhs_large.data(), 1, sol_large.data(), 1);

		// Reset RHS values
		std::fill(rhs_small.begin(), rhs_small.end(), one);
		std::fill(rhs_mid.begin(),   rhs_mid.end(),   one);
		std::fill(rhs_large.begin(), rhs_large.end(), one);

		// Output messages
		end_time_it = std::chrono::high_resolution_clock::now();
		//time_small  = std::chrono::duration_cast<std::chrono::seconds>(start_time_small - end_time_small).count();
		//time_mid    = std::chrono::duration_cast<std::chrono::seconds>(start_time_mid   - end_time_mid).count();
		//time_large  = std::chrono::duration_cast<std::chrono::seconds>(start_time_large - end_time_large).count();
		//std::cout << ">>>> Frequency = " << freq << " || " << "Time taken (s): Small = " << time_small << " || " << "Mid = " << time_mid << " || " << "Large = " << time_large << std::endl;

		// Accumulate time measurements
		//vec_time_small[i] = time_small;
		//vec_time_mid[i]   = time_mid;
		//vec_time_large[i] = time_large;
		i++;
	}
	end_time_loop = std::chrono::high_resolution_clock::now();
	//end_time_total = std::chrono::high_resolution_clock::now();
	anaysisTimer01.stop();

	// Get average time
	//float time_small_avg = cblas_sasum((int)freq_max, vec_time_small.data(), 1); time_small_avg /= freq_max;
	//float time_mid_avg   = cblas_sasum((int)freq_max, vec_time_mid.data(), 1);   time_mid_avg   /= freq_max;
	//float time_large_avg = cblas_sasum((int)freq_max, vec_time_large.data(), 1); time_large_avg /= freq_max;

	// Output messages
	std::cout << "\n" << ">>>> Frequency loop finished" << std::endl;
	//std::cout << ">>>>>> Time taken (s) = " << std::chrono::duration_cast<std::chrono::seconds>(start_time_loop - end_time_loop).count() << "\n" << std::endl;
	//std::cout << ">>>>>> Average time (s) for each matrix: Small = " << time_small_avg << " || " << " Mid = " << time_mid_avg << " || " << " Large = " << time_large_avg << "\n" << std::endl;

	// Output solutions
	io::writeSolVecComplex(sol_small, filepath_sol, filename_sol_small);
	io::writeSolVecComplex(sol_mid,   filepath_sol, filename_sol_mid);
	io::writeSolVecComplex(sol_large, filepath_sol, filename_sol_large);

	std::cout << ">>>>>> Total execution time = " << anaysisTimer01.getDurationSec() << "\n" << std::endl;
}
