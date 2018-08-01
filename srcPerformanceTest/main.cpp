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

int main(int argc, char *argv[]) {

	// Command line arguments
	if (argc < 3) {
		std::cerr << ">> Usage: " << argv[0] << " -s <small/mid/large>" << std::endl;
		return 1;
	}
	std::string SIZE = argv[2];
	std::cout << ">> Matrix Size: " << SIZE << std::endl;

	std::string filepath_input, filepath_sol;
	std::string filename_K, filename_M, filename_D, filename_sol;

#if defined(_WIN32) || defined(__WIN32__) 
	std::string filePathPrefix = "C:/software/examples/";
#endif
#if defined(__linux__) 
	std::string filePathPrefix = "/opt/software/examples/";
#endif

	if (SIZE == "small") {
		// Filepaths
		filepath_input = "MOR/small/";
		filepath_sol = "output/";
		// Filenames
		filename_K = "KSM_Stiffness_r21.mtx";
		filename_M = "KSM_Mass_r21.mtx";
		filename_D = "KSM_Damping_r21.mtx";
		filename_sol = "solution_mkl_small.dat";
	}
	else if (SIZE == "mid") {
		// Filepaths
		filepath_input = "MOR/mid/";
		filepath_sol = "output/";
		// Filenames
		filename_K = "KSM_Stiffness_r189.mtx";
		filename_M = "KSM_Mass_r189.mtx";
		filename_D = "KSM_Damping_r189.mtx";
		filename_sol = "solution_mkl_mid.dat";
	}

	else if (SIZE == "large") {
		// Filepaths
		filepath_input = "MOR/large/";
		filepath_sol = "output/";
		// Filenames
		filename_K = "KSM_Stiffness_r2520.mtx";
		filename_M = "KSM_Mass_r2520.mtx";
		filename_D = "KSM_Damping_r2520.mtx";
		filename_sol = "solution_mkl_large.dat";
	}
	else {
		std::cerr << ">> Incorrect matrix size, please check commandline argument\n" << std::endl;
		return 1;
	}

	// OpenMP Threads
	//int nt = mkl_get_max_threads();
	int nt = 1;
	mkl_set_num_threads(nt);
	std::cout << "\n>> Software will use the following number of threads: " << nt << " threads\n" << std::endl;

	// Parameters
	bool isComplex = true;
	double freq, freq_square;
	double freq_min = 1;
	double freq_max = 1000;
	const double alpha = 4 * PI*PI;
	MKL_Complex16 one;	// Dummy scaling factor for global matrix assembly
	one.real = 1;
	one.imag = 0;
	MKL_Complex16 rhs_val;
	rhs_val.real = 1;
	rhs_val.imag = 0;

	// Time measurement
	std::vector<float> vec_time((size_t)freq_max);

	timerTotal.start();

	// Matrices
	std::vector<MKL_Complex16> K, M, D;

	// Read MTX files
	io::readMtxDense(K, filePathPrefix + filepath_input, filename_K, isComplex);
	io::readMtxDense(M, filePathPrefix + filepath_input, filename_M, isComplex);
	io::readMtxDense(D, filePathPrefix + filepath_input, filename_D, isComplex);

	// Readjust matrix size (matrix size initially increased by 1 due to segmentation fault. See also io.cpp)
	K.pop_back();
	M.pop_back();
	D.pop_back();

	// Get matrix size
	int size = K.size();
	int row = sqrt(size);

	// Allocate global matrices
	std::vector<MKL_Complex16> A(size);

	// Initialise RHS vectors
	std::vector<MKL_Complex16> rhs(row, rhs_val);

	// Initialise solution vectors
	std::vector<MKL_Complex16> sol(row);

	// M = 4*pi^2*M (Single computation suffices)
	cblas_zdscal(size, alpha, M.data(), 1);
	std::cout << ">> M_tilde (" << SIZE << ")" << " computed with Intel MKL" << std::endl;

	// Pivots for LU Decomposition
	std::vector<lapack_int> pivot(size);

	int i = 0;
	timerLoop.start();
	// Loop over frequency
	for (size_t it = (size_t)freq_min; it <= (size_t)freq_max; it++) {
		timerIteration.start();
		// Compute scaling
		freq = (double)it;
		freq_square = -(freq*freq);

		timerMatrixComp.start();
		// Assemble global matrix ( A = K - f^2*M_tilde)
		cblas_zcopy(size, M.data(), 1, A.data(), 1);
		cblas_zdscal(size, freq_square, A.data(), 1);
		cblas_zaxpy(size, &one, K.data(), 1, A.data(), 1);
		// LU Decomposition
		LAPACKE_zgetrf(LAPACK_COL_MAJOR, row, row, A.data(), row, pivot.data());
		// Solve system
		LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', row, 1, A.data(), row, pivot.data(), rhs.data(), row);
		timerMatrixComp.stop();

		// Copy solution to solution vector
		cblas_zcopy(row, rhs.data(), 1, sol.data(), 1);

		// Reset RHS values
		std::fill(rhs.begin(), rhs.end(), one);

		// Output messages
		timerIteration.stop();
		std::cout << ">>>> Frequency = " << freq << " || " << "Time taken (" << SIZE << ") :" << timerMatrixComp.getDurationMicroSec()*1e-6 << std::endl;

		// Accumulate time measurements
		vec_time[i] = (float)timerMatrixComp.getDurationMicroSec()*1e-6;
		i++;
	}
	timerLoop.stop();
	timerTotal.stop();

	// Get average time
	float time_avg = cblas_sasum((int)freq_max, vec_time.data(), 1); time_avg /= freq_max;

	// Output messages
	std::cout << "\n" << ">>>> Frequency loop finished" << std::endl;
	std::cout << ">>>>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6 << "\n" << std::endl;
	std::cout << ">>>>>> Average time (s) for matrix computation (" << SIZE << ") : " << time_avg << "\n" << std::endl;

	// Output solutions
	io::writeSolVecComplex(sol, filepath_sol, filename_sol);

	std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}

