
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
	std::cout << "\n>> Software will use the following number of threads: " << nt << " threads\n" << std::endl;

	// Filepaths
	std::string filepath_small = "/opt/software/examples/MOR/small/";
	std::string filepath_mid = "/opt/software/examples/MOR/mid/";
	std::string filepath_large = "/opt/software/examples/MOR/large/";
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
	std::vector<float> vec_time_small((size_t)freq_max);
	std::vector<float> vec_time_mid((size_t)freq_max);
	std::vector<float> vec_time_large((size_t)freq_max);

	timerTotal.start();

	// Matrices
	std::vector<MKL_Complex16> K_small, M_small, D_small, K_mid, M_mid, D_mid, K_large, M_large, D_large;

	// Read MTX files
	io::readMtxDense(K_small, filepath_small, filename_K_small, isComplex);
	io::readMtxDense(M_small, filepath_small, filename_M_small, isComplex);
	io::readMtxDense(D_small, filepath_small, filename_D_small, isComplex);
	io::readMtxDense(K_mid,   filepath_mid,   filename_K_mid,   isComplex);
	io::readMtxDense(M_mid,   filepath_mid,   filename_M_mid,   isComplex);
	io::readMtxDense(D_mid,   filepath_mid,   filename_D_mid,   isComplex);
	io::readMtxDense(K_large, filepath_large, filename_K_large, isComplex);
	io::readMtxDense(M_large, filepath_large, filename_M_large, isComplex);
	io::readMtxDense(D_large, filepath_large, filename_D_large, isComplex);

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
	cblas_zdscal(size_small, alpha, M_small.data(), 1);
	std::cout << ">> M_tilde (small) computed with Intel MKL" << std::endl;
	cblas_zdscal(size_mid, alpha, M_mid.data(), 1);
	std::cout << ">> M_tilde (mid) computed with Intel MKL" << std::endl;
	cblas_zdscal(size_large, alpha, M_large.data(), 1);
	std::cout << ">> M_tilde (large) computed with Intel MKL" << std::endl;

	// Pivots for LU Decomposition
	std::vector<lapack_int> pivot_small(size_small);
	std::vector<lapack_int> pivot_mid(size_mid);
	std::vector<lapack_int> pivot_large(size_large);

	int i = 0;
	timerLoop.start();
	// Loop over frequency
	for (size_t it = (size_t)freq_min; it <= (size_t)freq_max; it++){
		timerIteration.start();
		// Compute scaling
		freq = (double)it;
		freq_square = -(freq*freq);

		/*------------
		Small matrics
		------------*/
		timerSmall.start();
		// Assemble global matrix ( A = K - f^2*M_tilde)
		cblas_zcopy(size_small, M_small.data(), 1, A_small.data(), 1);
		cblas_zdscal(size_small, freq_square, A_small.data(), 1);
		cblas_zaxpy(size_small, &one, K_small.data(), 1, A_small.data(), 1);
		// LU Decomposition
		LAPACKE_zgetrf(LAPACK_COL_MAJOR, row_small, row_small, A_small.data(), row_small, pivot_small.data());
		// Solve system
		LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', row_small, 1, A_small.data(), row_small, pivot_small.data(), rhs_small.data(), row_small);
		timerSmall.stop();

		/*------------
		Mid matrics
		------------*/
		timerMid.start();
		// Assemble global matrix ( A = K - f^2*M_tilde)
		cblas_zcopy(size_mid, M_mid.data(), 1, A_mid.data(), 1);
		cblas_zdscal(size_mid, freq_square, A_mid.data(), 1);
		cblas_zaxpy(size_mid, &one, K_mid.data(), 1, A_mid.data(), 1);
		// LU Decomposition
		LAPACKE_zgetrf(LAPACK_COL_MAJOR, row_mid, row_mid, A_mid.data(), row_mid, pivot_mid.data());
		// Solve system
		LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', row_mid, 1, A_mid.data(), row_mid, pivot_mid.data(), rhs_mid.data(), row_mid);
		timerMid.stop();

		/*------------
		Large matrics
		------------*/
		timerLarge.start();
		// Assemble global matrix ( A = K - f^2*M_tilde)
		cblas_zcopy(size_large, M_large.data(), 1, A_large.data(), 1);
		cblas_zdscal(size_large, freq_square, A_large.data(), 1);
		cblas_zaxpy(size_large, &one, K_large.data(), 1, A_large.data(), 1);
		// LU Decomposition
		LAPACKE_zgetrf(LAPACK_COL_MAJOR, row_large, row_large, A_large.data(), row_large, pivot_large.data());
		// Solve system
		LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', row_large, 1, A_large.data(), row_large, pivot_large.data(), rhs_large.data(), row_large);
		timerLarge.stop();

		// Copy solution to solution vector
		cblas_zcopy(row_small, rhs_small.data(), 1, sol_small.data(), 1);
		cblas_zcopy(row_mid,   rhs_mid.data(),   1, sol_mid.data(), 1);
		cblas_zcopy(row_large, rhs_large.data(), 1, sol_large.data(), 1);

		// Reset RHS values
		std::fill(rhs_small.begin(), rhs_small.end(), one);
		std::fill(rhs_mid.begin(),   rhs_mid.end(),   one);
		std::fill(rhs_large.begin(), rhs_large.end(), one);

		// Output messages
		timerIteration.stop();
		std::cout << ">>>> Frequency = " << freq << " || " << "Time taken (s): Small = " << timerSmall.getDurationMicroSec()*1e-6 << " || " << "Mid = " << timerMid.getDurationMicroSec()*1e-6  << " || " << "Large = " << timerLarge.getDurationMicroSec()*1e-6  << std::endl;

		// Accumulate time measurements
		vec_time_small[i] = (float)timerSmall.getDurationMicroSec()*1e-6 ;
		vec_time_mid[i]   = (float)timerMid.getDurationMicroSec()*1e-6 ;
		vec_time_large[i] = (float)timerLarge.getDurationMicroSec()*1e-6 ;
		i++;
	}
	timerLoop.stop();
	timerTotal.stop();

	// Get average time
	float time_small_avg = cblas_sasum((int)freq_max, vec_time_small.data(), 1); time_small_avg /= freq_max;
	float time_mid_avg   = cblas_sasum((int)freq_max, vec_time_mid.data(), 1);   time_mid_avg   /= freq_max;
	float time_large_avg = cblas_sasum((int)freq_max, vec_time_large.data(), 1); time_large_avg /= freq_max;

	// Output messages
	std::cout << "\n" << ">>>> Frequency loop finished" << std::endl;
	std::cout << ">>>>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6  << "\n" << std::endl;
	std::cout << ">>>>>> Average time (s) for each matrix: Small = " << time_small_avg << " || " << " Mid = " << time_mid_avg << " || " << " Large = " << time_large_avg << "\n" << std::endl;

	// Output solutions
	io::writeSolVecComplex(sol_small, filepath_sol, filename_sol_small);
	io::writeSolVecComplex(sol_mid,   filepath_sol, filename_sol_mid);
	io::writeSolVecComplex(sol_large, filepath_sol, filename_sol_large);

	std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6  << "\n" << std::endl;
}
