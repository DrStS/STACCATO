// Libraries
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// OpenMP
#include <omp.h>

// CUDA
#include <cuda_runtime.h>

// THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

// CUCOMPLEX
#include <cuComplex.h>

// CUBLAS
#include <cublas_v2.h>

// CUSOLVER
#include <cusolverDn.h>

// Header files
#include "io/io.cuh"
#include "helper/Timer.cuh"

// Definitions
#define	PI	3.14159265359

int main (int argc, char *argv[]){

	// Vector of filepaths
	std::string filepath[2];
	filepath[0] = "/opt/software/examples/MOR/r_approx_180/\0";
	filepath[1] = "/opt/software/examples/MOR/r_approx_300/\0";

	// Solution filepath
	std::string filepath_sol = "output/";

	// Solution filename
	std::string filename_sol = "solution.dat";

	// Array of matrix sizes (row)
	int row_sub[] = {126, 132, 168, 174, 180, 186, 192, 288, 294, 300, 306, 312};

	// Array of filenames
	std::string baseName_K = "KSM_Stiffness_r\0";
	std::string baseName_M = "KSM_Mass_r\0";
	std::string baseName_D = "KSM_Damping_r\0";
	std::string base_format = ".mtx\0";
	std::string filename_K[12];
	std::string filename_M[12];
	std::string filename_D[12];

	// Parameters
	bool isComplex = 1;
	double freq, freq_square;
	double freq_min = 1;
	double freq_max = 2000;
	const double alpha = 4*PI*PI;
	cuDoubleComplex one;	// Dummy scailing factor for global matrix assembly
	one.x = 1;
	one.y = 0;
	cuDoubleComplex rhs_val;
	rhs_val.x = (double)1.0;
	rhs_val.y = (double)0.0;
	int mat_repetition = 5;

	timerTotal.start();
	// Library initialisation
	cublasStatus_t cublasStatus;
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	cusolverStatus_t cusolverStatus = CUSOLVER_STATUS_ALLOC_FAILED;
	cusolverDnHandle_t cusolverHandle;
	cusolverStatus = cusolverDnCreate(&cusolverHandle);

	// Create matrix host_vectors
	thrust::host_vector<thrust::host_vector<cuDoubleComplex>> K_sub(12);
	thrust::host_vector<thrust::host_vector<cuDoubleComplex>> M_sub(12);
	thrust::host_vector<thrust::host_vector<cuDoubleComplex>> D_sub(12);

	// Read and process MTX file
	for (size_t i = 0; i < 7; i++){
		filename_K[i] = baseName_K + std::to_string(row_sub[i]) + base_format;
		filename_M[i] = baseName_M + std::to_string(row_sub[i]) + base_format;
		filename_D[i] = baseName_D + std::to_string(row_sub[i]) + base_format;
		io::readMtxDense(K_sub[i], filepath[0], filename_K[i], isComplex);
		io::readMtxDense(M_sub[i], filepath[0], filename_M[i], isComplex);
		io::readMtxDense(D_sub[i], filepath[0], filename_D[i], isComplex);
		K_sub[i].pop_back();
		M_sub[i].pop_back();
		D_sub[i].pop_back();
	}
	for (size_t i = 7; i < 12; i++){
		filename_K[i] = baseName_K + std::to_string(row_sub[i]) + base_format;
		filename_M[i] = baseName_M + std::to_string(row_sub[i]) + base_format;
		filename_D[i] = baseName_D + std::to_string(row_sub[i]) + base_format;
		io::readMtxDense(K_sub[i], filepath[1], filename_K[i], isComplex);
		io::readMtxDense(M_sub[i], filepath[1], filename_M[i], isComplex);
		io::readMtxDense(D_sub[i], filepath[1], filename_D[i], isComplex);
		K_sub[i].pop_back();
		M_sub[i].pop_back();
		D_sub[i].pop_back();
	}
	std::cout << ">> Matrices imported" << std::endl;

	// Get matrix sizes
	thrust::host_vector<int> size_sub(12);
	int size = 0;
	int row = 0;
	for (size_t i = 0; i < 12; i++){
		size_sub[i] = row_sub[i]*row_sub[i];
		size += size_sub[i];
		row  += row_sub[i];
	}
	row *= mat_repetition;
	size *= mat_repetition;

	// Combine matrices into a single array on host (to make use of GPU's high bandwidth. We could also import the matrices directly like this)
	thrust::host_vector<cuDoubleComplex> K(size);
	thrust::host_vector<cuDoubleComplex> M(size);
	thrust::host_vector<cuDoubleComplex> D(size);
	int  array_shift = 0;
	auto K_sub_ptr = &K_sub[0];
	auto M_sub_ptr = &M_sub[0];
	auto D_sub_ptr = &D_sub[0];
	for (size_t j = 0; j < mat_repetition; j++){
		for (size_t i = 0; i < 12; i++){
			K_sub_ptr = &K_sub[i];
			M_sub_ptr = &M_sub[i];
			D_sub_ptr = &D_sub[i];
			thrust::copy(K_sub_ptr->begin(), K_sub_ptr->end(), K.begin()+array_shift);
			thrust::copy(M_sub_ptr->begin(), M_sub_ptr->end(), M.begin()+array_shift);
			thrust::copy(D_sub_ptr->begin(), D_sub_ptr->end(), D.begin()+array_shift);
			array_shift += size_sub[i];
		}
	}
	std::cout <<">> Matrices combined" << std::endl;

	// Send matrices to device
	timerMatrixCpy.start();
	thrust::device_vector<cuDoubleComplex> d_K = K;
	thrust::device_vector<cuDoubleComplex> d_M = M;
	thrust::device_vector<cuDoubleComplex> d_D = D;
	timerMatrixCpy.stop();
	std::cout << ">> Matrices copied to device " << std::endl;
	std::cout << ">>>> Time taken = " << timerMatrixCpy.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;

	// Create RHS directly on device
	timerMatrixCpy.start();
	thrust::device_vector<cuDoubleComplex> d_rhs(row, rhs_val);
	thrust::device_vector<cuDoubleComplex> d_rhs_buf = d_rhs;
	timerMatrixCpy.stop();
	std::cout << ">> RHS copied to device " << std::endl;
	std::cout << ">>>> Time taken = " << timerMatrixCpy.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;

	// Create matrix device_vectors
	thrust::device_vector<cuDoubleComplex> d_A(size);

	// Get raw pointers to matrices
	cuDoubleComplex *d_ptr_K = thrust::raw_pointer_cast(d_K.data());
	cuDoubleComplex *d_ptr_M = thrust::raw_pointer_cast(d_M.data());
	cuDoubleComplex *d_ptr_D = thrust::raw_pointer_cast(d_D.data());
	cuDoubleComplex *d_ptr_A = thrust::raw_pointer_cast(d_A.data());

	// Get raw pointers to RHS vectors
	cuDoubleComplex *d_ptr_rhs = thrust::raw_pointer_cast(d_rhs.data());

	// Create solution vector on host
	thrust::host_vector<cuDoubleComplex> sol(row*freq_max);

	// Create solution vector on device
	thrust::device_vector<cuDoubleComplex> d_sol(row*freq_max);

	timerMatrixComp.start();
	// M = 4*pi^2*M (Single computation suffices)
	cublasStatus = cublasZdscal(cublasHandle, size, &alpha, d_ptr_M, 1);
	timerMatrixComp.stop();
	std::cout << ">> M_tilde computed with cuBLAS" << std::endl;
	std::cout << ">>>> Time taken = " << timerMatrixComp.getDurationMicroSec()*1e-6 << " (sec)\n" << std::endl;
	if (cublasStatus != CUBLAS_STATUS_SUCCESS){
		std::cout << "cublas failed!" << std::endl;
	}

	// LU decomposition prep
	thrust::host_vector<int> solverInfo(1);
	thrust::device_vector<int> d_solverInfo(1);
	int *d_ptr_solverInfo = thrust::raw_pointer_cast(d_solverInfo.data());

/*
	// Pivots
	thrust::device_vector<int> d_pivot(row);
	thrust::sequence(d_pivot.begin(), d_pivot.end()-(int)row/2, row-1, -1);
	int *d_ptr_pivot = thrust::raw_pointer_cast(d_pivot.data());
*/

	// Compute workspace size
	int totalSizeWorkspace = 0;
	thrust::host_vector<int> sizeWorkspace(12*mat_repetition);
	auto sizeWorkspace_ptr = &sizeWorkspace[0];
	for (size_t j = 0; j < mat_repetition; j++){
		for (size_t i = 0; i < 12; i++){
			sizeWorkspace_ptr = &sizeWorkspace[i+12*j];
			cusolverStatus = cusolverDnZgetrf_bufferSize(cusolverHandle, row_sub[i], row_sub[i], d_ptr_A, row_sub[i], sizeWorkspace_ptr);
			if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver workspace size computation failed\n" << std::endl;
			totalSizeWorkspace += sizeWorkspace[i];
		}
	}

	// Create workspace
	thrust::device_vector<cuDoubleComplex> d_workspace(totalSizeWorkspace);
	cuDoubleComplex *d_ptr_workspace = thrust::raw_pointer_cast(d_workspace.data());

	timerLoop.start();
	int sol_shift = 0;
	// Loop over frequency
	for (size_t it = (size_t)freq_min; it <= (size_t)freq_max; it++){
		timerIteration.start();

		// Compute scaling
		freq = (double)it;
		freq_square = -(freq*freq);

		// Assemble global matrix ( A = K - f^2*M_tilde )
		d_A = d_M;
		// Scale A with -f^2
		cublasStatus = cublasZdscal(cublasHandle, size, &freq_square, d_ptr_A, 1);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS){
			std::cout << "cublas failed!" << std::endl;
		}
		// Sum A with K
		cublasStatus = cublasZaxpy(cublasHandle, size, &one, d_ptr_K, 1, d_ptr_A, 1);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS){
			std::cout << "cublas failed during matrix assembly!" << std::endl;
		}

		array_shift = 0;
		size_t row_shift = 0;
		size_t workspace_shift = 0;
		for (size_t j = 0; j < mat_repetition; j++){
			for (size_t i = 0; i < 12; i++){
				// LU decomposition
				cusolverStatus = cusolverDnZgetrf(cusolverHandle, row_sub[i], row_sub[i], d_ptr_A + array_shift, row_sub[i], d_ptr_workspace + workspace_shift, NULL, d_ptr_solverInfo);
				if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver LU decomposition failed" << std::endl;
				solverInfo = d_solverInfo;
				if (solverInfo[0] != 0){
					std::cout << ">>>> LU decomposition failed" << std::endl;
					std::cout << ">>>> solverInfo = " << solverInfo[0] << std::endl;
				}

				// Solve x = A\b
				cusolverStatus = cusolverDnZgetrs(cusolverHandle, CUBLAS_OP_N, row_sub[i], 1, d_ptr_A + array_shift, row_sub[i], NULL, d_ptr_rhs + row_shift, row_sub[i], d_ptr_solverInfo);
				if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> System couldn't be solved" << std::endl;
				solverInfo = d_solverInfo;
				if (solverInfo[0] != 0) {
					std::cout << ">>>> System solution failure" << std::endl;
				}
				array_shift += size_sub[i];
				row_shift += row_sub[i];
				workspace_shift += sizeWorkspace[i];
			}
		}
		// Copy the solution to solution vector
		thrust::copy(d_rhs.begin(), d_rhs.end(), d_sol.begin() + sol_shift);
		sol_shift += row;
		// Reset RHS
		d_rhs = d_rhs_buf;
	}
	timerLoop.stop();

	std::cout << ">>>> Frequency loop finished" << std::endl;
	std::cout << ">>>>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6 << "\n" << std::endl;

	sol = d_sol;
	// Write out solution vectors
	io::writeSolVecComplex(sol, filepath_sol, filename_sol);

	// Destroy cuBLAS & cuSolver
	cublasDestroy(cublasHandle);
	cusolverDnDestroy(cusolverHandle);

	timerTotal.stop();
	std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}
