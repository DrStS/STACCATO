// Libraries
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// CUDA
#include <cuda_runtime.h>

// THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

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

/*
TODO
	1. Cuncurrent kernel execution

Possible optimization:
	1. Cuncurrent kernel execution
	2. Shared memory for the matrices
*/

int main (int argc, char *argv[]){

	// Command line arguments
	if (argc < 3){
		std::cerr << ">> Usage: " << argv[0] << " -s <small/mid/large>" << std::endl;
		return 1;
	}
	std::string SIZE = argv[2];
	std::cout << ">> Matrix Size: " << SIZE << std::endl;

	std::string filepath_input, filepath_sol;
	std::string filename_K, filename_M, filename_D, filename_sol;

	if (SIZE == "small"){
		// Filepaths
		filepath_input = "/opt/software/examples/MOR/small/";
		filepath_sol = "output/";
		// Filenames
		filename_K   = "KSM_Stiffness_r21.mtx";
		filename_M   = "KSM_Mass_r21.mtx";
		filename_D   = "KSM_Damping_r21.mtx";
		filename_sol = "solution_mkl_small.dat";
	}
	else if (SIZE == "mid"){
		// Filepaths
		filepath_input = "/opt/software/examples/MOR/mid/";
		filepath_sol = "output/";
		// Filenames
		filename_K   = "KSM_Stiffness_r189.mtx";
		filename_M   = "KSM_Mass_r189.mtx";
		filename_D   = "KSM_Damping_r189.mtx";
		filename_sol = "solution_mkl_mid.dat";
	}

	else if (SIZE == "large"){
		// Filepaths
		filepath_input = "/opt/software/examples/MOR/large/";
		filepath_sol = "output/";
		// Filenames
		filename_K   = "KSM_Stiffness_r2520.mtx";
		filename_M   = "KSM_Mass_r2520.mtx";
		filename_D   = "KSM_Damping_r2520.mtx";
		filename_sol = "solution_mkl_large.dat";
	}
	else{
		std::cerr << ">> Incorrect matrix size, please check commandline argument\n" << std::endl;
		return 1;
	}

	// Parameters
	bool isComplex = 1;
	double freq, freq_square;
	double freq_min = 1;
	double freq_max = 1000;
	const double alpha = 4*PI*PI;
	cuDoubleComplex one;	// Dummy scailing factor for global matrix assembly
	one.x = 1;
	one.y = 0;
	cuDoubleComplex rhs_val;
	rhs_val.x = (double)1.0;
	rhs_val.y = (double)0.0;

	// Time measurements
	thrust::host_vector<float> vec_time(freq_max);

	timerTotal.start();
	// Library initialisation
	cublasStatus_t cublasStatus;
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	cusolverStatus_t cusolverStatus = CUSOLVER_STATUS_ALLOC_FAILED;
	cusolverDnHandle_t cusolverHandle;
	cusolverStatus = cusolverDnCreate(&cusolverHandle);

	// Create matrix host_vectors
	thrust::host_vector<cuDoubleComplex> K, M, D;

	// Read MTX file
	io::readMtxDense(K, filepath_input, filename_K, isComplex);
	io::readMtxDense(M, filepath_input, filename_M, isComplex);
	io::readMtxDense(D, filepath_input, filename_D, isComplex);

	// Readjust matrix size (matrix size initially increased by 1 due to segmentation fault. See also io.cu)
	K.pop_back();
	M.pop_back();
	D.pop_back();

	// Get matrix sizes
	int size = K.size();
	int row  = sqrt(size);

	// Create RHS directly on device
	thrust::device_vector<cuDoubleComplex> d_rhs(row, rhs_val);
	thrust::device_vector<cuDoubleComplex> d_rhs_buf(row, rhs_val);

	// Send matrices to device
	timerMatrixCpy.start();
	thrust::device_vector<cuDoubleComplex> d_K = K;
	thrust::device_vector<cuDoubleComplex> d_M = M;
	thrust::device_vector<cuDoubleComplex> d_D = D;
	timerMatrixCpy.stop();
	std::cout << ">> " << SIZE << " matrices copied to device " << std::endl;
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
	thrust::host_vector<cuDoubleComplex> sol(row);

	// Create solution vector on device
	thrust::device_vector<cuDoubleComplex> d_sol(row);

	// M = 4*pi^2*M (Single computation suffices)
	cublasStatus = cublasZdscal(cublasHandle, size, &alpha, d_ptr_M, 1);
	std::cout << ">> M_tilde " << "(" << SIZE << ")" << " computed with cuBLAS" << std::endl;
	if (cublasStatus != CUBLAS_STATUS_SUCCESS){
		std::cout << "cublas failed!" << std::endl;
	}

	// LU decomposition prep
	thrust::host_vector<int> solverInfo(1);
	thrust::device_vector<int> d_solverInfo(1);
	int *d_ptr_solverInfo = thrust::raw_pointer_cast(d_solverInfo.data());

	// Pivots
	thrust::device_vector<int> d_pivot(row);
	thrust::sequence(d_pivot.begin(), d_pivot.end()-(int)row/2, row-1, -1);
	int *d_ptr_pivot = thrust::raw_pointer_cast(d_pivot.data());

	int i = 0;
	timerLoop.start();
	// Loop over frequency
	for (size_t it = (size_t)freq_min; it <= (size_t)freq_max; it++){
		timerIteration.start();

		// Compute scaling
		freq = (double)it;
		freq_square = -(freq*freq);

		timerMatrixComp.start();
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

		// Compute workspace size
		int sizeWorkspace;
		cusolverStatus = cusolverDnZgetrf_bufferSize(cusolverHandle, row, row, d_ptr_A, row, &sizeWorkspace);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver workspace size computation failed for " << SIZE << " system\n" << std::endl;

		// Create workspace
		thrust::device_vector<cuDoubleComplex> d_workspace(sizeWorkspace);
		cuDoubleComplex *d_ptr_workspace = thrust::raw_pointer_cast(d_workspace.data());

		// LU decomposition
		cusolverStatus = cusolverDnZgetrf(cusolverHandle, row, row, d_ptr_A, row, d_ptr_workspace, NULL, d_ptr_solverInfo);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver LU decomposition failed (" << SIZE << ")" << std::endl;
		solverInfo = d_solverInfo;
		if (solverInfo[0] != 0){
			std::cout << ">>>> LU decomposition failed for " << SIZE << " matrix" << std::endl;
		}

		// Solve x = A\b
		cusolverStatus = cusolverDnZgetrs(cusolverHandle, CUBLAS_OP_N, row, 1, d_ptr_A, row, NULL, d_ptr_rhs, row, d_ptr_solverInfo);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> System couldn't be solved" << std::endl;
		solverInfo = d_solverInfo;
		if (solverInfo[0] != 0) {
			std::cout << ">>>> System solution failure (" << SIZE << ")" << std::endl;
		}
		timerMatrixComp.stop();

		// Retrieve results from device to host
		sol = d_rhs;

		// Reset rhs values
		d_rhs = d_rhs_buf;

		// Output messages
		timerIteration.stop();
		std::cout << ">>>> Frequency = " << freq << " || " << "Time taken (s): Small = " << timerMatrixComp.getDurationMicroSec()*1e-6 << std::endl;

		// Accumulate time measurements
		vec_time[i] = timerMatrixComp.getDurationMicroSec()*1e-6;
		i++;
	}
	timerLoop.stop();

	// Compute average time per matrices
	float time_avg = thrust::reduce(vec_time.begin(), vec_time.end(), (float)0, thrust::plus<float>());
	time_avg /= freq_max;

	std::cout << "\n" << ">>>> Frequency loop finished" << std::endl;
	std::cout << ">>>>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6 << "\n" << std::endl;
	std::cout << ">>>>>> Average time (s) for computing " << SIZE << " matrix: = " << time_avg << "\n" << std::endl;

	// Write out solution vectors
	io::writeSolVecComplex(sol, filepath_sol, filename_sol);

	// Destroy cuBLAS & cuSolver
	cublasDestroy(cublasHandle);
	cusolverDnDestroy(cusolverHandle);

	timerTotal.stop();
	std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}
