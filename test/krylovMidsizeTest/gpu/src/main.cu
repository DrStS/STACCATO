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
	thrust::host_vector<std::string> filepath(2);
	filepath[0] = "/opt/software/examples/MOR/r_approx_180/\0";
	filepath[1] = "/opt/software/examples/MOR/r_approx_300/\0";
	// Vector of matrix sizes (row)
	thrust::host_vector<int> row_sub(12);
	row_sub[0] = 126;
	row_sub[1] = 132;
	row_sub[2] = 168;
	row_sub[3] = 174;
	row_sub[4] = 180;
	row_sub[5] = 186;
	row_sub[6] = 192;
	row_sub[7] = 288;
	row_sub[8] = 294;
	row_sub[9] = 300;
	row_sub[10] = 306;
	row_sub[11] = 312;
	// Vector of filenames
	std::string baseName_K = "KSM_Stiffness_r\0";
	std::string baseName_M = "KSM_Mass_r\0";
	std::string baseName_D = "KSM_Damping_r\0";
	std::string base_format = ".mtx\0";
	thrust::host_vector<std::string> filename_K(12);
	thrust::host_vector<std::string> filename_M(12);
	thrust::host_vector<std::string> filename_D(12);

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

	// Combine matrices into a single array on host (to make use of GPU's high bandwidth. We could also import the matrices directly like this)
	thrust::host_vector<cuDoubleComplex> K(size);
	thrust::host_vector<cuDoubleComplex> M(size);
	thrust::host_vector<cuDoubleComplex> D(size);

	thrust::copy((K_sub.data())->begin(), K_sub.data()->end(), K.begin());
	thrust::copy((M_sub.data())->begin(), M_sub.data()->end(), K.begin());
	thrust::copy((D_sub.data())->begin(), D_sub.data()->end(), K.begin());

/*
	for (size_t i = 1; i < 12; i++){
		thrust::copy(K_sub[i]->begin(), K_sub[i]->end(), K.begin()+i*size_sub[i-1]);
		thrust::copy(M_sub[i]->begin(), M_sub[i]->end(), M.begin()+i*size_sub[i-1]);
		thrust::copy(D_sub[i]->begin(), D_sub[i]->end(), D.begin()+i*size_sub[i-1]);
		std::cout << i << std::endl;
	}

	// Create RHS on host
	//thrust::host_vector<cuDoubleComplex> rhs(row);


	//thrust::device_vector<cuDoubleComplex> d_rhs(row);
	//thrust::device_vector<cuDoubleComplex> d_rhs_buf = d_rhs;

	//std::cout << "rhs created" << std::endl;

/*
	// Create RHS directly on device
	thrust::device_vector<cuDoubleComplex> d_rhs(row, rhs_val);
	thrust::device_vector<cuDoubleComplex> d_rhs_buf(row, rhs_val);

	// Combine matrices into a single array (do this on host to make use of GPU's high bandwidth)
	for (size_t i = 0; i < num_matrix; i++){
		thrust::copy(K_sub.begin(), K_sub.end(), K.begin()+i*size_sub);
		thrust::copy(M_sub.begin(), M_sub.end(), M.begin()+i*size_sub);
		thrust::copy(D_sub.begin(), D_sub.end(), D.begin()+i*size_sub);
	}

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
*/
}
