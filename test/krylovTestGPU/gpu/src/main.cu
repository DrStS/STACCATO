// Libraries
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
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

// Definitions
#define	PI	3.14159265359

/*
Validated:
	1. M_tilde
	2. LU Decomposition and solver

TODO
	1. Cuncurrent kernel execution

Possible optimization:
	1. Cuncurrent kernel execution
	2. Shared memory for the matrices
*/

int main (int argc, char *argv[]){

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
	std::string filename_sol_small = "solution_small.dat";
	std::string filename_sol_mid   = "solution_mid.dat";
	std::string filename_sol_large = "solution_large.dat";

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
	clock_t matrixCpyTime, time, time_loop, time_it, time_small, time_mid, time_large, time_total;
	thrust::host_vector<float> vec_time_small(freq_max);
	thrust::host_vector<float> vec_time_mid(freq_max);
	thrust::host_vector<float> vec_time_large(freq_max);

	time_total = clock();
	// Library initialisation
	cublasStatus_t cublasStatus;
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	cusolverStatus_t cusolverStatus = CUSOLVER_STATUS_ALLOC_FAILED;
	cusolverDnHandle_t cusolverHandle;
	cusolverStatus = cusolverDnCreate(&cusolverHandle);

	// Create matrix host_vectors
	thrust::host_vector<cuDoubleComplex> K_small;
	thrust::host_vector<cuDoubleComplex> M_small;
	thrust::host_vector<cuDoubleComplex> D_small;
	thrust::host_vector<cuDoubleComplex> K_mid;
	thrust::host_vector<cuDoubleComplex> M_mid;
	thrust::host_vector<cuDoubleComplex> D_mid;
	thrust::host_vector<cuDoubleComplex> K_large;
	thrust::host_vector<cuDoubleComplex> M_large;
	thrust::host_vector<cuDoubleComplex> D_large;

	// Read MTX file
	io::readMtxDense(K_small, filepath_small, filename_K_small, isComplex);
	io::readMtxDense(M_small, filepath_small, filename_M_small, isComplex);
	io::readMtxDense(D_small, filepath_small, filename_D_small, isComplex);
	io::readMtxDense(K_mid,   filepath_mid,   filename_K_mid,   isComplex);
	io::readMtxDense(M_mid,   filepath_mid,   filename_M_mid,   isComplex);
	io::readMtxDense(D_mid,   filepath_mid,   filename_D_mid,   isComplex);
	io::readMtxDense(K_large, filepath_large, filename_K_large, isComplex);
	io::readMtxDense(M_large, filepath_large, filename_M_large, isComplex);
	io::readMtxDense(D_large, filepath_large, filename_D_large, isComplex);

	// Readjust matrix size (matrix size initially increased by 1 due to segmentation fault. See also io.cu)
	K_small.pop_back();
	M_small.pop_back();
	D_small.pop_back();
	K_mid.pop_back();
	M_mid.pop_back();
	D_mid.pop_back();
	K_large.pop_back();
	M_large.pop_back();
	D_large.pop_back();

	// Get matrix sizes
	int size_small = K_small.size();
	int size_mid   = K_mid.size();
	int size_large = K_large.size();
	int row_small  = sqrt(size_small);
	int row_mid    = sqrt(size_mid);
	int row_large  = sqrt(size_large);

	// Create RHS directly on device
	thrust::device_vector<cuDoubleComplex> d_rhs_small(row_small, rhs_val);
	thrust::device_vector<cuDoubleComplex> d_rhs_mid(row_mid, rhs_val);
	thrust::device_vector<cuDoubleComplex> d_rhs_large(row_large, rhs_val);
	thrust::device_vector<cuDoubleComplex> d_rhs_small_buf(row_small, rhs_val);
	thrust::device_vector<cuDoubleComplex> d_rhs_mid_buf(row_mid, rhs_val);
	thrust::device_vector<cuDoubleComplex> d_rhs_large_buf(row_large, rhs_val);

	// Send matrices to device
	matrixCpyTime = clock();
	thrust::device_vector<cuDoubleComplex> d_K_small = K_small;
	thrust::device_vector<cuDoubleComplex> d_M_small = M_small;
	thrust::device_vector<cuDoubleComplex> d_D_small = D_small;
	matrixCpyTime = clock() - matrixCpyTime;
	std::cout << ">> Small matrices copied to device " << std::endl;
	std::cout << ">>>> Time taken = " << ((float)matrixCpyTime)/CLOCKS_PER_SEC << " (sec)" << "\n" << std::endl;
	matrixCpyTime = clock();
	thrust::device_vector<cuDoubleComplex> d_K_mid = K_mid;
	thrust::device_vector<cuDoubleComplex> d_M_mid = M_mid;
	thrust::device_vector<cuDoubleComplex> d_D_mid = D_mid;
	matrixCpyTime = clock() - matrixCpyTime;
	std::cout << ">> Mid matrices copied to device " << std::endl;
	std::cout << ">>>> Time taken = " << ((float)matrixCpyTime)/CLOCKS_PER_SEC << " (sec)" << "\n" << std::endl;
	matrixCpyTime = clock();
	thrust::device_vector<cuDoubleComplex> d_K_large = K_large;
	thrust::device_vector<cuDoubleComplex> d_M_large = M_large;
	thrust::device_vector<cuDoubleComplex> d_D_large = D_large;
	matrixCpyTime = clock() - matrixCpyTime;
	std::cout << ">> Large matrices copied to device " << std::endl;
	std::cout << ">>>> Time taken = " << ((float)matrixCpyTime)/CLOCKS_PER_SEC << " (sec)" << "\n" << std::endl;

	// Create matrix device_vectors
	thrust::device_vector<cuDoubleComplex> d_A_small(size_small);
	thrust::device_vector<cuDoubleComplex> d_A_mid(size_mid);
	thrust::device_vector<cuDoubleComplex> d_A_large(size_large);

	// Get raw pointers to matrices
	cuDoubleComplex *d_ptr_K_small = thrust::raw_pointer_cast(d_K_small.data());
	cuDoubleComplex *d_ptr_M_small = thrust::raw_pointer_cast(d_M_small.data());
	cuDoubleComplex *d_ptr_D_small = thrust::raw_pointer_cast(d_D_small.data());
	cuDoubleComplex *d_ptr_A_small = thrust::raw_pointer_cast(d_A_small.data());
	cuDoubleComplex *d_ptr_K_mid   = thrust::raw_pointer_cast(d_K_mid.data());
	cuDoubleComplex *d_ptr_M_mid   = thrust::raw_pointer_cast(d_M_mid.data());
	cuDoubleComplex *d_ptr_D_mid   = thrust::raw_pointer_cast(d_D_mid.data());
	cuDoubleComplex *d_ptr_A_mid   = thrust::raw_pointer_cast(d_A_mid.data());
	cuDoubleComplex *d_ptr_K_large = thrust::raw_pointer_cast(d_K_large.data());
	cuDoubleComplex *d_ptr_M_large = thrust::raw_pointer_cast(d_M_large.data());
	cuDoubleComplex *d_ptr_D_large = thrust::raw_pointer_cast(d_D_large.data());
	cuDoubleComplex *d_ptr_A_large = thrust::raw_pointer_cast(d_A_large.data());

	// Get raw pointers to RHS vectors
	cuDoubleComplex *d_ptr_rhs_small = thrust::raw_pointer_cast(d_rhs_small.data());
	cuDoubleComplex *d_ptr_rhs_mid   = thrust::raw_pointer_cast(d_rhs_mid.data());
	cuDoubleComplex *d_ptr_rhs_large = thrust::raw_pointer_cast(d_rhs_large.data());

	// Create solution vector on host
	thrust::host_vector<cuDoubleComplex> sol_small(row_small);
	thrust::host_vector<cuDoubleComplex> sol_mid(row_mid);
	thrust::host_vector<cuDoubleComplex> sol_large(row_large);

	// Create solution vector on device
	thrust::device_vector<cuDoubleComplex> d_sol_small(row_small);
	thrust::device_vector<cuDoubleComplex> d_sol_mid(row_mid);
	thrust::device_vector<cuDoubleComplex> d_sol_large(row_large);

	// M = 4*pi^2*M (Single computation suffices)
	time = clock();
	cublasStatus = cublasZdscal(cublasHandle, size_small, &alpha, d_ptr_M_small, 1);
	time = clock() - time;
	std::cout << ">> M_tilde (small) computed with cuBLAS" << std::endl;
	std::cout << ">>>> Time taken = " << ((float)time)/CLOCKS_PER_SEC << " (sec)" << "\n" << std::endl;
	if (cublasStatus != CUBLAS_STATUS_SUCCESS){
		std::cout << "cublas failed!" << std::endl;
	}
	time = clock();
	cublasStatus = cublasZdscal(cublasHandle, size_mid, &alpha, d_ptr_M_mid, 1);
	time = clock() - time;
	std::cout << ">> M_tilde (mid) computed with cuBLAS" << std::endl;
	std::cout << ">>>> Time taken = " << ((float)time)/CLOCKS_PER_SEC << " (sec)" << "\n" << std::endl;
	if (cublasStatus != CUBLAS_STATUS_SUCCESS){
		std::cout << ">>>>>> ERROR: cuBLAS failed!" << std::endl;
	}
	time = clock();
	cublasStatus = cublasZdscal(cublasHandle, size_large, &alpha, d_ptr_M_large, 1);
	time = clock() - time;
	std::cout << ">> M_tilde (large) computed with cuBLAS" << std::endl;
	std::cout << ">>>> Time taken = " << ((float)time)/CLOCKS_PER_SEC << " (sec)" << "\n" << std::endl;
	if (cublasStatus != CUBLAS_STATUS_SUCCESS){
		std::cout << ">>>>>> ERROR: cuBLAS failed!" << std::endl;
	}

	// LU decomposition prep
	thrust::host_vector<int> solverInfo(1);
	thrust::device_vector<int> d_solverInfo(1);
	int *d_ptr_solverInfo = thrust::raw_pointer_cast(d_solverInfo.data());

	// Pivots
	thrust::device_vector<int> d_pivot_small(row_small);
	thrust::device_vector<int> d_pivot_mid(row_mid);
	thrust::device_vector<int> d_pivot_large(row_large);
	thrust::sequence(d_pivot_small.begin(), d_pivot_small.end()-(int)row_small/2, row_small-1, -1);
	thrust::sequence(d_pivot_mid.begin(), d_pivot_mid.end()-(int)row_mid/2, row_mid-1, -1);
	thrust::sequence(d_pivot_large.begin(), d_pivot_large.end()-(int)row_large/2, row_large-1, -1);
	int *d_ptr_pivot_small = thrust::raw_pointer_cast(d_pivot_small.data());
	int *d_ptr_pivot_mid   = thrust::raw_pointer_cast(d_pivot_mid.data());
	int *d_ptr_pivot_large = thrust::raw_pointer_cast(d_pivot_large.data());

	int i = 0;
	time_loop = clock();
	// Loop over frequency
	for (size_t it = (size_t)freq_min; it <= (size_t)freq_max; it++){
		time_it = clock();

		// Compute scaling
		freq = (double)it;
		freq_square = -(freq*freq);

		/*------------
		Small matrices
		------------*/
		time_small = clock();
		// Assemble global matrix ( A = K - f^2*M_tilde )
		d_A_small = d_M_small;
		// Scale A with -f^2
		cublasStatus = cublasZdscal(cublasHandle, size_small, &freq_square, d_ptr_A_small, 1);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS){
			std::cout << "cublas failed!" << std::endl;
		}
		// Sum A with K
		cublasStatus = cublasZaxpy(cublasHandle, size_small, &one, d_ptr_K_small, 1, d_ptr_A_small, 1);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS){
			std::cout << "cublas failed during matrix assembly!" << std::endl;
		}

		// Compute workspace size
		int sizeWorkspace_small;
		cusolverStatus = cusolverDnZgetrf_bufferSize(cusolverHandle, row_small, row_small, d_ptr_A_small, row_small, &sizeWorkspace_small);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver workspace size computation failed for small system\n" << std::endl;

		// Create workspace
		thrust::device_vector<cuDoubleComplex> d_workspace_small(sizeWorkspace_small);
		cuDoubleComplex *d_ptr_workspace_small = thrust::raw_pointer_cast(d_workspace_small.data());

		// LU decomposition
		cusolverStatus = cusolverDnZgetrf(cusolverHandle, row_small, row_small, d_ptr_A_small, row_small, d_ptr_workspace_small, NULL, d_ptr_solverInfo);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver LU decomposition failed (small)" << std::endl;
		solverInfo = d_solverInfo;
		if (solverInfo[0] != 0){
			std::cout << ">>>> LU decomposition failed for small matrix" << std::endl;
		}

		// Solve x = A\b
		cusolverStatus = cusolverDnZgetrs(cusolverHandle, CUBLAS_OP_N, row_small, 1, d_ptr_A_small, row_small, NULL, d_ptr_rhs_small, row_small, d_ptr_solverInfo);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> System couldn't be solved" << std::endl;
		solverInfo = d_solverInfo;
		if (solverInfo[0] != 0) {
			std::cout << ">>>> System solution failure (small)" << std::endl;
		}
		time_small = clock() - time_small;

		/*------------
		Mid matrices
		------------*/
		time_mid = clock();
		// Assemble global matrix
		d_A_mid   = d_M_mid;
		cublasStatus = cublasZdscal(cublasHandle, size_mid, &freq_square, d_ptr_A_mid, 1);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS){
			std::cout << "cublas failed!" << std::endl;
		}
		cublasStatus = cublasZaxpy(cublasHandle, size_mid, &one, d_ptr_K_mid, 1, d_ptr_A_mid, 1);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS){
			std::cout << "cublas failed during matrix assembly!" << std::endl;
		}

		// Compute workspace size
		int sizeWorkspace_mid;
		cusolverStatus = cusolverDnZgetrf_bufferSize(cusolverHandle, row_mid, row_mid, d_ptr_A_mid, row_mid, &sizeWorkspace_mid);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver workspace size computation failed for mid system\n" << std::endl;

		// Create workspace
		thrust::device_vector<cuDoubleComplex> d_workspace_mid(sizeWorkspace_mid);
		cuDoubleComplex *d_ptr_workspace_mid = thrust::raw_pointer_cast(d_workspace_mid.data());

		// LU Decomposition
		cusolverStatus = cusolverDnZgetrf(cusolverHandle, row_mid, row_mid, d_ptr_A_mid, row_mid, d_ptr_workspace_mid, NULL, d_ptr_solverInfo);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver LU decomposition failed (mid)" << std::endl;
		solverInfo = d_solverInfo;
		if (solverInfo[0] != 0){
			std::cout << ">>>> LU decomposition failed for mid matrix" << std::endl;
		}

		// Solve x = A\b
		cusolverStatus = cusolverDnZgetrs(cusolverHandle, CUBLAS_OP_N, row_mid, 1, d_ptr_A_mid, row_mid, NULL, d_ptr_rhs_mid, row_mid, d_ptr_solverInfo);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> System couldn't be solved" << std::endl;
		solverInfo = d_solverInfo;
		if (solverInfo[0] != 0) {
			std::cout << ">>>> System solution failure (mid)" << std::endl;
		}
		time_mid = clock() - time_mid;

		/*------------
		Large matrices
		------------*/
		time_large = clock();
		// Assemble global matrix
		d_A_large = d_M_large;
		cublasStatus = cublasZdscal(cublasHandle, size_large, &freq_square, d_ptr_A_large, 1);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS){
			std::cout << "cublas failed!" << std::endl;
		}
		cublasStatus = cublasZaxpy(cublasHandle, size_large, &one, d_ptr_K_large, 1, d_ptr_A_large, 1);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS){
			std::cout << "cublas failed during matrix assembly!" << std::endl;
		}

		// Compute workspace size
		int sizeWorkspace_large;
		cusolverStatus = cusolverDnZgetrf_bufferSize(cusolverHandle, row_large, row_large, d_ptr_A_large, row_large, &sizeWorkspace_large);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver workspace size computation failed for large system\n" << std::endl;

		// Create workspace
		thrust::device_vector<cuDoubleComplex> d_workspace_large(sizeWorkspace_large);
		cuDoubleComplex *d_ptr_workspace_large = thrust::raw_pointer_cast(d_workspace_large.data());

		// LU Decomposition
		cusolverStatus = cusolverDnZgetrf(cusolverHandle, row_large, row_large, d_ptr_A_large, row_large, d_ptr_workspace_large, NULL, d_ptr_solverInfo);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver LU decomposition failed (large)" << std::endl;
		solverInfo = d_solverInfo;
		if (solverInfo[0] != 0){
			std::cout << ">>>> LU decomposition failed for large matrix" << std::endl;
		}

		// Solve x = A\b
		cusolverStatus = cusolverDnZgetrs(cusolverHandle, CUBLAS_OP_N, row_large, 1, d_ptr_A_large, row_large, NULL, d_ptr_rhs_large, row_large, d_ptr_solverInfo);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> System couldn't be solved" << std::endl;
		solverInfo = d_solverInfo;
		if (solverInfo[0] != 0) {
			std::cout << ">>>> System solution failure (large)" << std::endl;
		}
		time_large = clock() - time_large;

		// Retrieve results from device to host
		sol_small = d_rhs_small;
		sol_mid = d_rhs_mid;
		sol_large = d_rhs_large;

		// Reset rhs values
		d_rhs_small = d_rhs_small_buf;
		d_rhs_mid   = d_rhs_mid_buf;
		d_rhs_large = d_rhs_large_buf;

		// Output messages
		time_it = clock() - time_it;
		//std::cout << ">> Iteration finished for frequency = " << freq << " || Time taken = " << ((float)time_it)/CLOCKS_PER_SEC << std::endl;
		std::cout << ">>>> Frequency = " << freq << " || " << "Time taken (s): Small = " << ((float)time_small)/CLOCKS_PER_SEC << " || " << "Mid = " << ((float)time_mid)/CLOCKS_PER_SEC << " || " << "Large = " << ((float)time_large)/CLOCKS_PER_SEC << std::endl;

		// Accumulate time measurements
		vec_time_small[i] = ((float)time_small)/CLOCKS_PER_SEC;
		vec_time_mid[i]   = ((float)time_mid)/CLOCKS_PER_SEC;
		vec_time_large[i] = ((float)time_large)/CLOCKS_PER_SEC;
		i++;
	}
	time_loop = clock() - time_loop;

	// Compute average time per matrices
	float time_small_avg = thrust::reduce(vec_time_small.begin(), vec_time_small.end(), (float)0, thrust::plus<float>());
	float time_mid_avg   = thrust::reduce(vec_time_mid.begin(),   vec_time_mid.end(),   (float)0, thrust::plus<float>());
	float time_large_avg = thrust::reduce(vec_time_large.begin(), vec_time_large.end(), (float)0, thrust::plus<float>());
	time_small_avg /= freq_max;
	time_mid_avg   /= freq_max;
	time_large_avg /= freq_max;

	std::cout << "\n" << ">>>> Frequency loop finished" << std::endl;
	std::cout << ">>>>>> Time taken (s) = " << ((float)time_loop)/CLOCKS_PER_SEC << "\n" << std::endl;
	std::cout << ">>>>>> Average time (s) for each matrix: Small = " << time_small_avg << " || " << " Mid = " << time_mid_avg << " || " << " Large = " << time_large_avg << "\n" << std::endl;

	// Write out solution vectors
	io::writeSolVecComplex(sol_small, filepath_sol, filename_sol_small);
	io::writeSolVecComplex(sol_mid, filepath_sol, filename_sol_mid);
	io::writeSolVecComplex(sol_large, filepath_sol, filename_sol_large);

	// Destroy cuBLAS & cuSolver
	cublasDestroy(cublasHandle);
	cusolverDnDestroy(cusolverHandle);

	time_total = clock() - time_total;
	std::cout << ">>>>>> Total execution time = " << ((float)time_total)/CLOCKS_PER_SEC << "\n" << std::endl;

}
