// Libraries
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// OpenMP
#include <omp.h>

// THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// CUCOMPLEX
#include <cuComplex.h>

// CUBLAS
#include <cublas_v2.h>

// CUSPARSE
#include <cusparse.h>

// Header files
#include "io/io.cuh"
#include "helper/Timer.cuh"
#include "helper/helper.cuh"
#include "helper/math.cuh"

// Definitions
#define	PI	3.14159265359

int main (int argc, char *argv[]){

	// Command line arguments
	if (argc < 5){
		std::cerr << ">> Usage: " << argv[0] << " -f <maximum frequency> -m <matrix repetition>" << std::endl;
		std::cerr << ">> NOTE: There are 12 matrices and matrix repetition increases the total number of matrices (e.g. matrix repetition of 5 will use 60 matrices)" << std::endl;
		std::cerr << "         Frequency starts from 1 to maximum frequency" << std::endl;
		return 1;
	}

	double freq_max = atof(argv[2]);
	int mat_repetition = atoi(argv[4]);
	int num_matrix = mat_repetition*12;
	std::cout << ">> Maximum Frequency: " << freq_max << std::endl;
	std::cout << ">> Total number of matrices: " << num_matrix << "\n" << std::endl;

	// Vector of filepaths
	std::string filepath[2];
	filepath[0] = "/opt/software/examples/MOR/r_approx_180/\0";
	filepath[1] = "/opt/software/examples/MOR/r_approx_300/\0";

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

	// Parameters
	bool isComplex = 1;
	double freq, freq_square;
	double freq_min = 1;
	const double alpha = 4*PI*PI;
	cuDoubleComplex one;			// Dummy scailing factor for global matrix assembly
	one.x = 1;
	one.y = 0;
	cuDoubleComplex rhs_val;
	rhs_val.x = (double)1.0;
	rhs_val.y = (double)0.0;

	// OpenMP
	int num_threads = num_matrix;
	omp_set_num_threads(num_threads);

	timerTotal.start();

	// Library initialisation
	cublasStatus_t cublasStatus;
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	cusparseStatus_t cusparseStatus;
	cusparseHandle_t cusparseHandle;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	// Create matrix host_vectors
	thrust::host_vector<thrust::host_vector<cuDoubleComplex>> K_sub(12);
	thrust::host_vector<thrust::host_vector<cuDoubleComplex>> M_sub(12);
	thrust::host_vector<thrust::host_vector<cuDoubleComplex>> D_sub(12);

	// Read and process MTX file
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
	thrust::host_vector<int> row_sub(num_matrix);
	thrust::host_vector<int> size_sub(num_matrix);
	thrust::host_vector<size_t> ptr_mat_shift(num_matrix);
	thrust::host_vector<size_t> ptr_vec_shift(num_matrix);
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
	thrust::host_vector<cuDoubleComplex> K(nnz);
	thrust::host_vector<cuDoubleComplex> M(nnz);
	thrust::host_vector<cuDoubleComplex> D(nnz);
	auto K_sub_ptr = &K_sub[0];
	auto M_sub_ptr = &M_sub[0];
	auto D_sub_ptr = &D_sub[0];
	size_t array_shift = 0;
	for (size_t j = 0; j < mat_repetition; j++){
		for (size_t i = 0; i < 12; i++){
			K_sub_ptr = &K_sub[i];
			M_sub_ptr = &M_sub[i];
			D_sub_ptr = &D_sub[i];
			thrust::copy(K_sub_ptr->begin(), K_sub_ptr->end(), K.begin() + array_shift);
			thrust::copy(M_sub_ptr->begin(), M_sub_ptr->end(), M.begin() + array_shift);
			thrust::copy(D_sub_ptr->begin(), D_sub_ptr->end(), D.begin() + array_shift);
			array_shift += size_sub[i];
		}
	}

	std::cout <<">> Matrices combined\n" << std::endl;

	// Generate CSR format
	timerAux.start();
	thrust::host_vector<int> csrRowPtr(row+1);
	thrust::host_vector<int> csrColInd(nnz);
	generateCSR(csrRowPtr, csrColInd, row_sub, size_sub, row, nnz, num_matrix);
	thrust::device_vector<int> d_csrRowPtr = csrRowPtr;
	thrust::device_vector<int> d_csrColInd = csrColInd;
	timerAux.stop();
	std::cout <<">> CSR Format Generated" << std::endl;
	std::cout <<">>>> Time taken = " << timerAux.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;

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
	//thrust::device_vector<cuDoubleComplex> d_rhs_buf = d_rhs;
	timerMatrixCpy.stop();
	std::cout << ">> RHS copied to device " << std::endl;
	std::cout << ">>>> Time taken = " << timerMatrixCpy.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;

	// Create matrix device_vectors
	thrust::device_vector<cuDoubleComplex> d_A(nnz);

	// Get raw pointers to matrices
	cuDoubleComplex *d_ptr_K 	 = thrust::raw_pointer_cast(d_K.data());
	cuDoubleComplex *d_ptr_M 	 = thrust::raw_pointer_cast(d_M.data());
	cuDoubleComplex *d_ptr_D 	 = thrust::raw_pointer_cast(d_D.data());
	cuDoubleComplex *d_ptr_A 	 = thrust::raw_pointer_cast(d_A.data());

	// Get raw pointers to CSR arrays
	int *d_ptr_csrRowPtr = thrust::raw_pointer_cast(d_csrRowPtr.data());
	int *d_ptr_csrColInd = thrust::raw_pointer_cast(d_csrColInd.data());

	// Get raw pointers to RHS vectors
	cuDoubleComplex *d_ptr_rhs = thrust::raw_pointer_cast(d_rhs.data());

	// Create solution vector on host
	thrust::host_vector<cuDoubleComplex> sol(row*freq_max);

	// Create solution vector on device
	thrust::device_vector<cuDoubleComplex> d_z(row);				// Intermediate solution
	thrust::device_vector<cuDoubleComplex> d_sol(row*freq_max);		// Final solution

	// Get raw pointers to solution vector
	cuDoubleComplex *d_ptr_z   = thrust::raw_pointer_cast(d_z.data());
	cuDoubleComplex *d_ptr_sol = thrust::raw_pointer_cast(d_sol.data());

	timerMatrixComp.start();
	// M = 4*pi^2*M (Single computation suffices)
	cublasStatus = cublasZdscal(cublasHandle, nnz, &alpha, d_ptr_M, 1);
	assert(CUBLAS_STATUS_SUCCESS == cublasStatus);
	timerMatrixComp.stop();
	std::cout << ">> M_tilde computed with cuBLAS" << std::endl;
	std::cout << ">>>> Time taken = " << timerMatrixComp.getDurationMicroSec()*1e-6 << " (sec)\n" << std::endl;

	// Stream initialisation
	const int num_streams = num_threads;
	cudaStream_t streams[num_streams];
	for (size_t i = 0; i < num_streams; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

	/*-----------------------------
	LU Decomposition initialisation
	-----------------------------*/
	timerAux.start();
	// Matrix Descriptions
	cusparseMatDescr_t descr_A, descr_L, descr_U;
	cusparseCreateMatDescr(&descr_A);
	cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseCreateMatDescr(&descr_L);
	cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
	cusparseCreateMatDescr(&descr_U);
	cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
	// Solver Infos
	csrilu02Info_t solverInfo_A;
	csrsv2Info_t solverInfo_L, solverInfo_U;
	cusparseCreateCsrilu02Info(&solverInfo_A);
	cusparseCreateCsrsv2Info(&solverInfo_L);
	cusparseCreateCsrsv2Info(&solverInfo_U);
	// Transpose operations
	const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
	const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
	// Solver policies
	const cusparseSolvePolicy_t policy_A = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	// Zero Pivoting
	int structural_zero, numerical_zero;
	// Buffer space
	int bufferSize_A, bufferSize_L, bufferSize_U, bufferSize;
	cusparseStatus = cusparseZcsrilu02_bufferSize(cusparseHandle, row, nnz, descr_A, d_ptr_A, d_ptr_csrRowPtr, d_ptr_csrColInd, solverInfo_A, &bufferSize_A);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStatus);
	cusparseStatus = cusparseZcsrsv2_bufferSize(cusparseHandle, trans_L, row, nnz, descr_L, d_ptr_A, d_ptr_csrRowPtr, d_ptr_csrColInd, solverInfo_L, &bufferSize_L);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStatus);
	cusparseStatus = cusparseZcsrsv2_bufferSize(cusparseHandle, trans_U, row, nnz, descr_U, d_ptr_A, d_ptr_csrRowPtr, d_ptr_csrColInd, solverInfo_U, &bufferSize_U);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStatus);
	bufferSize = std::max(bufferSize_A, std::max(bufferSize_L, bufferSize_U));
	thrust::device_vector<int> d_buffer(bufferSize);
	void* d_ptr_buffer = thrust::raw_pointer_cast(d_buffer.data());
	// Perform analysis
	cusparseStatus = cusparseZcsrilu02_analysis(cusparseHandle, row, nnz, descr_A, d_ptr_A, d_ptr_csrRowPtr, d_ptr_csrColInd, solverInfo_A, policy_A, d_ptr_buffer);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStatus);
	cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle, solverInfo_A, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus) printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
	cusparseStatus = cusparseZcsrsv2_analysis(cusparseHandle, trans_L, row, nnz, descr_L, d_ptr_A, d_ptr_csrRowPtr, d_ptr_csrColInd, solverInfo_L, policy_L, d_ptr_buffer);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStatus);
	cusparseStatus = cusparseZcsrsv2_analysis(cusparseHandle, trans_U, row, nnz, descr_U, d_ptr_A, d_ptr_csrRowPtr, d_ptr_csrColInd, solverInfo_U, policy_U, d_ptr_buffer);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStatus);
	timerAux.stop();
	std::cout << ">> LU decomposition initialised" << std::endl;
	std::cout << ">>>> Time taken (s) = " << timerAux.getDurationMicroSec()*1e-6 << "\n" << std::endl;

	/*------------
	Frequency Loop
	------------*/
	timerLoop.start();
	int sol_shift = 0;
	for (size_t it = (size_t)freq_min; it <= (size_t)freq_max; it++){
		// Compute scaling
		freq = (double)it;
		freq_square = -(freq*freq);

		/*-----------------------------------------------
		// Assemble global matrix ( A = K - f^2*M_tilde )
		-----------------------------------------------*/
		d_A = d_M;
		// Scale A with -f^2
		cublasStatus = cublasZdscal(cublasHandle, nnz, &freq_square, d_ptr_A, 1);
		assert(CUBLAS_STATUS_SUCCESS == cublasStatus);
		// Sum A with K
		cublasStatus = cublasZaxpy(cublasHandle, nnz, &one, d_ptr_K, 1, d_ptr_A, 1);
		assert(CUBLAS_STATUS_SUCCESS == cublasStatus);

		/*--------------
		LU Decomposition
		--------------*/
		cusparseStatus = cusparseZcsrilu02(cusparseHandle, row, nnz, descr_A, d_ptr_A, d_ptr_csrRowPtr, d_ptr_csrColInd, solverInfo_A, policy_A, d_ptr_buffer);
		assert(CUSPARSE_STATUS_SUCCESS == cusparseStatus);
		cusparseStatus = cusparseXcsrilu02_zeroPivot(cusparseHandle, solverInfo_A, &numerical_zero);
		if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseStatus) printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);

		/*-----------
		Solve x = A\b
		-----------*/
		// Solve z = L\b
		cusparseStatus = cusparseZcsrsv2_solve(cusparseHandle, trans_L, row, nnz, &one, descr_L, d_ptr_A, d_ptr_csrRowPtr, d_ptr_csrColInd, solverInfo_L,
												d_ptr_rhs, d_ptr_z, policy_L, d_ptr_buffer);
		assert(CUSPARSE_STATUS_SUCCESS == cusparseStatus);
		// Solve x = U\z
		cusparseStatus = cusparseZcsrsv2_solve(cusparseHandle, trans_U, row, nnz, &one, descr_U, d_ptr_A, d_ptr_csrRowPtr, d_ptr_csrColInd, solverInfo_U,
												d_ptr_z, d_ptr_sol+sol_shift, policy_U, d_ptr_buffer);
		assert(CUSPARSE_STATUS_SUCCESS == cusparseStatus);
		// Update solution vector shift
		sol_shift += row;
	}
	timerLoop.stop();

	std::cout << ">>>> Frequency loop finished" << std::endl;
	std::cout << ">>>>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6 << "\n" << std::endl;

	sol = d_sol;

	// Write out solution vectors
	io::writeSolVecComplex(sol, filepath_sol, filename_sol);

	// Destroy cuBLAS & cuSparse
	cublasDestroy(cublasHandle);
	cusparseDestroy(cusparseHandle);

	// Destroy streams
	for (size_t i = 0; i < num_streams; i++) cudaStreamDestroy(streams[i]);

	timerTotal.stop();
	std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}
