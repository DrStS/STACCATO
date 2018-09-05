// Libraries
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

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

// CUSOLVER
#include <cusolverDn.h>

// Header files
#include "io/io.cuh"
#include "solver/assembly.cuh"
#include "helper/Timer.cuh"
#include "helper/helper.cuh"

// Definitions
#define	PI	3.14159265359

int main (int argc, char *argv[]){

    // Command line arguments
    if (argc < 5){
        std::cerr << ">> Usage: " << argv[0] << " -f <maximum frequency> -m <matrix repetition> -stream <number of CUDA streams>" << std::endl;
        std::cerr << ">> NOTE: There are 12 matrices and matrix repetition increases the total number of matrices (e.g. matrix repetition of 5 will use 60 matrices)" << std::endl;
        std::cerr << "         Frequency starts from 1 to maximum frequency" << std::endl;
        std::cerr << "         Default number of CUDA streams is 1" << std::endl;
        std::cerr << "         Ratio of number of matrices (12*matrix_repetition) to number of CUDA streams must be an integer" << std::endl;
        return 1;
    }

    double freq_max = atof(argv[2]);
    int mat_repetition = atoi(argv[4]);
    int num_matrix = mat_repetition*12;
    int num_streams = 1;

    if (argc > 6) num_streams = atoi(argv[6]);
    int num_threads = num_streams;
    std::cout << ">> Maximum Frequency: " << freq_max << std::endl;
    std::cout << ">> Total number of matrices: " << num_matrix << std::endl;
    std::cout << ">> Number of CUDA streams: " << num_streams << "\n" << std::endl;

    if ((num_matrix % num_streams) != 0) {
        std::cerr << ">> ERROR: Invalid number of streams\n" << std::endl;
        return 1;
    }

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
    cuDoubleComplex one;	// Dummy scailing factor for global matrix assembly
    one.x = 1;
    one.y = 0;
    cuDoubleComplex rhs_val;
    rhs_val.x = (double)1.0;
    rhs_val.y = (double)0.0;

    // OpenMP
    int tid;
    omp_set_num_threads(num_threads);

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
    int nnz = 0;
    int row = 0;
    size_t idx;
    for (size_t j = 0; j < mat_repetition; j++){
        for (size_t i = 0; i < 12; i++){
            idx = i + 12*j;
            row_sub[idx] = row_baseline[i];
            size_sub[idx] = row_sub[i]*row_sub[i];
            nnz += size_sub[idx];
            row  += row_sub[idx];
        }
    }

    // Combine matrices into a single array on host (to make use of GPU's high bandwidth. We could also import the matrices directly like this)
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
    thrust::device_vector<cuDoubleComplex> d_A(nnz*num_threads);

    // Get raw pointers to matrices
    cuDoubleComplex *d_ptr_K = thrust::raw_pointer_cast(d_K.data());
    cuDoubleComplex *d_ptr_M = thrust::raw_pointer_cast(d_M.data());
    cuDoubleComplex *d_ptr_D = thrust::raw_pointer_cast(d_D.data());

    // Vector of raw pointers to assembled matrix
    thrust::host_vector<cuDoubleComplex*> d_ptr_A(num_streams);
    size_t mat_shift = 0;
    cuDoubleComplex *d_ptr_A_base = thrust::raw_pointer_cast(d_A.data());
    for (size_t i = 0; i < num_streams; i++){
        d_ptr_A[i] = d_ptr_A_base + mat_shift;
        mat_shift += nnz;
    }

    // Get raw pointers to RHS vectors
    cuDoubleComplex *d_ptr_rhs = thrust::raw_pointer_cast(d_rhs.data());

    // Create solution vector on host
    thrust::host_vector<cuDoubleComplex> sol(row*freq_max);

    // Create solution vector on device
    thrust::device_vector<cuDoubleComplex> d_sol(row*freq_max);

    timerMatrixComp.start();
    // M = 4*pi^2*M (Single computation suffices)
    cublasStatus = cublasZdscal(cublasHandle, nnz, &alpha, d_ptr_M, 1);
    timerMatrixComp.stop();
    std::cout << ">> M_tilde computed with cuBLAS" << std::endl;
    std::cout << ">>>> Time taken = " << timerMatrixComp.getDurationMicroSec()*1e-6 << " (sec)\n" << std::endl;
    if (cublasStatus != CUBLAS_STATUS_SUCCESS){
        std::cout << "cublas failed!" << std::endl;
    }

    // LU decomposition prep
    thrust::device_vector<int> d_solverInfo(num_threads);
    thrust::device_vector<int*> d_ptr_solverInfo(num_threads);
    for (size_t i = 0; i < num_threads; i++) d_ptr_solverInfo[i] = thrust::raw_pointer_cast(d_solverInfo.data()) + i;

    // Compute workspace size
    int totalSizeWorkspace = 0;
    thrust::host_vector<int> sizeWorkspace(num_matrix);
    auto sizeWorkspace_ptr = &sizeWorkspace[0];
    for (size_t i = 0; i < num_matrix; i++){
        sizeWorkspace_ptr = &sizeWorkspace[i];
        cusolverStatus = cusolverDnZgetrf_bufferSize(cusolverHandle, row_sub[i], row_sub[i], d_ptr_A[i], row_sub[i], sizeWorkspace_ptr);
        if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) std::cout << ">> cuSolver workspace size computation failed\n" << std::endl;
        totalSizeWorkspace += sizeWorkspace[i];
    }

    // Create workspace
    thrust::device_vector<cuDoubleComplex> d_workspace(totalSizeWorkspace*num_threads);
    // Create vector of raw pointers to workspace
    thrust::host_vector<cuDoubleComplex*> d_ptr_workspace(num_threads);
    size_t workspace_shift = 0;
    for (size_t i = 0; i < num_threads; i++){
        d_ptr_workspace[i] = thrust::raw_pointer_cast(d_workspace.data()) + workspace_shift;
        workspace_shift += totalSizeWorkspace;
    }

    // Stream initialisation
    cudaStream_t streams[num_streams];
    for (size_t i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        std::cout << ">> Stream " << i << " created" << std::endl;
    }

    timerLoop.start();
    // Loop over frequency
    int i;
    int sol_shift = 0;
    std::cout << "\n>> Frequency loop started" << std::endl;
#pragma omp parallel private(tid, freq, freq_square, cublasStatus, i, cusolverStatus, array_shift) num_threads(num_threads)
    {
        // Get thread number
        tid = omp_get_thread_num();
        // Initialise shifts
        array_shift = 0;

#pragma omp for
        for (size_t it = (size_t)freq_min; it <= (size_t)freq_max; it++){

            /*--------------------
            Assemble global matrix
            --------------------*/
            // Compute scaling
            freq = (double)it;
            freq_square = -(freq*freq);
            assembly::assembleGlobalMatrix(tid, streams[tid], cublasStatus, cublasHandle, d_ptr_A[tid], d_ptr_K, d_ptr_M, nnz, one, freq_square);

            /*--------------
            LU decomposition
            --------------*/
            for (i = 0; i < num_matrix; i++){
                #pragma omp critical
                std::cout << "i = " << i << "   row_sub = " << row_sub[i] << "  array_shift = " << array_shift << "     d_ptr_A = " << d_ptr_A[tid] << std::endl;
/*
                // LU decomposition
                cusolverDnSetStream(cusolverHandle, streams[tid]);
                cusolverStatus = cusolverDnZgetrf(cusolverHandle, row_sub[i], row_sub[i], d_ptr_A[tid] + array_shift, row_sub[i], d_ptr_workspace[0], NULL, d_ptr_solverInfo[tid]);
                assert(cusolverStatus == CUSOLVER_STATUS_SUCCESS);

/*
                // Solve A\b
                cusolverDnSetStream(cusolverHandle, streams[tid]);
                cusolverStatus = cusolverDnZgetrs(cusolverHandle, CUBLAS_OP_N, row_sub[tid], 1, d_ptr_A_tmp[tid], row_sub[tid], NULL, d_ptr_rhs_tmp[tid], row_sub[tid], d_ptr_solverInfo[tid]);
                assert(cusolverStatus == CUSOLVER_STATUS_SUCCESS);
*/

                array_shift += size_sub[i];
            }

            // Synchronize streams
            for (size_t st = 0; st < num_streams; st++) cudaStreamSynchronize(streams[st]);
            // Copy the solution to solution vector
            thrust::copy(d_rhs.begin(), d_rhs.end(), d_sol.begin() + sol_shift);
            sol_shift += row;
            // Reset RHS
            d_rhs = d_rhs_buf;

        } // frequency loop
    } // omp parallel
    timerLoop.stop();

    std::cout << ">> Frequency loop finished" << std::endl;
    std::cout << ">>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6 << "\n" << std::endl;

    sol = d_sol;

    thrust::host_vector<cuDoubleComplex> A = d_A;
    io::writeSolVecComplex(A, filepath_sol, "A.dat");

    // Write out solution vectors
    //io::writeSolVecComplex(sol, filepath_sol, filename_sol);

    // Destroy cuBLAS & cuSolver
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    // Destroy streams
    for (size_t i = 0; i < num_streams; i++) cudaStreamDestroy(streams[i]);

    timerTotal.stop();
    std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}
