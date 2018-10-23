// Libraries
#include <iostream>
#include <string>
#include <cmath>

// OpenMP
#include <omp.h>

// THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

// CUCOMPLEX
#include <cuComplex.h>

// CUBLAS
#include <cublas_v2.h>

// NVTX: https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
#include <nvToolsExt.h>

// Header files
#include "config/config.cuh"
#include "io/io.cuh"
#include "solver/assembly.cuh"
#include "helper/Timer.cuh"
#include "helper/helper.cuh"

// Definitions
#define	PI	3.14159265359
#define MAX_NUM_THREADS 32

// Pinned Allocators
typedef thrust::system::cuda::experimental::pinned_allocator<int> pinnedAllocInt;
typedef thrust::system::cuda::experimental::pinned_allocator<cuDoubleComplex> pinnedAlloc;
typedef thrust::system::cuda::experimental::pinned_allocator<cuDoubleComplex*> pinnedAllocPtr;

int main (int argc, char *argv[]){

    /*--------------------
    COMMAND LINE ARGUMENTS
    --------------------*/
    double freq_max;
    int mat_repetition, subComponents, num_streams, num_threads, batchSize;
    // Configure test environment with command line arguments
    config::configureTest(argc, argv, freq_max, mat_repetition, subComponents, num_streams, num_threads, batchSize);

    /*---------------------
    FILEPATHS AND FILENAMES
    ---------------------*/
    // Vector of filepaths
    std::string filepath[2];
    filepath[0] = "/opt/software/examples/MOR/r_approx_180/\0";
    filepath[1] = "/opt/software/examples/MOR/r_approx_300/\0";
    // Solution filepath
    std::string filepath_sol = "output/";
    // Solution filename
    std::string filename_sol = "solution.dat";
    // Array of filenames
    std::string baseName_K = "KSM_Stiffness_r\0";
    std::string baseName_M = "KSM_Mass_r\0";
    std::string baseName_D = "KSM_Damping_r\0";
    std::string base_format = ".mtx\0";
    std::string filename_K[12];
    std::string filename_M[12];
    std::string filename_D[12];
    std::string filename_input[12];

    /*--------
    PARAMETERS
    --------*/
    bool isComplex = 1;
    double freq, freq_square;
    const double alpha = 4*PI*PI;
    cuDoubleComplex rhs_val;
    rhs_val.x = (double)1.0;
    rhs_val.y = (double)0.0;
    // Array of matrix sizes (row)
    int row_baseline[] = {126, 132, 168, 174, 180, 186, 192, 288, 294, 300, 306, 312};
    // Sort the array row_baseline
    //std::sort(row_baseline.begin(), row_baseline.end());

    /*----------------------------
    OPENMP & CUBLAS INITIALIZATION
    ----------------------------*/
    // OpenMP
    int tid;
    omp_set_num_threads(num_threads);
    // cuBLAS
    timerTotal.start();
    cublasHandle_t cublasHandle[MAX_NUM_THREADS];
    for (size_t i = 0; i < num_threads; ++i) cublasCreate(cublasHandle + i);

    /*-----------------------
    CHECK MEMORY REQUIREMENTS
    -----------------------*/
    config::check_memory(mat_repetition, freq_max, num_threads);

    /*-------------
    DATA STRUCTURES
    --------------*/
    // Create matrix host_vectors
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> K_sub(12);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> M_sub(12);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> D_sub(12);

    // Read and process MTX file
    for (size_t i = 0; i < 7; ++i){
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

    for (size_t i = 7; i < 12; ++i){
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
    thrust::host_vector<int> row_sub(subComponents);
    thrust::host_vector<int> nnz_sub(subComponents);
    int nnz = 0;
    int row = 0;
    size_t idx;
    for (size_t j = 0; j < mat_repetition; ++j){
        for (size_t i = 0; i < 12; ++i){
            idx = i + 12*j;
            row_sub[idx] = row_baseline[i];
            nnz_sub[idx] = row_sub[i]*row_sub[i];
            nnz += nnz_sub[idx];
            row += row_sub[idx];
        }
    }

    // Get maximum matrix size
    auto nnz_max_it = thrust::max_element(nnz_sub.begin(), nnz_sub.end());
    int nnz_max = *nnz_max_it;

    // Combine matrices into a single array on host (to make use of GPU's high bandwidth. We could also import the matrices directly like this)
    thrust::host_vector<cuDoubleComplex> K(nnz);
    thrust::host_vector<cuDoubleComplex> M(nnz);
    thrust::host_vector<cuDoubleComplex> D(nnz);
    auto K_sub_ptr = &K_sub[0];
    auto M_sub_ptr = &M_sub[0];
    auto D_sub_ptr = &D_sub[0];
    size_t array_shift = 0;
    for (size_t j = 0; j < mat_repetition; ++j){
        for (size_t i = 0; i < 12; ++i){
            K_sub_ptr = &K_sub[i];
            M_sub_ptr = &M_sub[i];
            D_sub_ptr = &D_sub[i];
            thrust::copy(K_sub_ptr->begin(), K_sub_ptr->end(), K.begin() + array_shift);
            thrust::copy(M_sub_ptr->begin(), M_sub_ptr->end(), M.begin() + array_shift);
            thrust::copy(D_sub_ptr->begin(), D_sub_ptr->end(), D.begin() + array_shift);
            array_shift += nnz_sub[i];
        }
    }

    std::cout <<">> Matrices combined" << std::endl;

    /*----------------------
    DATA STRUCTURES (DEVICE)
    ----------------------*/
    nvtxRangePushA("Data Structures (Device)");
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
    thrust::device_vector<cuDoubleComplex> d_rhs(row*freq_max, rhs_val);
    timerMatrixCpy.stop();
    std::cout << ">> RHS copied to device " << std::endl;
    std::cout << ">>>> Time taken = " << timerMatrixCpy.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;

    // Create matrix device_vectors
    thrust::device_vector<cuDoubleComplex> d_A(num_threads*freq_max*nnz_max);

    // Get vector of raw pointers to matrices
    cuDoubleComplex *d_ptr_K_base = thrust::raw_pointer_cast(d_K.data());
    cuDoubleComplex *d_ptr_M_base = thrust::raw_pointer_cast(d_M.data());
    cuDoubleComplex *d_ptr_D_base = thrust::raw_pointer_cast(d_D.data());

    // Get local shifts
    thrust::host_vector<cuDoubleComplex*> h_ptr_K(subComponents);
    thrust::host_vector<cuDoubleComplex*> h_ptr_M(subComponents);
    thrust::host_vector<cuDoubleComplex*> h_ptr_D(subComponents);
    size_t mat_shift = 0;
    size_t sol_shift = 0;
    thrust::host_vector<int> shift_local_rhs(subComponents);
    thrust::host_vector<int> shift_local_A(subComponents);
    for (size_t i = 0; i < subComponents; ++i){
        h_ptr_K[i] = d_ptr_K_base + mat_shift;
        h_ptr_M[i] = d_ptr_M_base + mat_shift;
        h_ptr_D[i] = d_ptr_D_base + mat_shift;
        shift_local_rhs[i] = sol_shift;
        shift_local_A[i]   = mat_shift;
        mat_shift += nnz_sub[i];
        sol_shift += row_sub[i];
    }

    // Get raw pointers to matrix A and rhs
    cuDoubleComplex *d_ptr_A_base = thrust::raw_pointer_cast(d_A.data());
    cuDoubleComplex *d_ptr_rhs_base = thrust::raw_pointer_cast(d_rhs.data());
    nvtxRangePop();

    timerMatrixComp.start();
    // M = 4*pi^2*M (Single computation suffices)
    cublas_check(cublasZdscal(cublasHandle[0], nnz, &alpha, d_ptr_M_base, 1));
    timerMatrixComp.stop();
    std::cout << ">> M_tilde computed with cuBLAS" << std::endl;
    std::cout << ">>>> Time taken = " << timerMatrixComp.getDurationMicroSec()*1e-6 << " (sec)\n" << std::endl;

    // Solver Info for batched LU decomposition
    thrust::device_vector<int> d_solverInfo(batchSize);
    int *d_ptr_solverInfo = thrust::raw_pointer_cast(d_solverInfo.data());
    int solverInfo_solve;

    // Stream initialisation
    cudaStream_t streams[num_streams];
    for (size_t i = 0; i < num_streams; ++i){
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        std::cout << ">> Stream " << i << " created" << std::endl;
    }

    /*--------------------
    Krylov Subspace Method
    --------------------*/
    nvtxRangePushA("Krylov Subspace Method");
    timerLoop.start();
    std::cout << "\n>> Matrix loop started for batched execution" << std::endl;
#pragma omp parallel private(tid, freq, freq_square) num_threads(num_threads)
    {
        // Get thread number
        tid = omp_get_thread_num();
        // Allocate vector of array pointers to A in each thread
        thrust::device_vector<cuDoubleComplex*> d_ptr_A(batchSize);
        thrust::device_vector<cuDoubleComplex*> d_ptr_rhs(batchSize);
        thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_A(batchSize);
        thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_rhs(batchSize);
        // Initialise shifts
        int shift_global_A, shift_batch_A, shift_global_rhs;
        shift_global_A = tid*freq_max*nnz_max;
        // Set cuBLAS stream
        cublasSetStream(cublasHandle[tid], streams[tid]);
    // Loop over each matrix size
    #pragma omp for
        for (size_t i = 0; i < subComponents; ++i){
            /*---------------------------------------------------------------
            Assemble Global Matrix & Update pointers to each matrix A and RHS
            ---------------------------------------------------------------*/
            // Initialise Shifts
            shift_global_rhs = 0;
            shift_batch_A = 0;
            // Loop over batch (assume batchSize = freq_max)
            for (size_t j = 0; j < batchSize; ++j){
                // Update matrix A pointer
                h_ptr_A[j] = d_ptr_A_base + shift_batch_A + shift_global_A;
                // Update rhs pointer
                h_ptr_rhs[j] = d_ptr_rhs_base + shift_local_rhs[i] + shift_global_rhs;
                // Compute frequency (assume batchSize = freq_max)
                freq = (j+1);
                freq_square = -(freq*freq);
                // Assemble matrix
                nvtxRangePushA("Matrix Assembly");
                assembly::assembleGlobalMatrixBatched(streams[tid], h_ptr_A[j], h_ptr_K[i], h_ptr_M[i], nnz_sub[i], freq_square);
                nvtxRangePop();
                // Update shifts
                shift_batch_A    += nnz_sub[i];
                shift_global_rhs += row;
            }
            nvtxRangePushA("Linear System");
            /*--------------
            LU Decomposition
            --------------*/
            d_ptr_A = h_ptr_A;
            cublas_check(cublasZgetrfBatched(cublasHandle[tid], row_sub[i], thrust::raw_pointer_cast(d_ptr_A.data()), row_sub[i], NULL, d_ptr_solverInfo, batchSize));
            /*-----------
            Solve x = A\b
            -----------*/
            d_ptr_rhs = h_ptr_rhs;
            cublas_check(cublasZgetrsBatched(cublasHandle[tid], CUBLAS_OP_N, row_sub[i], 1, thrust::raw_pointer_cast(d_ptr_A.data()), row_sub[i], NULL,
                                             thrust::raw_pointer_cast(d_ptr_rhs.data()), row_sub[i], &solverInfo_solve, batchSize));
            /*-----------------
            Synchronize Streams
            -----------------*/
            cudaStreamSynchronize(streams[tid]);
        } // matrix loop
    } // omp parallel
    nvtxRangePop();

    timerLoop.stop();
    nvtxRangePop();

    std::cout << ">> Matrix loop finished" << std::endl;
    std::cout << ">>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6 << "\n" << std::endl;

    // Copy solution from device to host
    thrust::host_vector<cuDoubleComplex> rhs = d_rhs;

/*
    io::writeSolVecComplex(rhs, filepath_sol, filename_sol);
    thrust::host_vector<cuDoubleComplex> A = d_A;
    io::writeSolVecComplex(A, filepath_sol, "A.dat");
*/

    // Destroy cuBLAS & streams
    for (size_t i = 0; i < num_threads; ++i){
        cublasDestroy(cublasHandle[i]);
        cudaStreamDestroy(streams[i]);
    }

    timerTotal.stop();
    std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}
