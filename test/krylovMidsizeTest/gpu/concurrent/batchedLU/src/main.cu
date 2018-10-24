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
#include "data/dataStructures.cuh"
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

// Namespace
using namespace staccato;

int main (int argc, char *argv[]){

    nvtxRangePushA("Initial Configuration (Host)");
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
    const double alpha = 4*PI*PI;
    cuDoubleComplex rhs_val;
    rhs_val.x = (double)1.0;
    rhs_val.y = (double)0.0;
    // Array of matrix sizes (row)
    int row_baseline[] = {126, 132, 168, 174, 180, 186, 192, 288, 294, 300, 306, 312};
    // Frequency vector
    thrust::host_vector<int, pinnedAllocInt> freq(batchSize);
    thrust::host_vector<int, pinnedAllocInt> freq_square(batchSize);

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

    nvtxRangePop(); // Initial Configuration

    /*--------------------
    DATA STRUCTURES (HOST)
    --------------------*/
    nvtxRangePushA("Data Structures (Host)");
    // Create matrix host_vectors
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> K_sub(12);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> M_sub(12);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> D_sub(12);
    thrust::host_vector<cuDoubleComplex> K;
    thrust::host_vector<cuDoubleComplex> M;
    thrust::host_vector<cuDoubleComplex> D;
    // Array information
    thrust::host_vector<int> row_sub(subComponents);
    thrust::host_vector<int> nnz_sub(subComponents);
    int nnz, row, nnz_max;
    thrust::host_vector<int> shift_local_A(subComponents);
    thrust::host_vector<int> shift_local_rhs(subComponents);
    // Set up host data structures
    data::constructHostDataStructure(filename_K, filename_M, filename_D, filepath, baseName_K, baseName_M, baseName_D, base_format, row_baseline,
                                     K_sub, M_sub, D_sub, shift_local_A, shift_local_rhs, row_sub, nnz_sub, nnz, row, nnz_max, mat_repetition, K, M, D);

    nvtxRangePop(); // Data Structures (Host)

    /*----------------------
    DATA STRUCTURES (DEVICE)
    ----------------------*/
    nvtxRangePushA("Data Structures (Device)");
    // Send matrices to device
    thrust::device_vector<cuDoubleComplex> d_K = K;
    thrust::device_vector<cuDoubleComplex> d_M = M;
    thrust::device_vector<cuDoubleComplex> d_D = D;
    // Create RHS vector directly on device (will be replaced with send operation)
    thrust::device_vector<cuDoubleComplex> d_rhs(row*freq_max, rhs_val);
    // Create matrix device_vectors
    thrust::device_vector<cuDoubleComplex> d_A(num_threads*freq_max*nnz_max);
    // Get raw pointers to device matrices & vectors
    cuDoubleComplex *d_ptr_K_base = thrust::raw_pointer_cast(d_K.data());
    cuDoubleComplex *d_ptr_M_base = thrust::raw_pointer_cast(d_M.data());
    cuDoubleComplex *d_ptr_D_base = thrust::raw_pointer_cast(d_D.data());
    cuDoubleComplex *d_ptr_A_base = thrust::raw_pointer_cast(d_A.data());
    cuDoubleComplex *d_ptr_rhs_base = thrust::raw_pointer_cast(d_rhs.data());
    // Create array of pointers for each sub-components from combined matrices on device
    thrust::host_vector<cuDoubleComplex*> h_ptr_K(subComponents);
    thrust::host_vector<cuDoubleComplex*> h_ptr_M(subComponents);
    thrust::host_vector<cuDoubleComplex*> h_ptr_D(subComponents);
    // Get information from device data structures
    data::getInfoDeviceDataStructure(h_ptr_K, h_ptr_M, h_ptr_D, d_ptr_K_base, d_ptr_M_base, d_ptr_D_base, nnz_sub, subComponents);

    nvtxRangePop(); // Data Structures (Device)


    /*--------------------------------
    Krylov Subspace Method Preparation
    --------------------------------*/
    nvtxRangePushA("Krylov Subspace Method Preparation");
    // M = 4*pi^2*M
    cublas_check(cublasZdscal(cublasHandle[0], nnz, &alpha, d_ptr_M_base, 1));
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

    nvtxRangePop(); // Krylov Subspace Method Preparation

    /*--------------------
    Krylov Subspace Method
    --------------------*/
    nvtxRangePushA("Krylov Subspace Method");
    timerLoop.start();
    std::cout << "\n>> Matrix loop started for batched execution" << std::endl;
#pragma omp parallel private(tid) num_threads(num_threads)
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
                freq[j] = (j+1);
                freq_square[j] = -(freq[j]*freq[j]);
                // Assemble matrix
                nvtxRangePushA("Matrix Assembly");
                assembly::assembleGlobalMatrixBatched(streams[tid], h_ptr_A[j], h_ptr_K[i], h_ptr_M[i], nnz_sub[i], freq_square[j]);
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


    io::writeSolVecComplex(rhs, filepath_sol, filename_sol);
/*
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
