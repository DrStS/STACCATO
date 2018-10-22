// Libraries
#include <iostream>
#include <string>
#include <cmath>

// OpenMP
#include <omp.h>

// THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
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
    int mat_repetition, num_matrix, num_streams, num_threads, batchSize;
    // Configure test environment with command line arguments
    config::configureTest(argc, argv, freq_max, mat_repetition, num_matrix, num_streams, num_threads, batchSize);

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

    /*--------
    PARAMETERS
    --------*/
    bool isComplex = 1;
    const double alpha = 4*PI*PI;
    cuDoubleComplex rhs_val;
    rhs_val.x = (double)1.0;
    rhs_val.y = (double)0.0;
    // Array of matrix sizes (row)
    int row_baseline[] = {126, 132, 168, 174, 180, 186, 192, 288, 294, 300, 306, 312};
    // Sort the array row_baseline
    //std::sort(row_baseline.begin(), row_baseline.end());
    // Frequency vector
    thrust::host_vector<int, pinnedAllocInt> freq(batchSize);
    thrust::host_vector<int, pinnedAllocInt> freq_square(batchSize);
    // Fill in frequency vectors
    thrust::sequence(freq.begin(), freq.end(), 1);
    thrust::transform(freq.begin(), freq.end(), freq.begin(), freq_square.begin(), thrust::multiplies<int>());

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
    thrust::host_vector<int> row_sub(num_matrix);
    thrust::host_vector<int> size_sub(num_matrix);
    int nnz = 0;
    int row = 0;
    size_t idx;
    for (size_t j = 0; j < mat_repetition; ++j){
        for (size_t i = 0; i < 12; ++i){
            idx = i + 12*j;
            row_sub[idx] = row_baseline[i];
            size_sub[idx] = row_sub[i]*row_sub[i];
            nnz += size_sub[idx];
            row  += row_sub[idx];
        }
    }
    int dofReduced = row*freq_max;

    // Get maximum matrix size
    auto nnz_max_it = thrust::max_element(size_sub.begin(), size_sub.end());
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
    thrust::device_vector<cuDoubleComplex> d_rhs(dofReduced, rhs_val);
    timerMatrixCpy.stop();
    std::cout << ">> RHS copied to device " << std::endl;
    std::cout << ">>>> Time taken = " << timerMatrixCpy.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;

    // Create matrix device_vectors
    thrust::device_vector<cuDoubleComplex> d_A(num_threads*freq_max*nnz_max);

    // remove later
    cuDoubleComplex eleven;
    eleven.x = 11;
    eleven.y = 11;
    thrust::fill(d_A.begin(), d_A.end(), eleven);

    // Get vector of raw pointers to matrices
    cuDoubleComplex *d_ptr_K_base = thrust::raw_pointer_cast(d_K.data());
    cuDoubleComplex *d_ptr_M_base = thrust::raw_pointer_cast(d_M.data());
    cuDoubleComplex *d_ptr_D_base = thrust::raw_pointer_cast(d_D.data());
    thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_K(num_matrix);
    thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_M(num_matrix);
    thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_D(num_matrix);
    size_t mat_shift = 0;
    size_t sol_shift = 0;
    thrust::host_vector<int> loop_shift(num_matrix);
    for (size_t i = 0; i < num_matrix; ++i){
        h_ptr_K[i] = d_ptr_K_base + mat_shift;
        h_ptr_M[i] = d_ptr_M_base + mat_shift;
        h_ptr_D[i] = d_ptr_D_base + mat_shift;
        loop_shift[i] = sol_shift;
        mat_shift += size_sub[i];
        sol_shift += row_sub[i];
    }

    // Get raw pointers to matrix A and rhs
    cuDoubleComplex *d_ptr_A_base = thrust::raw_pointer_cast(d_A.data());
    cuDoubleComplex *d_ptr_rhs_base = thrust::raw_pointer_cast(d_rhs.data());

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
#pragma omp parallel private(tid, array_shift) num_threads(num_threads)
    {
        // Get thread number
        tid = omp_get_thread_num();
        // Indices
        size_t j;
        // Allocate vector of array pointers to A in each thread
        thrust::device_vector<cuDoubleComplex*> d_ptr_A(batchSize);
        thrust::device_vector<cuDoubleComplex*> d_ptr_rhs(batchSize);
        thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_A(batchSize);
        thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_rhs(batchSize);
        // Initialise shifts
        int thread_shift, rhs_shift;
        rhs_shift = 0;
        thread_shift = tid*freq_max*nnz_max;
        // Set cuBLAS stream
        cublasSetStream(cublasHandle[tid], streams[tid]);

    // Loop over each matrix size
    #pragma omp for
        for (size_t i = 0; i < num_matrix; ++i){
        //for (size_t i = 0; i < 1; ++i){
            /*--------------------------------------
            Update pointers to each matrix A and RHS
            --------------------------------------*/
            // Initialise Shifts
            array_shift = 0;
            rhs_shift = 0;
            // Loop over frequency points
            for (j = 0; j < freq_max; ++j){
                // Update matrix A pointer
                h_ptr_A[j] = d_ptr_A_base + thread_shift + array_shift;
                // Update rhs pointer
                h_ptr_rhs[j] = d_ptr_rhs_base + rhs_shift + loop_shift[i];
                // Update shifts
                array_shift += size_sub[i];
                rhs_shift += row;
            }
            /*------------------------
            ASSEMBLE MATRICES IN BATCH
            ------------------------*/
            d_ptr_A = h_ptr_A;
            assembly::assembleGlobalMatrixBatched(streams[tid], thrust::raw_pointer_cast(d_ptr_A.data()), h_ptr_K[i], h_ptr_M[i],
                                                  size_sub[i], thrust::raw_pointer_cast(freq_square.data()), (int)freq_max, num_matrix);

            std::cout << "from main = " << d_ptr_A[1];

            /*--------------
            LU Decomposition
            --------------*/
/*
            d_ptr_A = h_ptr_A;
            cublas_check(cublasZgetrfBatched(cublasHandle[tid], row_sub[i], thrust::raw_pointer_cast(d_ptr_A.data()), row_sub[i], NULL, d_ptr_solverInfo, batchSize));
*/
            /*-----------
            Solve x = A\b
            -----------*/
/*
            d_ptr_rhs = h_ptr_rhs;
            cublas_check(cublasZgetrsBatched(cublasHandle[tid], CUBLAS_OP_N, row_sub[i], 1, thrust::raw_pointer_cast(d_ptr_A.data()), row_sub[i], NULL,
                                             thrust::raw_pointer_cast(d_ptr_rhs.data()), row_sub[i], &solverInfo_solve, batchSize));
*/
            /*-----------------
            Synchronize Streams
            -----------------*/
            cudaStreamSynchronize(streams[tid]);
        } // matrix loop
    } // omp parallel

    timerLoop.stop();
    nvtxRangePop();

    std::cout << ">> Matrix loop finished" << std::endl;
    std::cout << ">>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6 << "\n" << std::endl;

    // Copy solution from device to host
    thrust::host_vector<cuDoubleComplex> rhs = d_rhs;

    // Write out solution vectors
    thrust::host_vector<cuDoubleComplex> A = d_A;
    io::writeSolVecComplex(A, filepath_sol, "A.dat");

/*
    io::writeSolVecComplex(rhs, filepath_sol, filename_sol);
    */

    // Destroy cuBLAS & streams
    for (size_t i = 0; i < num_threads; ++i){
        cublasDestroy(cublasHandle[i]);
        cudaStreamDestroy(streams[i]);
    }

    timerTotal.stop();
    std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}
