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
    nvtxRangePushA("Initial Configuration");
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
    std::string input_filepath = "/opt/software/examples/MOR/inputs/\0";
    // Solution filepath
    std::string filepath_sol = "output/";
    // Solution filename
    std::string filename_sol = "solution.dat";
    // Array of filenames
    std::string baseName_K = "KSM_Stiffness_r\0";
    std::string baseName_M = "KSM_Mass_r\0";
    std::string baseName_D = "KSM_Damping_r\0";
    std::string baseName_input = "B\0";
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
    cuDoubleComplex one;
    one.x = 1.0;
    one.y = 1.0;
    cuDoubleComplex zero;
    zero.x = 0.0;
    zero.y = 0.0;
    // Array of matrix sizes (row)
    int row_baseline[] = {126, 132, 168, 174, 180, 186, 192, 288, 294, 300, 306, 312};
    // Sort the array row_baseline
    //std::sort(row_baseline.begin(), row_baseline.end());
    // Array of number of inputs
    int num_input_baseline[] = {6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72};

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
    cublasStatus_t cublasStatus;
    nvtxRangePop();

    /*-------------
    DATA STRUCTURES
    --------------*/
    nvtxRangePushA("Data Structures (Host)");
    // Create matrix host_vectors
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> K_sub(12);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> M_sub(12);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> D_sub(12);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> B_sub(12);

    // Read and process MTX file
    for (size_t i = 0; i < 7; ++i){
        filename_K[i] = baseName_K + std::to_string(row_baseline[i]) + base_format;
        filename_M[i] = baseName_M + std::to_string(row_baseline[i]) + base_format;
        filename_D[i] = baseName_D + std::to_string(row_baseline[i]) + base_format;
        filename_input[i] = baseName_input + std::to_string(num_input_baseline[i]) + base_format;
        io::readMtxDense(K_sub[i], filepath[0], filename_K[i], isComplex);
        io::readMtxDense(M_sub[i], filepath[0], filename_M[i], isComplex);
        io::readMtxDense(D_sub[i], filepath[0], filename_D[i], isComplex);
        io::readMtxDense(B_sub[i], input_filepath, filename_input[i], isComplex);
        K_sub[i].pop_back();
        M_sub[i].pop_back();
        D_sub[i].pop_back();
        B_sub[i].pop_back();
    }

    for (size_t i = 7; i < 12; ++i){
        filename_K[i] = baseName_K + std::to_string(row_baseline[i]) + base_format;
        filename_M[i] = baseName_M + std::to_string(row_baseline[i]) + base_format;
        filename_D[i] = baseName_D + std::to_string(row_baseline[i]) + base_format;
        filename_input[i] = baseName_input + std::to_string(num_input_baseline[i]) + base_format;
        io::readMtxDense(K_sub[i], filepath[1], filename_K[i], isComplex);
        io::readMtxDense(M_sub[i], filepath[1], filename_M[i], isComplex);
        io::readMtxDense(D_sub[i], filepath[1], filename_D[i], isComplex);
        io::readMtxDense(B_sub[i], input_filepath, filename_input[i], isComplex);
        K_sub[i].pop_back();
        M_sub[i].pop_back();
        D_sub[i].pop_back();
        B_sub[i].pop_back();
    }
    std::cout << ">> Matrices imported" << std::endl;

    // Get matrix sizes
    thrust::host_vector<int> row_sub(num_matrix);
    thrust::host_vector<int> size_sub(num_matrix);
    thrust::host_vector<int> size_sub_B(num_matrix);
    thrust::host_vector<int, pinnedAllocInt> num_input_sub(num_matrix);
    int nnz = 0;
    int row = 0;
    int nnz_B = 0;
    size_t idx;
    for (size_t j = 0; j < mat_repetition; ++j){
        for (size_t i = 0; i < 12; ++i){
            idx = i + 12*j;
            row_sub[idx] = row_baseline[i];
            size_sub[idx] = row_sub[i]*row_sub[i];
            num_input_sub[idx] = num_input_baseline[i];
            nnz += size_sub[idx];
            row  += row_sub[idx];
            size_sub_B[idx] = row_sub[i]*num_input_baseline[i];
            nnz_B += size_sub_B[idx];
        }
    }

    // Get maximum matrix size & number of inputs
    auto nnz_max_it = thrust::max_element(size_sub.begin(), size_sub.end());
    auto nnz_max_B_it = thrust::max_element(size_sub_B.begin(), size_sub_B.end());
    int nnz_max = *nnz_max_it;
    int nnz_max_B = *nnz_max_B_it;

    // Combine matrices into a single array on host (to make use of GPU's high bandwidth. We could also import the matrices directly like this)
    thrust::host_vector<cuDoubleComplex> K(nnz);
    thrust::host_vector<cuDoubleComplex> M(nnz);
    thrust::host_vector<cuDoubleComplex> D(nnz);
    thrust::host_vector<cuDoubleComplex> B(nnz_B);
    auto K_sub_ptr = &K_sub[0];
    auto M_sub_ptr = &M_sub[0];
    auto D_sub_ptr = &D_sub[0];
    auto B_sub_ptr = &B_sub[0];
    size_t array_shift = 0;
    size_t B_shift = 0;
    for (size_t j = 0; j < mat_repetition; ++j){
        for (size_t i = 0; i < 12; ++i){
            K_sub_ptr = &K_sub[i];
            M_sub_ptr = &M_sub[i];
            D_sub_ptr = &D_sub[i];
            B_sub_ptr = &B_sub[i];
            thrust::copy(K_sub_ptr->begin(), K_sub_ptr->end(), K.begin() + array_shift);
            thrust::copy(M_sub_ptr->begin(), M_sub_ptr->end(), M.begin() + array_shift);
            thrust::copy(D_sub_ptr->begin(), D_sub_ptr->end(), D.begin() + array_shift);
            thrust::copy(B_sub_ptr->begin(), B_sub_ptr->end(), B.begin() + B_shift);
            array_shift += size_sub[i];
            B_shift += size_sub_B[i];
        }
    }

    std::cout <<">> Matrices combined" << std::endl;
    nvtxRangePop();

    /*----------------------
    DATA STRUCTURES (DEVICE)
    ----------------------*/
    int dofReduced = row*freq_max;
    nvtxRangePushA("Data Structures (Device)");
    // Send matrices to device
    timerMatrixCpy.start();
    thrust::device_vector<cuDoubleComplex> d_K = K;
    thrust::device_vector<cuDoubleComplex> d_M = M;
    thrust::device_vector<cuDoubleComplex> d_D = D;
    thrust::device_vector<cuDoubleComplex> d_B_orig = B;
    timerMatrixCpy.stop();
    std::cout << ">> Matrices copied to device " << std::endl;
    std::cout << ">>>> Time taken = " << timerMatrixCpy.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;

    // Create RHS directly on device
    timerMatrixCpy.start();
    thrust::device_vector<cuDoubleComplex> d_rhs(dofReduced, rhs_val);
    timerMatrixCpy.stop();
    std::cout << ">> RHS copied to device " << std::endl;
    std::cout << ">>>> Time taken = " << timerMatrixCpy.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;

    // Create B (sourcing term) matrix on device
    thrust::device_vector<cuDoubleComplex> d_B(num_threads*freq_max*nnz_max_B);

    // Create C (Transpose of B) matrix on device
    thrust::device_vector<cuDoubleComplex> d_C(num_threads*freq_max*nnz_max_B);

    // Create H (Re-projection matrix) on device
    thrust::device_vector<cuDoubleComplex> d_H(freq_max*nnz_B);

    // Create matrix device_vectors
    thrust::device_vector<cuDoubleComplex> d_A(num_threads*freq_max*nnz_max);

    // Get vector of raw pointers to matrices
    cuDoubleComplex *d_ptr_K_base = thrust::raw_pointer_cast(d_K.data());
    cuDoubleComplex *d_ptr_M_base = thrust::raw_pointer_cast(d_M.data());
    cuDoubleComplex *d_ptr_D_base = thrust::raw_pointer_cast(d_D.data());
    cuDoubleComplex *d_ptr_B_orig_base = thrust::raw_pointer_cast(d_B_orig.data());
    thrust::host_vector<cuDoubleComplex*> h_ptr_K(num_matrix);
    thrust::host_vector<cuDoubleComplex*> h_ptr_M(num_matrix);
    thrust::host_vector<cuDoubleComplex*> h_ptr_D(num_matrix);
    thrust::host_vector<cuDoubleComplex*> h_ptr_B_orig(num_matrix);
    size_t mat_shift = 0;
    size_t sol_shift = 0;
    B_shift   = 0;
    thrust::host_vector<int> loop_shift(num_matrix);
    thrust::host_vector<int> H_thread_shift(num_matrix);
    for (size_t i = 0; i < num_matrix; ++i){
        h_ptr_K[i] = d_ptr_K_base + mat_shift;
        h_ptr_M[i] = d_ptr_M_base + mat_shift;
        h_ptr_D[i] = d_ptr_D_base + mat_shift;
        h_ptr_B_orig[i] = d_ptr_B_orig_base + B_shift;
        loop_shift[i] = sol_shift;
        H_thread_shift[i] = B_shift;
        mat_shift += size_sub[i];
        sol_shift += row_sub[i];
        B_shift   += size_sub_B[i];

    }

    // Get raw pointers to matrix A and rhs
    cuDoubleComplex *d_ptr_A_base = thrust::raw_pointer_cast(d_A.data());
    cuDoubleComplex *d_ptr_rhs_base = thrust::raw_pointer_cast(d_rhs.data());
    cuDoubleComplex *d_ptr_B_base = thrust::raw_pointer_cast(d_B.data());
    cuDoubleComplex *d_ptr_C_base = thrust::raw_pointer_cast(d_C.data());
    cuDoubleComplex *d_ptr_H_base = thrust::raw_pointer_cast(d_H.data());

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
#pragma omp parallel private(tid, array_shift, B_shift, freq, freq_square) num_threads(num_threads)
    {
        // Get thread number
        tid = omp_get_thread_num();
        // Indices
        size_t j;
        // Allocate vector of array pointers to A in each thread
        thrust::device_vector<cuDoubleComplex*> d_ptr_A(batchSize);
        thrust::device_vector<cuDoubleComplex*> d_ptr_rhs(batchSize);
        thrust::device_vector<cuDoubleComplex*> d_ptr_B(batchSize);
        thrust::device_vector<cuDoubleComplex*> d_ptr_C(batchSize);
        thrust::device_vector<cuDoubleComplex*> d_ptr_H(batchSize);
        thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_A(batchSize);
        thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_rhs(batchSize);
        thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_B(batchSize);
        thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_C(batchSize);
        thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_H(batchSize);

        // Initialise shifts
        int thread_shift, thread_shift_B, B_array_shift, H_freq_shift, rhs_shift;
        thread_shift = tid*freq_max*nnz_max;
        thread_shift_B = tid*freq_max*nnz_max_B;
        // Set cuBLAS stream
        cublasSetStream(cublasHandle[tid], streams[tid]);

    // Loop over each matrix size
    #pragma omp for
        for (size_t i = 0; i < num_matrix; ++i){
            /*---------------------------------------------------------------
            Assemble Global Matrix & Update pointers to each matrix A and RHS
            ---------------------------------------------------------------*/
            // Initialise Shifts
            array_shift = 0;
            B_array_shift = 0;
            rhs_shift = 0;
            H_freq_shift = 0;
            // Loop over batch (assume batchSize = freq_max)
            for (j = 0; j < batchSize; ++j){
                // Update matrix A pointer
                h_ptr_A[j] = d_ptr_A_base + thread_shift + array_shift;
                // Update (multiple) vector B pointer
                B_shift = thread_shift_B + B_array_shift;
                h_ptr_B[j] = d_ptr_B_base + B_shift;
                // Update vector C pointer
                h_ptr_C[j] = d_ptr_C_base + B_shift;
                // Update rhs pointer
                h_ptr_rhs[j] = d_ptr_rhs_base + rhs_shift + loop_shift[i];
                // Update H pointer
                h_ptr_H[j] = d_ptr_H_base + H_thread_shift[i] + H_freq_shift;
                // Compute frequency (assume batchSize = freq_max)
                freq = (j+1);
                freq_square = -(freq*freq);
                // Assemble matrix
                nvtxRangePushA("Matrix Assembly");
                assembly::assembleGlobalMatrixBatched(streams[tid], h_ptr_A[j], h_ptr_K[i], h_ptr_M[i], size_sub[i], freq_square);
                nvtxRangePop();
                // Construct matrix B
                thrust::copy_n(thrust::device, h_ptr_B_orig[i], size_sub_B[i], h_ptr_B[j]);
                // Update shifts
                array_shift += size_sub[i];
                B_array_shift += size_sub_B[i];
                rhs_shift += row;
                H_freq_shift += nnz_B;
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
            nvtxRangePop();
            /*----------------
            INTERFACE JACOBIAN
            ----------------*/
            nvtxRangePushA("Interface Jacobian");
            // Solve A\B
            d_ptr_B = h_ptr_B;
            cublas_check(cublasZgetrsBatched(cublasHandle[tid], CUBLAS_OP_N, row_sub[i], num_input_sub[i], thrust::raw_pointer_cast(d_ptr_A.data()), row_sub[i], NULL,
                                             thrust::raw_pointer_cast(d_ptr_B.data()), row_sub[i], &solverInfo_solve, batchSize));
            // GEMM
            d_ptr_C = h_ptr_C;
            d_ptr_H = h_ptr_H;
/*
            cublas_check(cublasZgemmBatched(cublasHandle[tid], CUBLAS_OP_T, CUBLAS_OP_N, num_input_sub[i], num_input_sub[i], size_sub_B[i], &one, thrust::raw_pointer_cast(d_ptr_C.data()),
                                            num_input_sub[i], thrust::raw_pointer_cast(d_ptr_B.data()), size_sub_B[i], &zero, thrust::raw_pointer_cast(d_ptr_H.data()), num_input_sub[i],
                                            batchSize);
*/
            nvtxRangePop();
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
    //thrust::host_vector<cuDoubleComplex> H = d_H;
    //io::writeSolVecComplex(H, filepath_sol, "H.dat");

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
