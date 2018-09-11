// Libraries
#include <iostream>
#include <string>
#include <cmath>

// OpenMP
#include <omp.h>

// THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

// CUCOMPLEX
#include <cuComplex.h>

// CUBLAS
#include <cublas_v2.h>

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
        std::cerr << ">> Usage: " << argv[0] << " -f <maximum frequency> -m <matrix repetition> -stream <number of CUDA streams> -batch <batch count>" << std::endl;
        std::cerr << ">> NOTE: There are 12 matrices and matrix repetition increases the total number of matrices (e.g. matrix repetition of 5 will use 60 matrices)" << std::endl;
        std::cerr << "         Frequency starts from 1 to maximum frequency" << std::endl;
        std::cerr << "         Default number of CUDA streams is 1" << std::endl;
        std::cerr << "         Ratio of number of matrix sizes to number of CUDA streams must be an integer" << std::endl;
        std::cerr << "         Default number of batch count is freq max" << std::endl;
        return 1;
    }

    double freq_max = atof(argv[2]);
    int mat_repetition = atoi(argv[4]);
    int num_matrix = mat_repetition*12;
    int num_streams = 1;

    if (argc > 6) num_streams = atoi(argv[6]);
    int num_threads = num_streams;

    int batchCount = freq_max;
    if (argc > 8) batchCount = atoi(argv[8]);

    std::cout << ">> Maximum Frequency: " << freq_max << std::endl;
    std::cout << ">> Total number of matrices: " << num_matrix << std::endl;
    std::cout << ">> Number of CUDA streams: " << num_streams << std::endl;
    std::cout << ">> Number of batched matrices: " << batchCount << "\n" << std::endl;

    if (((int)freq_max % num_streams) != 0) {
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

    // Sort the array row_baseline
    //std::sort(row_baseline.begin(), row_baseline.end());

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
    omp_set_dynamic(0);
    omp_set_nested(0);

    timerTotal.start();
    // Library initialisation
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

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
    thrust::device_vector<cuDoubleComplex> d_rhs(row*freq_max, rhs_val);
    timerMatrixCpy.stop();
    std::cout << ">> RHS copied to device " << std::endl;
    std::cout << ">>>> Time taken = " << timerMatrixCpy.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;

    // Create matrix device_vectors
    thrust::device_vector<cuDoubleComplex> d_A(freq_max*nnz_max, one);

    // Get raw pointers to matrices
    cuDoubleComplex *d_ptr_K = thrust::raw_pointer_cast(d_K.data());
    cuDoubleComplex *d_ptr_M = thrust::raw_pointer_cast(d_M.data());
    cuDoubleComplex *d_ptr_D = thrust::raw_pointer_cast(d_D.data());

    // Get array of raw pointers to A for batched LU
    cuDoubleComplex* d_ptr_A_base = thrust::raw_pointer_cast(d_A.data());
    thrust::device_vector<cuDoubleComplex*> d_ptr_A(batchCount);

    // Get array of raw pointers to RHS for batched LU
    cuDoubleComplex* d_ptr_rhs_base = thrust::raw_pointer_cast(d_rhs.data());
    cuDoubleComplex* d_ptr_rhs[batchCount];

    timerMatrixComp.start();
    // M = 4*pi^2*M (Single computation suffices)
    cublas_check(cublasZdscal(cublasHandle, nnz, &alpha, d_ptr_M, 1));
    timerMatrixComp.stop();
    std::cout << ">> M_tilde computed with cuBLAS" << std::endl;
    std::cout << ">>>> Time taken = " << timerMatrixComp.getDurationMicroSec()*1e-6 << " (sec)\n" << std::endl;

    // Solver Info for batched LU decomposition
    thrust::device_vector<int> d_solverInfo(batchCount);
    int *d_ptr_solverInfo = thrust::raw_pointer_cast(d_solverInfo.data());

    // Stream initialisation
    cudaStream_t streams[num_streams];
    for (size_t i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        std::cout << ">> Stream " << i << " created" << std::endl;
    }

    // Loop over matrices
    timerLoop.start();
    std::cout << "\n>> Matrix loop started for batched execution" << std::endl;
    int mat_shift = 0;
    for (size_t i = 0; i < num_matrix; i++){
        /*---------------------------------------------------------------
        Assemble Global Matrix & Update pointers to each matrix A and RHS
        ---------------------------------------------------------------*/
        array_shift = 0;
        int rhs_shift = 0;
        int loop_shift = i*row;
        for (size_t j = 0; j < batchCount; j++){
            // Update matrix A pointer
            d_ptr_A[j] = d_ptr_A_base + array_shift;
            // Compute frequency
            freq = (i+1);
            freq_square = -freq*freq;
            // Assemble matrix
            assembly::assembleGlobalMatrix4Batched(streams[0], cublasHandle, d_ptr_A[j], d_ptr_K + mat_shift, d_ptr_M + mat_shift, size_sub[i], one, freq_square);
            //d_ptr_rhs[j] = d_ptr_rhs_base + loop_shift + rhs_shift;
            array_shift += size_sub[i];
            //rhs_shift += row_sub[i];
        }

        /*--------------
        LU Decomposition
        --------------*/
        std::cout << "matrix = " << i << std::endl;
        cublas_check(cublasZgetrfBatched(cublasHandle, row_sub[i], thrust::raw_pointer_cast(d_ptr_A.data()), row_sub[i], NULL, d_ptr_solverInfo, batchCount));

        /*-----------
        Solve x = A\b
        -----------*/
/*
        cublas_check(cublasZgetrsBatched(cublasHandle, CUBLAS_OP_N, row_sub[i], 1, d_ptr_A, row_sub[i], NULL, d_ptr_rhs, row_sub[i], d_ptr_solverInfo, batchCount));
*/

        // Update matrix shift
        mat_shift += size_sub[i];
    } // matrix loop

    timerLoop.stop();

    std::cout << ">> Matrix loop finished" << std::endl;
    std::cout << ">>>> Time taken (s) = " << timerLoop.getDurationMicroSec()*1e-6 << "\n" << std::endl;

    // Copy solution from device to host
    thrust::host_vector<cuDoubleComplex> rhs = d_rhs;

    // Write out solution vectors
    io::writeSolVecComplex(rhs, filepath_sol, filename_sol);

    thrust::host_vector<cuDoubleComplex> A = d_A;
    io::writeSolVecComplex(A, filepath_sol, "A.dat");

    // Destroy cuBLAS
    cublasDestroy(cublasHandle);

    // Destroy streams
    for (size_t i = 0; i < num_streams; i++) cudaStreamDestroy(streams[i]);

    timerTotal.stop();
    std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}
