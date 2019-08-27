/*  Copyright &copy; 2019, Stefan Sicklinger, Munich
*
*  All rights reserved.
*
*  This file is part of STACCATO.
*
*  STACCATO is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  STACCATO is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with STACCATO.  If not, see http://www.gnu.org/licenses/.
*/
/*************************************************************************************************
* \file IJSA.cu
* Written by Ji-Ho Yang
* This file executes Interface Jacobian Substructuring Algorithm on GPU
* \date 7/12/2019
**************************************************************************************************/

/*----------------------------------------------------------------------*\
|                                                                        |
|   Interface Jacobian Substructuring on GPU                             |
|                                                                        |
|   Written by: Ji-Ho Yang                                               |
|               M.Sc. candidate in Computational Science & Engineering   |
|               Technische Universitaet Muenchen                         |
|                                                                        |
|   Work conducted as master's thesis at BMW AG                          |
|   under the supervision of Rupert Ullmann and Stefan Sickinlger        |
|                                                                        |
|   This code is part of STACCATO                                        |
|   https://github.com/DrStS/STACCATO                                    |
|                                                                        |
|   Lastest Update: 12/07/2019                                           |
|                                                                        |
\*----------------------------------------------------------------------*/

// Libraries
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <iterator>
#include <numeric>

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

// NVTX
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
typedef thrust::system::cuda::experimental::pinned_allocator<int*> pinnedAllocIntPtr;

// Namespace
using namespace staccato;

/*
    A = LocalNumDofTotal * LocalNumDofTotal
    B = LocalNumDofTotal x num_input
    C = num_input x row_sim
    H = num_input x num_input
*/

/*
    TODO

    1. Load on interface DOFs for residual computation
    2. Thrust vectors (both host and device) are FUNCTION SCOPE - ALWAYS TRANSFER DATA TO DEVICE ON HOST FUNCTION SCOPE! THIS WAY YOU KEEP DEVICE MEMORY FREE
    3. JSON IO

*/

/*
    Note

    1. H_exp already includes all the possible inputs (needs to be extracted too)
    2. For i8 model, all the simulation models happen to have all their input DOFs to be coupled (totalNumInput = interfaceNumInput)
    3. For experimental models, H_exp_internal happen to have bigger nnz than interface due to large internal DOFs in subsystem 17
    4. For i8 model we have point load on sub-system 1 (experimental model), node 1 (interface node)
    5. Everything's zero-indexed
*/

/*

    i8 Model:

    LocalTotalDOFs:     [36, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 240]
                         --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ---
                         E   E   E   E   E   E   S   S   S   S   S   E   E   E   E   E   E

    LocalInterfaceDOFs: [30, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 30]
                         --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
                         E   E   E   E   E   E   S   S   S   S   S   E   E   E   E   E   E

    LocalLoadDOFs:      [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ]
                         -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
                         E   E   E   E   E   E   S   S   S   S   S   E   E   E   E   E   E

    LocalInternalDOFs:  [6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 210]
                         -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  ---
                         E   E   E   E   E   E   S   S   S   S   S   E   E   E   E   E   E

*/

/*

    g20 Model:

    LocalTotalDOFs:     [36, 12, 12, 12, 12, 12, 12, 12, 12, 12, 24, 12, 12, 12, 12, 12, 12, 24, 12, 84, 12, 12, 12, 12, 48]
                         --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
                         S   E   E   E   E   E   S   S   S   S   S   E   E   E   E   E   E   E   E   S   E   E   E   E   E

    LocalInterfaceDOFs: [30, 12, 12, 12, 12, 12, 12, 12, 12, 12, 24, 12, 12, 12, 12, 12, 12, 24, 12, 54, 12, 12, 12, 12, 48]
                         --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
                         S   E   E   E   E   E   S   S   S   S   S   E   E   E   E   E   E   E   E   S   E   E   E   E   E

    LocalLoadDOFs:      [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ]
                         -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
                         S   E   E   E   E   E   S   S   S   S   S   E   E   E   E   E   E   E   E   S   E   E   E   E   E

    LocalInternalDOFs:  [6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  30, 0,  0,  0,  0,  0 ]
                         -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   --  -   -   -   -   -
                         S   E   E   E   E   E   S   S   S   S   S   E   E   E   E   E   E   E   E   S   E   E   E   E   E

*/

/*

    DATA STRUCTURES (for 3 subsystem and 2 frequency points for instance)

    Matrices are in column-major format

    A = [A0 A0 A1 A1 A2 A2]
         ----- ----- -----
         f0 f1 f0 f1 f0 f1

    H_sim = [H_sim0 H_sim0 H_sim1 H_sim1 H_sim2 H_sim2]
             ------ ------ ------ ------ ------ ------
               f0     f1     f0     f1     f0     f1

    H_exp = [H_exp0 H_exp0 H_exp1 H_exp1 H_exp2 H_exp2]
             ------ ------ ------ ------ ------ ------
               f0     f1     f0     f1     f0     f1

    rhs_sim = [rhs_sim0 rhs_sim0 rhs_sim1 rhs_sim1]
               -------- -------- -------- --------
                  f0       f1       f0       f1

    rhs_exp = [rhs_exp0 rhs_exp0 rhs_exp1 rhs_exp1]
               -------- -------- -------- --------
                  f0       f1       f0       f1

    Y_sim = [Y_sim0 Y_sim0 Y_sim1 Y_sim1]
             ------ ------ ------ ------
               f0     f1     f0     f1

    Y_exp = [Y_exp0 Y_exp0 Y_exp1 Y_exp1]
             ------ ------ ------ ------
               f0     f1     f0     f1

    residual = [r_exp0 r_exp1 r_exp2 r_sim0 r_sim1 r_exp0 r_exp1 r_exp2 r_sim0 r_sim1]
                ---------------------------------- ----------------------------------
                               f0                                 f1

    J = [  J     J     J  ]
         ----- ----- -----
          f0    f1    f2

    Y = [Y0_E Y1_S Y2_E Y0_E Y1_S Y2_E]
         -------------- --------------
               f0             f1

*/

int main (int argc, char *argv[]){

    timerTotal.start();

    PUSH_RANGE("Initial Configuration (Host)", 1)
    timerInit.start();

    /*--------------------
    COMMAND LINE ARGUMENTS
    --------------------*/
    double freq_min, freq_max;
    int subSystems, num_streams, num_threads, batchSize;
    bool postProcess;
    subSystems = 17; // i8
    //subSystems = 25; // g20
    // Configure test environment with command line arguments
    config::configureTest(argc, argv, freq_min, freq_max, num_streams, num_threads, batchSize, subSystems, postProcess);

    /*---------------------
    FILEPATHS AND FILENAMES
    ---------------------*/
    // Vector of filepaths
    std::string filepath_sim = "/opt/software/examples/i8/sim/\0";
    std::string filepath_exp = "/opt/software/examples/i8/exp/\0";
    std::string filepath_jac = "/opt/software/examples/i8/jac/\0";
/*
    std::string filepath_sim = "/opt/software/examples/g20/sim/\0";
    std::string filepath_exp = "/opt/software/examples/g20/exp/\0";
    std::string filepath_jac = "/opt/software/examples/g20/jac/\0";
*/
    // Solution filepath
    std::string filepath_sol = "output/";
    // Solution filename
    std::string filename_sol = "solution.dat";
    // Array of filenames
    std::string baseName_K    = "K\0";
    std::string baseName_M    = "M\0";
    std::string baseName_D    = "D\0";
    std::string baseName_B    = "B\0";
    std::string baseName_C    = "C\0";
    std::string baseName_H    = "H\0";
    std::string baseName_dRdY = "dRdS\0";
    std::string baseName_dRdU = "dRdU\0";
    std::string base_format   = ".mtx\0";

    /*--------
    PARAMETERS
    --------*/
    double alpha = 4*PI*PI;
    cuDoubleComplex one, zero, u_init;
    void *onePtr, *zeroPtr;
    onePtr = &one;
    zeroPtr = &zero;
    one.x     = 1.0;
    one.y     = 0.0;
    zero.x    = 0.0;
    zero.y    = 0.0;
    u_init    = zero;       // Initial guess of 0 for U vector

    /*----------------
    SYSTEM INFORMATION
    ----------------*/
    // Array of global subsystem ID (maps from global subsystem ID to local subsystemID (e.g. subSystemID_sim or subSystemID_exp)) - PREDEFINE
    std::vector<int> subSystemID_Global2Local = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11}; // i8
    //std::vector<int> subSystemID_Global2Local = {0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 6, 13, 14, 15, 16, 17}; // g20
    // Array of subsystem IDs with simulation model - PREDEFINE
    std::vector<int> subSystemID_sim = {6, 7, 8, 9, 10}; // i8
    //std::vector<int> subSystemID_sim = {0, 6, 7, 8, 9, 10, 19}; // g20
    // Array of subsystem IDs with experimental data - PREDEFINE
    std::vector<int> subSystemID_exp = {0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16}; // i8
    //std::vector<int> subSystemID_exp = {1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24}; // g20
    // System flags - PREDEFINE
    std::vector<bool> subSystem_flag = {EXP, EXP, EXP, EXP, EXP, EXP, SIM, SIM, SIM, SIM, SIM, EXP, EXP, EXP, EXP, EXP, EXP}; // i8
    //std::vector<bool> subSystem_flag = {SIM, EXP, EXP, EXP, EXP, EXP, SIM, SIM, SIM, SIM, SIM, EXP, EXP, EXP, EXP, EXP, EXP, EXP, EXP, SIM, EXP, EXP, EXP, EXP, EXP}; // g20

    // Number of components with simulation model & experimental data
    int num_simModel = subSystemID_sim.size();
    int num_expModel = subSystemID_exp.size();
    // Local interface DOFs - PREDEFINE
    // i8
    std::vector<std::vector<int>> LocalNumInterfaceDOFs(subSystems);
    LocalNumInterfaceDOFs[0]  = {30}; LocalNumInterfaceDOFs[1]  = {12}; LocalNumInterfaceDOFs[2]  = {12}; LocalNumInterfaceDOFs[3]  = {12}; LocalNumInterfaceDOFs[4]  = {12};
    LocalNumInterfaceDOFs[5]  = {12}; LocalNumInterfaceDOFs[6]  = {12}; LocalNumInterfaceDOFs[7]  = {12}; LocalNumInterfaceDOFs[8]  = {12}; LocalNumInterfaceDOFs[9]  = {12};
    LocalNumInterfaceDOFs[10] = {12}; LocalNumInterfaceDOFs[11] = {12}; LocalNumInterfaceDOFs[12] = {12}; LocalNumInterfaceDOFs[13] = {12}; LocalNumInterfaceDOFs[14] = {12};
    LocalNumInterfaceDOFs[15] = {12}; LocalNumInterfaceDOFs[16] = {30};

    std::vector<std::vector<int>> LocalInterfaceDOFs_start(subSystems);
    LocalInterfaceDOFs_start[0]  = {6}; LocalInterfaceDOFs_start[1]  = {0}; LocalInterfaceDOFs_start[2]  = {0}; LocalInterfaceDOFs_start[3]  = {0}; LocalInterfaceDOFs_start[4]  = {0};
    LocalInterfaceDOFs_start[5]  = {0}; LocalInterfaceDOFs_start[6]  = {0}; LocalInterfaceDOFs_start[7]  = {0}; LocalInterfaceDOFs_start[8]  = {0}; LocalInterfaceDOFs_start[9]  = {0};
    LocalInterfaceDOFs_start[10] = {0}; LocalInterfaceDOFs_start[11] = {0}; LocalInterfaceDOFs_start[12] = {0}; LocalInterfaceDOFs_start[13] = {0}; LocalInterfaceDOFs_start[14] = {0};
    LocalInterfaceDOFs_start[15] = {0}; LocalInterfaceDOFs_start[16] = {0};

/*
    // g20
    std::vector<std::vector<int>> LocalNumInterfaceDOFs(subSystems);
    LocalNumInterfaceDOFs[0]  = {30}; LocalNumInterfaceDOFs[1]  = {12}; LocalNumInterfaceDOFs[2]  = {12}; LocalNumInterfaceDOFs[3]  = {12}; LocalNumInterfaceDOFs[4]  = {12};
    LocalNumInterfaceDOFs[5]  = {12}; LocalNumInterfaceDOFs[6]  = {12}; LocalNumInterfaceDOFs[7]  = {12}; LocalNumInterfaceDOFs[8]  = {12}; LocalNumInterfaceDOFs[9]  = {12};
    LocalNumInterfaceDOFs[10] = {24}; LocalNumInterfaceDOFs[11] = {12}; LocalNumInterfaceDOFs[12] = {12}; LocalNumInterfaceDOFs[13] = {12}; LocalNumInterfaceDOFs[14] = {12};
    LocalNumInterfaceDOFs[15] = {12}; LocalNumInterfaceDOFs[16] = {12}; LocalNumInterfaceDOFs[17] = {24}; LocalNumInterfaceDOFs[18] = {12}; LocalNumInterfaceDOFs[19] = {30, 24};
    LocalNumInterfaceDOFs[20] = {12}; LocalNumInterfaceDOFs[21] = {12}; LocalNumInterfaceDOFs[22] = {12}; LocalNumInterfaceDOFs[23] = {12}; LocalNumInterfaceDOFs[24] = {48};

    std::vector<std::vector<int>> LocalInterfaceDOFs_start(subSystems);
    LocalInterfaceDOFs_start[0]  = {6}; LocalInterfaceDOFs_start[1]  = {0}; LocalInterfaceDOFs_start[2]  = {0}; LocalInterfaceDOFs_start[3]  = {0}; LocalInterfaceDOFs_start[4]  = {0};
    LocalInterfaceDOFs_start[5]  = {0}; LocalInterfaceDOFs_start[6]  = {0}; LocalInterfaceDOFs_start[7]  = {0}; LocalInterfaceDOFs_start[8]  = {0}; LocalInterfaceDOFs_start[9]  = {0};
    LocalInterfaceDOFs_start[10] = {0}; LocalInterfaceDOFs_start[11] = {0}; LocalInterfaceDOFs_start[12] = {0}; LocalInterfaceDOFs_start[13] = {0}; LocalInterfaceDOFs_start[14] = {0};
    LocalInterfaceDOFs_start[15] = {0}; LocalInterfaceDOFs_start[16] = {0}; LocalInterfaceDOFs_start[17] = {0}; LocalInterfaceDOFs_start[18] = {0}; LocalInterfaceDOFs_start[19] = {6, 60};
    LocalInterfaceDOFs_start[20] = {0}; LocalInterfaceDOFs_start[21] = {0}; LocalInterfaceDOFs_start[22] = {0}; LocalInterfaceDOFs_start[23] = {0}; LocalInterfaceDOFs_start[24] = {0};
*/

    std::vector<std::vector<int>> LocalInterfaceDOFs(subSystems);
    for (size_t i = 0; i < subSystems; ++i){
        // Get Local number of interface DOFs & resize vector accordingly
        int NumInterfaceDOFs = 0;
        for (size_t j = 0; j < LocalNumInterfaceDOFs[i].size(); ++j) NumInterfaceDOFs += LocalNumInterfaceDOFs[i][j];
        LocalInterfaceDOFs[i].resize(NumInterfaceDOFs);
        // Fill in the interface DOF indices
        int interfaceShift = 0;
        auto it = LocalInterfaceDOFs[i].begin();
        for (size_t j = 0; j < LocalNumInterfaceDOFs[i].size(); ++j){
            auto nx_begin = std::next(it, interfaceShift);
            auto nx_end   = std::next(it, interfaceShift+LocalNumInterfaceDOFs[i][j]);
            std::iota(nx_begin, nx_end, LocalInterfaceDOFs_start[i][j]);
            interfaceShift += LocalNumInterfaceDOFs[i][j];
        }
    }

    // Define arrays of DOFs
    std::vector<std::vector<int>> LocalTotalDOFs(subSystems), LocalInternalDOFs(subSystems);
    std::vector<int> GlobalTotalDOFs, GlobalInterfaceDOFs, GlobalInternalDOFs;

    /*-------
    FREQUENCY
    -------*/
    // Frequency vector
    thrust::host_vector<int, pinnedAllocInt> freq(batchSize), freq_square(batchSize);
    // Fill in frequency vectors
    thrust::sequence(freq.begin(), freq.end(), freq_min);
    thrust::transform(freq.begin(), freq.end(), freq.begin(), freq_square.begin(), thrust::multiplies<int>());

    /*-----------
    EXTERNAL LOAD
    -----------*/
    std::vector<std::vector<int>> LocalLoadDOFs(subSystems);
    std::vector<std::vector<double>> LocalLoadVals(subSystems);
    // PREDEFINE (for performance purposes, define them in ascending order)
    LocalLoadDOFs[0] = {2};
    LocalLoadVals[0] = {1};
    // Error messages
    for (size_t i = 0; i < subSystems; ++i){
        if (LocalLoadDOFs[i].size() != LocalLoadVals[i].size()){
            std::cerr << ">> Error: Number of DOFs with external load must match the number of load values" << std::endl;
            std::cerr << "   Exiting program ..." << std::endl;
            std::exit(1);
        }
    }

    /*----------------------------
    OPENMP & CUBLAS INITIALIZATION
    ----------------------------*/
    // OpenMP
    int tid;
    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);
    omp_set_nested(1);
    omp_set_num_threads(num_threads);
    // cuBLAS
    cublasHandle_t cublasHandle[MAX_NUM_THREADS];
    for (size_t i = 0; i < num_threads; ++i) cublasCreate(cublasHandle + i);
    // Tensor Core Option
    cublasMath_t cublasMathMode = CUBLAS_TENSOR_OP_MATH;
    cublasSetMathMode(cublasHandle[0], cublasMathMode);
    cudaDataType_t cudaArrayDataType = CUDA_C_64F;
    cublasGemmAlgo_t cudaAlgoType = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    timerInit.stop();
    std::cout << ">> Initial Configuration done" << std::endl;
    std::cout << ">>>> Time taken = " << timerInit.getDurationMicroSec()*1e-6 << " sec" << "\n" << std::endl;
    POP_RANGE // Initial Configuration

    /*--------------------
    DATA STRUCTURES (HOST)
    --------------------*/
    PUSH_RANGE("Data Structures (Host)", 1)
    timerDataHost.start();
    // Create matrix host_vectors (simulation models)
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> K_sub(num_simModel), M_sub(num_simModel), D_sub(num_simModel);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> B_sub(num_simModel), C_sub(num_simModel);
    // Create matrix host_vectors (experimental data)
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> H_exp_sub(num_expModel*batchSize);
    // Creat matrix host_vectors (Jacobian related)
    thrust::host_vector<cuDoubleComplex> dRdY, dRdU;
    // Create extracted host matrices
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> B_interface_sub(num_simModel)              , C_interface_sub(num_simModel);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> B_load_sub(num_simModel)                   , C_internal_sub(num_simModel);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> H_exp_interface_sub(num_expModel*batchSize), H_exp_interfaceLoad_sub(num_expModel*batchSize);
    thrust::host_vector<thrust::host_vector<cuDoubleComplex>> H_exp_internal_sub(num_expModel*batchSize) , H_exp_internalLoad_sub(num_expModel*batchSize);
    thrust::host_vector<cuDoubleComplex> dRdY_interface;
    // Create combined host matrices
    thrust::host_vector<cuDoubleComplex> K, M, D;
    thrust::host_vector<cuDoubleComplex> B_interface, B_load, C_interface, C_internal;
    thrust::host_vector<cuDoubleComplex> H_exp_interface, H_exp_interfaceLoad, H_exp_internal, H_exp_internalLoad;

    // Array information
    thrust::host_vector<int> row_A_sub(num_simModel);
    thrust::host_vector<int> nnz_A_sub(num_simModel);
    thrust::host_vector<int> nnz_B_interface_sub(num_simModel), nnz_B_load_sub(num_simModel);
    thrust::host_vector<int> nnz_C_interface_sub(num_simModel), nnz_C_internal_sub(num_simModel);
    thrust::host_vector<int> row_H_exp_sub(num_expModel);
    int nnz_A, nnz_A_max;
    int nnz_B_interface, nnz_B_load, nnz_C_interface, nnz_C_internal;
    int nnz_B_interface_max, nnz_B_load_max;
    int nnz_C_interface_max, nnz_C_internal_max;
    int row_dRdY, row_dRdU, col_dRdY, col_dRdU, nnz_dRdY, nnz_dRdU, numEntry_dRdY, numEntry_dRdU;

    // Set up host data structures
    timerHostIO.start();
    data::constructHostDataStructure(filepath_sim, filepath_exp, filepath_jac, base_format, baseName_K, baseName_M, baseName_D, baseName_B, baseName_C, baseName_H, baseName_dRdY, baseName_dRdU,
                                     num_simModel, num_expModel, subSystemID_sim, subSystemID_exp, freq, row_A_sub, row_H_exp_sub, nnz_A_sub, nnz_A, nnz_A_max, row_dRdY, row_dRdU, col_dRdY, col_dRdU,
                                     nnz_dRdY, nnz_dRdU, numEntry_dRdY, numEntry_dRdU, K_sub, M_sub, D_sub, B_sub, C_sub, H_exp_sub, dRdY, dRdU);
    timerHostIO.stop();
    std::cout << ">> Matrices Imported on Host" << std::endl;
    std::cout << ">>>> Time taken = " << timerHostIO.getDurationMicroSec()*1e-6 << " sec" << "\n" << std::endl;


    data::extractHostMatrices(LocalTotalDOFs, LocalInterfaceDOFs, LocalInternalDOFs, LocalLoadDOFs, GlobalTotalDOFs, GlobalInterfaceDOFs, GlobalInternalDOFs,
                              subSystem_flag, subSystemID_Global2Local, row_A_sub, row_H_exp_sub, col_dRdY,
                              batchSize, B_sub, C_sub, H_exp_sub, B_interface_sub, B_load_sub, C_interface_sub, C_internal_sub,
                              H_exp_interface_sub, H_exp_interfaceLoad_sub, H_exp_internal_sub, H_exp_internalLoad_sub, dRdY, dRdY_interface,
                              nnz_B_interface, nnz_B_load, nnz_C_interface, nnz_C_internal, nnz_B_interface_max, nnz_B_load_max, nnz_C_interface_max, nnz_C_internal_max,
                              nnz_B_interface_sub, nnz_B_load_sub, nnz_C_interface_sub, nnz_C_internal_sub);

    data::combineHostMatrices(K_sub, M_sub, D_sub, B_interface_sub, B_load_sub, C_interface_sub, C_internal_sub,
                              H_exp_interface_sub, H_exp_interfaceLoad_sub, H_exp_internal_sub, H_exp_internalLoad_sub,
                              K, M, D, B_interface, B_load, C_interface, C_internal,
                              H_exp_interface, H_exp_interfaceLoad, H_exp_internal, H_exp_internalLoad,
                              nnz_A, nnz_A_sub, nnz_B_interface, nnz_B_interface_sub, nnz_B_load, nnz_B_load_sub, nnz_C_interface, nnz_C_interface_sub, nnz_C_internal, nnz_C_internal_sub,
                              num_simModel, num_expModel, batchSize);

    timerDataHost.stop();
    std::cout << ">> Host data structure constructed" << std::endl;
    std::cout << ">>>> Time taken = " << timerDataHost.getDurationMicroSec()*1e-6 << " sec" << "\n" << std::endl;
    POP_RANGE // Data Structures (Host)

    /*----------------------
    DATA STRUCTURES (DEVICE)
    ----------------------*/
    PUSH_RANGE("Data Structures (Device)", 2)
    timerDataDevice.start();

    timerDeviceIO.start();
    // Send matrices to device
    thrust::device_vector<cuDoubleComplex> d_K                   = K;
    thrust::device_vector<cuDoubleComplex> d_M                   = M;
    thrust::device_vector<cuDoubleComplex> d_D                   = D;
    thrust::device_vector<cuDoubleComplex> d_B_interface         = B_interface;
    thrust::device_vector<cuDoubleComplex> d_B_load              = B_load;
    thrust::device_vector<cuDoubleComplex> d_C_interface         = C_interface;
    thrust::device_vector<cuDoubleComplex> d_C_internal          = C_internal;
    thrust::device_vector<cuDoubleComplex> d_H_exp_interface     = H_exp_interface;
    thrust::device_vector<cuDoubleComplex> d_H_exp_interfaceLoad = H_exp_interfaceLoad;
    thrust::device_vector<cuDoubleComplex> d_H_exp_internal      = H_exp_internal;
    thrust::device_vector<cuDoubleComplex> d_H_exp_internalLoad  = H_exp_internalLoad;
    thrust::device_vector<cuDoubleComplex> d_dRdY_interface      = dRdY_interface;
    thrust::device_vector<cuDoubleComplex> d_dRdU                = dRdU;
    timerDeviceIO.stop();
    std::cout << ">> Data Sent to Device" << std::endl;
    std::cout << ">>>> Time taken = " << timerDeviceIO.getDurationMicroSec()*1e-6 << " sec" << "\n" << std::endl;

    // Send DOFs arrays to device
    thrust::host_vector<int> GlobalInterfaceDOFs_thrust(GlobalInterfaceDOFs.size());
    thrust::host_vector<int> GlobalInternalDOFs_thrust(GlobalInternalDOFs.size());
    for (size_t i = 0; i < GlobalInterfaceDOFs.size(); ++i) GlobalInterfaceDOFs_thrust[i] = GlobalInterfaceDOFs[i];
    for (size_t i = 0; i < GlobalInternalDOFs.size(); ++i) GlobalInternalDOFs_thrust[i]   = GlobalInternalDOFs[i];
    thrust::device_vector<int> d_GlobalInterfaceDOFs = GlobalInterfaceDOFs_thrust;
    thrust::device_vector<int> d_GlobalInternalDOFs  = GlobalInternalDOFs_thrust;

    // Create device_vectors
    thrust::device_vector<cuDoubleComplex> d_A_batch(num_threads*batchSize*nnz_A_max);
    thrust::device_vector<cuDoubleComplex> d_B_interface_batch(num_threads*batchSize*nnz_B_interface_max);
    thrust::device_vector<cuDoubleComplex> d_B_load_batch(num_threads*batchSize*nnz_B_load_max);
    thrust::device_vector<cuDoubleComplex> d_H_sim_interface;
    thrust::device_vector<cuDoubleComplex> d_H_sim_interfaceLoad;
    thrust::device_vector<cuDoubleComplex> d_H_sim_internal;
    thrust::device_vector<cuDoubleComplex> d_H_sim_internalLoad;
    thrust::device_vector<cuDoubleComplex> d_residual(GlobalInterfaceDOFs.size() * batchSize, zero);
    thrust::device_vector<cuDoubleComplex> d_J(GlobalInterfaceDOFs.size() * GlobalInterfaceDOFs.size() * batchSize);

    // Get raw pointers to device matrices & vectors
    cuDoubleComplex *d_ptr_K_base                   = thrust::raw_pointer_cast(d_K.data());
    cuDoubleComplex *d_ptr_M_base                   = thrust::raw_pointer_cast(d_M.data());
    cuDoubleComplex *d_ptr_D_base                   = thrust::raw_pointer_cast(d_D.data());
    cuDoubleComplex *d_ptr_B_interface_base         = thrust::raw_pointer_cast(d_B_interface.data());
    cuDoubleComplex *d_ptr_B_load_base              = thrust::raw_pointer_cast(d_B_load.data());
    cuDoubleComplex *d_ptr_C_interface_base         = thrust::raw_pointer_cast(d_C_interface.data());
    cuDoubleComplex *d_ptr_C_internal_base          = thrust::raw_pointer_cast(d_C_internal.data());
    cuDoubleComplex *d_ptr_A_batch_base             = thrust::raw_pointer_cast(d_A_batch.data());
    cuDoubleComplex *d_ptr_B_interface_batch_base   = thrust::raw_pointer_cast(d_B_interface_batch.data());
    cuDoubleComplex *d_ptr_B_load_batch_base        = thrust::raw_pointer_cast(d_B_load_batch.data());
    cuDoubleComplex *d_ptr_H_exp_interface_base     = thrust::raw_pointer_cast(d_H_exp_interface.data());
    cuDoubleComplex *d_ptr_H_exp_interfaceLoad_base = thrust::raw_pointer_cast(d_H_exp_interfaceLoad.data());
    cuDoubleComplex *d_ptr_H_exp_internal_base      = thrust::raw_pointer_cast(d_H_exp_internal.data());
    cuDoubleComplex *d_ptr_H_exp_internalLoad_base  = thrust::raw_pointer_cast(d_H_exp_internalLoad.data());
    cuDoubleComplex *d_ptr_dRdY_interface_base      = thrust::raw_pointer_cast(d_dRdY_interface.data());
    cuDoubleComplex *d_ptr_dRdU_base                = thrust::raw_pointer_cast(d_dRdU.data());
    cuDoubleComplex *d_ptr_J_base                   = thrust::raw_pointer_cast(d_J.data());
    cuDoubleComplex *d_ptr_residual_base            = thrust::raw_pointer_cast(d_residual.data());

    // Get raw pointers to index arrays
    int *d_ptr_GlobalInterfaceDOFs_base = thrust::raw_pointer_cast(d_GlobalInterfaceDOFs.data());
    int *d_ptr_GlobalInternalDOFs_base  = thrust::raw_pointer_cast(d_GlobalInternalDOFs.data());

    // Copy d_dRdU into d_J
    int array_shift_J = 0;
    for (size_t f = 0; f < batchSize; ++f){
        thrust::copy(thrust::device, d_dRdU.begin(), d_dRdU.end(), d_J.begin() + array_shift_J);
        array_shift_J += GlobalInterfaceDOFs.size() * GlobalInterfaceDOFs.size();
    }

    // Create device vectors of pointers for each sub-components from combined matrices on device
    thrust::device_vector<cuDoubleComplex*> d_ptr_K(num_simModel),           d_ptr_M(num_simModel),      d_ptr_D(num_simModel);
    thrust::device_vector<cuDoubleComplex*> d_ptr_B_interface(num_simModel), d_ptr_B_load(num_simModel);
    thrust::device_vector<cuDoubleComplex*> d_ptr_C_interface(num_simModel), d_ptr_C_internal(num_simModel);

    // Get information from device data structures
    int nnz_H_sim_interface, nnz_H_sim_interfaceLoad, nnz_H_sim_internal, nnz_H_sim_internalLoad;
    std::vector<int> shift_total_dRdY_interface(subSystems, 0), shift_total_dRdU(subSystems, 0), shift_global_J(subSystems, 0), shift_global_residual(subSystems, 0);
    data::getInfoDeviceDataStructure(d_ptr_K, d_ptr_M, d_ptr_D, d_ptr_B_interface, d_ptr_B_load, d_ptr_C_interface, d_ptr_C_internal,
                                     d_ptr_K_base, d_ptr_M_base, d_ptr_D_base, d_ptr_B_interface_base, d_ptr_B_load_base, d_ptr_C_interface_base, d_ptr_C_internal_base,
                                     nnz_A_sub, nnz_B_interface_sub, nnz_B_load_sub, nnz_C_interface_sub, nnz_C_internal_sub,
                                     nnz_H_sim_interface, nnz_H_sim_interfaceLoad, nnz_H_sim_internal, nnz_H_sim_internalLoad,
                                     LocalInterfaceDOFs, LocalLoadDOFs, LocalInternalDOFs, subSystemID_sim, num_simModel);

    // Get total shifts for dRdY_interface and dRdU
    int _shift_dRdY_tmp, _shift_residual_tmp;
    _shift_dRdY_tmp = 0;
    _shift_residual_tmp = 0;
    for (size_t i = 0; i < subSystems; ++i){
        shift_total_dRdY_interface[i] = _shift_dRdY_tmp;
        shift_total_dRdU[i]           = _shift_dRdY_tmp;
        shift_global_J[i]             = _shift_dRdY_tmp;
        shift_global_residual[i]      = _shift_residual_tmp;
        _shift_dRdY_tmp              += GlobalInterfaceDOFs.size() * LocalInterfaceDOFs[i].size();
        _shift_residual_tmp          += LocalInterfaceDOFs[i].size();
    }

    // Resize matrices accordingly
    d_H_sim_interface.resize(batchSize*nnz_H_sim_interface);
    d_H_sim_interfaceLoad.resize(batchSize*nnz_H_sim_interfaceLoad);
    d_H_sim_internal.resize(batchSize*nnz_H_sim_internal);
    d_H_sim_internalLoad.resize(batchSize*nnz_H_sim_internalLoad);

    // Get base pointers
    cuDoubleComplex *d_ptr_H_sim_interface_base     = thrust::raw_pointer_cast(d_H_sim_interface.data());
    cuDoubleComplex *d_ptr_H_sim_interfaceLoad_base = thrust::raw_pointer_cast(d_H_sim_interfaceLoad.data());
    cuDoubleComplex *d_ptr_H_sim_internal_base      = thrust::raw_pointer_cast(d_H_sim_internal.data());
    cuDoubleComplex *d_ptr_H_sim_internalLoad_base  = thrust::raw_pointer_cast(d_H_sim_internalLoad.data());

    // Construct RHS
    int idx_sim, idx_exp;
    int DOF_sim_interface = 0;
    int DOF_sim_load      = 0;
    int DOF_sim_internal  = 0;
    int DOF_exp_interface = 0;
    int DOF_exp_load      = 0;
    int DOF_exp_internal  = 0;
    int DOF_total         = 0;
    for (size_t i = 0; i < num_simModel; ++i){
        idx_sim = subSystemID_sim[i];
        DOF_sim_interface += LocalInterfaceDOFs[idx_sim].size();
        DOF_sim_load      += LocalLoadDOFs[idx_sim].size();
        DOF_sim_internal  += LocalInternalDOFs[idx_sim].size();
    }
    for (size_t i = 0; i < num_expModel; ++i){
        idx_exp = subSystemID_exp[i];
        DOF_exp_interface += LocalInterfaceDOFs[idx_exp].size();
        DOF_exp_load      += LocalLoadDOFs[idx_exp].size();
        DOF_exp_internal  += LocalInternalDOFs[idx_exp].size();
    }

    // RHS vectors (input vectors)
    thrust::device_vector<cuDoubleComplex> d_rhs_sim_interface(DOF_sim_interface*batchSize, u_init);
    thrust::device_vector<cuDoubleComplex> d_rhs_exp_interface(DOF_exp_interface*batchSize, u_init);
    // Load vectors
    thrust::device_vector<cuDoubleComplex> d_rhs_sim_load;
    thrust::device_vector<cuDoubleComplex> d_rhs_exp_load;
    for (size_t i = 0; i < subSystems; ++i){
        if (LocalLoadDOFs[i].size() != 0){
            for (size_t f = 0; f < batchSize; ++f){
                if (subSystem_flag[i] == SIM){
                    for (size_t j = 0; j < LocalLoadVals[i].size(); ++j){
                        cuDoubleComplex _val;
                        _val.x = LocalLoadVals[i][j];
                        _val.y = 0;
                        d_rhs_sim_load.push_back(_val);
                    }
                }
                else{
                    for (size_t j = 0; j < LocalLoadVals[i].size(); ++j){
                        cuDoubleComplex _val;
                        _val.x = LocalLoadVals[i][j];
                        _val.y = 0;
                        d_rhs_exp_load.push_back(_val);
                    }
                }
            }
        }
    }

    // Get base pointers
    cuDoubleComplex *d_ptr_rhs_sim_interface_base = thrust::raw_pointer_cast(d_rhs_sim_interface.data());
    cuDoubleComplex *d_ptr_rhs_sim_load_base      = thrust::raw_pointer_cast(d_rhs_sim_load.data());
    cuDoubleComplex *d_ptr_rhs_exp_interface_base = thrust::raw_pointer_cast(d_rhs_exp_interface.data());
    cuDoubleComplex *d_ptr_rhs_exp_load_base      = thrust::raw_pointer_cast(d_rhs_exp_load.data());

    // Solution vectors
    DOF_total = DOF_sim_interface + DOF_sim_internal + DOF_exp_interface + DOF_exp_internal;
    thrust::device_vector<cuDoubleComplex> d_Y_sim_interface(DOF_sim_interface*batchSize);
    thrust::device_vector<cuDoubleComplex> d_Y_exp_interface(DOF_exp_interface*batchSize);
    thrust::device_vector<cuDoubleComplex> d_Y_sim_internal(DOF_sim_internal*batchSize);
    thrust::device_vector<cuDoubleComplex> d_Y_exp_internal(DOF_exp_internal*batchSize);
    thrust::device_vector<cuDoubleComplex> d_Y_interface((DOF_sim_interface+DOF_exp_interface)*batchSize);
    thrust::device_vector<cuDoubleComplex> d_Y(DOF_total*batchSize);

    // Get base pointers
    cuDoubleComplex *d_ptr_Y_sim_interface_base = thrust::raw_pointer_cast(d_Y_sim_interface.data());
    cuDoubleComplex *d_ptr_Y_exp_interface_base = thrust::raw_pointer_cast(d_Y_exp_interface.data());
    cuDoubleComplex *d_ptr_Y_sim_internal_base  = thrust::raw_pointer_cast(d_Y_sim_internal.data());
    cuDoubleComplex *d_ptr_Y_exp_internal_base  = thrust::raw_pointer_cast(d_Y_exp_internal.data());
    cuDoubleComplex *d_ptr_Y_interface_base		= thrust::raw_pointer_cast(d_Y_interface.data());
    cuDoubleComplex *d_ptr_Y_base               = thrust::raw_pointer_cast(d_Y.data());

    timerDataDevice.stop();
    std::cout << ">> Device data structure constructed" << std::endl;
    std::cout << ">>>> Time taken = " << timerDataDevice.getDurationMicroSec()*1e-6 << " sec" << "\n" << std::endl;
    POP_RANGE // Data Structures (Device)

    /*-------------------------------------------
    Interface Jacobian Substructuring Preparation
    -------------------------------------------*/
    PUSH_RANGE("Interface Jacobian Substructuring Preparation", 2)
    timerIJCAprep.start();
    // M = 4*pi^2*M
    cublas_check(cublasZdscal(cublasHandle[0], nnz_A, &alpha, d_ptr_M_base, 1));
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
    timerIJCAprep.stop();
    std::cout << "\n>> Ready to start to Interface Jacobian Substructuring" << std::endl;
    std::cout << ">>>> Time taken = " << timerIJCAprep.getDurationMicroSec()*1e-6 << " sec" << "\n" << std::endl;
    POP_RANGE // Interface Jacobian Substructuring Preparation

    /*-------------------------------
    Interface Jacobian Substructuring
    -------------------------------*/
    PUSH_RANGE("Interface Jacobian Substructuring", 3)
    timerIJCA.start();
    std::cout << ">> Interface Jacobian Substructuring started\n" << std::endl;

    /*---------------
    Simulation Models
    ---------------*/
    if (num_simModel != 0){
    timerSim.start();
    std::cout << ">> Solving Simulation Models" << std::endl;
    #pragma omp parallel private(tid) num_threads(num_threads)
        {
            // Get thread number
            tid = omp_get_thread_num();

            // Allocate vector of array pointers to matrices in each thread
            thrust::device_vector<cuDoubleComplex*> d_ptr_A_batch(batchSize);
            thrust::device_vector<cuDoubleComplex*> d_ptr_B_interface_batch(batchSize), d_ptr_B_load_batch(batchSize);
            thrust::device_vector<cuDoubleComplex*> d_ptr_C_interface_batch(batchSize), d_ptr_C_internal_batch(batchSize);

            thrust::device_vector<void*> d_ptr_B_interface_batch_GEMM(batchSize)      , d_ptr_B_load_batch_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_C_interface_batch_GEMM(batchSize)      , d_ptr_C_internal_batch_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_H_sim_interface_GEMM(batchSize)        , d_ptr_H_sim_interfaceLoad_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_H_sim_internal_GEMM(batchSize)         , d_ptr_H_sim_internalLoad_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_rhs_sim_interface_GEMM(batchSize)      , d_ptr_rhs_sim_load_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_Y_sim_interface_GEMM(batchSize)        , d_ptr_Y_sim_internal_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_dRdY_interface_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_dRdU_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_residual_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_J_GEMM(batchSize);

            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_A_batch(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_B_interface_batch(batchSize)   , h_ptr_B_load_batch(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_C_interface_batch(batchSize)   , h_ptr_C_internal_batch(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_H_sim_interface(batchSize)     , h_ptr_H_sim_interfaceLoad(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_H_sim_internal(batchSize)      , h_ptr_H_sim_internalLoad(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_rhs_sim_interface(batchSize)   , h_ptr_rhs_sim_load(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_Y_sim_interface(batchSize)     , h_ptr_Y_sim_internal(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_dRdY_interface(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_dRdU(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_residual(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_J(batchSize);

            // Initialise shifts
            int _idx, _idx_global_shift;
            int shift_global_A;
            int shift_batch_A;
            int shift_global_B_interface, shift_global_B_load;
            int shift_batch_B_interface , shift_batch_B_load;
            int shift_global_H_sim_interface, shift_global_H_sim_interfaceLoad, shift_global_H_sim_internal, shift_global_H_sim_internalLoad;
            int shift_batch_H_sim_interface , shift_batch_H_sim_interfaceLoad , shift_batch_H_sim_internal , shift_batch_H_sim_internalLoad;
            int shift_global_rhs_sim_interface, shift_global_rhs_sim_load;
            int shift_batch_rhs_sim_interface, shift_batch_rhs_sim_load;
            int shift_global_Y_sim_interface, shift_global_Y_sim_internal;
            int shift_batch_Y_sim_interface, shift_batch_Y_sim_internal;
            int shift_batch_residual;
            int shift_batch_J;

            shift_global_A           = tid*batchSize*nnz_A_max;
            shift_global_B_interface = tid*batchSize*nnz_B_interface_max;
            shift_global_B_load      = tid*batchSize*nnz_B_load_max;

            // Set cuBLAS stream
            cublasSetStream(cublasHandle[tid], streams[tid]);

            // Set tensor core math mode
            cublasSetMathMode(cublasHandle[tid], cublasMathMode);

        // Loop over simulation models
        #pragma omp for
            for (size_t i = 0; i < num_simModel; ++i){

                /*-------------
                Update pointers
                -------------*/
                // Get simulation model idx
                _idx = subSystemID_sim[i];

                // Initialise Shifts
                shift_global_H_sim_interface     = 0;
                shift_global_H_sim_interfaceLoad = 0;
                shift_global_H_sim_internal      = 0;
                shift_global_H_sim_internalLoad  = 0;
                shift_batch_H_sim_interface      = 0;
                shift_batch_H_sim_interfaceLoad  = 0;
                shift_batch_H_sim_internal       = 0;
                shift_batch_H_sim_internalLoad   = 0;
                shift_batch_A                    = 0;
                shift_batch_B_interface          = 0;
                shift_batch_B_load               = 0;
                shift_global_rhs_sim_interface   = 0;
                shift_global_rhs_sim_load        = 0;
                shift_batch_rhs_sim_interface    = 0;
                shift_batch_rhs_sim_load         = 0;
                shift_global_Y_sim_interface     = 0;
                shift_global_Y_sim_internal      = 0;
                shift_batch_Y_sim_interface      = 0;
                shift_batch_Y_sim_internal       = 0;
                shift_batch_residual             = 0;
                shift_batch_J                    = 0;

                // Get global shifts
                for (size_t k = 0; k < i; ++k){
                    _idx_global_shift = subSystemID_sim[k];
                    for (size_t f = 0; f < batchSize; ++f){
                        shift_global_H_sim_interface     += LocalInterfaceDOFs[_idx_global_shift].size() * LocalInterfaceDOFs[_idx_global_shift].size();
                        shift_global_H_sim_interfaceLoad += LocalInterfaceDOFs[_idx_global_shift].size() * LocalLoadDOFs[_idx_global_shift].size();
                        shift_global_H_sim_internal      += LocalInternalDOFs[_idx_global_shift].size()  * LocalInterfaceDOFs[_idx_global_shift].size();
                        shift_global_H_sim_internalLoad  += LocalInternalDOFs[_idx_global_shift].size()  * LocalLoadDOFs[_idx_global_shift].size();
                        shift_global_rhs_sim_interface   += LocalInterfaceDOFs[_idx_global_shift].size();
                        shift_global_rhs_sim_load        += LocalLoadDOFs[_idx_global_shift].size();
                        shift_global_Y_sim_interface     += LocalInterfaceDOFs[_idx_global_shift].size();
                        shift_global_Y_sim_internal      += LocalInternalDOFs[_idx_global_shift].size();
                    }
                }

                // Loop over batch (assume batchSize = freq_max)
                for (size_t j = 0; j < batchSize; ++j){
                    // Update pointers for batched operations
                    h_ptr_A_batch[j]                 = d_ptr_A_batch_base             + shift_batch_A                    + shift_global_A;
                    h_ptr_B_interface_batch[j]       = d_ptr_B_interface_batch_base   + shift_global_B_interface         + shift_batch_B_interface;
                    h_ptr_B_load_batch[j]            = d_ptr_B_load_batch_base        + shift_global_B_load              + shift_batch_B_load;
                    h_ptr_C_interface_batch[j]       = d_ptr_C_interface[i];
                    h_ptr_C_internal_batch[j]        = d_ptr_C_internal[i];
                    h_ptr_H_sim_interface[j]         = d_ptr_H_sim_interface_base     + shift_global_H_sim_interface     + shift_batch_H_sim_interface;
                    h_ptr_H_sim_interfaceLoad[j]     = d_ptr_H_sim_interfaceLoad_base + shift_global_H_sim_interfaceLoad + shift_batch_H_sim_interfaceLoad;
                    h_ptr_H_sim_internal[j]          = d_ptr_H_sim_internal_base      + shift_global_H_sim_internal      + shift_batch_H_sim_internal;
                    h_ptr_H_sim_internalLoad[j]      = d_ptr_H_sim_internalLoad_base  + shift_global_H_sim_internalLoad  + shift_batch_H_sim_internalLoad;
                    h_ptr_rhs_sim_interface[j]       = d_ptr_rhs_sim_interface_base   + shift_global_rhs_sim_interface   + shift_batch_rhs_sim_interface;
                    h_ptr_rhs_sim_load[j]            = d_ptr_rhs_sim_load_base        + shift_global_rhs_sim_load        + shift_batch_rhs_sim_load;
                    h_ptr_Y_sim_interface[j]         = d_ptr_Y_sim_interface_base     + shift_global_Y_sim_interface     + shift_batch_Y_sim_interface;
                    h_ptr_Y_sim_internal[j]          = d_ptr_Y_sim_internal_base      + shift_global_Y_sim_internal      + shift_batch_Y_sim_internal;
                    h_ptr_residual[j]                = d_ptr_residual_base            + shift_global_residual[_idx]      + shift_batch_residual;
                    h_ptr_dRdY_interface[j]          = d_ptr_dRdY_interface_base      + shift_total_dRdY_interface[_idx];
                    h_ptr_dRdU[j]                    = d_ptr_dRdU_base                + shift_total_dRdU[_idx];
                    h_ptr_J[j]                       = d_ptr_J_base                   + shift_global_J[_idx]             + shift_batch_J;

                    // Update batch shifts
                    shift_batch_A                   += nnz_A_sub[i];
                    shift_batch_B_interface         += nnz_B_interface_sub[i];
                    shift_batch_B_load              += nnz_B_load_sub[i];
                    shift_batch_H_sim_interface     += LocalInterfaceDOFs[_idx].size() * LocalInterfaceDOFs[_idx].size();
                    shift_batch_H_sim_interfaceLoad += LocalInterfaceDOFs[_idx].size() * LocalLoadDOFs[_idx].size();
                    shift_batch_H_sim_internal      += LocalInternalDOFs[_idx].size()  * LocalInterfaceDOFs[_idx].size();
                    shift_batch_H_sim_internalLoad  += LocalInternalDOFs[_idx].size()  * LocalLoadDOFs[_idx].size();
                    shift_batch_rhs_sim_interface   += LocalInterfaceDOFs[_idx].size();
                    shift_batch_rhs_sim_load        += LocalLoadDOFs[_idx].size();
                    shift_batch_Y_sim_interface     += LocalInterfaceDOFs[_idx].size();
                    shift_batch_Y_sim_internal      += LocalInternalDOFs[_idx].size();
                    shift_batch_residual            += GlobalInterfaceDOFs.size();
                    shift_batch_J                   += GlobalInterfaceDOFs.size() * GlobalInterfaceDOFs.size();
                }

                /*---------------------------------
                Compute dY/dU for Simulation Models
                ---------------------------------*/
                // Assembly Matrices in Batch
                PUSH_RANGE("Matrix Assembly", 4)
                d_ptr_A_batch = h_ptr_A_batch;
                assembly::assembleGlobalMatrixBatched(streams[tid], thrust::raw_pointer_cast(d_ptr_A_batch.data()), d_ptr_K[i], d_ptr_M[i], d_ptr_D[i],
                                                      nnz_A_sub[i], thrust::raw_pointer_cast(freq.data()), thrust::raw_pointer_cast(freq_square.data()), batchSize);
                POP_RANGE // Matrix Assembly

                // LU Decomposition
                PUSH_RANGE("LU Decomposition", 5)
                d_ptr_A_batch = h_ptr_A_batch;
                cublas_check(cublasZgetrfBatched(cublasHandle[tid], row_A_sub[i], thrust::raw_pointer_cast(d_ptr_A_batch.data()), row_A_sub[i], NULL, d_ptr_solverInfo, batchSize));
                POP_RANGE // LU Decomposition

                PUSH_RANGE("H_sim Computation", 6)

                // Solve A\B (Forward-Backward Substitution)
                PUSH_RANGE("Forward-Backward Substitution", 7)

                // Construct batch matrices B_interface
                d_ptr_B_interface_batch = h_ptr_B_interface_batch;
                assembly::constructMatricesBatched(streams[tid], d_ptr_B_interface[i], thrust::raw_pointer_cast(d_ptr_B_interface_batch.data()), nnz_B_interface_sub[i], (int)batchSize);

                // Construct batch matrices B_load
                if (nnz_B_load_sub[i] != 0){
                    d_ptr_B_load_batch = h_ptr_B_load_batch;
                    assembly::constructMatricesBatched(streams[tid], d_ptr_B_load[i], thrust::raw_pointer_cast(d_ptr_B_load_batch.data()), nnz_B_load_sub[i], (int)batchSize);
                }

                // Interface
                d_ptr_A_batch = h_ptr_A_batch;
                d_ptr_B_interface_batch = h_ptr_B_interface_batch;
                cublas_check(cublasZgetrsBatched(cublasHandle[tid], CUBLAS_OP_N, row_A_sub[i], LocalInterfaceDOFs[_idx].size(), thrust::raw_pointer_cast(d_ptr_A_batch.data()), row_A_sub[i],
                                                 NULL, thrust::raw_pointer_cast(d_ptr_B_interface_batch.data()), row_A_sub[i], &solverInfo_solve, batchSize));

                // Load
                d_ptr_B_load_batch = h_ptr_B_load_batch;
                cublas_check(cublasZgetrsBatched(cublasHandle[tid], CUBLAS_OP_N, row_A_sub[i], LocalLoadDOFs[_idx].size(), thrust::raw_pointer_cast(d_ptr_A_batch.data()), row_A_sub[i],
                                                 NULL, thrust::raw_pointer_cast(d_ptr_B_load_batch.data()), row_A_sub[i], &solverInfo_solve, batchSize));
                POP_RANGE // Forward-Backward Substitution

                PUSH_RANGE("GEMM (H_sim)", 8)

                // Interface
                d_ptr_B_interface_batch_GEMM = h_ptr_B_interface_batch;
                d_ptr_C_interface_batch_GEMM = h_ptr_C_interface_batch;
                d_ptr_H_sim_interface_GEMM   = h_ptr_H_sim_interface;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInterfaceDOFs[_idx].size(), LocalInterfaceDOFs[_idx].size(), row_A_sub[i],
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_C_interface_batch_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 thrust::raw_pointer_cast(d_ptr_B_interface_batch_GEMM.data()), cudaArrayDataType, row_A_sub[i],
                                                 zeroPtr, thrust::raw_pointer_cast(d_ptr_H_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));

                // Interface-Load
                d_ptr_B_load_batch_GEMM        = h_ptr_B_load_batch;
                d_ptr_C_interface_batch_GEMM   = h_ptr_C_interface_batch;
                d_ptr_H_sim_interfaceLoad_GEMM = h_ptr_H_sim_interfaceLoad;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInterfaceDOFs[_idx].size(), LocalLoadDOFs[_idx].size(), row_A_sub[i],
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_C_interface_batch_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 thrust::raw_pointer_cast(d_ptr_B_load_batch_GEMM.data()), cudaArrayDataType, row_A_sub[i],
                                                 zeroPtr, thrust::raw_pointer_cast(d_ptr_H_sim_interfaceLoad_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));

                // Post-processing
                if (postProcess){
                    if (LocalInternalDOFs[_idx].size() != 0){
                        // Internal
                        d_ptr_B_interface_batch_GEMM = h_ptr_B_interface_batch;
                        d_ptr_C_internal_batch_GEMM  = h_ptr_C_internal_batch;
                        d_ptr_H_sim_internal_GEMM    = h_ptr_H_sim_internal;

                        cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInternalDOFs[_idx].size(), LocalInterfaceDOFs[_idx].size(), row_A_sub[i],
                                                         onePtr, thrust::raw_pointer_cast(d_ptr_C_internal_batch_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[_idx].size(),
                                                         thrust::raw_pointer_cast(d_ptr_B_interface_batch_GEMM.data()), cudaArrayDataType, row_A_sub[i],
                                                         zeroPtr, thrust::raw_pointer_cast(d_ptr_H_sim_internal_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[_idx].size(), batchSize,
                                                         cudaArrayDataType, cudaAlgoType));

                        // Internal-Load
                        d_ptr_B_load_batch_GEMM        = h_ptr_B_load_batch;
                        d_ptr_C_internal_batch_GEMM    = h_ptr_C_internal_batch;
                        d_ptr_H_sim_internalLoad_GEMM  = h_ptr_H_sim_internalLoad;
                        cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInternalDOFs[_idx].size(), LocalLoadDOFs[_idx].size(), row_A_sub[i],
                                                         onePtr, thrust::raw_pointer_cast(d_ptr_C_internal_batch_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[_idx].size(),
                                                         thrust::raw_pointer_cast(d_ptr_B_load_batch_GEMM.data()), cudaArrayDataType, row_A_sub[i],
                                                         zeroPtr, thrust::raw_pointer_cast(d_ptr_H_sim_internalLoad_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[_idx].size(), batchSize,
                                                         cudaArrayDataType, cudaAlgoType));
                    }
                }

                POP_RANGE // GEMM (H_sim)

                POP_RANGE // H_sim Computation

                /*------------------------
                Compute initial solution Y
                ------------------------*/
                PUSH_RANGE("GEMM (Y_sim)", 9)
                // Interface
                d_ptr_H_sim_interface_GEMM   = h_ptr_H_sim_interface;
                d_ptr_rhs_sim_interface_GEMM = h_ptr_rhs_sim_interface;
                d_ptr_Y_sim_interface_GEMM   = h_ptr_Y_sim_interface;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInterfaceDOFs[_idx].size(), 1, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_H_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 thrust::raw_pointer_cast(d_ptr_rhs_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 zeroPtr, thrust::raw_pointer_cast(d_ptr_Y_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));

                // Load (sum the interface solution on top of this)
                d_ptr_H_sim_interfaceLoad_GEMM = h_ptr_H_sim_interfaceLoad;
                d_ptr_rhs_sim_load_GEMM        = h_ptr_rhs_sim_load;
                d_ptr_Y_sim_internal_GEMM      = h_ptr_Y_sim_internal;
                if (LocalLoadDOFs[_idx].size() != 0){
                    cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInterfaceDOFs[_idx].size(), 1, LocalLoadDOFs[_idx].size(),
                                                     onePtr, thrust::raw_pointer_cast(d_ptr_H_sim_interfaceLoad_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                     thrust::raw_pointer_cast(d_ptr_rhs_sim_load_GEMM.data()), cudaArrayDataType, LocalLoadDOFs[_idx].size(),
                                                     onePtr, thrust::raw_pointer_cast(d_ptr_Y_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(), batchSize,
                                                     cudaArrayDataType, cudaAlgoType));
                }

                /*--------------
                Compute Residual
                --------------*/
                // r = dRdY * Y + r
                d_ptr_dRdY_interface_GEMM  = h_ptr_dRdY_interface;
                d_ptr_Y_sim_interface_GEMM = h_ptr_Y_sim_interface;
                d_ptr_residual_GEMM        = h_ptr_residual;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, GlobalInterfaceDOFs.size(), 1, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_dRdY_interface_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(),
                                                 thrust::raw_pointer_cast(d_ptr_Y_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_residual_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));

                // r = dRdU * U + r
                d_ptr_dRdU_GEMM              = h_ptr_dRdU;
                d_ptr_rhs_sim_interface_GEMM = h_ptr_rhs_sim_interface;
                d_ptr_residual_GEMM          = h_ptr_residual;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, GlobalInterfaceDOFs.size(), 1, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_dRdU_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(),
                                                 thrust::raw_pointer_cast(d_ptr_rhs_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_residual_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));


                /*------------------------
                Compute Interface Jacobian
                ------------------------*/
                d_ptr_dRdY_interface_GEMM  = h_ptr_dRdY_interface;
                d_ptr_H_sim_interface_GEMM = h_ptr_H_sim_interface;
                d_ptr_J_GEMM               = h_ptr_J;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, GlobalInterfaceDOFs.size(), LocalInterfaceDOFs[_idx].size(), LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_dRdY_interface_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(),
                                                 thrust::raw_pointer_cast(d_ptr_H_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_J_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));

                POP_RANGE

                /*-----------------
                Synchronize Streams
                -----------------*/
                cudaStreamSynchronize(streams[tid]);
            } // Simulation model loop
        } // omp parallel
        timerSim.stop();
        std::cout << ">>>> Time taken = " << timerSim.getDurationMicroSec()*1e-6 << " sec" << "\n" << std::endl;
    } // if simulation model

    /*-----------------
    Experimental Models
    -----------------*/
    if (num_expModel != 0){
        timerExp.start();
        std::cout << ">> Solving Experimental Models" << std::endl;
    #pragma omp parallel private(tid) num_threads(num_threads)
        {
            // Get thread number
            tid = omp_get_thread_num();

            // Allocate vector of array pointers to matrices in each thread
            thrust::device_vector<void*> d_ptr_H_exp_interface_GEMM(batchSize)  , d_ptr_H_exp_interfaceLoad_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_H_exp_internal_GEMM(batchSize)   , d_ptr_H_exp_internalLoad_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_rhs_exp_interface_GEMM(batchSize), d_ptr_rhs_exp_load_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_Y_exp_interface_GEMM(batchSize)  , d_ptr_Y_exp_internal_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_dRdY_interface_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_dRdU_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_residual_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_J_GEMM(batchSize);

            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_H_exp_interface(batchSize)  , h_ptr_H_exp_interfaceLoad(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_H_exp_internal(batchSize)   , h_ptr_H_exp_internalLoad(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_rhs_exp_interface(batchSize), h_ptr_rhs_exp_load(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_Y_exp_interface(batchSize)  , h_ptr_Y_exp_internal(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_dRdY_interface(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_dRdU(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_residual(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_J(batchSize);

            // Initialise shifts
            int _idx, _idx_global_shift;
            int shift_global_H_exp_interface, shift_global_H_exp_interfaceLoad, shift_global_H_exp_internal, shift_global_H_exp_internalLoad;
            int shift_batch_H_exp_interface , shift_batch_H_exp_interfaceLoad , shift_batch_H_exp_internal , shift_batch_H_exp_internalLoad;
            int shift_global_rhs_exp_interface, shift_global_rhs_exp_load;
            int shift_batch_rhs_exp_interface, shift_batch_rhs_exp_load;
            int shift_global_Y_exp_interface, shift_global_Y_exp_internal;
            int shift_batch_Y_exp_interface, shift_batch_Y_exp_internal;
            int shift_batch_residual;
            int shift_batch_J;

            // Set cuBLAS stream
            cublasSetStream(cublasHandle[tid], streams[tid]);
            // Set tensor core math mode
            cublasSetMathMode(cublasHandle[tid], cublasMathMode);

        // Loop over experimental models
        #pragma omp for
            for (size_t i = 0; i < num_expModel; ++i){

                /*-------------
                Update pointers
                -------------*/
                // Get experimental model idx
                _idx = subSystemID_exp[i];

                // Initialise Shifts
                shift_global_H_exp_interface     = 0;
                shift_global_H_exp_interfaceLoad = 0;
                shift_global_H_exp_internal      = 0;
                shift_global_H_exp_internalLoad  = 0;
                shift_batch_H_exp_interface      = 0;
                shift_batch_H_exp_interfaceLoad  = 0;
                shift_batch_H_exp_internal       = 0;
                shift_batch_H_exp_internalLoad   = 0;
                shift_global_rhs_exp_interface   = 0;
                shift_global_rhs_exp_load        = 0;
                shift_batch_rhs_exp_interface    = 0;
                shift_batch_rhs_exp_load         = 0;
                shift_global_Y_exp_interface     = 0;
                shift_global_Y_exp_internal      = 0;
                shift_batch_Y_exp_interface      = 0;
                shift_batch_Y_exp_internal       = 0;
                shift_batch_residual             = 0;
                shift_batch_J                    = 0;

                // Get global shifts
                for (size_t k = 0; k < i; ++k){
                    _idx_global_shift = subSystemID_exp[k];
                    for (size_t f = 0; f < batchSize; ++f){
                        shift_global_H_exp_interface     += H_exp_interface_sub[k*batchSize+f].size();
                        shift_global_H_exp_interfaceLoad += H_exp_interfaceLoad_sub[k*batchSize+f].size();
                        shift_global_H_exp_internal      += H_exp_internal_sub[k*batchSize+f].size();
                        shift_global_H_exp_internalLoad  += H_exp_internalLoad_sub[k*batchSize+f].size();
                        shift_global_rhs_exp_interface   += LocalInterfaceDOFs[_idx_global_shift].size();
                        shift_global_rhs_exp_load        += LocalLoadDOFs[_idx_global_shift].size();
                        shift_global_Y_exp_interface     += LocalInterfaceDOFs[_idx_global_shift].size();
                        shift_global_Y_exp_internal      += LocalInternalDOFs[_idx_global_shift].size();
                    }
                }

                // Get batch shifts
                for (size_t j = 0; j < batchSize; ++j){
                    // Update pointers for batched operations
                    h_ptr_H_exp_interface[j]     = d_ptr_H_exp_interface_base     + shift_global_H_exp_interface     + shift_batch_H_exp_interface;
                    h_ptr_H_exp_interfaceLoad[j] = d_ptr_H_exp_interfaceLoad_base + shift_global_H_exp_interfaceLoad + shift_batch_H_exp_interfaceLoad;
                    h_ptr_H_exp_internal[j]      = d_ptr_H_exp_internal_base      + shift_global_H_exp_internal      + shift_batch_H_exp_internal;
                    h_ptr_H_exp_internalLoad[j]  = d_ptr_H_exp_internalLoad_base  + shift_global_H_exp_internalLoad  + shift_batch_H_exp_internalLoad;
                    h_ptr_rhs_exp_interface[j]   = d_ptr_rhs_exp_interface_base   + shift_global_rhs_exp_interface   + shift_batch_rhs_exp_interface;
                    h_ptr_rhs_exp_load[j]        = d_ptr_rhs_exp_load_base        + shift_global_rhs_exp_load        + shift_batch_rhs_exp_load;
                    h_ptr_Y_exp_interface[j]     = d_ptr_Y_exp_interface_base     + shift_global_Y_exp_interface     + shift_batch_Y_exp_interface;
                    h_ptr_Y_exp_internal[j]      = d_ptr_Y_exp_internal_base      + shift_global_Y_exp_internal      + shift_batch_Y_exp_internal;
                    h_ptr_residual[j]            = d_ptr_residual_base            + shift_global_residual[_idx]      + shift_batch_residual;
                    h_ptr_dRdY_interface[j]      = d_ptr_dRdY_interface_base      + shift_total_dRdY_interface[_idx];
                    h_ptr_dRdU[j]                = d_ptr_dRdU_base                + shift_total_dRdU[_idx];
                    h_ptr_J[j]                   = d_ptr_J_base                   + shift_global_J[_idx]             + shift_batch_J;

                    // Update batch shifts
                    shift_batch_H_exp_interface     += H_exp_interface_sub[i*batchSize+j].size();
                    shift_batch_H_exp_interfaceLoad += H_exp_interfaceLoad_sub[i*batchSize+j].size();
                    shift_batch_H_exp_internal      += H_exp_internal_sub[i*batchSize+j].size();
                    shift_batch_H_exp_internalLoad  += H_exp_internalLoad_sub[i*batchSize+j].size();
                    shift_batch_rhs_exp_interface   += LocalInterfaceDOFs[_idx].size();
                    shift_batch_rhs_exp_load        += LocalLoadDOFs[_idx].size();
                    shift_batch_Y_exp_interface     += LocalInterfaceDOFs[_idx].size();
                    shift_batch_Y_exp_internal      += LocalInternalDOFs[_idx].size();
                    shift_batch_residual            += GlobalInterfaceDOFs.size();
                    shift_batch_J                   += GlobalInterfaceDOFs.size() * GlobalInterfaceDOFs.size();
                }

                PUSH_RANGE("GEMM (Y_exp)", 10)
                // Interface
                d_ptr_H_exp_interface_GEMM   = h_ptr_H_exp_interface;
                d_ptr_rhs_exp_interface_GEMM = h_ptr_rhs_exp_interface;
                d_ptr_Y_exp_interface_GEMM   = h_ptr_Y_exp_interface;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInterfaceDOFs[_idx].size(), 1, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_H_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 thrust::raw_pointer_cast(d_ptr_rhs_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 zeroPtr, thrust::raw_pointer_cast(d_ptr_Y_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));

                // Load (sum the interface solution on top of this)
                d_ptr_H_exp_interfaceLoad_GEMM = h_ptr_H_exp_interfaceLoad;
                d_ptr_rhs_exp_load_GEMM        = h_ptr_rhs_exp_load;
                d_ptr_Y_exp_interface_GEMM     = h_ptr_Y_exp_interface;
                if (LocalLoadDOFs[_idx].size() != 0){
                    cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInterfaceDOFs[_idx].size(), 1, LocalLoadDOFs[_idx].size(),
                                                     onePtr, thrust::raw_pointer_cast(d_ptr_H_exp_interfaceLoad_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                     thrust::raw_pointer_cast(d_ptr_rhs_exp_load_GEMM.data()), cudaArrayDataType, LocalLoadDOFs[_idx].size(),
                                                     onePtr, thrust::raw_pointer_cast(d_ptr_Y_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(), batchSize,
                                                     cudaArrayDataType, cudaAlgoType));
                }

                /*--------------
                Compute Residual
                --------------*/
                // r = dRdY * Y + r
                d_ptr_dRdY_interface_GEMM  = h_ptr_dRdY_interface;
                d_ptr_Y_exp_interface_GEMM = h_ptr_Y_exp_interface;
                d_ptr_residual_GEMM        = h_ptr_residual;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, GlobalInterfaceDOFs.size(), 1, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_dRdY_interface_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(),
                                                 thrust::raw_pointer_cast(d_ptr_Y_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_residual_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));

                // r = dRdU * U + r
                d_ptr_dRdU_GEMM              = h_ptr_dRdU;
                d_ptr_rhs_exp_interface_GEMM = h_ptr_rhs_exp_interface;
                d_ptr_residual_GEMM          = h_ptr_residual;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, GlobalInterfaceDOFs.size(), 1, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_dRdU_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(),
                                                 thrust::raw_pointer_cast(d_ptr_rhs_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_residual_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));

                /*------------------------
                Compute Interface Jacobian
                ------------------------*/
                d_ptr_dRdY_interface_GEMM  = h_ptr_dRdY_interface;
                d_ptr_H_exp_interface_GEMM = h_ptr_H_exp_interface;
                d_ptr_J_GEMM               = h_ptr_J;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, GlobalInterfaceDOFs.size(), LocalInterfaceDOFs[_idx].size(), LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_dRdY_interface_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(),
                                                 thrust::raw_pointer_cast(d_ptr_H_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[_idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_J_GEMM.data()), cudaArrayDataType, GlobalInterfaceDOFs.size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));
                POP_RANGE

                /*-----------------
                Synchronize Streams
                -----------------*/
                cudaStreamSynchronize(streams[tid]);
            } // Experimental model loop
        } // omp parallel
        timerExp.stop();
        std::cout << ">>>> Time taken = " << timerExp.getDurationMicroSec()*1e-6 << " sec" << "\n" << std::endl;
    } // if experimental model

    /*----------------
    Interface Coupling
    ----------------*/
    // Get batch array pointers
    int _shift_batch_J        = 0;
    int _shift_batch_residual = 0;
    thrust::device_vector<cuDoubleComplex*> d_ptr_J_batch(batchSize), d_ptr_residual(batchSize);
    for (size_t f = 0; f < batchSize; ++f){
        d_ptr_J_batch[f]        = d_ptr_J_base + _shift_batch_J;
        d_ptr_residual[f]       = d_ptr_residual_base + _shift_batch_residual;
        _shift_batch_J         += GlobalInterfaceDOFs.size() * GlobalInterfaceDOFs.size();
        _shift_batch_residual  += GlobalInterfaceDOFs.size();
    }

    /*-------------------
    LU Decomposition on J
    -------------------*/
    cublas_check(cublasZgetrfBatched(cublasHandle[0], GlobalInterfaceDOFs.size(), thrust::raw_pointer_cast(d_ptr_J_batch.data()), GlobalInterfaceDOFs.size(), NULL, d_ptr_solverInfo, batchSize));

    /*-----------------------
    Compute correction factor
    -----------------------*/
    cublas_check(cublasZgetrsBatched(cublasHandle[0], CUBLAS_OP_N, GlobalInterfaceDOFs.size(), 1, thrust::raw_pointer_cast(d_ptr_J_batch.data()), GlobalInterfaceDOFs.size(),
                                     NULL, thrust::raw_pointer_cast(d_ptr_residual.data()), GlobalInterfaceDOFs.size(), &solverInfo_solve, batchSize));

    /*------------------
    Post-processing Prep
    ------------------*/
    std::vector<int> shift_global_GlobalInterfaceDOFs(subSystems), shift_global_GlobalInternalDOFs(subSystems);
    int _shift_interface = 0;
    int _shift_internal  = 0;
    for (size_t i = 0; i < subSystems; ++i){
        shift_global_GlobalInterfaceDOFs[i] = _shift_interface;
        shift_global_GlobalInternalDOFs[i]  = _shift_internal;
        _shift_interface                   += LocalInterfaceDOFs[i].size();
        _shift_internal                    += LocalInternalDOFs[i].size();
    }

    /*-----------------------------------
    Update solution for simulation models
    -----------------------------------*/
    if (num_simModel != 0){
    //#pragma omp parallel private(tid) num_threads(num_threads)
    #pragma omp parallel private(tid) num_threads(1)    // NOTE: currently only supports single thread run (error in RHS update)
        {
            // Get thread number
            tid = omp_get_thread_num();

            // Array of pointers for batched operation
            thrust::device_vector<cuDoubleComplex*> d_ptr_residual_batch(batchSize)			, d_ptr_rhs_sim_interface_batch(batchSize);
            thrust::device_vector<cuDoubleComplex*> d_ptr_Y_sim_interface_batch(batchSize)  , d_ptr_Y_sim_internal_batch(batchSize);
            thrust::device_vector<cuDoubleComplex*> d_ptr_Y_interface_batch(batchSize)		, d_ptr_Y_internal_batch(batchSize);
            thrust::device_vector<cuDoubleComplex*> d_ptr_Y(batchSize);
            thrust::device_vector<void*> d_ptr_H_sim_interface_GEMM(batchSize)     			, d_ptr_H_sim_interfaceLoad_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_H_sim_internal_GEMM(batchSize)      			, d_ptr_H_sim_internalLoad_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_Y_sim_interface_GEMM(batchSize)     			, d_ptr_Y_sim_internal_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_rhs_sim_interface_GEMM(batchSize)            , d_ptr_rhs_sim_load_GEMM(batchSize);

            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_residual_batch(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_rhs_sim_interface_batch(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_rhs_sim_load_batch(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_H_sim_interface(batchSize), h_ptr_H_sim_interfaceLoad(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_H_sim_internal(batchSize) , h_ptr_H_sim_internalLoad(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_Y_interface(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_Y_sim_interface(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_Y_sim_internal(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_Y(batchSize);

            // Pointers for mappers
            int *d_ptr_GlobalInterfaceDOFs, *d_ptr_GlobalInternalDOFs;

            // Shifts
            int shift_batch_residual;
            int shift_global_rhs_sim_interface, shift_batch_rhs_sim_interface;
            int shift_global_H_sim_interface, shift_global_H_sim_interfaceLoad, shift_global_H_sim_internal, shift_global_H_sim_internalLoad;
            int shift_batch_H_sim_interface , shift_batch_H_sim_interfaceLoad , shift_batch_H_sim_internal , shift_batch_H_sim_internalLoad;
            int shift_global_Y_interface, shift_batch_Y_interface;
            int shift_global_Y_sim_interface, shift_batch_Y_sim_interface;
            int shift_global_Y_sim_internal , shift_batch_Y_sim_internal;
            int shift_global_rhs_sim_load, shift_batch_rhs_sim_load;
            int shift_batch_Y;

            // Indices
            int idx, idx_global_shift;

            #pragma omp for
            for (size_t i = 0; i < num_simModel; ++i){
                // Get global sub-system index
                idx = subSystemID_sim[i];

                /*-------------
                Update pointers
                -------------*/
                shift_batch_residual 		     = 0;
                shift_global_rhs_sim_interface   = 0;
                shift_batch_rhs_sim_interface    = 0;
                shift_global_rhs_sim_load        = 0;
                shift_batch_rhs_sim_load         = 0;
                shift_global_H_sim_interface     = 0;
                shift_batch_H_sim_interface    	 = 0;
                shift_global_H_sim_interfaceLoad = 0;
                shift_batch_H_sim_interfaceLoad  = 0;
                shift_global_H_sim_internal		 = 0;
                shift_batch_H_sim_internal		 = 0;
                shift_global_H_sim_internalLoad  = 0;
                shift_batch_H_sim_internalLoad   = 0;
                shift_global_Y_interface		 = 0;
                shift_batch_Y_interface			 = 0;
                shift_global_Y_sim_interface     = 0;
                shift_batch_Y_sim_interface      = 0;
                shift_global_Y_sim_internal      = 0;
                shift_batch_Y_sim_internal       = 0;
                shift_batch_Y                    = 0;

                // Global shifts
                for (size_t k = 0; k < i; ++k){
                    idx_global_shift = subSystemID_sim[k];
                    for (size_t f = 0; f < batchSize; ++f){
                        shift_global_rhs_sim_interface   += LocalInterfaceDOFs[idx_global_shift].size();
                        shift_global_rhs_sim_load        += LocalLoadDOFs[idx_global_shift].size();
                        shift_global_H_sim_interface     += LocalInterfaceDOFs[idx_global_shift].size() * LocalInterfaceDOFs[idx_global_shift].size();
                        shift_global_H_sim_interfaceLoad += LocalInterfaceDOFs[idx_global_shift].size() * LocalLoadDOFs[idx_global_shift].size();
                        shift_global_H_sim_internal      += LocalInternalDOFs[idx_global_shift].size()  * LocalInterfaceDOFs[idx_global_shift].size();
                        shift_global_H_sim_internalLoad  += LocalInternalDOFs[idx_global_shift].size()  * LocalLoadDOFs[idx_global_shift].size();
                        shift_global_Y_sim_interface     += LocalInterfaceDOFs[idx_global_shift].size();
                        shift_global_Y_sim_internal      += LocalInternalDOFs[idx_global_shift].size();
                    }
                }

                for (size_t k = 0; k < idx; ++k){
                    shift_global_Y_interface += LocalInterfaceDOFs[k].size();
                }

                // Batch shifts
                for (size_t f = 0; f < batchSize; ++f){
                    h_ptr_residual_batch[f]          = d_ptr_residual_base            + shift_global_residual[idx]       + shift_batch_residual;
                    h_ptr_rhs_sim_interface_batch[f] = d_ptr_rhs_sim_interface_base   + shift_global_rhs_sim_interface   + shift_batch_rhs_sim_interface;
                    h_ptr_rhs_sim_load_batch[f]      = d_ptr_rhs_sim_load_base        + shift_global_rhs_sim_load        + shift_batch_rhs_sim_load;
                    h_ptr_H_sim_interface[f]         = d_ptr_H_sim_interface_base     + shift_global_H_sim_interface     + shift_batch_H_sim_interface;
                    h_ptr_H_sim_interfaceLoad[f]     = d_ptr_H_sim_interfaceLoad_base + shift_global_H_sim_interfaceLoad + shift_batch_H_sim_interfaceLoad;
                    h_ptr_H_sim_internal[f]          = d_ptr_H_sim_internal_base      + shift_global_H_sim_internal      + shift_batch_H_sim_internal;
                    h_ptr_H_sim_internalLoad[f]      = d_ptr_H_sim_internalLoad_base  + shift_global_H_sim_internalLoad  + shift_batch_H_sim_internalLoad;
                    h_ptr_Y_interface[f]			 = d_ptr_Y_interface_base		  + shift_global_Y_interface         + shift_batch_Y_interface;
                    h_ptr_Y_sim_interface[f]         = d_ptr_Y_sim_interface_base     + shift_global_Y_sim_interface     + shift_batch_Y_sim_interface;
                    h_ptr_Y_sim_internal[f]          = d_ptr_Y_sim_internal_base      + shift_global_Y_sim_internal      + shift_batch_Y_sim_internal;
                    h_ptr_Y[f]                       = d_ptr_Y_base                   + shift_batch_Y;

                    shift_batch_residual            += GlobalInterfaceDOFs.size();
                    shift_batch_rhs_sim_interface   += LocalInterfaceDOFs[idx].size();
                    shift_batch_rhs_sim_load        += LocalLoadDOFs[idx].size();
                    shift_batch_H_sim_interface     += LocalInterfaceDOFs[idx].size() * LocalInterfaceDOFs[idx].size();
                    shift_batch_H_sim_interfaceLoad += LocalInterfaceDOFs[idx].size() * LocalLoadDOFs[idx].size();
                    shift_batch_H_sim_internal      += LocalInternalDOFs[idx].size()  * LocalInterfaceDOFs[idx].size();
                    shift_batch_H_sim_internalLoad  += LocalInternalDOFs[idx].size()  * LocalLoadDOFs[idx].size();
                    shift_batch_Y_interface			+= GlobalInterfaceDOFs.size();
                    shift_batch_Y_sim_interface     += LocalInterfaceDOFs[idx].size();
                    shift_batch_Y_sim_internal      += LocalInternalDOFs[idx].size();
                    shift_batch_Y                   += DOF_total;
                }

                // Pointer for global index mappers
                d_ptr_GlobalInterfaceDOFs = d_ptr_GlobalInterfaceDOFs_base + shift_global_GlobalInterfaceDOFs[idx];
                d_ptr_GlobalInternalDOFs  = d_ptr_GlobalInternalDOFs_base  + shift_global_GlobalInternalDOFs[idx];

                /*--------
                Update RHS
                --------*/
                d_ptr_residual_batch          = h_ptr_residual_batch;
                d_ptr_rhs_sim_interface_batch = h_ptr_rhs_sim_interface_batch;
                PUSH_RANGE("Update RHS", 11)
                assembly::updateRhsBatched(streams[tid], thrust::raw_pointer_cast(d_ptr_residual_batch.data()), thrust::raw_pointer_cast(d_ptr_rhs_sim_interface_batch.data()),
                                           LocalInterfaceDOFs[idx].size(), batchSize);
                POP_RANGE

                /*---------------
                Update solution Y
                ---------------*/
                PUSH_RANGE("GEMM (Y_sim update)", 12)
                // Interface
                d_ptr_H_sim_interface_GEMM   = h_ptr_H_sim_interface;
                d_ptr_rhs_sim_interface_GEMM = h_ptr_rhs_sim_interface_batch;
                d_ptr_Y_sim_interface_GEMM   = h_ptr_Y_sim_interface;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInterfaceDOFs[idx].size(), 1, LocalInterfaceDOFs[idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_H_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(),
                                                 thrust::raw_pointer_cast(d_ptr_rhs_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(),
                                                 zeroPtr, thrust::raw_pointer_cast(d_ptr_Y_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));

                // Load (sum the interface solution on top of this)
                d_ptr_H_sim_interfaceLoad_GEMM = h_ptr_H_sim_interfaceLoad;
                d_ptr_rhs_sim_load_GEMM        = h_ptr_rhs_sim_load_batch;
                d_ptr_Y_sim_interface_GEMM     = h_ptr_Y_sim_interface;
                if (LocalLoadDOFs[idx].size() != 0){
                    cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInterfaceDOFs[idx].size(), 1, LocalLoadDOFs[idx].size(),
                                                     onePtr, thrust::raw_pointer_cast(d_ptr_H_sim_interfaceLoad_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(),
                                                     thrust::raw_pointer_cast(d_ptr_rhs_sim_load_GEMM.data()), cudaArrayDataType, LocalLoadDOFs[idx].size(),
                                                     onePtr, thrust::raw_pointer_cast(d_ptr_Y_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(), batchSize,
                                                     cudaArrayDataType, cudaAlgoType));
                }

                // Post-processing
                if (postProcess){
                    if (LocalInternalDOFs[idx].size() != 0){
                        // Internal (H_internal * rhs_interface)
                        d_ptr_H_sim_internal_GEMM    = h_ptr_H_sim_internal;
                        d_ptr_rhs_sim_interface_GEMM = h_ptr_rhs_sim_interface_batch;
                        d_ptr_Y_sim_internal_GEMM    = h_ptr_Y_sim_internal;
                        cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInternalDOFs[idx].size(), 1, LocalInterfaceDOFs[idx].size(),
                                                         onePtr, thrust::raw_pointer_cast(d_ptr_H_sim_internal_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[idx].size(),
                                                         thrust::raw_pointer_cast(d_ptr_rhs_sim_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(),
                                                         zeroPtr, thrust::raw_pointer_cast(d_ptr_Y_sim_internal_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[idx].size(), batchSize,
                                                         cudaArrayDataType, cudaAlgoType));

                        // Load (sum the internal solution on top of this) (H_internalLoad * rhs_load)
                        d_ptr_H_sim_internalLoad_GEMM  = h_ptr_H_sim_internalLoad;
                        d_ptr_rhs_sim_load_GEMM        = h_ptr_rhs_sim_load_batch;
                        d_ptr_Y_sim_internal_GEMM      = h_ptr_Y_sim_internal;
                        if (LocalLoadDOFs[idx].size() != 0){
                            cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInternalDOFs[idx].size(), 1, LocalLoadDOFs[idx].size(),
                                                             onePtr, thrust::raw_pointer_cast(d_ptr_H_sim_internalLoad_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[idx].size(),
                                                             thrust::raw_pointer_cast(d_ptr_rhs_sim_load_GEMM.data()), cudaArrayDataType, LocalLoadDOFs[idx].size(),
                                                             onePtr, thrust::raw_pointer_cast(d_ptr_Y_sim_internal_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[idx].size(), batchSize,
                                                             cudaArrayDataType, cudaAlgoType));
                        }
                    }
                }
                POP_RANGE

                /*----------------------------------
                Collocate solution Y to single array
                ----------------------------------*/
                d_ptr_Y_sim_interface_batch	  = h_ptr_Y_sim_interface;
                d_ptr_Y_interface_batch		  = h_ptr_Y_interface;
                PUSH_RANGE("Collocate Final Solution (sim)", 13)
                assembly::collocateFinalSolution(streams[tid], thrust::raw_pointer_cast(d_ptr_Y_sim_interface_batch.data()),
                                                 thrust::raw_pointer_cast(d_ptr_Y_interface_batch.data()), LocalInterfaceDOFs[idx].size(), batchSize);
                if (postProcess){
                    d_ptr_Y_sim_interface_batch = h_ptr_Y_sim_interface;
                    d_ptr_Y                     = h_ptr_Y;
                    assembly::collocateFinalSolutionPostProcess(streams[tid], thrust::raw_pointer_cast(d_ptr_Y_sim_interface_batch.data()),
                                                                thrust::raw_pointer_cast(d_ptr_Y.data()), d_ptr_GlobalInterfaceDOFs,
                                                                LocalInterfaceDOFs[idx].size(), batchSize);
                    if (LocalInternalDOFs[idx].size() != 0){
                        d_ptr_Y_sim_internal_batch = h_ptr_Y_sim_internal;
                        d_ptr_Y                    = h_ptr_Y;
                        assembly::collocateFinalSolutionPostProcess(streams[tid], thrust::raw_pointer_cast(d_ptr_Y_sim_internal_batch.data()),
                                                                    thrust::raw_pointer_cast(d_ptr_Y.data()), d_ptr_GlobalInternalDOFs,
                                                                    LocalInternalDOFs[idx].size(), batchSize);
                    }
                }
                POP_RANGE

                /*-----------------
                Synchronize Streams
                -----------------*/
                cudaStreamSynchronize(streams[tid]);
            } // Simulation model loop
        } // omp parallel
    } // if simulation model

    /*-------------------------------------
    Update solution for experimental models
    -------------------------------------*/
    if (num_expModel != 0){
    //#pragma omp parallel private(tid) num_threads(num_threads)
    #pragma omp parallel private(tid) num_threads(1)    // NOTE: currently only supports single thread run (error in RHS update)
        {
            // Get thread number
            tid = omp_get_thread_num();

            // Array of pointers for batched operation
            thrust::device_vector<cuDoubleComplex*> d_ptr_residual_batch(batchSize)		  , d_ptr_rhs_exp_interface_batch(batchSize);
            thrust::device_vector<cuDoubleComplex*> d_ptr_Y_exp_interface_batch(batchSize), d_ptr_Y_exp_internal_batch(batchSize);
            thrust::device_vector<cuDoubleComplex*> d_ptr_Y_interface_batch(batchSize)	  , d_ptr_Y_internal_batch(batchSize);
            thrust::device_vector<cuDoubleComplex*> d_ptr_Y(batchSize);
            thrust::device_vector<void*> d_ptr_H_exp_interface_GEMM(batchSize)     		  , d_ptr_H_exp_interfaceLoad_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_H_exp_internal_GEMM(batchSize)      		  , d_ptr_H_exp_internalLoad_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_Y_exp_interface_GEMM(batchSize)     		  , d_ptr_Y_exp_internal_GEMM(batchSize);
            thrust::device_vector<void*> d_ptr_rhs_exp_interface_GEMM(batchSize)          , d_ptr_rhs_exp_load_GEMM(batchSize);

            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_residual_batch(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_rhs_exp_interface_batch(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_rhs_exp_load_batch(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_H_exp_interface(batchSize), h_ptr_H_exp_interfaceLoad(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_H_exp_internal(batchSize) , h_ptr_H_exp_internalLoad(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_Y_interface(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_Y_exp_interface(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_Y_exp_internal(batchSize);
            thrust::host_vector<cuDoubleComplex*, pinnedAllocPtr> h_ptr_Y(batchSize);

            // Pointers for mappers
            int *d_ptr_GlobalInterfaceDOFs, *d_ptr_GlobalInternalDOFs;

            // Shifts
            int shift_batch_residual;
            int shift_global_rhs_exp_interface, shift_batch_rhs_exp_interface;
            int shift_global_H_exp_interface, shift_global_H_exp_interfaceLoad, shift_global_H_exp_internal, shift_global_H_exp_internalLoad;
            int shift_batch_H_exp_interface , shift_batch_H_exp_interfaceLoad , shift_batch_H_exp_internal , shift_batch_H_exp_internalLoad;
            int shift_global_Y_interface, shift_batch_Y_interface;
            int shift_global_Y_exp_interface, shift_batch_Y_exp_interface;
            int shift_global_Y_exp_internal , shift_batch_Y_exp_internal;
            int shift_global_rhs_exp_load, shift_batch_rhs_exp_load;
            int shift_batch_Y;

            // Indices
            int idx, idx_global_shift;

            #pragma omp for
            for (size_t i = 0; i < num_expModel; ++i){
                // Get global sub-system index
                idx = subSystemID_exp[i];

                /*-------------
                Update pointers
                -------------*/
                shift_batch_residual 		     = 0;
                shift_global_rhs_exp_interface   = 0;
                shift_batch_rhs_exp_interface    = 0;
                shift_global_rhs_exp_load        = 0;
                shift_batch_rhs_exp_load         = 0;
                shift_global_H_exp_interface     = 0;
                shift_batch_H_exp_interface    	 = 0;
                shift_global_H_exp_interfaceLoad = 0;
                shift_batch_H_exp_interfaceLoad  = 0;
                shift_global_H_exp_internal		 = 0;
                shift_batch_H_exp_internal		 = 0;
                shift_global_H_exp_internalLoad  = 0;
                shift_batch_H_exp_internalLoad   = 0;
                shift_global_Y_interface		 = 0;
                shift_batch_Y_interface			 = 0;
                shift_global_Y_exp_interface     = 0;
                shift_batch_Y_exp_interface      = 0;
                shift_global_Y_exp_internal      = 0;
                shift_batch_Y_exp_internal       = 0;
                shift_batch_Y                    = 0;

                // Global shifts
                for (size_t k = 0; k < i; ++k){
                    idx_global_shift = subSystemID_exp[k];
                    for (size_t f = 0; f < batchSize; ++f){
                        shift_global_rhs_exp_interface   += LocalInterfaceDOFs[idx_global_shift].size();
                        shift_global_rhs_exp_load        += LocalLoadDOFs[idx_global_shift].size();
                        shift_global_H_exp_interface     += LocalInterfaceDOFs[idx_global_shift].size() * LocalInterfaceDOFs[idx_global_shift].size();
                        shift_global_H_exp_interfaceLoad += LocalInterfaceDOFs[idx_global_shift].size() * LocalLoadDOFs[idx_global_shift].size();
                        shift_global_H_exp_internal      += LocalInternalDOFs[idx_global_shift].size()  * LocalInterfaceDOFs[idx_global_shift].size();
                        shift_global_H_exp_internalLoad  += LocalInternalDOFs[idx_global_shift].size()  * LocalLoadDOFs[idx_global_shift].size();
                        shift_global_Y_exp_interface     += LocalInterfaceDOFs[idx_global_shift].size();
                        shift_global_Y_exp_internal      += LocalInternalDOFs[idx_global_shift].size();
                    }
                }

                for (size_t k = 0; k < idx; ++k) shift_global_Y_interface += LocalInterfaceDOFs[k].size();

                // Batch shifts
                for (size_t f = 0; f < batchSize; ++f){
                    h_ptr_residual_batch[f]          = d_ptr_residual_base            + shift_global_residual[idx]       + shift_batch_residual;
                    h_ptr_rhs_exp_interface_batch[f] = d_ptr_rhs_exp_interface_base   + shift_global_rhs_exp_interface   + shift_batch_rhs_exp_interface;
                    h_ptr_rhs_exp_load_batch[f]      = d_ptr_rhs_exp_load_base        + shift_global_rhs_exp_load        + shift_batch_rhs_exp_load;
                    h_ptr_H_exp_interface[f]         = d_ptr_H_exp_interface_base     + shift_global_H_exp_interface     + shift_batch_H_exp_interface;
                    h_ptr_H_exp_interfaceLoad[f]     = d_ptr_H_exp_interfaceLoad_base + shift_global_H_exp_interfaceLoad + shift_batch_H_exp_interfaceLoad;
                    h_ptr_H_exp_internal[f]          = d_ptr_H_exp_internal_base      + shift_global_H_exp_internal      + shift_batch_H_exp_internal;
                    h_ptr_H_exp_internalLoad[f]      = d_ptr_H_exp_internalLoad_base  + shift_global_H_exp_internalLoad  + shift_batch_H_exp_internalLoad;
                    h_ptr_Y_interface[f]			 = d_ptr_Y_interface_base		  + shift_global_Y_interface         + shift_batch_Y_interface;
                    h_ptr_Y_exp_interface[f]         = d_ptr_Y_exp_interface_base     + shift_global_Y_exp_interface     + shift_batch_Y_exp_interface;
                    h_ptr_Y_exp_internal[f]          = d_ptr_Y_exp_internal_base      + shift_global_Y_exp_internal      + shift_batch_Y_exp_internal;
                    h_ptr_Y[f]                       = d_ptr_Y_base                   + shift_batch_Y;

                    shift_batch_residual            += GlobalInterfaceDOFs.size();
                    shift_batch_rhs_exp_interface   += LocalInterfaceDOFs[idx].size();
                    shift_batch_rhs_exp_load        += LocalLoadDOFs[idx].size();
                    shift_batch_H_exp_interface     += LocalInterfaceDOFs[idx].size() * LocalInterfaceDOFs[idx].size();
                    shift_batch_H_exp_interfaceLoad += LocalInterfaceDOFs[idx].size() * LocalLoadDOFs[idx].size();
                    shift_batch_H_exp_internal      += LocalInternalDOFs[idx].size()  * LocalInterfaceDOFs[idx].size();
                    shift_batch_H_exp_internalLoad  += LocalInternalDOFs[idx].size()  * LocalLoadDOFs[idx].size();
                    shift_batch_Y_interface			+= GlobalInterfaceDOFs.size();
                    shift_batch_Y_exp_interface     += LocalInterfaceDOFs[idx].size();
                    shift_batch_Y_exp_internal      += LocalInternalDOFs[idx].size();
                    shift_batch_Y                   += DOF_total;
                }

                // Pointer for global index mappers
                d_ptr_GlobalInterfaceDOFs = d_ptr_GlobalInterfaceDOFs_base + shift_global_GlobalInterfaceDOFs[idx];
                d_ptr_GlobalInternalDOFs  = d_ptr_GlobalInternalDOFs_base  + shift_global_GlobalInternalDOFs[idx];

                /*--------
                Update RHS
                --------*/
                d_ptr_residual_batch          = h_ptr_residual_batch;
                d_ptr_rhs_exp_interface_batch = h_ptr_rhs_exp_interface_batch;
                PUSH_RANGE("Update RHS (exp)", 15)
                assembly::updateRhsBatched(streams[tid], thrust::raw_pointer_cast(d_ptr_residual_batch.data()), thrust::raw_pointer_cast(d_ptr_rhs_exp_interface_batch.data()),
                                           LocalInterfaceDOFs[idx].size(), batchSize);
                POP_RANGE

                /*---------------
                Update solution Y
                ---------------*/
                PUSH_RANGE("GEMM (Y_exp final update)", 16)
                // Interface
                d_ptr_H_exp_interface_GEMM   = h_ptr_H_exp_interface;
                d_ptr_rhs_exp_interface_GEMM = h_ptr_rhs_exp_interface_batch;
                d_ptr_Y_exp_interface_GEMM   = h_ptr_Y_exp_interface;
                cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInterfaceDOFs[idx].size(), 1, LocalInterfaceDOFs[idx].size(),
                                                 onePtr, thrust::raw_pointer_cast(d_ptr_H_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(),
                                                 thrust::raw_pointer_cast(d_ptr_rhs_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(),
                                                 zeroPtr, thrust::raw_pointer_cast(d_ptr_Y_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(), batchSize,
                                                 cudaArrayDataType, cudaAlgoType));

                // Load (sum the interface solution on top of this)
                d_ptr_H_exp_interfaceLoad_GEMM = h_ptr_H_exp_interfaceLoad;
                d_ptr_rhs_exp_load_GEMM        = h_ptr_rhs_exp_load_batch;
                d_ptr_Y_exp_interface_GEMM     = h_ptr_Y_exp_interface;
                if (LocalLoadDOFs[idx].size() != 0){
                    cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInterfaceDOFs[idx].size(), 1, LocalLoadDOFs[idx].size(),
                                                     onePtr, thrust::raw_pointer_cast(d_ptr_H_exp_interfaceLoad_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(),
                                                     thrust::raw_pointer_cast(d_ptr_rhs_exp_load_GEMM.data()), cudaArrayDataType, LocalLoadDOFs[idx].size(),
                                                     onePtr, thrust::raw_pointer_cast(d_ptr_Y_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(), batchSize,
                                                     cudaArrayDataType, cudaAlgoType));
                }

                // Post-processing
                if (postProcess){
                    if (LocalInternalDOFs[idx].size() != 0){
                        // Internal (H_internal * rhs_interface)
                        d_ptr_H_exp_internal_GEMM    = h_ptr_H_exp_internal;
                        d_ptr_rhs_exp_interface_GEMM = h_ptr_rhs_exp_interface_batch;
                        d_ptr_Y_exp_internal_GEMM    = h_ptr_Y_exp_internal;
                        cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInternalDOFs[idx].size(), 1, LocalInterfaceDOFs[idx].size(),
                                                         onePtr, thrust::raw_pointer_cast(d_ptr_H_exp_internal_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[idx].size(),
                                                         thrust::raw_pointer_cast(d_ptr_rhs_exp_interface_GEMM.data()), cudaArrayDataType, LocalInterfaceDOFs[idx].size(),
                                                         zeroPtr, thrust::raw_pointer_cast(d_ptr_Y_exp_internal_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[idx].size(), batchSize,
                                                         cudaArrayDataType, cudaAlgoType));

                        // Load (sum the internal solution on top of this) (H_internalLoad * rhs_load)
                        d_ptr_H_exp_internalLoad_GEMM  = h_ptr_H_exp_internalLoad;
                        d_ptr_rhs_exp_load_GEMM        = h_ptr_rhs_exp_load_batch;
                        d_ptr_Y_exp_internal_GEMM      = h_ptr_Y_exp_internal;
                        if (LocalLoadDOFs[idx].size() != 0){
                            cublas_check(cublasGemmBatchedEx(cublasHandle[tid], CUBLAS_OP_N, CUBLAS_OP_N, LocalInternalDOFs[idx].size(), 1, LocalLoadDOFs[idx].size(),
                                                             onePtr, thrust::raw_pointer_cast(d_ptr_H_exp_internalLoad_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[idx].size(),
                                                             thrust::raw_pointer_cast(d_ptr_rhs_exp_load_GEMM.data()), cudaArrayDataType, LocalLoadDOFs[idx].size(),
                                                             onePtr, thrust::raw_pointer_cast(d_ptr_Y_exp_internal_GEMM.data()), cudaArrayDataType, LocalInternalDOFs[idx].size(), batchSize,
                                                             cudaArrayDataType, cudaAlgoType));
                        }
                    }
                }
                POP_RANGE

                /*----------------------------------
                Collocate solution Y to single array
                ----------------------------------*/
                d_ptr_Y_exp_interface_batch	  = h_ptr_Y_exp_interface;
                d_ptr_Y_interface_batch		  = h_ptr_Y_interface;
                PUSH_RANGE("Collocate Final Solution (exp)", 17)
                assembly::collocateFinalSolution(streams[tid], thrust::raw_pointer_cast(d_ptr_Y_exp_interface_batch.data()),
                                                 thrust::raw_pointer_cast(d_ptr_Y_interface_batch.data()), LocalInterfaceDOFs[idx].size(), batchSize);
                if (postProcess){
                    d_ptr_Y_exp_interface_batch = h_ptr_Y_exp_interface;
                    d_ptr_Y                     = h_ptr_Y;
                    assembly::collocateFinalSolutionPostProcess(streams[tid], thrust::raw_pointer_cast(d_ptr_Y_exp_interface_batch.data()),
                                                                thrust::raw_pointer_cast(d_ptr_Y.data()), d_ptr_GlobalInterfaceDOFs,
                                                                LocalInterfaceDOFs[idx].size(), batchSize);
                    if (LocalInternalDOFs[idx].size() != 0){
                        d_ptr_Y_exp_internal_batch = h_ptr_Y_exp_internal;
                        d_ptr_Y                    = h_ptr_Y;
                        assembly::collocateFinalSolutionPostProcess(streams[tid], thrust::raw_pointer_cast(d_ptr_Y_exp_internal_batch.data()),
                                                                    thrust::raw_pointer_cast(d_ptr_Y.data()), d_ptr_GlobalInternalDOFs,
                                                                    LocalInternalDOFs[idx].size(), batchSize);
                    }
                }
                POP_RANGE
                /*-----------------
                Synchronize Streams
                -----------------*/
                cudaStreamSynchronize(streams[tid]);
            } // Simulation model loop
        } // omp parallel
    } // if experimental model

    timerIJCA.stop();
    std::cout << ">> Interface Jacobian Substructuring finished" << std::endl;
    std::cout << ">>>> Time taken = " << timerIJCA.getDurationMicroSec()*1e-6 << " sec" << "\n" << std::endl;
    POP_RANGE // Interface Jacobian Substructuring

    /*------------------------
    Retrieve & Write Solutions
    ------------------------*/
    PUSH_RANGE("Solution Transfer to Host", 18)
    timerDataD2H.start();
    thrust::host_vector<cuDoubleComplex> Y_interface = d_Y_interface;
    io::writeMtxDenseComplex(Y_interface, GlobalInterfaceDOFs.size(), batchSize, filepath_sol, "Y_interface.mtx");
    if (postProcess){
        thrust::host_vector<cuDoubleComplex> Y = d_Y;
        io::writeMtxDenseComplex(Y, Y.size()/batchSize, batchSize, filepath_sol, "Y.mtx");
    }
    timerDataD2H.stop();
    POP_RANGE // Solution Transfer to Host
    std::cout << ">> Solutions copied to host and written" << std::endl;
    std::cout << ">>>> Time taken = " << timerDataD2H.getDurationMicroSec()*1e-6 << " sec" << "\n" << std::endl;

    // Destroy cuBLAS & streams
    for (size_t i = 0; i < num_threads; ++i){
        cublasDestroy(cublasHandle[i]);
        cudaStreamDestroy(streams[i]);
    }

    timerTotal.stop();
    std::cout << ">>>> End of program" << std::endl;
    std::cout << ">>>>>> Total execution time (s) = " << timerTotal.getDurationMicroSec()*1e-6 << "\n" << std::endl;
}
