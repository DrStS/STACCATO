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
* \file dataStructures.cu
* Written by Ji-Ho Yang
* This file constructs data structures
* \date 7/12/2019
**************************************************************************************************/

// Libraries
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <thrust/host_vector.h>

#include <iostream>

// Header Files
#include "dataStructures.cuh"
#include "../io/io.cuh"

// Namespace
using namespace staccato;

void data::constructHostDataStructure(
                                      std::string filepath_sim, std::string filepath_exp, std::string filepath_jac, std::string base_format,
                                      std::string baseName_K, std::string baseName_M, std::string baseName_D, std::string baseName_B, std::string baseName_C, std::string baseName_H,
                                      std::string baseName_dRdY, std::string baseName_dRdU,
                                      int num_simModel, int num_expModel, std::vector<int> subSystemID_sim, std::vector<int> subSystemID_exp,
                                      thrust::host_vector<int> freq,
                                      thrust::host_vector<int> &row_A_sub, thrust::host_vector<int> &row_H_exp_sub,
                                      thrust::host_vector<int> &nnz_A_sub, int &nnz_A, int &nnz_A_max,
                                      int &row_dRdY, int &row_dRdU, int &col_dRdY, int &col_dRdU, int &nnz_dRdY, int &nnz_dRdU, int &numEntry_dRdY, int &numEntry_dRdU,
                                      thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &K_sub,
                                      thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &M_sub,
                                      thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &D_sub,
                                      thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &B_sub,
                                      thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &C_sub,
                                      thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &H_exp_sub,
                                      thrust::host_vector<cuDoubleComplex> &dRdY, thrust::host_vector<cuDoubleComplex> &dRdU
                                     )
{
    /*------------------------
    READ AND PROCESS MTX FILES
    ------------------------*/
    // Temporary variables
    std::string _filename_K[num_simModel], _filename_M[num_simModel], _filename_D[num_simModel];
    std::string _filename_B[num_simModel], _filename_C[num_simModel];
    std::string _filename_H[num_expModel];
    std::string _filename_dRdY, _filename_dRdU;
    int _row_A, _col_A, _nnz_A;
    int _row_B, _col_B, _nnz_B;
    int _row_C, _col_C, _nnz_C;
    int _row_H_exp, _col_H_exp, _nnz_H_exp;
    // Initialisation
    nnz_A = 0;

    // Read matrices for simulation model
    for (size_t i = 0; i < num_simModel; ++i){
        _filename_K[i] = baseName_K + std::to_string(subSystemID_sim[i]+1) + base_format;
        _filename_M[i] = baseName_M + std::to_string(subSystemID_sim[i]+1) + base_format;
        _filename_D[i] = baseName_D + std::to_string(subSystemID_sim[i]+1) + base_format;
        _filename_B[i] = baseName_B + std::to_string(subSystemID_sim[i]+1) + base_format;
        _filename_C[i] = baseName_C + std::to_string(subSystemID_sim[i]+1) + base_format;

        // i8
        io::readMtxDense(K_sub[i], filepath_sim, _filename_K[i], _row_A, _col_A, _nnz_A, true);
        io::readMtxDense(M_sub[i], filepath_sim, _filename_M[i], _row_A, _col_A, _nnz_A, false);
        io::readMtxDense(D_sub[i], filepath_sim, _filename_D[i], _row_A, _col_A, _nnz_A, false);
        io::readMtxDense(B_sub[i], filepath_sim, _filename_B[i], _row_B, _col_B, _nnz_B, false);
        io::readMtxDense(C_sub[i], filepath_sim, _filename_C[i], _row_C, _col_C, _nnz_C, false);

/*
        // g20
        io::readMtxDense(K_sub[i], filepath_sim, _filename_K[i], _row_A, _col_A, _nnz_A, true);
        io::readMtxDense(M_sub[i], filepath_sim, _filename_M[i], _row_A, _col_A, _nnz_A, true);
        io::readMtxDense(D_sub[i], filepath_sim, _filename_D[i], _row_A, _col_A, _nnz_A, false);
        io::readMtxDense(B_sub[i], filepath_sim, _filename_B[i], _row_B, _col_B, _nnz_B, true);
        io::readMtxDense(C_sub[i], filepath_sim, _filename_C[i], _row_C, _col_C, _nnz_C, true);
*/

        K_sub[i].pop_back();
        M_sub[i].pop_back();
        D_sub[i].pop_back();
        B_sub[i].pop_back();
        C_sub[i].pop_back();

        // Get rows, cols, and nnz
        row_A_sub[i] = _row_A;
        nnz_A_sub[i] = _row_A*_row_A;
        nnz_A       += nnz_A_sub[i];
    }

    // Get maximum matrix size
    auto nnz_A_max_it = thrust::max_element(nnz_A_sub.begin(), nnz_A_sub.end());
    nnz_A_max         = *nnz_A_max_it;

    // Read matrices for experimental model
    int _idx;
    bool _isComplex = true;
    for (size_t i = 0; i < num_expModel; ++i){
        for (size_t j = 0; j < freq.size(); ++j){
            _idx = i*freq.size() + j;
            _filename_H[i] = baseName_H + std::to_string(subSystemID_exp[i]+1) + "_\0" + std::to_string(freq[j]) + base_format;

            // i8
            if (i == 0) _isComplex = false;
            io::readMtxDense(H_exp_sub[_idx], filepath_exp, _filename_H[i], _row_H_exp, _col_H_exp, _nnz_H_exp, _isComplex);
            _isComplex = true;

/*
            // g20
            io::readMtxDense(H_exp_sub[_idx], filepath_exp, _filename_H[i], _row_H_exp, _col_H_exp, _nnz_H_exp, true);
*/

            H_exp_sub[_idx].pop_back();
        }
        // Get row nnz
        row_H_exp_sub[i] = _row_H_exp;
    }

    // Read matrices for interface Jacobian
    _filename_dRdY = baseName_dRdY + base_format;
    _filename_dRdU = baseName_dRdU + base_format;
    io::readMtxSparse(dRdY, filepath_jac, _filename_dRdY, row_dRdY, col_dRdY, nnz_dRdY, false);
    io::readMtxSparse(dRdU, filepath_jac, _filename_dRdU, row_dRdU, col_dRdU, nnz_dRdU, false);
    dRdY.pop_back();
    dRdU.pop_back();

    // Get information of the interface Jacobian related matrices
    numEntry_dRdY = row_dRdY*col_dRdY;
    numEntry_dRdU = row_dRdU*col_dRdU;
}

void data::extractHostMatrices(
                               std::vector<std::vector<int>> &LocalTotalDOFs,
                               std::vector<std::vector<int>> LocalInterfaceDOFs,
                               std::vector<std::vector<int>> &LocalInternalDOFs,
                               std::vector<std::vector<int>> LocalLoadDOFs,
                               std::vector<int> &GlobalTotalDOFs,
                               std::vector<int> &GlobalInterfaceDOFs,
                               std::vector<int> &GlobalInternalDOFs,
                               std::vector<bool> subSystem_flag,
                               std::vector<int> subSystemID_Global2Local,
                               thrust::host_vector<int> row_A_sub, thrust::host_vector<int> row_H_exp_sub, int col_dRdY, int batchSize,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>>  B_sub          , thrust::host_vector<thrust::host_vector<cuDoubleComplex>>  C_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>>  H_exp_sub      ,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &B_interface_sub, thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &B_load_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &C_interface_sub, thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &C_internal_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &H_exp_interface_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &H_exp_interfaceLoad_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &H_exp_internal_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &H_exp_internalLoad_sub,
                               thrust::host_vector<cuDoubleComplex> dRdY, thrust::host_vector<cuDoubleComplex> &dRdY_interface,
                               int &nnz_B_interface, int &nnz_B_load, int &nnz_C_interface, int &nnz_C_internal,
                               int &nnz_B_interface_max, int &nnz_B_load_max, int &nnz_C_interface_max, int &nnz_C_internal_max,
                               thrust::host_vector<int> &nnz_B_interface_sub, thrust::host_vector<int> &nnz_B_load_sub,
                               thrust::host_vector<int> &nnz_C_interface_sub, thrust::host_vector<int> &nnz_C_internal_sub
                              )
{
    // Initialisation
    nnz_B_interface = 0; nnz_B_load = 0;
    nnz_C_interface = 0; nnz_C_internal = 0;

    // Get vector of local DOFs excluding interface ones
    int _subSystems = LocalInterfaceDOFs.size();
    for (size_t i = 0; i < _subSystems; ++i){
        int _idx = subSystemID_Global2Local[i];
        if (subSystem_flag[i] == SIM){
            int _DOFs = B_sub[_idx].size() / row_A_sub[_idx];
            // Total DOFs - Simulation
            LocalTotalDOFs[i].resize(_DOFs);
            std::iota(LocalTotalDOFs[i].begin(), LocalTotalDOFs[i].end(), 0);
            // Non-interface DOFs - Simulation
            LocalInternalDOFs[i].resize(_DOFs - LocalInterfaceDOFs[i].size());
            std::set_difference(LocalTotalDOFs[i].begin(), LocalTotalDOFs[i].end(), LocalInterfaceDOFs[i].begin(), LocalInterfaceDOFs[i].end(), LocalInternalDOFs[i].begin());
        }

        else{
            int _row_H_exp_sub = row_H_exp_sub[_idx];
            // Total DOFs - Experiment
            LocalTotalDOFs[i].resize(_row_H_exp_sub);
            std::iota(LocalTotalDOFs[i].begin(), LocalTotalDOFs[i].end(), 0);
            // Non-interface DOFs - Experiment
            LocalInternalDOFs[i].resize(_row_H_exp_sub - LocalInterfaceDOFs[i].size());
            std::set_difference(LocalTotalDOFs[i].begin(), LocalTotalDOFs[i].end(), LocalInterfaceDOFs[i].begin(), LocalInterfaceDOFs[i].end(), LocalInternalDOFs[i].begin());
        }
    }

    // Get vector of global DOFs
    int _totalNumDOFs = 0;
    GlobalTotalDOFs.resize(col_dRdY);

    // Interface
    std::iota(GlobalTotalDOFs.begin(), GlobalTotalDOFs.end(), 0);
    for (size_t i = 0; i < _subSystems; ++i){
        for (size_t j = 0; j < LocalInterfaceDOFs[i].size(); ++j){
            GlobalInterfaceDOFs.push_back(LocalInterfaceDOFs[i][j] + _totalNumDOFs);
        }
        _totalNumDOFs += LocalTotalDOFs[i].size();
    }

    // Internal
    int _internalDOFs = 0;
    for (size_t i = 0; i < _subSystems; ++i){
        _internalDOFs += LocalInternalDOFs[i].size();
    }
    GlobalInternalDOFs.resize(_internalDOFs);
    std::set_difference(GlobalTotalDOFs.begin(), GlobalTotalDOFs.end(), GlobalInterfaceDOFs.begin(), GlobalInterfaceDOFs.end(), GlobalInternalDOFs.begin());

    // Extract sub-matrices
    for (size_t k = 0; k < _subSystems; ++k){
        int _sys = subSystemID_Global2Local[k];
        if (subSystem_flag[k] == SIM){
            // Extract interface sub-matrix of C (extract rows) -> C_interface = C(LocalInterfaceDOFs, :)
            C_interface_sub[_sys].resize(LocalInterfaceDOFs[k].size() * row_A_sub[_sys]);
            int _row = C_sub[_sys].size() / row_A_sub[_sys];                                            // Number of total inputs per subsystem
            int _idx_sub = 0;
            for (size_t j = 0; j < row_A_sub[_sys]; ++j){                                               // Loop through columns (all the DOFs)
                for (size_t i = 0; i < LocalInterfaceDOFs[k].size(); ++i){                              // Loop through rows (only the interface inputs)
                    int _idx = LocalInterfaceDOFs[k][i] + _row*j;
                    C_interface_sub[_sys][_idx_sub] = C_sub[_sys][_idx];
                    _idx_sub++;
                }
            }
            // Collect nnz
            nnz_C_interface_sub[_sys] = _idx_sub;

            // Extract interface sub-matrix of B (extract columns) -> B_interface = B(:, LocalInterfaceDOFs)
            B_interface_sub[_sys].resize(row_A_sub[_sys] * LocalInterfaceDOFs[k].size());
            _row = row_A_sub[_sys];                                                                     // Number of total DOFs per subsystem
            _idx_sub = 0;
            for (size_t j = 0; j < LocalInterfaceDOFs[k].size(); ++j){                                  // Loop through columns (only the interface inputs)
                for (size_t i = 0; i < row_A_sub[_sys]; ++i){                                           // Loop through rows (all the DOFs)
                    int _idx = i + _row*LocalInterfaceDOFs[k][j];
                    B_interface_sub[_sys][_idx_sub] = B_sub[_sys][_idx];
                    _idx_sub++;
                }
            }
            // Collect nnz
            nnz_B_interface_sub[_sys] = _idx_sub;

            // Extract internal DOF sub-matrix of C -> C_internal = C(internalDOFs, :)
            C_internal_sub[_sys].resize(LocalInternalDOFs[k].size() * row_A_sub[_sys]);
            _row = C_sub[_sys].size() / row_A_sub[_sys];                                                // Number of total inputs per subsystem
            _idx_sub = 0;
            for (size_t j = 0; j < row_A_sub[_sys]; ++j){                                               // Loop through columns (all the DOFs)
                for (size_t i = 0; i < LocalInternalDOFs[k].size(); ++i){                               // Loop through rows (only the internal inputs)
                    int _idx = LocalInternalDOFs[k][i] + _row*j;
                    C_internal_sub[_sys][_idx_sub] = C_sub[_sys][_idx];
                    _idx_sub++;
                }
            }
            // Collect nnz
            nnz_C_internal_sub[_sys] = _idx_sub;

            // Extract external load sub-matrix B -> B_load = B(:, LocalLoadDOFs);
            B_load_sub[_sys].resize(row_A_sub[_sys] * LocalLoadDOFs[k].size());
            _row = row_A_sub[_sys];                                                                     // Number of total DOFs per subsystem
            _idx_sub = 0;
            for (size_t j = 0; j < LocalLoadDOFs[k].size(); ++j){                                       // Loop through columns (only the load DOFs)
                for (size_t i = 0; i < row_A_sub[_sys]; ++i){                                           // Loop through rows (all the DOFs)
                    int _idx = i + _row*LocalLoadDOFs[k][j];
                    B_load_sub[_sys][_idx_sub] = B_sub[_sys][_idx];
                    _idx_sub++;
                }
            }
            // Collect nnz
            nnz_B_load_sub[_sys] = _idx_sub;

            // Accumulate nnz
            nnz_B_interface += nnz_B_interface_sub[_sys];
            nnz_C_interface += nnz_C_interface_sub[_sys];
            nnz_B_load      += nnz_B_load_sub[_sys];
            nnz_C_internal      += nnz_C_internal_sub[_sys];
        }

        else{
            for (size_t f = 0; f < batchSize; ++f){
                // Extract interface sub-matrix of experimentally measured H (for all the frequency points) -> H_exp_interface = H_exp(LocalInterfaceDOFs, LocalInterfaceDOFs)
                int _row = std::sqrt(H_exp_sub[_sys*batchSize+f].size());                               // Number of total inputs per subsystem
                H_exp_interface_sub[_sys*batchSize+f].resize(LocalInterfaceDOFs[k].size() * LocalInterfaceDOFs[k].size());
                int _idx_sub = 0;
                for (size_t j = 0; j < LocalInterfaceDOFs[k].size(); ++j){                              // Loop through columns (only the interface inputs)
                    for (size_t i = 0; i < LocalInterfaceDOFs[k].size(); ++i){                          // Loop through rows (only the interface inputs)
                        int _idx = LocalInterfaceDOFs[k][i] + _row*LocalInterfaceDOFs[k][j];
                        H_exp_interface_sub[_sys*batchSize+f][_idx_sub] = H_exp_sub[_sys*batchSize+f][_idx];
                        _idx_sub++;
                    }
                }

                // Interface-Load coupled sub-matrix (for all the frequency points) -> H_exp_interfaceLoad = H_exp(LocalInterfaceDOFs, LocalLoadDOFs)
                H_exp_interfaceLoad_sub[_sys*batchSize+f].resize(LocalInterfaceDOFs[k].size() * LocalLoadDOFs[k].size());
                _idx_sub = 0;
                for (size_t j = 0; j < LocalLoadDOFs[k].size(); ++j){                                   // Loop through columns (only the load DOFs)
                    for (size_t i = 0; i < LocalInterfaceDOFs[k].size(); ++i){                          // Loop through rows (only the interface inputs)
                        int _idx = LocalInterfaceDOFs[k][i] + _row*LocalLoadDOFs[k][j];
                        H_exp_interfaceLoad_sub[_sys*batchSize+f][_idx_sub] = H_exp_sub[_sys*batchSize+f][_idx];
                        _idx_sub++;
                    }
                }

                // Internal-Interface sub-matrix (for all the frequency points) -> H_exp_internal = H_exp(LocalInternalDOFs, LocalInterfaceDOFs)
                H_exp_internal_sub[_sys*batchSize+f].resize(LocalInternalDOFs[k].size() * LocalInterfaceDOFs[k].size());
                _idx_sub = 0;
                for (size_t j = 0; j < LocalInterfaceDOFs[k].size(); ++j){                              // Loop through columns (only the interface inputs)
                    for (size_t i = 0; i < LocalInternalDOFs[k].size(); ++i){                           // Loop through rows (only the internal inputs)
                        int _idx = LocalInternalDOFs[k][i] + _row*LocalInterfaceDOFs[k][j];
                        H_exp_internal_sub[_sys*batchSize+f][_idx_sub] = H_exp_sub[_sys*batchSize+f][_idx];
                        _idx_sub++;
                    }
                }

                // Internal-Load sub-matrix (for all the frequency points) -> H_exp_internalLoad = H_exp(LocalInternalDOFs, LocalLoadDOFs)
                H_exp_internalLoad_sub[_sys*batchSize+f].resize(LocalInternalDOFs[k].size() * LocalLoadDOFs[k].size());
                _idx_sub = 0;
                for (size_t j = 0; j < LocalLoadDOFs[k].size(); ++j){                                   // Loop through columns (only the load DOFs)
                    for (size_t i = 0; i < LocalInternalDOFs[k].size(); ++i){                           // Loop through rows (only the internal inputs)
                        int _idx = LocalInternalDOFs[k][i] + _row*LocalLoadDOFs[k][j];
                        H_exp_internalLoad_sub[_sys*batchSize+f][_idx_sub] = H_exp_sub[_sys*batchSize+f][_idx];
                        _idx_sub++;
                    }
                }
            }
        }

        // Extract interface sub-matrix of dRdY -> dRdY_interface = dRdY(:, GlobalInterfaceDOFs)
        dRdY_interface.resize(GlobalInterfaceDOFs.size() * GlobalInterfaceDOFs.size());
        int _row = GlobalInterfaceDOFs.size();
        int _idx_sub = 0;
        for (size_t j = 0; j < _row; ++j){                                                              // Loop through columns (only the interface inputs)
            for (size_t i = 0; i < _row; ++i){                                                          // Loop through rows (entire rows - but the size is the same as the columns)
                int _idx = i + _row*GlobalInterfaceDOFs[j];
                dRdY_interface[_idx_sub] = dRdY[_idx];
                _idx_sub++;
            }
        }
    }
    // Get maximum nnz_B
    auto nnz_B_interface_max_it = thrust::max_element(nnz_B_interface_sub.begin(), nnz_B_interface_sub.end());
    auto nnz_B_load_max_it      = thrust::max_element(nnz_B_load_sub.begin()     , nnz_B_load_sub.end());
    auto nnz_C_interface_max_it = thrust::max_element(nnz_C_interface_sub.begin(), nnz_C_interface_sub.end());
    auto nnz_C_internal_max_it      = thrust::max_element(nnz_C_internal_sub.begin()     , nnz_C_internal_sub.end());

    nnz_B_interface_max = *nnz_B_interface_max_it;
    nnz_B_load_max      = *nnz_B_load_max_it;
    nnz_C_interface_max = *nnz_C_interface_max_it;
    nnz_C_internal_max      = *nnz_C_internal_max_it;
}

void data::combineHostMatrices(
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> K_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> M_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> D_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> B_interface_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> B_load_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> C_interface_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> C_internal_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> H_exp_interface_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> H_exp_interfaceLoad_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> H_exp_internal_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> H_exp_internalLoad_sub,
                               thrust::host_vector<cuDoubleComplex> &K, thrust::host_vector<cuDoubleComplex> &M, thrust::host_vector<cuDoubleComplex> &D,
                               thrust::host_vector<cuDoubleComplex> &B_interface, thrust::host_vector<cuDoubleComplex> &B_load,
                               thrust::host_vector<cuDoubleComplex> &C_interface, thrust::host_vector<cuDoubleComplex> &C_internal,
                               thrust::host_vector<cuDoubleComplex> &H_exp_interface, thrust::host_vector<cuDoubleComplex> &H_exp_interfaceLoad,
                               thrust::host_vector<cuDoubleComplex> &H_exp_internal , thrust::host_vector<cuDoubleComplex> &H_exp_internalLoad,
                               int nnz_A, thrust::host_vector<int> nnz_A_sub,
                               int nnz_B_interface, thrust::host_vector<int> nnz_B_interface_sub,
                               int nnz_B_load, thrust::host_vector<int> nnz_B_load_sub,
                               int nnz_C_interface, thrust::host_vector<int> nnz_C_interface_sub,
                               int nnz_C_internal, thrust::host_vector<int> nnz_C_internal_sub,
                               int num_simModel, int num_expModel, int batchSize
                              )
{
    /*---------------
    Simulation Models
    ---------------*/
    // Resize device vectors
    K.resize(nnz_A);
    M.resize(nnz_A);
    D.resize(nnz_A);
    B_interface.resize(nnz_B_interface);
    B_load.resize(nnz_B_load);
    C_interface.resize(nnz_C_interface);
    C_internal.resize(nnz_C_internal);

    // Combine matrices into a single array
    auto K_sub_ptr           = &K_sub[0];
    auto M_sub_ptr           = &M_sub[0];
    auto D_sub_ptr           = &D_sub[0];
    auto B_interface_sub_ptr = &B_interface_sub[0];
    auto B_load_sub_ptr      = &B_load_sub[0];
    auto C_interface_sub_ptr = &C_interface_sub[0];
    auto C_internal_sub_ptr      = &C_internal_sub[0];

    size_t array_shift_A           = 0;
    size_t array_shift_B_interface = 0;
    size_t array_shift_B_load      = 0;
    size_t array_shift_C_interface = 0;
    size_t array_shift_C_internal      = 0;

    for (size_t i = 0; i < num_simModel; ++i){
        K_sub_ptr           = &K_sub[i];
        M_sub_ptr           = &M_sub[i];
        D_sub_ptr           = &D_sub[i];
        B_interface_sub_ptr = &B_interface_sub[i];
        B_load_sub_ptr      = &B_load_sub[i];
        C_interface_sub_ptr = &C_interface_sub[i];
        C_internal_sub_ptr      = &C_internal_sub[i];

        thrust::copy(K_sub_ptr->begin()          , K_sub_ptr->end()          , K.begin()           + array_shift_A);
        thrust::copy(M_sub_ptr->begin()          , M_sub_ptr->end()          , M.begin()           + array_shift_A);
        thrust::copy(D_sub_ptr->begin()          , D_sub_ptr->end()          , D.begin()           + array_shift_A);
        thrust::copy(B_interface_sub_ptr->begin(), B_interface_sub_ptr->end(), B_interface.begin() + array_shift_B_interface);
        thrust::copy(B_load_sub_ptr->begin()     , B_load_sub_ptr->end()     , B_load.begin()      + array_shift_B_load);
        thrust::copy(C_interface_sub_ptr->begin(), C_interface_sub_ptr->end(), C_interface.begin() + array_shift_C_interface);
        thrust::copy(C_internal_sub_ptr->begin() , C_internal_sub_ptr->end() , C_internal.begin()  + array_shift_C_internal);

        array_shift_A           += nnz_A_sub[i];
        array_shift_B_interface += nnz_B_interface_sub[i];
        array_shift_B_load      += nnz_B_load_sub[i];
        array_shift_C_interface += nnz_C_interface_sub[i];
        array_shift_C_internal  += nnz_C_internal_sub[i];
    }

    /*-----------------
    Experimental Models
    -----------------*/
    // Temporary variables
    int _nnz_H_exp_interface, _nnz_H_exp_interfaceLoad, _nnz_H_exp_internal, _nnz_H_exp_internalLoad;

    // Initialisation
    _nnz_H_exp_interface = 0; _nnz_H_exp_interfaceLoad = 0; _nnz_H_exp_internal = 0; _nnz_H_exp_internalLoad = 0;

    // Get nnz
    for (size_t i = 0; i < num_expModel; ++i){
        for (size_t f = 0; f < batchSize; ++f){
            _nnz_H_exp_interface     += H_exp_interface_sub[i*batchSize+f].size();
            _nnz_H_exp_interfaceLoad += H_exp_interfaceLoad_sub[i*batchSize+f].size();
            _nnz_H_exp_internal      += H_exp_internal_sub[i*batchSize+f].size();
            _nnz_H_exp_internalLoad  += H_exp_internalLoad_sub[i*batchSize+f].size();
        }
    }

    // Resize
    H_exp_interface.resize(_nnz_H_exp_interface);
    H_exp_interfaceLoad.resize(_nnz_H_exp_interfaceLoad);
    H_exp_internal.resize(_nnz_H_exp_internal);
    H_exp_internalLoad.resize(_nnz_H_exp_internalLoad);

    // Combine matrices into a single array
    auto H_exp_interface_sub_ptr     = &H_exp_interface_sub[0];
    auto H_exp_interfaceLoad_sub_ptr = &H_exp_interfaceLoad_sub[0];
    auto H_exp_internal_sub_ptr      = &H_exp_internal_sub[0];
    auto H_exp_internalLoad_sub_ptr  = &H_exp_internalLoad_sub[0];

    size_t array_shift_H_exp_interface     = 0;
    size_t array_shift_H_exp_interfaceLoad = 0;
    size_t array_shift_H_exp_internal      = 0;
    size_t array_shift_H_exp_internalLoad  = 0;

    for (size_t i = 0; i < num_expModel; ++i){
        for (size_t f = 0; f < batchSize; ++f){
            H_exp_interface_sub_ptr     = &H_exp_interface_sub[i*batchSize+f];
            H_exp_interfaceLoad_sub_ptr = &H_exp_interfaceLoad_sub[i*batchSize+f];
            H_exp_internal_sub_ptr      = &H_exp_internal_sub[i*batchSize+f];
            H_exp_internalLoad_sub_ptr  = &H_exp_internalLoad_sub[i*batchSize+f];

            thrust::copy(H_exp_interface_sub_ptr->begin()    , H_exp_interface_sub_ptr->end()    , H_exp_interface.begin()     + array_shift_H_exp_interface);
            thrust::copy(H_exp_interfaceLoad_sub_ptr->begin(), H_exp_interfaceLoad_sub_ptr->end(), H_exp_interfaceLoad.begin() + array_shift_H_exp_interfaceLoad);
            thrust::copy(H_exp_internal_sub_ptr->begin()     , H_exp_internal_sub_ptr->end()     , H_exp_internal.begin()      + array_shift_H_exp_internal);
            thrust::copy(H_exp_internalLoad_sub_ptr->begin() , H_exp_internalLoad_sub_ptr->end() , H_exp_internalLoad.begin()  + array_shift_H_exp_internalLoad);

            array_shift_H_exp_interface     += H_exp_interface_sub[i*batchSize+f].size();
            array_shift_H_exp_interfaceLoad += H_exp_interfaceLoad_sub[i*batchSize+f].size();
            array_shift_H_exp_internal      += H_exp_internal_sub[i*batchSize+f].size();
            array_shift_H_exp_internalLoad  += H_exp_internalLoad_sub[i*batchSize+f].size();
        }
    }
}

void data::getInfoDeviceDataStructure(
                                      thrust::device_vector<cuDoubleComplex*> &d_ptr_K,
                                      thrust::device_vector<cuDoubleComplex*> &d_ptr_M,
                                      thrust::device_vector<cuDoubleComplex*> &d_ptr_D,
                                      thrust::device_vector<cuDoubleComplex*> &d_ptr_B_interface,
                                      thrust::device_vector<cuDoubleComplex*> &d_ptr_B_load,
                                      thrust::device_vector<cuDoubleComplex*> &d_ptr_C_interface,
                                      thrust::device_vector<cuDoubleComplex*> &d_ptr_C_internal,
                                      cuDoubleComplex *d_ptr_K_base,
                                      cuDoubleComplex *d_ptr_M_base,
                                      cuDoubleComplex *d_ptr_D_base,
                                      cuDoubleComplex *d_ptr_B_interface_base,
                                      cuDoubleComplex *d_ptr_B_load_base,
                                      cuDoubleComplex *d_ptr_C_interface_base,
                                      cuDoubleComplex *d_ptr_C_internal_base,
                                      thrust::host_vector<int> nnz_A_sub,
                                      thrust::host_vector<int> nnz_B_interface_sub,
                                      thrust::host_vector<int> nnz_B_load_sub,
                                      thrust::host_vector<int> nnz_C_interface_sub,
                                      thrust::host_vector<int> nnz_C_internal_sub,
                                      int &nnz_H_sim_interface, int &nnz_H_sim_interfaceLoad,
                                      int &nnz_H_sim_internal , int &nnz_H_sim_internalLoad,
                                      std::vector<std::vector<int>> LocalInterfaceDOFs,
                                      std::vector<std::vector<int>> LocalLoadDOFs,
                                      std::vector<std::vector<int>> LocalInternalDOFs,
                                      std::vector<int> subSystemID_sim,
                                      int num_simModel
                                     )
{
    // Get pointers to each sub-components in combined matrices on device
    int _mat_shift_A           = 0;
    int _mat_shift_B_interface = 0;
    int _mat_shift_B_load      = 0;
    int _mat_shift_C_interface = 0;
    int _mat_shift_C_internal      = 0;

    nnz_H_sim_interface = 0; nnz_H_sim_interfaceLoad = 0;
    nnz_H_sim_internal  = 0; nnz_H_sim_internalLoad  = 0;

    int _idx;
    for (size_t i = 0; i < num_simModel; ++i){
        d_ptr_K[i]              = d_ptr_K_base           + _mat_shift_A;
        d_ptr_M[i]              = d_ptr_M_base           + _mat_shift_A;
        d_ptr_D[i]              = d_ptr_D_base           + _mat_shift_A;
        d_ptr_B_interface[i]    = d_ptr_B_interface_base + _mat_shift_B_interface;
        d_ptr_B_load[i]         = d_ptr_B_load_base      + _mat_shift_B_load;
        d_ptr_C_interface[i]    = d_ptr_C_interface_base + _mat_shift_C_interface;
        d_ptr_C_internal[i]     = d_ptr_C_internal_base      + _mat_shift_C_internal;
        _mat_shift_A           += nnz_A_sub[i];
        _mat_shift_B_interface += nnz_B_interface_sub[i];
        _mat_shift_B_load      += nnz_B_load_sub[i];
        _mat_shift_C_interface += nnz_C_interface_sub[i];
        _mat_shift_C_internal  += nnz_C_internal_sub[i];
        // Collect & accumulate nnz
        _idx = subSystemID_sim[i];
        nnz_H_sim_interface     += LocalInterfaceDOFs[_idx].size() * LocalInterfaceDOFs[_idx].size();
        nnz_H_sim_interfaceLoad += LocalInterfaceDOFs[_idx].size() * LocalLoadDOFs[_idx].size();
        nnz_H_sim_internal      += LocalInternalDOFs[_idx].size()  * LocalInterfaceDOFs[_idx].size();
        nnz_H_sim_internalLoad  += LocalInternalDOFs[_idx].size()  * LocalLoadDOFs[_idx].size();
    }
}
