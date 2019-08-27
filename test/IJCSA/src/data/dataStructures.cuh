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
* \file dataStructures.cuh
* Written by Ji-Ho Yang
* This file constructs data structures
* \date 7/12/2019
**************************************************************************************************/

#pragma once

// Definitions
#define SIM 0
#define EXP 1

// Libraries
#include <cuComplex.h>

namespace staccato{
    namespace data{
        void constructHostDataStructure(
                                        std::string filepath_sim,   std::string filepath_exp, std::string filepath_jac, std::string base_format,
                                        std::string baseName_K,   std::string baseName_M,   std::string baseName_D,   std::string baseName_B,   std::string baseName_C, std::string baseName_H,
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
                                       );

        void extractHostMatrices(
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
                                );

        void combineHostMatrices(
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
                                );

        void getInfoDeviceDataStructure(
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
                                       );
    } // namespace::data
} // namespace::staccato
