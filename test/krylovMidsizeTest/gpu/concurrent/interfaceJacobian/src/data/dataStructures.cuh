#pragma once

// Libraries
#include <cuComplex.h>

namespace staccato{
    namespace data{
        void getInfoHostDataStructure(
                                      thrust::host_vector<int> &shift_local_A, thrust::host_vector<int> &shift_local_rhs, thrust::host_vector<int> &shift_local_B,
                                      thrust::host_vector<int> &shift_local_H, thrust::host_vector<int> &row_sub, thrust::host_vector<int> &nnz_sub, thrust::host_vector<int> &nnz_sub_B,
                                      thrust::host_vector<int> &nnz_sub_H, thrust::host_vector<int> &num_input_sub, int &nnz, int &nnz_B, int &nnz_H, int &row, int &nnz_max, int &nnz_max_B,
                                      int mat_repetition, int row_baseline[], int num_input_baseline[]
                                     );

        void getInfoDeviceDataStructure(
                                        thrust::device_vector<cuDoubleComplex*> &d_ptr_K,
                                        thrust::device_vector<cuDoubleComplex*> &d_ptr_M,
                                        thrust::device_vector<cuDoubleComplex*> &d_ptr_D,
                                        thrust::device_vector<cuDoubleComplex*> &d_ptr_B,
                                        thrust::device_vector<cuDoubleComplex*> &d_ptr_C,
                                        cuDoubleComplex *d_ptr_K_base,
                                        cuDoubleComplex *d_ptr_M_base,
                                        cuDoubleComplex *d_ptr_D_base,
                                        cuDoubleComplex *h_ptr_B_base,
                                        cuDoubleComplex *h_ptr_C_base,
                                        thrust::host_vector<int> nnz_sub, thrust::host_vector<int> nnz_sub_B,
                                        int subComponents
                                       );
        void combineHostMatrices(
                                 thrust::host_vector<thrust::host_vector<cuDoubleComplex>> K_sub,
                                 thrust::host_vector<thrust::host_vector<cuDoubleComplex>> M_sub,
                                 thrust::host_vector<thrust::host_vector<cuDoubleComplex>> D_sub,
                                 thrust::host_vector<thrust::host_vector<cuDoubleComplex>> B_sub,
                                 thrust::host_vector<thrust::host_vector<cuDoubleComplex>> C_sub,
                                 thrust::host_vector<cuDoubleComplex> &K, thrust::host_vector<cuDoubleComplex> &M, thrust::host_vector<cuDoubleComplex> &D,
                                 thrust::host_vector<cuDoubleComplex> &B, thrust::host_vector<cuDoubleComplex> &C,
                                 int nnz, int nnz_B, int mat_repetition, thrust::host_vector<int> nnz_sub, thrust::host_vector<int> nnz_sub_B
                                );
        void constructHostDataStructure(
                                        std::string filename_K[], std::string filename_M[], std::string filename_D[], std::string filename_B[], std::string filename_C[],
                                        std::string filepath[],   std::string input_filepath,
                                        std::string baseName_K,   std::string baseName_M,   std::string baseName_D,   std::string baseName_B,   std::string baseName_C, std::string base_format,
                                        int row_baseline[], int num_input_baseline[],
                                        thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &K_sub,
                                        thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &M_sub,
                                        thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &D_sub,
                                        thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &B_sub,
                                        thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &C_sub
                                       );
    } // namespace::data
} // namespace::staccato
