#pragma once

// Libraries
#include <cuComplex.h>

namespace staccato{
    namespace data{
        void getInfoHostDataStructure(
                                      thrust::host_vector<int> &shift_local_A, thrust::host_vector<int> &shift_local_rhs,
                                      thrust::host_vector<int> &row_sub, thrust::host_vector<int> &nnz_sub,
                                      int &nnz, int &row, int &nnz_max,
                                      int mat_repetition, int row_baseline[]
                                     );

        void getInfoDeviceDataStructure(
                                        thrust::device_vector<cuDoubleComplex*> &d_ptr_K,
                                        thrust::device_vector<cuDoubleComplex*> &d_ptr_M,
                                        thrust::device_vector<cuDoubleComplex*> &d_ptr_D,
                                        cuDoubleComplex *d_ptr_K_base,
                                        cuDoubleComplex *d_ptr_M_base,
                                        cuDoubleComplex *d_ptr_D_base,
                                        thrust::host_vector<int> nnz_sub,
                                        int subComponents
                                       );
        void combineHostMatrices(
                                 thrust::host_vector<thrust::host_vector<cuDoubleComplex>> K_sub,
                                 thrust::host_vector<thrust::host_vector<cuDoubleComplex>> M_sub,
                                 thrust::host_vector<thrust::host_vector<cuDoubleComplex>> D_sub,
                                 thrust::host_vector<cuDoubleComplex> &K, thrust::host_vector<cuDoubleComplex> &M, thrust::host_vector<cuDoubleComplex> &D,
                                 int nnz, int mat_repetition, thrust::host_vector<int> nnz_sub
                                );
        void constructHostDataStructure(
                                        std::string filename_K[], std::string filename_M[], std::string filename_D[],
                                        std::string filepath[],
                                        std::string baseName_K,   std::string baseName_M,   std::string baseName_D, std::string base_format,
                                        int row_baseline[],
                                        thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &K_sub,
                                        thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &M_sub,
                                        thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &D_sub
                                       );
    } // namespace::data
} // namespace::staccato
