// Libraries
#include <string>
#include <thrust/host_vector.h>

// Header Files
#include "dataStructures.cuh"
#include "../io/io.cuh"

// Namespace
using namespace staccato;

void data::getInfoHostDataStructure(
                                    thrust::host_vector<int> &shift_local_A, thrust::host_vector<int> &shift_local_rhs,
                                    thrust::host_vector<int> &row_sub, thrust::host_vector<int> &nnz_sub,
                                    int &nnz, int &row, int &nnz_max, int mat_repetition, int row_baseline[]
                                   )
{
    // Get matrix sizes and local shifts
    nnz = 0;
    row = 0;
    size_t idx;
    int mat_shift   = 0;
    int sol_shift   = 0;
    for (size_t j = 0; j < mat_repetition; ++j){
        for (size_t i = 0; i < 12; ++i){
            // Index for combined matrix
            idx = i + 12*j;
            // Sub-component matrix & vector sizes
            row_sub[idx]       = row_baseline[i];
            nnz_sub[idx]       = row_sub[i]*row_sub[i];
            // Accumulate total matrix & vector sizes
            nnz   += nnz_sub[idx];
            row   += row_sub[idx];
            // (Local) shifts for each sub-components from combined matrix
            shift_local_A[idx]   = mat_shift;
            shift_local_rhs[idx] = sol_shift;
            // Update shifts
            mat_shift   += nnz_sub[idx];
            sol_shift   += row_sub[idx];
        }
    }
    // Get maximum matrix size
    auto nnz_max_it   = thrust::max_element(nnz_sub.begin(), nnz_sub.end());
    nnz_max   = *nnz_max_it;
}

void data::getInfoDeviceDataStructure(
                                      thrust::device_vector<cuDoubleComplex*> &d_ptr_K,
                                      thrust::device_vector<cuDoubleComplex*> &d_ptr_M,
                                      thrust::device_vector<cuDoubleComplex*> &d_ptr_D,
                                      cuDoubleComplex *d_ptr_K_base,
                                      cuDoubleComplex *d_ptr_M_base,
                                      cuDoubleComplex *d_ptr_D_base,
                                      thrust::host_vector<int> nnz_sub,
                                      int subComponents
                                     )
{
    // Get pointers to each sub-components in combined matrices on device
    int mat_shift   = 0;
    for (size_t i = 0; i < subComponents; ++i){
        d_ptr_K[i] = d_ptr_K_base + mat_shift;
        d_ptr_M[i] = d_ptr_M_base + mat_shift;
        d_ptr_D[i] = d_ptr_D_base + mat_shift;
        mat_shift   += nnz_sub[i];
    }
}

void data::combineHostMatrices(
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> K_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> M_sub,
                               thrust::host_vector<thrust::host_vector<cuDoubleComplex>> D_sub,
                               thrust::host_vector<cuDoubleComplex> &K, thrust::host_vector<cuDoubleComplex> &M, thrust::host_vector<cuDoubleComplex> &D,
                               int nnz, int mat_repetition, thrust::host_vector<int> nnz_sub
                              )
{
    K.resize(nnz);
    M.resize(nnz);
    D.resize(nnz);
    // Combine matrices into a single array
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
            array_shift   += nnz_sub[i];
        }
    }
}

void data::constructHostDataStructure(
                                      std::string filename_K[], std::string filename_M[], std::string filename_D[],
                                      std::string filepath[],
                                      std::string baseName_K,   std::string baseName_M,   std::string baseName_D, std::string base_format,
                                      int row_baseline[],
                                      thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &K_sub,
                                      thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &M_sub,
                                      thrust::host_vector<thrust::host_vector<cuDoubleComplex>> &D_sub
                                     )
{
    /*------------------------
    READ AND PROCESS MTX FILES
    ------------------------*/
    for (size_t i = 0; i < 7; ++i){
        filename_K[i] = baseName_K + std::to_string(row_baseline[i]) + base_format;
        filename_M[i] = baseName_M + std::to_string(row_baseline[i]) + base_format;
        filename_D[i] = baseName_D + std::to_string(row_baseline[i]) + base_format;
        io::readMtxDense(K_sub[i], filepath[0], filename_K[i], true);
        io::readMtxDense(M_sub[i], filepath[0], filename_M[i], true);
        io::readMtxDense(D_sub[i], filepath[0], filename_D[i], true);
        K_sub[i].pop_back();
        M_sub[i].pop_back();
        D_sub[i].pop_back();
    }
    for (size_t i = 7; i < 12; ++i){
        filename_K[i] = baseName_K + std::to_string(row_baseline[i]) + base_format;
        filename_M[i] = baseName_M + std::to_string(row_baseline[i]) + base_format;
        filename_D[i] = baseName_D + std::to_string(row_baseline[i]) + base_format;
        io::readMtxDense(K_sub[i], filepath[1], filename_K[i], true);
        io::readMtxDense(M_sub[i], filepath[1], filename_M[i], true);
        io::readMtxDense(D_sub[i], filepath[1], filename_D[i], true);
        K_sub[i].pop_back();
        M_sub[i].pop_back();
        D_sub[i].pop_back();
    }
}
