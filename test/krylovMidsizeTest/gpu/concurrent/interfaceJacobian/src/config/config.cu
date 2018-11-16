//Libraries
#include <iostream>
#include <iomanip>
#include <cuComplex.h>

//Header Files
#include "config.cuh"

// Namespace
using namespace staccato;

void config::configureTest(int argc, char *argv[], double &freq_max, int &mat_repetition, int &subComponents, int &num_streams, int &num_threads, int &batchSize){
    // Usage
    if (argc < 5){
        std::cerr << ">> Usage: " << argv[0] << " -f <maximum frequency> -m <matrix repetition> -stream <number of CUDA streams> -batch <batch size>" << std::endl;
        std::cerr << ">> NOTE: There are 12 matrices and matrix repetition increases the total number of sub-components (e.g. matrix repetition of 5 will use 60 sub-components)" << std::endl;
        std::cerr << "         Frequency starts from 1 to maximum frequency" << std::endl;
        std::cerr << "         Default number of CUDA streams is 1" << std::endl;
        std::cerr << "         Default number of batch size is freq max (currently only supports batchSize = freq_max)" << std::endl;
        std::exit(1);
    }
    // Set parameters
    freq_max = atof(argv[2]);
    mat_repetition = atoi(argv[4]);
    subComponents = mat_repetition*12;
    num_streams = 1;
    if (argc > 6) num_streams = atoi(argv[6]);
    num_threads = num_streams;
    batchSize = freq_max;
    if (argc > 8) batchSize = atoi(argv[8]);
    // Output messages
    std::cout << ">> Maximum Frequency: " << freq_max << std::endl;
    std::cout << ">> Total number of sub-components: " << subComponents << std::endl;
    std::cout << ">> Number of CUDA streams: " << num_streams << std::endl;
    std::cout << ">> Number of batched matrices: " << batchSize << "\n" << std::endl;
}

void config::check_memory(int mat_repetition, double freq_max, int num_threads){
    /*-----------------
    MEMORY REQUIREMENTS
    -----------------*/
    /*
    1. K, M, D = nnz * 3 * mat_repetition
    2. B, C = nnz_B * 2 * mat_repetition
    3. rhs (sol) = row * freq_max
    4. H = freq_max * nnz_H
    5. A = nt * freq_max * nnz_max
    6. B_batch, C_batch = nt * freq_max * nnz_max_B * 2
    7. d_ptr_K, d_ptr_M, d_ptr_D, d_ptr_B, d_ptr_C = subComponents * 5
    8. d_ptr_A_batch, d_ptr_rhs, d_ptr_B_batch, d_ptr_C_batch, d_ptr_H = freq_max * 5
    */
    unsigned int memory_nnz, memory_nnz_B, memory_row, memory_nnz_H, memory_nnz_max, memory_nnz_max_B, memory_ptr;
    memory_nnz       = sizeof(cuDoubleComplex) * 611424;       // 1
    memory_nnz_B     = sizeof(cuDoubleComplex) * 120060;       // 2
    memory_row       = sizeof(cuDoubleComplex) * 2658;         // 3
    memory_nnz_H     = sizeof(cuDoubleComplex) * 23400;        // 4
    memory_nnz_max   = sizeof(cuDoubleComplex) * 97344;        // 5
    memory_nnz_max_B = sizeof(cuDoubleComplex) * 22464;        // 6
    memory_ptr       = sizeof(cuDoubleComplex*);               // 7
    double memory_required = (
                              memory_nnz*3*mat_repetition +
                              memory_nnz_B*2*mat_repetition +
                              memory_row*freq_max +
                              memory_nnz_H*freq_max +
                              num_threads*freq_max*memory_nnz_max +
                              num_threads*freq_max*memory_nnz_max_B*2 +
                              memory_ptr * 12 * mat_repetition * 5 +
                              memory_ptr * freq_max * 5
                             )*1E-9;

    if (memory_required > 32){
        std::cerr << ">> NOT ENOUGH MEMORY ON GPU" << std::endl;
        std::cerr << ">>>> Memory Required = " << std::setprecision(3) << memory_required << "GB" << std::endl;
        std::cerr << ">>>> Hardware Limit = 32GB" << std::endl;
        std::exit(1);
    }
    else std::cout << ">> Memory Required = " << std::setprecision(3) << memory_required << "GB\n" << std::endl;
}
