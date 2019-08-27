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
* \file config.cu
* Written by Ji-Ho Yang
* This file reads command line inputs
* \date 7/12/2019
**************************************************************************************************/

//Libraries
#include <iostream>
#include <iomanip>
#include <cuComplex.h>

//Header Files
#include "config.cuh"

// Namespace
using namespace staccato;

void config::configureTest(int argc, char *argv[], double &freq_min, double &freq_max, int &num_streams, int &num_threads, int &batchSize, int subSystems, bool &postProcess){
    // Usage
    if (argc < 4){
        std::cerr << ">> Usage: " << argv[0] << " -fi <minimum frequency> -fm <maximum frequency> -stream <number of CUDA streams> -postProcess <bool>" << std::endl;
        std::cerr << ">> NOTE: Default number of CUDA streams is 1" << std::endl;
        std::cerr << "         Default number of frequency points is (maximum_frequency - minimum_frequency + 1)" << std::endl;
        std::cerr << "         Default number of batch size is freq max (currently only supports batchSize = num_freq)" << std::endl;
        std::cerr << "         Post-processing is off by default\n" << std::endl;
        std::cerr << ">> Example Usage: ./IJCA -fi 100 -fm 500 -stream 3 -postProcess\n" << std::endl;
        std::exit(1);
    }
    // Set parameters
    freq_min = atof(argv[2]);
    freq_max = atof(argv[4]);
    if (freq_min > freq_max){
        std::cerr << ">> Error: Minimum frequency must not be bigger than maximum frequency" << std::endl;
        std::cerr << "   Aborting..." << std::endl;
        std::exit(1);
    }
    num_streams = 1;
    if (argc > 6) num_streams = atoi(argv[6]);
    num_threads = num_streams;
    batchSize = freq_max-freq_min+1;
    if (argc > 7){
        postProcess = bool(argv[7]);
    }
    // Output messages
    std::cout << ">> Number of sub-systems: " << subSystems << std::endl;
    std::cout << ">> Minimum Frequency: " << freq_min << std::endl;
    std::cout << ">> Maximum Frequency: " << freq_max << std::endl;
    std::cout << ">> Number of Frequency Points: " << batchSize << std::endl;
    std::cout << ">> Number of CUDA streams: " << num_streams << std::endl;
    std::cout << ">> Post-processing: " << postProcess << std::endl;
    std::cout << ">> Number of batched matrices: " << batchSize << "\n" << std::endl;
}

//void config::check_memory(int mat_repetition, double freq_max, int num_threads){
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
/*
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
*/
