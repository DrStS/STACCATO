//Libraries
#include <iostream>

//Header Files
#include "config.cuh"

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
