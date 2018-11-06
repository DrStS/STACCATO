// Libraries
#include <iostream>
#include <string>
#include <omp.h>
#include <mkl.h>

// Header Files
#include "config.hpp"

void config::configureTest(int argc, char *argv[], double &freq_max, int &mat_repetition, int &num_matrix, int &tid, int &nt_mkl, int &nt){
    /*--------------------
    Command line arguments
    --------------------*/
    // Usage
    if (argc < 5){
        std::cerr << ">> Usage: " << argv[0] << " -f <maximum frequency> -m <matrix repetition> -mkl <mkl threads> -openmp <OpenMP threads>" << std::endl;
        std::cerr << ">> NOTE: There are 12 matrices and matrix repetition increases the total number of matrices (e.g. matrix repetition of 5 will use 60 matrices)" << std::endl;
        std::cerr << "         Frequency starts from 1 to maximum frequency" << std::endl;
        std::cerr << "         Default number of MKL threads is 1" << std::endl;
        std::cerr << "         Default number of OpenMP threads is omp_get_max_threads()" << std::endl;
        std::exit(1);
    }
    // Frequency & Matrix repetition
    freq_max = atof(argv[2]);
    mat_repetition = atoi(argv[4]);
    num_matrix = mat_repetition*12;
    std::cout << ">> Maximum Frequency: " << freq_max << std::endl;
    std::cout << ">> Total number of matrices: " << num_matrix << "\n" << std::endl;

    // OpenMP
    tid;
    nt_mkl = 1;
    nt = omp_get_max_threads();
    if (argc > 6) nt_mkl = atoi(argv[6]);
    if (argc > 8){
        nt_mkl = atoi(argv[6]);
        nt = atoi(argv[8]);
    }
    omp_set_num_threads(nt);
    mkl_set_num_threads(nt_mkl);
    std::cout << "\n>> Software will use the following number of threads: " << nt << " OpenMP threads, " << nt_mkl << " MKL threads\n" << std::endl;
    if ((int)freq_max % nt != 0) {
        std::cerr << ">> ERROR: Invalid number of OpenMP threads" << std::endl;
        std::cerr << ">>        The ratio of OpenMP threads to maximum frequency must be an integer" << std::endl;
        std::exit(2);
    }
    omp_set_nested(true);
    mkl_set_dynamic(false);
    //mkl_set_threading_layer(MKL_THREADING_INTEL);

    /*---------------
    Print MKL Version
    ---------------*/
    int len = 198;
    char buf[198];
    mkl_get_version_string(buf, len);
    printf("%s\n", buf);
    printf("\n");
}
