// Libraries
#include <iostream>
#include <string>
#include <omp.h>
#include <mkl.h>

// Header Files
#include "config.hpp"

void config::configureTest(int argc, char *argv[], double &freq_max, int &mat_repetition, int &num_matrix, int &tid, int &nt_mkl, int &nt, 
                           std::string &parallel_mode, std::string &sparse_mode, std::string &arg_parallel, std::string &arg_sparse){
    /*--------------------
    Command line arguments
    --------------------*/
    // Usage
    if (argc < 5){
        std::cerr << ">> Usage: " << argv[0] << " -f <maximum frequency> -m <matrix repetition> -parallel=<yes/no> -sparse=<yes/no> -mkl <mkl threads> -openmp <OpenMP threads>" << std::endl;
        std::cerr << ">> NOTE: There are 12 matrices and matrix repetition increases the total number of matrices (e.g. matrix repetition of 5 will use 60 matrices)" << std::endl;
        std::cerr << "         Frequency starts from 1 to maximum frequency" << std::endl;
        std::cerr << "         '-parallel=yes' parallelises frequency loop and '-parallel=no' executes it with default master thread. Default is sequential" << std::endl;
        std::cerr << "         '-sparse=yes' calls PARDISO for block-diagonal matrix system (currently only supports sequential version) and '-sparse=no' calls LAPACKE for multiple dense matrices. Default is dense" << std::endl;
        std::cerr << "         Default number of MKL threads is mkl_get_max_threads()" << std::endl;
        std::cerr << "         Default number of OpenMP threads is 1" << std::endl;
        std::exit(1);
    }
    // Frequency & Matrix repetition
    freq_max = atof(argv[2]);
    mat_repetition = atoi(argv[4]);
    num_matrix = mat_repetition*12;
    std::cout << ">> Maximum Frequency: " << freq_max << std::endl;
    std::cout << ">> Total number of matrices: " << num_matrix << "\n" << std::endl;
    // Parallel & sparse mode
    parallel_mode = "Sequential";
    sparse_mode = "Multiple Dense Matrices";
    if (argc > 5) {
        arg_parallel=argv[5];
        if (arg_parallel == "-parallel=yes") parallel_mode = "Parallel";
        else if (arg_parallel == "-parallel=no") parallel_mode = "Sequential";
    }
    if (argc > 6) {
        arg_parallel=argv[5];
        arg_sparse=argv[6];
        if (arg_parallel == "-parallel=yes") parallel_mode = "Parallel";
        else if (arg_parallel == "-parallel=no") parallel_mode = "Sequential";
        if (arg_sparse == "-sparse=yes") sparse_mode = "Sparse Block Diagonal System";
        else if (arg_sparse == "-sparse=no") sparse_mode = "Multiple Dense Matrices";
    }
    std::cout << ">> Matrix system: " << sparse_mode << std::endl;
    std::cout << ">> Frequency Loop: " << parallel_mode << std::endl;

    // OpenMP
    tid;
    nt_mkl = mkl_get_max_threads();
    nt = 1;
    if (argc > 8) nt_mkl = atoi(argv[8]);
    if (argc > 10){
        nt_mkl = atoi(argv[8]);
        nt = atoi(argv[10]);
    }
    omp_set_num_threads(nt);
    mkl_set_num_threads(nt_mkl);
    std::cout << "\n>> Software will use the following number of threads: " << nt << " OpenMP threads, " << nt_mkl << " MKL threads\n" << std::endl;
    if ((int)freq_max % nt != 0) {
        std::cerr << ">> ERROR: Invalid number of OpenMP threads" << std::endl;
        std::cerr << ">>        The ratio of OpenMP threads to maximum frequency must be an integer" << std::endl;
        std::exit(2);
    }
    if (parallel_mode == "Parallel"){
        omp_set_nested(true);
        mkl_set_dynamic(false);
        //mkl_set_threading_layer(MKL_THREADING_INTEL);
    }
    if (parallel_mode == "Sequential" && nt > 1){
        std::cerr << ">> ERROR: Incompatible number of OpenMP threads: " << nt << " with " << parallel_mode << " mode" << std::endl;
        std::exit(3);
    }

    /*---------------
    Print MKL Version
    ---------------*/
    int len = 198;
    char buf[198];
    mkl_get_version_string(buf, len);
    printf("%s\n", buf);
    printf("\n");
}
