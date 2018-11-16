#pragma once

#include <string>

namespace config{
    void configureTest(int argc, char *argv[], double &freq_max, int &mat_repetition, int &num_matrix, int &tid, int &nt_mkl, int &nt, 
                       std::string &parallel_mode, std::string &sparse_mode, std::string &arg_parallel, std::string &arg_sparse);
}
