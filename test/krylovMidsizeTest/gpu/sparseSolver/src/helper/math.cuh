#pragma once

// Libraries
#include <thrust/host_vector.h>

using namespace thrust;

void generateCSR(host_vector<int> &csrRowPtr, host_vector<int> &csrColInd, host_vector<int> &row_sub, host_vector<int> &size_sub, int row, int nnz, int num_matrix);
