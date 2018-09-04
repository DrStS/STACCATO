#pragma once

// Libraries
#include <vector>

using namespace std;

namespace dataStructure{
    void generateCSR(vector<int> &csrRowPtr, vector<int> &csrColInd, vector<int> &row_sub, vector<int> &size_sub, int row, int nnz, int num_matrix);
}
