// Libraries
#include <thrust/host_vector.h>

// Header Files
#include "math.cuh"

using namespace thrust;

void generateCSR(host_vector<int> &csrRowPtr, host_vector<int> &csrColInd, host_vector<int> &row_sub, host_vector<int> &size_sub, int row, int nnz, int num_matrix){
	size_t initVal = 0;
	size_t arrayShift, matShift, rowShift, idx;
	// First matrix (loop unrolled)
	thrust::sequence(thrust::host, csrColInd.begin(), csrColInd.begin()+row_sub[0], initVal);
	for (size_t i = 1; i < row_sub[0]; i++){
		rowShift = i*row_sub[0];
		// Column Index
		thrust::sequence(thrust::host, csrColInd.begin()+rowShift, csrColInd.begin()+rowShift+row_sub[0], initVal);
		// Row Pointer
		csrRowPtr[i] = csrRowPtr[i-1] + row_sub[0];
	}
	arrayShift = row_sub[0];
	initVal += row_sub[0];
	matShift = size_sub[0];

	// Rest of the matrices
	for (size_t mat = 1; mat < num_matrix; mat++){
		idx = arrayShift;
		// Row iteration loop unrolled
		csrRowPtr[idx] = csrRowPtr[idx-1] + row_sub[mat-1];
		thrust::sequence(thrust::host, csrColInd.begin()+matShift, csrColInd.begin()+row_sub[mat]+matShift, initVal);
		for (size_t i = 1; i < row_sub[mat]; i++){
			rowShift = i*row_sub[mat];
			idx = i+arrayShift;
			// Column Index
			thrust::sequence(thrust::host, csrColInd.begin()+rowShift+matShift, csrColInd.begin()+rowShift+row_sub[mat]+matShift, initVal);
			// Row Pointer
			csrRowPtr[idx] = csrRowPtr[idx-1] + row_sub[mat];
		}
		initVal += row_sub[mat];
		arrayShift += row_sub[mat];
		matShift += size_sub[mat];
	}
	csrRowPtr[row] = nnz;
}
