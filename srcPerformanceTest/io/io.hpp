#pragma once

// Libraries
#include <string>
#include <vector>

// MKL
#include <mkl.h>

namespace io{
	void readMtxDense(std::vector<MKL_Complex16> &A, std::string _filepath, std::string _filename, bool _isComplex);
	void writeSolVecComplex(std::vector<MKL_Complex16> &sol, std::string _filepath, std::string _filename);
}
