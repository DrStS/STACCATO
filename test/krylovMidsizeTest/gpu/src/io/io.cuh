#ifndef IO_HPP
#define IO_HPP

// Libraries
#include <string>

// THRUST
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// CUCOMPLEX
#include <cuComplex.h>

namespace io{
	void readMtxDense(thrust::host_vector<cuDoubleComplex> &A, std::string _filepath, std::string _filename, bool _isComplex);
	void writeSolVecComplex(thrust::host_vector<cuDoubleComplex> &sol, std::string _filepath, std::string _filename);
}

#endif
