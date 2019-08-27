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
* \file io.cuh
* Written by Ji-Ho Yang
* This file contains functions for I/O
* \date 7/12/2019
**************************************************************************************************/

#pragma once

// Libraries
#include <string>

// THRUST
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// CUCOMPLEX
#include <cuComplex.h>

namespace staccato{
    namespace io{
        void readMtxDense(thrust::host_vector<cuDoubleComplex> &A, std::string _filepath, std::string _filename, int &row, int &col, int &nnz, bool _isComplex);
        void readMtxSparse(thrust::host_vector<cuDoubleComplex> &A, std::string _filepath, std::string _filename, int &row, int &col, int &nnz, bool _isComplex);
        void writeSolVecComplex(thrust::host_vector<cuDoubleComplex> &sol, std::string _filepath, std::string _filename);
        void writeMtxDenseComplex(thrust::host_vector<cuDoubleComplex> &mat, int row, int col, std::string _filepath, std::string _filename);
    } // namespace::staccato
} // namespace::io
