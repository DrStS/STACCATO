/*  Copyright &copy; 2016, Dr. Stefan Sicklinger, Munich \n
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
/***********************************************************************************************//**
 * \file AuxiliaryParameters.h
 * This file holds the class of AuxiliaryParameters.
 * \date 4/2/2016
 **************************************************************************************************/
#pragma once



#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
# define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
# define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
# define CURRENT_FUNCTION __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
# define CURRENT_FUNCTION __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
# define CURRENT_FUNCTION __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
# define CURRENT_FUNCTION __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
# define CURRENT_FUNCTION __func__
#else
# define CURRENT_FUNCTION "(unknown)"
#endif

#include <string>
#include <mkl.h>

typedef MKL_Complex16 STACCATOComplexDouble;
namespace STACCATO {
/********//**
 * \brief Class AuxiliaryParameters provides a central place for STACCATO wide parameters
 ***********/
class AuxiliaryParameters {
public:
    /// How many threads are used for linear solver part
    static const int solverMKLThreads;

    /// How many threads are used for the element loop
    static const int denseVectorMatrixThreads;

    /// Machine epsilon (the difference between 1 and the least value greater than 1 that is representable).
    static const double machineEpsilon;

    /// Git hash is determined during configure by cmake
    static const std::string gitSHA1;

    /// Git tag is determined during configure by cmake
    static const std::string gitTAG;
};

} /* namespace STACCATO */
