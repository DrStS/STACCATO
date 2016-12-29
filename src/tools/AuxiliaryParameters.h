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
#ifndef AUXILIARYPARAMETERS_H_
#define AUXILIARYPARAMETERS_H_
#include <string>

namespace STACCATO {
/********//**
 * \brief Class AuxiliaryParameters provides a central place for STACCATO wide parameters
 ***********/
class AuxiliaryParameters {
public:
    /// How many threads are used for MKL thread parallel routines
    static const int mklSetNumThreads;

    /// How many threads are used for mortar mapper thread parallel routines
    static const int mapperSetNumThreads;

    /// Machine epsilon (the difference between 1 and the least value greater than 1 that is representable).
    static const double machineEpsilon;

    /// Git hash is determined during configure by cmake
    static const std::string gitSHA1;

    /// Git tag is determined during configure by cmake
    static const std::string gitTAG;
};

} /* namespace STACCATO */
#endif /* AUXILIARYFUNCTIONS_H_ */
