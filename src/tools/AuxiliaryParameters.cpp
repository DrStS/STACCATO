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
#include "AuxiliaryParameters.h"
#include <limits>
#define GIT_SHA1 "5966255fc29f3c6b105d82cdcea6ab467d3cf8a9" //Cmake will set this value
#define GIT_TAG  "GIT-NOTFOUND"  //Cmake will set this value

namespace STACCATO {
const int AuxiliaryParameters::solverMKLThreads= 4;
const int AuxiliaryParameters::denseVectorMatrixThreads= 4;
const double AuxiliaryParameters::machineEpsilon= std::numeric_limits<double>::epsilon();
const std::string AuxiliaryParameters::gitSHA1(GIT_SHA1);
const std::string AuxiliaryParameters::gitTAG(GIT_TAG);
} /* namespace STACCATO */
