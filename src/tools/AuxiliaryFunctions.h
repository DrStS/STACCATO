/*  Copyright &copy; 2018, Stefan Sicklinger, Munich
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
* \file AuxiliaryFunctions.h
* This file holds the class of AuxiliaryFunctions.
* \date 19/4/2018
**************************************************************************************************/
#pragma once

#include <string>
#include <vector>
#include "AuxiliaryParameters.h"

class AuxiliaryFunctions
{
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Harikrishnan Sreekumar
	***********/
	AuxiliaryFunctions();
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	~AuxiliaryFunctions();
	/***********************************************************************************************
	* \brief Write a double vector to the DAT file with format: <value>
	* \param[in] _fileName
	* \param[in] _vector::double
	* \author Harikrishnan Sreekumar
	***********/
	static void writeDoubleVectorDatFormat(std::string _fileName, std::vector<double> &_vector);
	/***********************************************************************************************
	* \brief Write a int vector to the DAT file with format: <value>
	* \param[in] _fileName
	* \param[in] _vector::int
	* \author Harikrishnan Sreekumar
	***********/
	static void writeIntegerVectorDatFormat(std::string _fileName, std::vector<int> &_vector);
	/***********************************************************************************************
	* \brief Write a STACCATOComplexDouble vector to the DAT file with format: <real_value> <imag_value>
	* \param[in] _fileName
	* \param[in] _vector::int
	* \author Harikrishnan Sreekumar
	***********/
	static void writeMKLComplexVectorDatFormat(std::string _fileName, std::vector< STACCATOComplexDouble > &_vector);

};

