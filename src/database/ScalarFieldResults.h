/*  Copyright &copy; 2017, Stefan Sicklinger, Munich
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
* \file ScalarFieldResults.h
* This file holds the class of ScalarFieldResults.
* \date 3/5/2018
**************************************************************************************************/
#pragma once

#include <Results.h>

using namespace STACCATO_Results;

class ScalarFieldResults: public Results
{
public:
	/***********************************************************************************************
	* \brief Constructor
	* \param[in] _resultType STACCATO_Results_type
	* \param[in] _analysisType STACCATO_Analysis_type
	* \author Harikrishnan Sreekumar
	***********/
	ScalarFieldResults(STACCATO_Results_type _resultType, STACCATO_Analysis_type _analysisType);
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	~ScalarFieldResults();
	/***********************************************************************************************
	* \brief Build Label with Nomenclature
	* \author Harikrishnan Sreekumar
	***********/
	void buildLabelMap();
private:
	/// Result Label to Component Enum Map
	std::map<std::string, STACCATO_ScalarField_components> myResultLabelMap;

	/// result Vector node index to result value
	std::vector<std::vector<double>> resultsRe;
	/// result Vector node index to result value
	std::vector<std::vector<double>> resultsIm;
};
