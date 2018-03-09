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
* \file OutputDatabase.h
* This file holds the class of OutputDatabase.
* \date 3/6/2018
**************************************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <map>
#include "STACCATO_Enum.h"

#include "VectorFieldResults.h"
#include "ScalarFieldResults.h"

class OutputDatabase {
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Harikrishnan Sreekumar
	***********/
	OutputDatabase();
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	~OutputDatabase();
	/***********************************************************************************************
	* \brief Add a vector field result to the database
	* \param[in] _vectorField
	* \author Harikrishnan Sreekumar
	***********/
	void addVectorFieldToDatabase(VectorFieldResults* _vectorField);
	/***********************************************************************************************
	* \brief Get vector field result
	* \param[out] vectorField
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<VectorFieldResults>& getVectorFieldFromDatabase() { return myVectorFieldResults; }
	/***********************************************************************************************
	* \brief Add Analyis Description for each VectorField
	* \param[in] _resultsTimeDescription
	* \author Harikrishnan Sreekumar
	***********/
	void addVectorFieldAnalysisDescription(std::string _resultsAnalyisDescription, STACCATO_Analysis_type _type);
private:
	std::vector<ScalarFieldResults> myScalarFieldResults;
	std::vector<VectorFieldResults> myVectorFieldResults;

	std::vector<std::string> myVectorFieldAnalysisDectription;
};