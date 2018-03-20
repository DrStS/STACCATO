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
* \file VectorFieldResults.h
* This file holds the class of VectorFieldResults.
* \date 3/5/2018
**************************************************************************************************/
#pragma once

#include <Results.h>

using namespace STACCATO_Results;

class VectorFieldResults : public Results
{
public:
	/***********************************************************************************************
	* \brief Constructor
	* \param[in] _resultType STACCATO_Results_type
	* \param[in] _analysisType STACCATO_Analysis_type
	* \author Harikrishnan Sreekumar
	***********/
	VectorFieldResults();
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	~VectorFieldResults();
	/***********************************************************************************************
	* \brief Build Label with Nomenclature
	* \author Harikrishnan Sreekumar
	***********/
	void buildLabelMap();
	/***********************************************************************************************
	* \brief Add a result to database
	* \param[in] _type
	* \param[in] _valueVec
	* \author Stefan Sicklinger
	***********/
	void addResultScalarFieldAtNodes(STACCATO_VectorField_components _component, std::vector<double> _valueVec);
	/***********************************************************************************************
	* \brief Get result from database
	* \param[in] _type
	* \param[out] reference to std vector
	* \author Stefan Sicklinger
	***********/
	std::vector<double>&  getResultScalarFieldAtNodes(STACCATO_VectorField_components _component, int index);
	/***********************************************************************************************
	* \brief Get scaled result from database
	* \param[in] _type
	* \param[out] reference to std vector
	* \author Stefan Sicklinger
	***********/
	const std::vector<double>&  getResultScaledScalarFieldAtNodes(STACCATO_VectorField_components _component, int index, double scale);
private:
	// Component Enum to Results map
	std::map<STACCATO_VectorField_components, std::vector<std::vector<double>>> myFieldMap;

	std::vector<std::vector<std::vector<double>>> myMasterVector;


	std::map<STACCATO_VectorField_components, int> componentIndexMap;
public:
	// Label to Component Enum Map
	std::map< std::string, STACCATO_VectorField_components> myResultLabelMap;
	std::vector<double> result;
};
