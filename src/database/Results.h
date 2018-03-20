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
* \file Results.h
* This file holds the class of Results.
* \date 3/5/2018
**************************************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <map>

#include <STACCATO_Enum.h>

using namespace STACCATO_Results;

class Results {
public:
	/***********************************************************************************************
	* \brief Constructor
	* \param[in] _resultType STACCATO_Results_type
	* \param[in] _analysisType STACCATO_Analysis_type
	* \author Harikrishnan Sreekumar
	***********/
	Results();
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	~Results();
	/***********************************************************************************************
	* \brief Get Type of LoadCase
	* \param[out] myAnalsisType
	* \author Harikrishnan Sreekumar
	***********/
	STACCATO_Results::STACCATO_ResultsCase_type getResultsLoadCaseType() { return myResultCase; }
	/***********************************************************************************************
	* \brief Set Type of Results
	* \param[in] _type
	* \author Harikrishnan Sreekumar
	***********/
	void setResultsType(STACCATO_Results_type _type);
	/***********************************************************************************************
	* \brief Set Type of Evaluation
	* \param[in] _evaluationType Nodal or elemental
	* \author Harikrishnan Sreekumar
	***********/
	void setResultsEvaluationType(STACCATO_ResultsEvaluation_type _evaluationType);
	/***********************************************************************************************
	* \brief Build Label with Nomenclature
	* \author Harikrishnan Sreekumar
	***********/
	virtual void buildLabelMap() = 0;
protected:
	/// Analysis Type
	STACCATO_Analysis_type myAnalsisType;
	/// Results Type
	STACCATO_Results_type myResultType;
	/// Result Case Type
	STACCATO_ResultsCase_type myResultCase;
	/// Result Evaluation Type
	STACCATO_ResultsEvaluation_type myResultEvaluationType;
public:
	/// Result Label
	std::string myLabel;
};
