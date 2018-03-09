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
	Results(STACCATO_Results_type _resultType, STACCATO_Analysis_type _analysisType);
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	~Results();
	/***********************************************************************************************
	* \brief Set Type of Analysis
	* \param[in] _analysisType
	* \author Harikrishnan Sreekumar
	***********/
	void setResultsAnalysisType(STACCATO_Analysis_type _analysisType);
	/***********************************************************************************************
	* \brief Get Type of Analysis
	* \param[out] myAnalsisType
	* \author Harikrishnan Sreekumar
	***********/
	STACCATO_Results::STACCATO_Analysis_type getResultsAnalysisType() { return myAnalsisType; }
	/***********************************************************************************************
	* \brief Set Type of Result
	* \param[in] _resultType
	* \author Harikrishnan Sreekumar
	***********/
	void setResultsType(STACCATO_Results_type _resultType);
	/***********************************************************************************************
	* \brief Set Type of Result Case
	* \param[in] _resultCase
	* \author Harikrishnan Sreekumar
	***********/
	void setResultsCase(STACCATO_ResultsCase_type _resultCase);
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
	/***********************************************************************************************
	* \brief Add a result time description
	* \param[in] _resultsTimeDescription
	* \author Stefan Sicklinger
	***********/
	void addResultsTimeDescription(std::string _resultsTimeDescription);
	/***********************************************************************************************
	* \brief Get results for all time description
	* \author Harikrishnan Sreekumar
	***********/
	std::map<int, std::string>& getResultsTimeDescription();
	/***********************************************************************************************
	* \brief Add a result case description
	* \param[in] _resultsTimeDescription
	* \author Stefan Sicklinger
	***********/
	void addResultsCaseDescription(std::string _resultsCaseDescription);
	/***********************************************************************************************
	* \brief Get results for all case description
	* \author Harikrishnan Sreekumar
	***********/
	std::map<int, std::string>& getResultsCaseDescription();
protected:
	/// Analysis Type
	STACCATO_Analysis_type myAnalsisType;
	/// Results Type
	STACCATO_Results_type myResultType;
	/// Result Case Type
	STACCATO_ResultsCase_type myResultCase;
	/// Result Evaluation Type
	STACCATO_ResultsEvaluation_type myResultEvaluationType;

	/// result vector description in time domain 
	std::map<int, std::string> resultsTimeDescription;
	/// result vector case description
	std::map<int, std::string> resultsCaseDescription;
public:
	/// Result Label
	std::string myLabel;
	/// Time Unit
	std::string myTimeUnit;
	/// Case Unit
	std::string myCaseUnit;

	/// Result Case Label to Enum Map
	std::map<std::string, STACCATO_ResultsCase_type> myResultCaseLabelMap;
};
