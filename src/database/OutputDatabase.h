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

class VectorFieldResults;
class ScalarFieldResults;
using namespace STACCATO_Results;

class OutputDatabase {
public:
	struct LoadCase {
		std::string name;										// XML Prefix Name
		STACCATO_ResultsCase_type type;								// Type of LoadCase - To indicate for Subcase
		std::string unit;											// SubCase Unit
		int startIndex;												// StartIndex Within the AnalysisVectorIndex
	};

	struct TimeStep {
		std::string timeDescription;								// Vector of Strings: With STATIC - "STATIC" or With DYNAMIC - Vector of Frequency Steps
		std::string unit;											// TimeStep Unit
		std::vector<LoadCase> caseList;								// Container of all LoadCases
		int startIndex;												// StartIndex Within the AnalysisVectorIndex
	};

	struct Analysis {
		std::string name;											// Name of Analysis
		STACCATO_Analysis_type type;								// Staccato Type of Anaylsis
		std::vector<TimeStep> timeSteps;							// Container of all TimeSteps
		int startIndex;												// StartIndex Within the VectorFieldResults container
	};
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
	void addNewAnalysisVectorField( Analysis _analysis, VectorFieldResults* _vectorField);
	/***********************************************************************************************
	* \brief Get vector field result
	* \param[out] vectorField
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<VectorFieldResults>& getVectorFieldFromDatabase() { return myVectorFieldResults; }
	/***********************************************************************************************
	* \brief Analyis Description for each VectorField
	* \param[in] _resultsTimeDescription
	* \param[out] description syntax
	* \author Harikrishnan Sreekumar
	***********/
	std::string getSyntaxAnalysisDescription(std::string _resultsAnalyisDescription, STACCATO_Analysis_type _type);
	/***********************************************************************************************
	* \brief Find Analysis with it's name
	* \param[in] _analysisName
	* \param[out] index
	* \author Harikrishnan Sreekumar
	***********/
	int findAnalysis(std::string _analysisName);
	/***********************************************************************************************
	* \brief Find TimeStep with it's name and Analysis Name
	* \param[in] _analysisIndex
	* \param[in] _timeStepName
	* \param[out] index
	* \author Harikrishnan Sreekumar
	***********/
	int findTimeStep(int _analysisIndex, std::string _timeStepName);
	/***********************************************************************************************
	* \brief Find LoadCase with it's name prefix, Analysis Name and LoadCasePrefix Name
	* \param[in] _analysisIndex
	* \param[in] _timeStepIndex
	* \param[in] _loadCasePrefixName
	* \param[out] index
	* \author Harikrishnan Sreekumar
	***********/
	int findLoadCase(int _analysisIndex, int _timeStepIndex, std::string _loadCasePrefixName);
	/***********************************************************************************************
	* \brief Get Start Index for Analysis
	* \param[out] index
	* \author Harikrishnan Sreekumar
	***********/
	int getVectorFieldIndex(int _analysisIndex);
	/***********************************************************************************************
	* \brief Get Start Index for TimeStep
	* \param[out] index
	* \author Harikrishnan Sreekumar
	***********/
	int getVectorFieldIndex(int _analysisIndex, int _timeStepIndex);
	/***********************************************************************************************
	* \brief Get Start Index for LoadCase
	* \param[out] index
	* \author Harikrishnan Sreekumar
	***********/
	int getVectorFieldIndex(int _analysisIndex, int _timeStepIndex, int _loadCaseIndex);
	/***********************************************************************************************
	* \brief Get Number of Analysis
	* \param[out] size
	* \author Harikrishnan Sreekumar
	***********/
	int getNumberOfAnalyses();
	/***********************************************************************************************
	* \brief Get Number of TimeSteps
	* \param[out] size
	* \author Harikrishnan Sreekumar
	***********/
	int getNumberOfTimeSteps(int _analysisIndex);
	/***********************************************************************************************
	* \brief Get Number of LoadCases
	* \param[out] size
	* \author Harikrishnan Sreekumar
	***********/
	int getNumberOfLoadCases(int _analysisIndex, int _timeStepIndex);
	/***********************************************************************************************
	* \brief Get Analysis Description
	* \param[out] description vector
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<std::string> getAnalysisDescription();
	/***********************************************************************************************
	* \brief Get Time Description
	* \param[out] description vector
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<std::string> getTimeDescription(int _analysisIndex);
	/***********************************************************************************************
	* \brief Get Unit for Time Step
	* \param[int] analysis type
	* \param[out] syntax
	* \author Harikrishnan Sreekumar
	***********/
	std::string getSyntaxForTime(STACCATO_Analysis_type);
	/***********************************************************************************************
	* \brief Get Unit for SubLoad Step
	* \param[int] case type
	* \param[out] syntax
	* \author Harikrishnan Sreekumar
	***********/
	std::string getSyntaxForSubLoadCase(STACCATO_ResultsCase_type);
	/***********************************************************************************************
	* \brief Get Unit for Time Step
	* \param[out] unit
	* \author Harikrishnan Sreekumar
	***********/
	std::string getUnit(int _analysisIndex, int _timeStepIndex);
	/***********************************************************************************************
	* \brief Get Unit for Load Step
	* \param[out] unit
	* \author Harikrishnan Sreekumar
	***********/
	std::string getUnit(int _analysisIndex, int _timeStepIndex, int _loadCaseIndex);
	/***********************************************************************************************
	* \brief Get Size of current vector field
	* \param[out] size = next index
	* \author Harikrishnan Sreekumar
	***********/
	int getStartIndexForNewAnalysis();
	private:
		/// Avoid copy of a OutputDatabase object copy constructor 
		OutputDatabase(const OutputDatabase&);
		/// Avoid copy of a OutputDatabase object assignment operator
		OutputDatabase& operator=(const OutputDatabase&);
private:
	std::vector<ScalarFieldResults> myScalarFieldResults;
	std::vector<VectorFieldResults> myVectorFieldResults;

public:
	std::vector<Analysis> myAnalyses;
};