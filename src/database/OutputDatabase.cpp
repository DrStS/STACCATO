/*  Copyright &copy; 2016, Stefan Sicklinger, Munich
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
#include "OutputDatabase.h"
#include "ScalarFieldResults.h"
#include "VectorFieldResults.h"
#include <iostream>

OutputDatabase::OutputDatabase() {
}

OutputDatabase::~OutputDatabase() {

}

void OutputDatabase::addNewAnalysisVectorField(	Analysis _analysis, VectorFieldResults* _vectorField) {

	myVectorFieldResults.push_back(*_vectorField);

	myAnalyses.push_back(_analysis);

	std::cout << ">> OutputDatabase is allocated for an expected Vector Field Result.\n";
}

std::string OutputDatabase::getSyntaxAnalysisDescription(std::string _resultsAnalyisDescription, STACCATO_Analysis_type _type) {
	switch (_type)
	{
	case STACCATO_Analysis_Static:
		_resultsAnalyisDescription += "_Static";
		return _resultsAnalyisDescription;
	case STACCATO_Analysis_DynamicReal:
		_resultsAnalyisDescription += "_DynamicR";
		return _resultsAnalyisDescription;
	case STACCATO_Analysis_Dynamic:
		_resultsAnalyisDescription += "_Dynamic";
		return _resultsAnalyisDescription;
	default:
		std::cerr << "Invalid Analyis Type!\n";
		_resultsAnalyisDescription += "_Ukn";
		return _resultsAnalyisDescription;
	}
}

int OutputDatabase::findAnalysis(std::string _analysisName) {
	for (int i = 0; i < myAnalyses.size(); i++)	{
		if (myAnalyses[i].name == _analysisName)	{
			return i;
		}
	}
	return -1;
}

int OutputDatabase::findTimeStep(int _analysisIndex, std::string _timeStepDescription) {
	for (int j = 0; j < myAnalyses[_analysisIndex].timeSteps.size(); j++) {
		if (myAnalyses[_analysisIndex].timeSteps[j].timeDescription == _timeStepDescription) {
			return j;
		}
	}
	return -1;
}

int OutputDatabase::findLoadCase(int _analysisIndex, int _timeStepIndex, std::string _loadCasePrefixName) {
	for (int k = 0; k < myAnalyses[_analysisIndex].timeSteps[_timeStepIndex].caseList.size(); k++) {
		if (myAnalyses[_analysisIndex].timeSteps[_timeStepIndex].caseList[k].name == _loadCasePrefixName) {
			return k;
		}
	}
	return -1;
}

int OutputDatabase::getVectorFieldIndex(int _analysisIndex) {
	return myAnalyses[_analysisIndex].startIndex;
}

int OutputDatabase::getVectorFieldIndex(int _analysisIndex, int _timeStepIndex) {
	return myAnalyses[_analysisIndex].timeSteps[_timeStepIndex].startIndex;
}

int OutputDatabase::getVectorFieldIndex(int _analysisIndex, int _timeStepIndex, int _loadCaseIndex) {
	return myAnalyses[_analysisIndex].timeSteps[_timeStepIndex].caseList[_loadCaseIndex].startIndex;
}

int OutputDatabase::getNumberOfAnalyses() {
	return myAnalyses.size();
}

int OutputDatabase::getNumberOfTimeSteps(int _analysisIndex) {
	return myAnalyses[_analysisIndex].timeSteps.size();
}

int OutputDatabase::getNumberOfLoadCases(int _analysisIndex, int _timeStepIndex) {
	return myAnalyses[_analysisIndex].timeSteps[_timeStepIndex].caseList.size();
}

std::vector<std::string> OutputDatabase::getAnalysisDescription() {
	std::vector<std::string> analysisdescription;
	for (int i = 0; i < myAnalyses.size(); i++) {
		analysisdescription.push_back(myAnalyses[i].name);
	}
	return analysisdescription;
}

std::vector<std::string> OutputDatabase::getTimeDescription(int _analysisIndex) {
	std::vector<std::string> timeDescription;

	for (int i = 0; i < myAnalyses[_analysisIndex].timeSteps.size(); i++)	{
		timeDescription.push_back(myAnalyses[_analysisIndex].timeSteps[i].timeDescription);
	}
	return timeDescription;
}

std::string OutputDatabase::getSyntaxForTime(STACCATO_Analysis_type _analysisType) {
	switch (_analysisType)
	{
	case STACCATO_Analysis_Static:
		return "";
	case STACCATO_Analysis_DynamicReal:
	case STACCATO_Analysis_Dynamic:
		return " Hz";
	default:
		return " Uk";		// UnknownDataType
	}
}

std::string OutputDatabase::getSyntaxForSubLoadCase(STACCATO_ResultsCase_type _caseType) {
	switch (_caseType)
	{
	case STACCATO_Case_None:
		return "";
	case STACCATO_Case_Load:
		return " deg";
	default:
		std::cerr << "Invalid Result Case Type!\n";
		return " Uk";
	}
}

int OutputDatabase::getStartIndexForNewAnalysis() {
	return myVectorFieldResults.size();
}

std::string OutputDatabase::getUnit(int _analysisIndex, int _timeStepIndex) {
	return myAnalyses[_analysisIndex].timeSteps[_timeStepIndex].unit;
}

std::string OutputDatabase::getUnit(int _analysisIndex, int _timeStepIndex, int _loadCaseIndex) {
	return myAnalyses[_analysisIndex].timeSteps[_timeStepIndex].caseList[_loadCaseIndex].unit;
}