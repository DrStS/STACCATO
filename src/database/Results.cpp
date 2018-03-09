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
#include <Results.h>

Results::Results(STACCATO_Results_type _resultType, STACCATO_Analysis_type _analysisType) {
	setResultsType(_resultType);
	setResultsAnalysisType(_analysisType);

	myResultCase = STACCATO_Case_None;			// no sub-cases
	myResultEvaluationType = STACCATO_Evaluation_Nodal;

	myTimeUnit = "";
	myCaseUnit = "";
}

Results::~Results() {
	
}

void Results::setResultsAnalysisType(STACCATO_Analysis_type _analysisType) {
	myAnalsisType = _analysisType;

	switch (_analysisType)
	{
	case STACCATO_Analysis_Static:
		break;
	case STACCATO_Analysis_DynamicReal:
	case STACCATO_Analysis_Dynamic:
		myTimeUnit = " Hz";
		break;
	default:
		break;
	}
}

void Results::setResultsType(STACCATO_Results_type _resultType) {
	myResultType = _resultType;
}

void Results::setResultsCase(STACCATO_ResultsCase_type _resultCase) {
	myResultCase = _resultCase;
	buildLabelMap();
}

void Results::setResultsEvaluationType(STACCATO_ResultsEvaluation_type _evaluationType) {
	myResultEvaluationType = _evaluationType;
}

void Results::addResultsTimeDescription(std::string _resultsTimeDescription) {
	resultsTimeDescription[resultsTimeDescription.size() * resultsCaseDescription.size()] = _resultsTimeDescription;
}

std::map<int, std::string>& Results::getResultsTimeDescription() {		// Getter Function to return all frequency steps
	return resultsTimeDescription;
}

void Results::addResultsCaseDescription(std::string _resultsCaseDescription) {
	resultsCaseDescription[resultsCaseDescription.size()] = _resultsCaseDescription;
}

std::map<int, std::string>& Results::getResultsCaseDescription() {
	return resultsCaseDescription;
}