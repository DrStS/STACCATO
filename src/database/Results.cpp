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

Results::Results() {
	myResultCase = STACCATO_Case_None;			// no sub-cases
	myResultEvaluationType = STACCATO_Evaluation_Nodal;
}

Results::~Results() {
	
}

void Results::setAnalysisType(STACCATO_Analysis_type _type) {
	myAnalsisType = _type;
	buildLabelMap();
}

void Results::setResultsEvaluationType(STACCATO_ResultsEvaluation_type _evaluationType) {
	myResultEvaluationType = _evaluationType;
}

void Results::setResultsType(STACCATO_Results_type _type) {
	myResultType = _type;
	buildLabelMap();
}