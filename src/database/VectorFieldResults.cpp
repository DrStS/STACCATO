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
#include <VectorFieldResults.h>
#include "STACCATO_Enum.h"
#include <iostream>

VectorFieldResults::VectorFieldResults(STACCATO_Results_type _resultType, STACCATO_Analysis_type _analysisType) : Results(_resultType, _analysisType) {
	buildLabelMap();
}

VectorFieldResults::~VectorFieldResults()
{
}

void VectorFieldResults::buildLabelMap() {
	switch (myResultType) {
	case STACCATO_Result_Displacement:
		setResultsEvaluationType(STACCATO_Evaluation_Nodal);

		myLabel = "U";
		myResultLabelMap[myLabel + "1"] = STACCATO_x_Re;
		myResultLabelMap[myLabel + "2"] = STACCATO_y_Re;
		myResultLabelMap[myLabel + "3"] = STACCATO_z_Re;
		myResultLabelMap[myLabel + "Mag"] = STACCATO_Magnitude_Re;

		if (myAnalsisType == STACCATO_Analysis_Dynamic) {
			myResultLabelMap[myLabel + "1_Im"] = STACCATO_x_Im;
			myResultLabelMap[myLabel + "2_Im"] = STACCATO_y_Im;
			myResultLabelMap[myLabel + "3_Im"] = STACCATO_z_Im;
			myResultLabelMap[myLabel + "Mag_Im"] = STACCATO_Magnitude_Im;
		}
		break;
	default:
		std::cerr << "Invalid Vector Field!\n";
	}

	myResultCaseLabelMap.clear();
	switch (myResultCase)
	{
	case STACCATO_Case_None:
		myResultCaseLabelMap["None"] = myResultCase;
		break;
	case STACCATO_Case_Load:
		myCaseUnit = " deg";
		myResultCaseLabelMap["Load Case"] = myResultCase;
		break;
	default:
		std::cerr << "Invalid Result Case Type!\n";
	}
}

void VectorFieldResults::addResultScalarFieldAtNodes(STACCATO_VectorField_components _component, std::vector<double> _valueVec) {
	myFieldMap[_component].push_back(_valueVec);
}

std::vector<double>&  VectorFieldResults::getResultScalarFieldAtNodes(STACCATO_VectorField_components _component, int index) {
	return myFieldMap[_component][index];
}
