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
#include <OutputDatabase.h>
#include <iostream>

OutputDatabase::OutputDatabase() {
}

OutputDatabase::~OutputDatabase() {

}

void OutputDatabase::addVectorFieldToDatabase(VectorFieldResults* _vectorField) {
	myVectorFieldResults.push_back(*_vectorField);
	std::cout << ">> OutputDatabase is allocated for an expected Vector Field Result.\n";
}

void OutputDatabase::addVectorFieldAnalysisDescription(std::string _resultsAnalyisDescription, STACCATO_Analysis_type _type) {
	switch (_type)
	{
	case STACCATO_Analysis_Static:
		_resultsAnalyisDescription += "_Static";
		break;
	case STACCATO_Analysis_DynamicReal:
		_resultsAnalyisDescription += "_DynamicR";
		break;
	case STACCATO_Analysis_Dynamic:
		_resultsAnalyisDescription += "_Dynamic";
		break;
	default:
		std::cerr << "Invalid Analyis Type!\n";
		break;
	}
	myVectorFieldAnalysisDectription.push_back(_resultsAnalyisDescription);
}