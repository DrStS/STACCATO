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
#include <ScalarFieldResults.h>
#include "STACCATO_Enum.h"
#include <iostream>

ScalarFieldResults::ScalarFieldResults(STACCATO_Results_type _resultType, STACCATO_Analysis_type _analysisType) : Results(_resultType, _analysisType) {

}

ScalarFieldResults::~ScalarFieldResults()
{
}

void ScalarFieldResults::buildLabelMap() {
	switch (myResultType) {
	default:
		std::cerr << "Invalid Scalar Field!\n";
	}
}