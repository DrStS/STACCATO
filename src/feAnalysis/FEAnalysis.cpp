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
#include "FeAnalysis.h"
#include "Message.h"
#include "HMesh.h"
#include "FeMetaDatabase.h"
#include "MathLibrary.h"


FeAnalysis::FeAnalysis(HMesh& _HMesh, FeMetaDatabase& _FeMetaDatabase) : myHMesh(&_HMesh), myFeMetaDatabase(&_FeMetaDatabase) {

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();
	unsigned int i = 0;
	MathLibrary::SparseMatrix<double> * Adyn = new MathLibrary::SparseMatrix<double>(2 * numNodes * 2, true);
	for (i = 0; i < numElements; i++)
	{

	}

}

FeAnalysis::~FeAnalysis() {
}



