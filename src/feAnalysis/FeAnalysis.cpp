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
#include "FeElement.h"
#include "MathLibrary.h"


FeAnalysis::FeAnalysis(HMesh& _hMesh, FeMetaDatabase& _feMetaDatabase) : myHMesh(&_hMesh), myFeMetaDatabase(& _feMetaDatabase) {

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();

	for (int i = 0; i < numElements; i++)
	{
		double Emat[9];


		c = E / (1 - vu*vu);

		D = c*[1  vu  0; vu  1    0; 0    0   0.5*(1 - vu)];


		double Ke[64];
		FeElement* oneEle = new FeElement();

		//3D -> 2D
		double eleCorrds[8] = { _hMesh.getNodeCoords()[0], _hMesh.getNodeCoords()[1], _hMesh.getNodeCoords()[3], _hMesh.getNodeCoords()[4], _hMesh.getNodeCoords()[9], _hMesh.getNodeCoords()[10], _hMesh.getNodeCoords()[6], _hMesh.getNodeCoords()[7] };
		//&(_hMesh.getNodeCoords()[0])
		oneEle->computeElementStiffness(eleCorrds, Emat, Ke);
	}

}

FeAnalysis::~FeAnalysis() {
}



