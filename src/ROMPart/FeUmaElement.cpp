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
#include "FeUmaElement.h"
#include "Material.h"
#include "Message.h"
#include "MathLibrary.h"

#ifdef SIMULIA_API_ON
#include <ads_CoreFESystemC.h>
#include <ads_CoreMeshC.h>
#include <uma_System.h>
#include <uma_SparseMatrix.h>
#include <uma_ArrayInt.h>
#endif

#include <iostream>
#include <stdio.h>

FeUmaElement::FeUmaElement(Material *_material) : FeElement(_material) {
	myKe.resize(1000);
	myMe.resize(1000);
}

FeUmaElement::~FeUmaElement() {
}

void FeUmaElement::computeElementMatrix(const double* _eleCoords) {
	char * _fileName = "C:/software/repos/staccato/model/B31_fe_X1.sim";

	char * simFile = _fileName;
	printf("SIM file: %s\n", simFile);


#ifdef SIMULIA_API_ON
	uma_System system(simFile);

	std::cout << "\n>> Importing SIM to HMesh ..." << std::endl;

	char* matrixName = "GenericSystem_stiffness";
	if (!system.HasMatrix(matrixName)) {
		return;
		printf("\nSparse matrix %s not found\n", matrixName);
	}
	uma_SparseMatrix smtx;
	system.SparseMatrix(smtx, matrixName);

	// Map column DOFS to user nodes and dofs
	if (smtx.TypeColumns() != uma_Enum::DOFS)
		return;

	if (!smtx) {
		printf("\nSparse matrix %s cannot be not accessed\n", matrixName);
		return;
	}
	uma_SparseIterator iter(smtx);
	int row, col; double val;

	int count = 0;
	for (iter.First(); !iter.IsDone(); iter.Next(), count++) {
		iter.Entry(row, col, val);
		if (row < 30 && col < 30) {
			myKe[row * 30 + col] = val;
		}
	}
#endif
}