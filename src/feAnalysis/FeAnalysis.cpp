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

#define _USE_MATH_DEFINES
#include <math.h>
#include "Message.h"
#include "FeAnalysis.h"
#include "HMesh.h"
#include "FeMetaDatabase.h"
#include "FeElement.h"
#include "FePlainStress4NodeElement.h"
#include "FeTetrahedron10NodeElement.h"
#include "Material.h"

#include "MathLibrary.h"
#include "Timer.h"
#include "MemWatcher.h"


FeAnalysis::FeAnalysis(HMesh& _hMesh, FeMetaDatabase& _feMetaDatabase) : myHMesh(&_hMesh), myFeMetaDatabase(& _feMetaDatabase) {

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();


	Material * elasticMaterial = new Material();

	int totalDoF = myHMesh->getTotalNumOfDoFsRaw();
	int dimension=3;
	MathLibrary::SparseMatrix<double> *A = new MathLibrary::SparseMatrix<double>(totalDoF, true);
	std::vector<double> b;
	std::vector<double> sol;

	b.resize(totalDoF);
	sol.resize(totalDoF);
	

	if (myHMesh->getElementTypes()[0] == STACCATO_PlainStrain4Node2D || myHMesh->getElementTypes()[0] == STACCATO_PlainStress4Node2D){
		dimension = 2;
	}

	const int maxDoFsPerElement = 128;
	int eleDoFs[maxDoFsPerElement];
	int numDoFsPerElement = 0;


	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	anaysisTimer01.start();

	
	
	std::vector<FeElement*> allElements(numElements);


	for (int iElement = 0; iElement < numElements; iElement++)
	{
	if (myHMesh->getElementTypes()[iElement] == STACCATO_PlainStress4Node2D) {
			allElements[iElement] = new FePlainStress4NodeElement(elasticMaterial);		
    } else	if (myHMesh->getElementTypes()[iElement] == STACCATO_Tetrahedron10Node3D) {
		allElements[iElement] = new FeTetrahedron10NodeElement(elasticMaterial);
	}

		int numNodesPerElement = myHMesh->getNumNodesPerElement()[iElement];
		double * eleCoord = new double[numNodesPerElement*dimension];
		numDoFsPerElement = 0;

		//Loop over nodes of current element
		for (int j = 0; j < numNodesPerElement; j++)
		{
			int nodeIndex = myHMesh->getElementIndexToNodesIndices()[iElement][j];
			if (dimension == 3){
				eleCoord[j*dimension + 0] = myHMesh->getNodeCoords()[nodeIndex * 3 + 0];
				eleCoord[j*dimension + 1] = myHMesh->getNodeCoords()[nodeIndex * 3 + 1];
				eleCoord[j*dimension + 2] = myHMesh->getNodeCoords()[nodeIndex * 3 + 2];
			}
			else if (dimension == 2) {
				// Extract x and y coord only; for 2D; z=0
				eleCoord[j*dimension + 0] = myHMesh->getNodeCoords()[nodeIndex * 3 + 0];
				eleCoord[j*dimension + 1] = myHMesh->getNodeCoords()[nodeIndex * 3 + 1];
			}

			// Generate DoF table
			int numDoFsPerNode = myHMesh->getNumDoFsPerNode(nodeIndex);
			for (int l = 0; l < numDoFsPerNode; l++){
				eleDoFs[j*numDoFsPerNode+l] = myHMesh->getNodeIndexToDoFIndices()[nodeIndex][l];
				numDoFsPerElement++;
			}

		}
		allElements[iElement]->computeElementMatrix(eleCoord);
		delete eleCoord;

		double freq = 101;
		double omega = 2 * M_PI*freq;
		//Assembly routine symmetric stiffness
		for (int i = 0; i < numDoFsPerElement; i++){
			for (int j = 0; j < numDoFsPerElement; j++){
				if(eleDoFs[j]>= eleDoFs[i]){
					(*A)(eleDoFs[i], eleDoFs[j]) += allElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j];
				}
			}
		}
		//K - omega*omega*M
		//Assembly routine symmetric mass
		for (int i = 0; i < numDoFsPerElement; i++){
			for (int j = 0; j < numDoFsPerElement; j++){
				if (eleDoFs[j] >= eleDoFs[i]) {
					(*A)(eleDoFs[i], eleDoFs[j]) -= allElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j] * omega*omega;
					//std::cout << "A(" << eleDoFs[i] << "," << eleDoFs[j] << ")=" << (*A)(eleDoFs[i], eleDoFs[j]) << std::endl;
				}
			}
		}
	}

	//Add cload rhs contribution 
	double cload = 1.0;
	for (int j = 0; j < numNodes; j++)
	{	
		int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
		for (int l = 0; l < numDoFsPerNode; l++) {
			if (myHMesh->getNodeLabels()[j] == 1) {//16
				int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
				b[dofIndex] =+ cload;
				cload++;

				std::cout << dofIndex << std::endl;
			}
		}
	}

	anaysisTimer01.stop();
	infoOut << "Duration for element loop: " << anaysisTimer01.getDurationMilliSec() <<" milliSec"<<std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	anaysisTimer01.start();
	anaysisTimer02.start();
	(*A).check();
	anaysisTimer01.stop();
	infoOut << "Duration for direct solver check: " << anaysisTimer01.getDurationSec() << " sec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	anaysisTimer01.start();
	(*A).factorize();
	anaysisTimer01.stop();
	infoOut << "Duration for direct solver factorize: " << anaysisTimer01.getDurationSec() << " sec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	anaysisTimer01.start();
	(*A).solve(&sol[0], &b[0]);
	anaysisTimer01.stop();
	anaysisTimer02.stop();
	infoOut << "Duration for direct solver substitution : " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	infoOut << "Total duration for direct solver: " << anaysisTimer02.getDurationSec() << " sec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;


	// Store results
	for (int j = 0; j < numNodes; j++)
	{
		int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
		for (int l = 0; l < numDoFsPerNode; l++) {
			int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
			if (l == 0) {
				myHMesh->addResultScalarFieldAtNodes(STACCATO_Ux_Re, sol[dofIndex]);
			}
			if (l == 1) {
				myHMesh->addResultScalarFieldAtNodes(STACCATO_Uy_Re, sol[dofIndex]);
			}
			if (l == 2) {
				myHMesh->addResultScalarFieldAtNodes(STACCATO_Uz_Re, sol[dofIndex]);
			}
		}
		if (dimension==2) {
			myHMesh->addResultScalarFieldAtNodes(STACCATO_Uz_Re, 0.0);
		}
		
	}
	infoOut	<<	sol[0]	<<	std::endl;
	infoOut << sol[1] << std::endl;
	infoOut << sol[2] << std::endl;
	delete A;
}

FeAnalysis::~FeAnalysis() {
}



