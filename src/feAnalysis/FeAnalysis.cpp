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
#include "FeAnalysis.h"
#include "Message.h"
#include "HMesh.h"
#include "FeMetaDatabase.h"
#include "FeElement.h"
#include "MathLibrary.h"
#include "Timer.h"
#include "MemWatcher.h"


FeAnalysis::FeAnalysis(HMesh& _hMesh, FeMetaDatabase& _feMetaDatabase) : myHMesh(&_hMesh), myFeMetaDatabase(& _feMetaDatabase) {

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();

	//Hack mat class is missing
	double ni = 0.3;
	double E = 210000;
	double tmp = E / (1 - ni*ni);
	double Emat[9] = { tmp, tmp*ni, 0, tmp*ni, tmp, 0, 0, 0, tmp*0.5*(1 - ni) };

	int totalDoF = myHMesh->getTotalNumOfDoFsRaw();
	int dimension;
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

	FeElement* oneEle = new FeElement();
	for (int i = 0; i < numElements; i++)
	{
		double Ke[64] = { 0 };
		double Me[64] = { 0 }; 
		int numNodesPerElement = myHMesh->getNumNodesPerElement()[i];
		double * eleCoord = new double[numNodesPerElement*dimension];
		numDoFsPerElement = 0;

		//Loop over nodes of current element
		for (int j = 0; j < numNodesPerElement; j++)
		{
			int nodeIndex = myHMesh->getElementIndexToNodesIndices()[i][j];
			if (dimension == 2){
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
		oneEle->computeElementMatrix(eleCoord, Emat, Ke, Me);
		delete eleCoord;

		double freq = 101;
		double omega = 2 * M_PI*freq;
		//Assembly routine symmetric stiffness
		for (int i = 0; i < numDoFsPerElement; i++){
			for (int j = 0; j < numDoFsPerElement; j++){
				if(eleDoFs[j]>= eleDoFs[i]){
					(*A)(eleDoFs[i], eleDoFs[j]) += Ke[i*numDoFsPerElement + j];
				}
			}
		}
		//K - omega*omega*M
		//Assembly routine symmetric mass
		for (int i = 0; i < numDoFsPerElement; i++){
			for (int j = 0; j < numDoFsPerElement; j++){
				if (eleDoFs[j] >= eleDoFs[i]) {
					(*A)(eleDoFs[i], eleDoFs[j]) -= Me[i*numDoFsPerElement + j] * omega*omega;
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
			if (myHMesh->getNodeLabels()[j] == 16) {
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
	infoOut << "Duration for direct solver factorize: " << anaysisTimer01.getDurationSec() << " sec" << std::endl;
	infoOut << "Total duration for direct solver: " << anaysisTimer02.getDurationSec() << " sec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	infoOut<<sol[0]<<std::endl;

	delete A;
}

FeAnalysis::~FeAnalysis() {
}



