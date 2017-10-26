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


FeAnalysis::FeAnalysis(HMesh& _hMesh, FeMetaDatabase& _feMetaDatabase) : myHMesh(&_hMesh), myFeMetaDatabase(&_feMetaDatabase) {

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();

	Material * elasticMaterial = new Material();

	anaysisTimer01.start();
	myHMesh->buildDoFGraph();
	anaysisTimer01.stop();
	infoOut << "Duration for building DoF graph: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	anaysisTimer01.start();

	std::vector<FeElement*> allElements(numElements);
	int lastIndex = 0;
	for (int iElement = 0; iElement < numElements; iElement++)
	{
		if (myHMesh->getElementTypes()[iElement] == STACCATO_PlainStress4Node2D) {
			allElements[iElement] = new FePlainStress4NodeElement(elasticMaterial);
		}
		else	if (myHMesh->getElementTypes()[iElement] == STACCATO_Tetrahedron10Node3D) {
			allElements[iElement] = new FeTetrahedron10NodeElement(elasticMaterial);
		}
		int numNodesPerElement = myHMesh->getNumNodesPerElement()[iElement];
		double*  eleCoords = &myHMesh->getNodeCoordsSortElement()[lastIndex];
		allElements[iElement]->computeElementMatrix(eleCoords);
		lastIndex += numNodesPerElement*myHMesh->getDomainDimension();
	}
	anaysisTimer01.stop();
	infoOut << "Duration for element loop: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	
	
	std::vector<double> freq;
	freq.push_back(10);
	freq.push_back(100);
	freq.push_back(1000);
	freq.push_back(2000);
	freq.push_back(3000);
	freq.push_back(4000);
	freq.push_back(5000);
	freq.push_back(6000);
	freq.push_back(7000);
	freq.push_back(8000);

	anaysisTimer01.start();

	int totalDoF = myHMesh->getTotalNumOfDoFsRaw();
	// Memory for output
	std::vector<double> resultUxRe;
	std::vector<double> resultUyRe;
	std::vector<double> resultUzRe;
	resultUxRe.resize(numNodes);
	resultUyRe.resize(numNodes);
	resultUzRe.resize(numNodes);
	
	
	// Allocate global matrix and vector memory
	
	std::vector<double> b;
	std::vector<double> sol;
	b.resize(totalDoF);
	sol.resize(totalDoF);
	for (int iFreqCounter=0; iFreqCounter < 10; iFreqCounter++) {
	lastIndex = 0;
	MathLibrary::SparseMatrix<double> *A = new MathLibrary::SparseMatrix<double>(totalDoF, true);
	std::cout << "test1" << std::endl;
	for (int iElement = 0; iElement < numElements; iElement++)
	{
		int numDoFsPerElement = myHMesh->getNumDoFsPerElement()[iElement];
		int*  eleDoFs = &myHMesh->getElementDoFList()[lastIndex];
		lastIndex += numDoFsPerElement;
		double omega = 2 * M_PI*freq[iFreqCounter];
		//Assembly routine symmetric stiffness
		for (int i = 0; i < numDoFsPerElement; i++) {
			for (int j = 0; j < numDoFsPerElement; j++) {
				if (eleDoFs[j] >= eleDoFs[i]) {
					(*A)(eleDoFs[i], eleDoFs[j]) += allElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j];
				}
			}
		}
		//K - omega*omega*M
		//Assembly routine symmetric mass
		for (int i = 0; i < numDoFsPerElement; i++) {
			for (int j = 0; j < numDoFsPerElement; j++) {
				if (eleDoFs[j] >= eleDoFs[i]) {
					(*A)(eleDoFs[i], eleDoFs[j]) -= allElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j] * omega*omega;
				}
			}
		}
	}
	std::cout << "test2" << std::endl;
	//Add cload rhs contribution 
	double cload = 1.0;
	for (int j = 0; j < numNodes; j++)
	{
		int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
		for (int l = 0; l < numDoFsPerNode; l++) {
			if (myHMesh->getNodeLabels()[j] == 1) {
				int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
				b[dofIndex] = +cload;
				cload++;
			}
		}
	}
	anaysisTimer01.stop();
	infoOut << "Duration for assembly loop: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
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
				resultUxRe[j]=sol[dofIndex];
			}
			if (l == 1) {
				resultUyRe[j] = sol[dofIndex];
			}
			if (l == 2) {
				resultUzRe[j] = sol[dofIndex];
			}
		}
		if (myHMesh->getDomainDimension() == 2) {
			resultUzRe[j] = 0.0;
		}

	}
	// Store results to database
	myHMesh->addResultScalarFieldAtNodes(STACCATO_Ux_Re, resultUxRe);
	myHMesh->addResultScalarFieldAtNodes(STACCATO_Uy_Re, resultUyRe);
	myHMesh->addResultScalarFieldAtNodes(STACCATO_Uz_Re, resultUzRe);
	myHMesh->addResultsTimeDescription(std::to_string(freq[iFreqCounter]));

	
	infoOut << sol[0] << std::endl;
	infoOut << sol[1] << std::endl;
	infoOut << sol[2] << std::endl;
	delete A;
	(*A).cleanPardiso();
	}



}

FeAnalysis::~FeAnalysis() {
}



