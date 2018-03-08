/*  Copyright &copy; 2018, Stefan Sicklinger, Munich
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
#include "BoundaryCondition.h"
#include "MetaDatabase.h"
#include "HMesh.h"
#include <iostream>
#include "Timer.h"

#define _USE_MATH_DEFINES
#include <math.h>

BoundaryCondition::BoundaryCondition(HMesh& _hMesh) : myHMesh(& _hMesh) {
	nRHS = 1;
	isRotate = false;
}

BoundaryCondition::~BoundaryCondition() {
}


void BoundaryCondition::addConcentratedForce(std::vector<double> &_rhsReal){


	unsigned int numNodes = myHMesh->getNumNodes();

	STACCATO_XML::LOADS_const_iterator iLoads(MetaDatabase::getInstance()->xmlHandle->LOADS().begin());
	for (int k = 0; k < iLoads->LOAD().size(); k++) {

		if (std::string(iLoads->LOAD()[k].Type()->c_str()) == "ConcentratedForce") {
			// Find NODESET
			std::vector<int> nodeSet = myHMesh->convertNodeSetNameToLabels(std::string(iLoads->LOAD()[k].NODESET().begin()->Name()->c_str()));

			if (nodeSet.empty())
				std::cerr << ">> Error while Loading: NODESET " << std::string(iLoads->LOAD()[k].NODESET().begin()->Name()->c_str()) << " not Found.\n";
			else
				std::cout << ">> " << std::string(iLoads->LOAD()[k].Type()->c_str()) << " " << iLoads->LOAD()[k].NODESET().begin()->Name()->c_str() << " is loaded.\n";

			bool flagLabel = true;

			for (int m = 0; m < nodeSet.size(); m++) {
				int numDoFsPerNode = myHMesh->getNumDoFsPerNode(myHMesh->convertNodeLabelToNodeIndex(nodeSet[m]));
				for (int l = 0; l < numDoFsPerNode; l++) {
						if (myHMesh->getNodeLabels()[myHMesh->convertNodeLabelToNodeIndex(nodeSet[m])] == nodeSet[m]) {
							flagLabel = false;

							std::complex<double> temp_Fx(std::atof(iLoads->LOAD()[k].REAL().begin()->X()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->X()->data()));
							std::complex<double> temp_Fy(std::atof(iLoads->LOAD()[k].REAL().begin()->Y()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->Y()->data()));
							std::complex<double> temp_Fz(std::atof(iLoads->LOAD()[k].REAL().begin()->Z()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->Z()->data()));

							int dofIndex = myHMesh->getNodeIndexToDoFIndices()[myHMesh->convertNodeLabelToNodeIndex(nodeSet[m])][l];
							switch (l) {
							case 0:
								_rhsReal[dofIndex] += temp_Fx.real();
								break;
							case 1:
								_rhsReal[dofIndex] += temp_Fy.real();
								break;
							case 2:
								_rhsReal[dofIndex] += temp_Fz.real();
								break;
							default:
								break;
							}
						}
				}
			}
			if (flagLabel)
				std::cerr << ">> Error while Loading: NODE of NODESET " << std::string(iLoads->LOAD()[k].NODESET().begin()->Name()->c_str()) << " not found.\n";

		}
		else if (std::string(iLoads->LOAD()[k].Type()->c_str()) == "DistributingCouplingForce" || std::string(iLoads->LOAD()[k].Type()->c_str()) == "RotatingDistributingCouplingForce") {

			if (std::string(iLoads->LOAD()[k].Type()->c_str()) == "RotatingDistributingCouplingForce") {
				isRotate = true;

				std::cout << ">> Looking for Rotating Distributed Coupling ...\n";

				// Routine to accomodate Step Distribution
				double start_theta = std::atof(iLoads->LOAD()[k].ROTATE().begin()->START_THETA()->c_str());
				myHMesh->addResultsSubFrameDescription(start_theta);		// Push back starting frequency in any case

				if (std::string(iLoads->LOAD()[k].ROTATE().begin()->Type()->data()) == "STEP") {		// Step Distribute
					double end_theta = std::atof(iLoads->LOAD()[k].ROTATE().begin()->END_THETA()->c_str());
					double step_theta = std::atof(iLoads->LOAD()[k].ROTATE().begin()->STEP_THETA()->c_str());
					double push_theta = start_theta + step_theta;

					while (push_theta <= end_theta) {
						myHMesh->addResultsSubFrameDescription(push_theta);
						push_theta += step_theta;
					}
				}

				// Resize RHS vector
				int newSize = myHMesh->getResultsSubFrameDescription().size()*_rhsReal.size();
				_rhsReal.resize(newSize);

				nRHS = myHMesh->getResultsSubFrameDescription().size();
			}

			std::vector<int> refNode = myHMesh->convertNodeSetNameToLabels(std::string(iLoads->LOAD()[k].REFERENCENODESET().begin()->Name()->c_str()));
			std::vector<int> couplingNodes = myHMesh->convertNodeSetNameToLabels(std::string(iLoads->LOAD()[k].COUPLINGNODESET().begin()->Name()->c_str()));

			int i = 0;
			do {
				std::vector<std::complex<double>> loadVector(6);
				loadVector[0] = { std::atof(iLoads->LOAD()[k].REAL().begin()->X()->data()), 0 };
				loadVector[1] = { std::atof(iLoads->LOAD()[k].REAL().begin()->Y()->data()), 0 };
				loadVector[2] = { std::atof(iLoads->LOAD()[k].REAL().begin()->Z()->data()), 0 };
				loadVector[3] = 0;
				loadVector[4] = 0;
				loadVector[5] = 0;
				if (std::string(iLoads->LOAD()[k].Type()->c_str()) == "RotatingDistributingCouplingForce")
				{
					std::cout << ">> Computing for theta = " << myHMesh->getResultsSubFrameDescription()[i] << " degree ...\n";
					double load[] = { std::atof(iLoads->LOAD()[k].REAL().begin()->X()->data()) , std::atof(iLoads->LOAD()[k].REAL().begin()->Y()->data()) ,std::atof(iLoads->LOAD()[k].REAL().begin()->Z()->data()) };
					double loadMagnitude = MathLibrary::computeDenseEuclideanNorm(load, 3);
					loadVector[0] = { sin(myHMesh->getResultsSubFrameDescription()[i]*M_PI/180)*loadMagnitude, 0 };
					loadVector[1] = { std::atof(iLoads->LOAD()[k].REAL().begin()->Y()->data()), 0 };
					loadVector[2] = { cos(myHMesh->getResultsSubFrameDescription()[i]*M_PI/180)*loadMagnitude, 0 };
					loadVector[3] = 0;
					loadVector[4] = 0;
					loadVector[5] = 0;
					std::cout << ">> Load at theta " << myHMesh->getResultsSubFrameDescription()[i] << " is " << loadVector[0] << " : " << loadVector[1] << " : " << loadVector[2]<< std::endl;
				}			

				std::vector<std::complex<double>> distributedCouplingLoad;
				if (refNode.size() == 1 && couplingNodes.size() != 0)
					distributedCouplingLoad = computeDistributingCouplingLoad(refNode, couplingNodes, loadVector);
				else
					std::cerr << ">> Error in DistributingCouplingForce Input.\n" << std::endl;

				bool flagLabel = false;
				for (int j = 0; j < couplingNodes.size(); j++)
				{
					int numDoFsPerNode = myHMesh->getNumDoFsPerNode(myHMesh->convertNodeLabelToNodeIndex(couplingNodes[j]));
					for (int l = 0; l < numDoFsPerNode; l++) {
						if (myHMesh->getNodeLabels()[myHMesh->convertNodeLabelToNodeIndex(couplingNodes[j])] == couplingNodes[j]) {
							flagLabel = true;
							int dofIndex = myHMesh->getNodeIndexToDoFIndices()[myHMesh->convertNodeLabelToNodeIndex(couplingNodes[j])][l];
							switch (l) {
							case 0:
								_rhsReal[i*myHMesh->getTotalNumOfDoFsRaw() + dofIndex] += distributedCouplingLoad[j * 3 + 0].real();
								break;
							case 1:
								_rhsReal[i*myHMesh->getTotalNumOfDoFsRaw() + dofIndex] += distributedCouplingLoad[j * 3 + 1].real();
								break;
							case 2:
								_rhsReal[i*myHMesh->getTotalNumOfDoFsRaw() + dofIndex] += distributedCouplingLoad[j * 3 + 2].real();
								break;
							default:
								break;
							}
						}
					}
				}
				
				if (flagLabel)
					std::cout << ">> Building RHS with DistributedCouplingForce Finished." << std::endl;
				else
					std::cerr << ">> Error in building RHS with DistributedCouplingForce." << std::endl;
				i++;
			} while (i < myHMesh->getResultsSubFrameDescription().size());
		}
	}
	std::cout << ">> Building RHS Finished." << std::endl;
}

void BoundaryCondition::addConcentratedForce(std::vector<MKL_Complex16> &_rhsComplex) {
	std::cout << "Complex Routine!";

	unsigned int numNodes = myHMesh->getNumNodes();

	STACCATO_XML::LOADS_const_iterator iLoads(MetaDatabase::getInstance()->xmlHandle->LOADS().begin());
	for (int k = 0; k < iLoads->LOAD().size(); k++) {

		if (std::string(iLoads->LOAD()[k].Type()->c_str()) == "ConcentratedForce") {
			// Find NODESET
			std::vector<int> nodeSet = myHMesh->convertNodeSetNameToLabels(std::string(iLoads->LOAD()[k].NODESET().begin()->Name()->c_str()));

			if (nodeSet.empty())
				std::cerr << ">> Error while Loading: NODESET " << std::string(iLoads->LOAD()[k].NODESET().begin()->Name()->c_str()) << " not Found.\n";
			else
				std::cout << ">> " << std::string(iLoads->LOAD()[k].Type()->c_str()) << " " << iLoads->LOAD()[k].NODESET().begin()->Name()->c_str() << " is loaded.\n";

			int flagLabel = 0;

			for (int j = 0; j < numNodes; j++)
			{
				int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
				for (int l = 0; l < numDoFsPerNode; l++) {
					for (int m = 0; m < nodeSet.size(); m++) {
						if (myHMesh->getNodeLabels()[j] == myHMesh->getNodeLabels()[myHMesh->convertNodeLabelToNodeIndex(nodeSet[m])]) {
							flagLabel = 1;

							std::complex<double> temp_Fx(std::atof(iLoads->LOAD()[k].REAL().begin()->X()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->X()->data()));
							std::complex<double> temp_Fy(std::atof(iLoads->LOAD()[k].REAL().begin()->Y()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->Y()->data()));
							std::complex<double> temp_Fz(std::atof(iLoads->LOAD()[k].REAL().begin()->Z()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->Z()->data()));

							int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
							switch (l) {
							case 0:
								_rhsComplex[dofIndex].real += temp_Fx.real();
								_rhsComplex[dofIndex].imag += temp_Fx.imag();
								break;
							case 1:
								_rhsComplex[dofIndex].real += temp_Fy.real();
								_rhsComplex[dofIndex].imag += temp_Fy.imag();
								break;
							case 2:
								_rhsComplex[dofIndex].real += temp_Fz.real();
								_rhsComplex[dofIndex].imag += temp_Fz.imag();
								break;
							default:
								break;
							}
						}
					}
				}
			}
			if (flagLabel == 0)
				std::cerr << ">> Error while Loading: NODE of NODESET " << std::string(iLoads->LOAD()[k].NODESET().begin()->Name()->c_str()) << " not found.\n";

		}
		else if (std::string(iLoads->LOAD()[k].Type()->c_str()) == "DistributingCouplingForce") {
			std::cout << ">> Looking for Distributed Coupling ...\n";

			std::vector<int> refNode = myHMesh->convertNodeSetNameToLabels(std::string(iLoads->LOAD()[k].REFERENCENODESET().begin()->Name()->c_str()));
			std::vector<int> couplingNodes = myHMesh->convertNodeSetNameToLabels(std::string(iLoads->LOAD()[k].COUPLINGNODESET().begin()->Name()->c_str()));

			std::cout << "RF Size: " << refNode.size() << std::endl;

			std::cout << "CN Size: " << couplingNodes.size() << std::endl;

			std::vector<std::complex<double>> loadVector(6);
			loadVector[0] = { std::atof(iLoads->LOAD()[k].REAL().begin()->X()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->X()->data()) };
			loadVector[1] = { std::atof(iLoads->LOAD()[k].REAL().begin()->Y()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->Y()->data()) };
			loadVector[2] = { std::atof(iLoads->LOAD()[k].REAL().begin()->Z()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->Z()->data()) };
			loadVector[3] = 0;
			loadVector[4] = 0;
			loadVector[5] = 0;

			std::vector<std::complex<double>> distributedCouplingLoad;
			if (refNode.size() == 1 && couplingNodes.size() != 0)
				distributedCouplingLoad = computeDistributingCouplingLoad(refNode, couplingNodes, loadVector);
			else
				std::cerr << ">> Error in DistributingCouplingForce Input.\n" << std::endl;
			
			bool flagLabel = false;
			for (int j = 0; j < numNodes; j++)
			{
				int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
				for (int l = 0; l < numDoFsPerNode; l++) {
					for (int m = 0; m < couplingNodes.size(); m++) {
						if (myHMesh->getNodeLabels()[j] == couplingNodes[m]) {
							flagLabel = true;
							int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
							switch (l) {
							case 0:
								_rhsComplex[dofIndex].real += distributedCouplingLoad[m * 3 + 0].real();
								_rhsComplex[dofIndex].imag += distributedCouplingLoad[m * 3 + 0].imag();
								break;
							case 1:
								_rhsComplex[dofIndex].real += distributedCouplingLoad[m * 3 + 1].real();
								_rhsComplex[dofIndex].imag += distributedCouplingLoad[m * 3 + 1].imag();
								break;
							case 2:
								_rhsComplex[dofIndex].real += distributedCouplingLoad[m * 3 + 2].real();
								_rhsComplex[dofIndex].imag += distributedCouplingLoad[m * 3 + 2].imag();
								break;
							default:
								break;
							}
						}
					}
				}
			}
			if (flagLabel)
				std::cout << ">> Building RHS with DistributedCouplingForce Finished." << std::endl;
			else
				std::cerr << ">> Error in building RHS with DistributedCouplingForce." << std::endl;
		}
	}
	std::cout << ">> Building RHS Finished." << std::endl;
}

std::vector<std::complex<double>> BoundaryCondition::computeDistributingCouplingLoad(std::vector<int> &_referenceNode, std::vector<int> &_couplingNodes, std::vector<std::complex<double>> &_loadVector) {	
	int numCouplingNodes = _couplingNodes.size();
	double weightingFactor = 1.0 / (double)numCouplingNodes;
	double xMean[] = {0, 0, 0};

	std::string analysisType = MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin()->TYPE()->data();

	std::vector<double> forceVector;
	std::vector<double> forceVectorIm;
	std::vector<double> momentVector;
	std::vector<double> momentVectorIm;
	for (int i = 0; i < 3; i++)	{
		forceVector.push_back(_loadVector[i].real());
		momentVector.push_back(_loadVector[i + 3].real());
		if (analysisType == "STEADYSTATE_DYNAMIC") {
			forceVectorIm.push_back(_loadVector[i].imag());
			momentVectorIm.push_back(_loadVector[i + 3].imag());
		}
	}

	std::vector<double> referencePositionVector(3);
	referencePositionVector[0] = myHMesh->getNodeCoords()[(_referenceNode[0] - 1) * 3 + 0];
	referencePositionVector[1] = myHMesh->getNodeCoords()[(_referenceNode[0] - 1) * 3 + 1];
	referencePositionVector[2] = myHMesh->getNodeCoords()[(_referenceNode[0] - 1) * 3 + 2];

	std::vector<double> momentRefereceD = MathLibrary::computeVectorCrossProduct(referencePositionVector, forceVector);
	MathLibrary::computeDenseVectorAddition(&momentVector[0], &momentRefereceD[0], 1, 3);

	std::vector<double> momentRefereceDIm;
	if (analysisType == "STEADYSTATE_DYNAMIC") {
		momentRefereceDIm = MathLibrary::computeVectorCrossProduct(referencePositionVector, forceVectorIm);
		MathLibrary::computeDenseVectorAddition(&momentVector[0], &momentRefereceDIm[0], 1, 3);		
	}
	
	std::vector<double> r(numCouplingNodes*3,0); 
	std::vector<double> T(9,0);
	std::vector<std::complex<double>> RForces(numCouplingNodes*3,0);
	
	// xMean Calculation
	for (int i = 0; i < numCouplingNodes; i++) {
		double* x = new double[3];
		x[0] = myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 0];
		x[1] = myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 1];
		x[2] = myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 2];
		
		MathLibrary::computeDenseVectorAddition(x, xMean, weightingFactor, 3);
	}
	
	// r vector Calculation
	for (int i = 0; i < numCouplingNodes; i++) {
		double* x = new double[3];
		x[0] = myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 0];
		x[1] = myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 1];
		x[2] = myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 2];

		MathLibrary::computeDenseVectorAddition(xMean, x, -1, 3);
		r[i * 3 + 0] = x[0];
		r[i * 3 + 1] = x[1];
		r[i * 3 + 2] = x[2];
	}

	// Building T Matrix
	for (int m = 0; m < 3; m++)	{
		for (int n = 0; n < 3; n++) {
			for (int i = 0; i < numCouplingNodes; i++) {
				T[m * 3 + n] += -weightingFactor*r[i * 3 + m] * r[i * 3 + n];
				if (m == n) {		// Diagonal Index Check
					for (int j = 0; j < 3; j++)					{
						// Diagonals - Identity Summation
						T[m * 3 + n] += weightingFactor*r[i * 3 + j] * r[i * 3 + j];
					}
				}
			}
		}
	}

	std::vector<double> temp = MathLibrary::solve3x3LinearSystem(T, momentRefereceD, 1e-14);
	std::vector<double> tempIm;
	if (analysisType == "STEADYSTATE_DYNAMIC") {
		tempIm = MathLibrary::solve3x3LinearSystem(T, momentRefereceDIm, 1e-14);
	}

	
	for (int i = 0; i < numCouplingNodes; i++) {
		std::vector<double> rCurrent = { r[i * 3 + 0] , r[i * 3 + 1],r[i * 3 + 2] };
		std::vector<double> temp2 = MathLibrary::computeVectorCrossProduct(temp, rCurrent);
		std::vector<double> temp2Im;
		if (analysisType == "STEADYSTATE_DYNAMIC") {
			temp2Im = MathLibrary::computeVectorCrossProduct(tempIm, rCurrent);
		}
		for (int j = 0; j < 3; j++) {
			RForces[i * 3 + j].real(weightingFactor*(forceVector[j] + temp2[j]));
			if (analysisType == "STEADYSTATE_DYNAMIC") {
				RForces[i * 3 + j].imag(weightingFactor*(forceVectorIm[j] + temp2Im[j]));
			}
		}
	}
	return RForces;
}
