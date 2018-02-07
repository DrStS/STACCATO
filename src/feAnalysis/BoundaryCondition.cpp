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
#include <complex>
#include "Timer.h"

BoundaryCondition::BoundaryCondition(HMesh& _hMesh) : myHMesh(& _hMesh) {
	
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
			}
			if (flagLabel == 0)
				std::cerr << ">> Error while Loading: NODE of NODESET " << std::string(iLoads->LOAD()[k].NODESET().begin()->Name()->c_str()) << " not found.\n";

		}
		else if (std::string(iLoads->LOAD()[k].Type()->c_str()) == "DistributingCouplingForce") {
			std::cout << ">> Looking for Distributed Coupling ...\n";

			std::vector<int> refNode = myHMesh->convertNodeSetNameToLabels(std::string(iLoads->LOAD()[k].REFERENCENODESET().begin()->Name()->c_str()));
			std::vector<int> couplingNodes = myHMesh->convertNodeSetNameToLabels(std::string(iLoads->LOAD()[k].COUPLINGNODESET().begin()->Name()->c_str()));

			std::vector<double> loadVector(6);
			loadVector[0] = std::atof(iLoads->LOAD()[k].REAL().begin()->X()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->X()->data());
			loadVector[1] = std::atof(iLoads->LOAD()[k].REAL().begin()->Y()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->Y()->data());
			loadVector[2] = std::atof(iLoads->LOAD()[k].REAL().begin()->Z()->data()), std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->Z()->data());
			loadVector[3] = 0;
			loadVector[4] = 0;
			loadVector[5] = 0;

			std::vector<double> distributedCouplingLoad;
			if (refNode.size() == 1 && couplingNodes.size() != 0)
				distributedCouplingLoad = computeDistributingCouplingLoad(refNode, couplingNodes, loadVector);
			else
				std::cerr << ">> Error in DistributingCouplingForce Input.\n"<<std::endl;

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
								_rhsReal[dofIndex] += distributedCouplingLoad[m*3+0];
								break;
							case 1:
								_rhsReal[dofIndex] += distributedCouplingLoad[m*3+1];
								break;
							case 2:
								_rhsReal[dofIndex] += distributedCouplingLoad[m*3+2];
								break;
							default:
								break;
							}
						}
					}
				}
			}
			if(flagLabel)
				std::cout << ">> Building RHS with DistributedCouplingForce Finished." << std::endl;
			else
				std::cerr << ">> Error in building RHS with DistributedCouplingForce." << std::endl;
		}
	}
	std::cout << ">> Building RHS Finished." << std::endl;
}

void BoundaryCondition::addConcentratedForce(std::vector<MKL_Complex16> &_rhsComplex) {

	unsigned int numNodes = myHMesh->getNumNodes();
	
	STACCATO_XML::LOADS_const_iterator iLoads(MetaDatabase::getInstance()->xmlHandle->LOADS().begin());
	for (int k = 0; k < iLoads->LOAD().size(); k++) {

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
						if (std::string(iLoads->LOAD()[k].Type()->c_str()) == "ConcentratedForce") {
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
		}
		if (flagLabel == 0)
			std::cerr << ">> Error while Loading: NODE of NODESET " << std::string(iLoads->LOAD()[k].NODESET().begin()->Name()->c_str()) << " not found.\n";
	}
	std::cout << ">> Building RHS Finished." << std::endl;
}

std::vector<double> BoundaryCondition::computeDistributingCouplingLoad(std::vector<int> &_referenceNode, std::vector<int> &_couplingNodes, std::vector<double> &_loadVector) {
	int numCouplingNodes = _couplingNodes.size();
	double weightingFactor = 1.0 / (double)numCouplingNodes;
	double xMean[] = {0, 0, 0};

	std::vector<double> forceVector;
	std::vector<double> momentVector;
	for (int i = 0; i < 3; i++)	{
		forceVector.push_back(_loadVector[i]);
		momentVector.push_back(_loadVector[i + 3]);
	}
	
	std::vector<double> referencePositionVector(3);
	referencePositionVector[0] = 0.0;
	referencePositionVector[1] = 500.0;
	referencePositionVector[2] = 0.0;

	std::vector<double> momentRefereceD = computeVectorCrossProduct(referencePositionVector, forceVector);
	MathLibrary::computeDenseVectorAddition(&momentVector[0], &momentRefereceD[0], 1, 3);
	
	std::vector<double> r(numCouplingNodes*3,0);
	std::vector<double> T(9,0);
	std::vector<double> RForces(numCouplingNodes*3,0);
	
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

	std::vector<double> temp = solve3x3LinearSystem(T, momentRefereceD, 1e-14);
	
	for (int i = 0; i < numCouplingNodes; i++) {
		std::vector<double> rCurrent = { r[i * 3 + 0] , r[i * 3 + 1],r[i * 3 + 2] };
		std::vector<double> temp2 = computeVectorCrossProduct(temp, rCurrent);
		for (int j = 0; j < 3; j++)
			RForces[i*3+j] = weightingFactor*(forceVector[j] + temp2[j]);
	}
	
	return RForces;
}

std::vector<double> BoundaryCondition::computeVectorCrossProduct(std::vector<double> &_v1, std::vector<double> &_v2) {
	std::vector<double> crossProduct(3);
	crossProduct[0] = _v1[1] * _v2[2] - _v2[1] * _v1[2];
	crossProduct[1] = -(_v1[0] * _v2[2] - _v2[0] * _v1[2]);
	crossProduct[2] = _v1[0] * _v2[1] - _v2[0] * _v1[1];
	return crossProduct;
}

std::vector<double> BoundaryCondition::solve3x3LinearSystem(std::vector<double>& _A, std::vector<double>& _b, double _EPS) {
	std::vector<double> A(9,0);
	std::vector<double> b(3,0);

	double detA = det3x3(_A);
	if (fabs(detA) < _EPS)
		return{};
	for (int i = 0; i < 3; i++)
		b[i] = _b[i];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 9; j++)
			A[j] = _A[j];
		for (int j = 0; j < 3; j++)
			A[j * 3 + i] = b[j];
		_b[i] = det3x3(A) / detA;
	}
	return _b;
}

double BoundaryCondition::det3x3(std::vector<double>& _A) {
	return _A[0] * _A[4] * _A[8] + _A[1] * _A[5] * _A[6] + _A[2] * _A[3] * _A[7]
		- _A[0] * _A[5] * _A[7] - _A[1] * _A[3] * _A[8] - _A[2] * _A[4] * _A[6];
}