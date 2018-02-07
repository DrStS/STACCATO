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

			std::vector<std::vector<double>> distributedCouplingLoad;
			if (refNode.size() == 1 && couplingNodes.size() != 0)
				distributedCouplingLoad = computeDistributingCouplingLoad(refNode, couplingNodes, loadVector);
			else
				std::cerr << " >> Error in DistributingCouplingForce Input.\n"<<std::endl;

			/* -- Testing -- */
			/*std::cout << ">> Printing RHS.. \n";
			for (int i = 0; i < distributedCouplingLoad.size(); i++)
			{
				for (int j = 0; j < 3; j++)
				{
					std::cout << " << " << distributedCouplingLoad[i][j];

				}
				std::cout << "\n" ;
			}
			std::cout << std::endl;
			*/
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
								_rhsReal[dofIndex] += distributedCouplingLoad[m][0];
								break;
							case 1:
								_rhsReal[dofIndex] += distributedCouplingLoad[m][1];
								break;
							case 2:
								_rhsReal[dofIndex] += distributedCouplingLoad[m][2];
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

std::vector<std::vector<double>> BoundaryCondition::computeDistributingCouplingLoad(std::vector<int> &_referenceNode, std::vector<int> &_couplingNodes, std::vector<double> &_loadVector) {
	std::cout << ">> Computing ...\n";
	
	int numCouplingNodes = _couplingNodes.size();// / 3;
	std::cout << ">> n  :" << numCouplingNodes << std::endl;
	double weightingFactor = 1.0 / (double)numCouplingNodes;
	std::cout << ">> WF :" << weightingFactor << std::endl;
	std::vector<double> xMean(3);
	xMean[0] = 0.0;
	xMean[1] = 0.0;
	xMean[2] = 0.0;

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

	std::cout << ">> Check Point: Ref Vector (x,y,z) " << referencePositionVector[0] << " << " << referencePositionVector[1] << " << " << referencePositionVector[2]<< std::endl;

	std::vector<double> momentRefereceD = computeVectorAddition(momentVector, computeVectorCrossProduct(referencePositionVector, forceVector));

	std::vector<std::vector<double>> r;
	std::vector<std::vector<double>> T(3);
	for (int i = 0; i < 3; i++) {
		T[i].resize(3);
		T[i][0] = 0;
		T[i][1] = 0;
		T[i][2] = 0;
	}
	std::vector<std::vector<double>> RForces;
	std::cout << ">> Computation Stage ...\n";
	for (int i = 0; i < numCouplingNodes; i++) {
		std::vector<double> x;
		x.clear();
		x.push_back(myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 0]);
		x.push_back(myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 1]);
		x.push_back(myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 2]);

		//xMean[0] += weightingFactor * x[0];
		//xMean[1] += weightingFactor * x[1];
		//xMean[2] += weightingFactor * x[2];
		xMean = computeVectorAddition(xMean, computeVectorScalarMultiplication(x, weightingFactor));
	}
	std::cout << ">> Check Point: xMean Vector (x,y,z) " << xMean[0] << " << " << xMean[1] << " << " << xMean[2] << std::endl;
	for (int i = 0; i < numCouplingNodes; i++) {
		std::vector<double> x;
		x.clear();
		x.push_back(myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 0]);
		x.push_back(myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 1]);
		x.push_back(myHMesh->getNodeCoords()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[i]) * 3 + 2]);

		r.push_back(computeVectorSubstraction(x, xMean));
	}
	std::cout << ">> rSize :" << r.size() << std::endl;

	for (int i = 0; i < numCouplingNodes; i++) {
		for (int j = 0; j < 3; j++)
		{
			// Diagonals - Identity Summation
			T[0][0] += weightingFactor*r[i][j] * r[i][j];
			T[1][1] += weightingFactor*r[i][j] * r[i][j];
			T[2][2] += weightingFactor*r[i][j] * r[i][j];
		}
		// Diagonals - r'*r summation
		T[0][0] += - r[i][0] * r[i][0];
		T[1][1] += - r[i][1] * r[i][1];
		T[2][2] += - r[i][2] * r[i][2];

		
	}
	
	/*--Testing-- */
	std::cout << ">> Printing T.. \n";
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::cout << " << " << T[i][j];

		}
		std::cout << "\n";
	}
	std::cout << std::endl;

	// Off-Diagonal Entries
	for (int m = 0; m < 3; m++)
	{
		for (int n = 0; n < 3; n++) {
			if (m != n) {		// Off-Diagonal Index Check
				for (int i = 0; i < numCouplingNodes; i++) {
					T[m][n] += -r[i][m] * r[i][n];
				}
			}
		}
	}

	/*--Testing-- */
		std::cout << ">> Printing T.. \n";
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::cout << " << " << T[i][j];

		}
		std::cout << "\n";
	}
	std::cout << std::endl;

	std::cout << ">> Solving with T ...";
	std::vector<double> temp = solve3x3LinearSystem(T, momentRefereceD, 1e-14);
	if(temp.size()!=0)
		std::cout << " Finished.\n";
	else
		std::cout << " Failed.\n";

	for (int i = 0; i < numCouplingNodes; i++) {
		RForces.push_back(computeVectorScalarMultiplication(computeVectorCrossProduct(temp, r[i]), weightingFactor));
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

std::vector<double> BoundaryCondition::computeVectorAddition(std::vector<double> &_v1, std::vector<double> &_v2) {

	std::vector<double> sum(3);
	sum[0] = _v1[0] + _v2[0];
	sum[1] = _v1[1] + _v2[1];
	sum[2] = _v1[2] + _v2[2];
	return sum;
}

std::vector<double> BoundaryCondition::computeVectorSubstraction(std::vector<double> &_v1, std::vector<double> &_v2) {
	std::vector<double> difference(3);
	difference[0] = _v1[0] - _v2[0];
	difference[1] = _v1[1] - _v2[1];
	difference[2] = _v1[2] - _v2[2];
	return difference;
}

std::vector<double> BoundaryCondition::computeVectorScalarMultiplication(std::vector<double> &_v, int _a) {
	std::vector<double> product(3);
	product[0] = _a*_v[0];
	product[1] = _a*_v[1];
	product[2] = _a*_v[2];
	return product;
}

std::vector<double> BoundaryCondition::solve3x3LinearSystem(std::vector<std::vector<double>>& _A, std::vector<double>& _b, double _EPS) {
	std::vector<std::vector<double>> A(3);
	std::vector<double> b(3);
	for (int i = 0; i < 3; i++) {
		A[i].resize(3);
		A[i][0] = 0;
		A[i][1] = 0;
		A[i][2] = 0;
	}

	double detA = det3x3(_A);
	if (fabs(detA) < _EPS)
		return {};
	for (int i = 0; i < 3; i++)
		b[i] = _b[i];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
			A[i][j] = _A[i][j];
		for (int j = 0; j < 3; j++)
			A[j][i] = b[j];
		_b[i] = det3x3(A) / detA;
	}
	return _b;
}

double BoundaryCondition::det3x3(std::vector<std::vector<double>>& _A) {
	return _A[0][0] * _A[1][1] * _A[2][2] + _A[0][1] * _A[1][2] * _A[2][0] + _A[0][2] * _A[1][0] * _A[2][1]
		- _A[0][0] * _A[1][2] * _A[2][1] - _A[0][1] * _A[1][0] * _A[2][2] - _A[0][2] * _A[1][1] * _A[2][0];
}