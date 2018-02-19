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
/***********************************************************************************************//**
 * \file BoundaryCondition.h
 * This file holds the class BoundaryCondition
 * \date 2/2/2018
 **************************************************************************************************/

#ifndef BOUNDARYCONDITION_H_
#define BOUNDARYCONDITION_H_

#include <string>
#include <vector>
#include <assert.h>
#include "MathLibrary.h"

class HMesh;
/********//**
 * \brief This implements all boundary condtions
 **************************************************************************************************/
template<class T>
class BoundaryCondition {
public:
	/***********************************************************************************************
	 * \brief Constructor
	 * \author Stefan Sicklinger
	 ***********/
	BoundaryCondition(HMesh& _hMesh) : myHMesh(&_hMesh) {};
	/***********************************************************************************************
	 * \brief Destructor
	 *
	 * \author Stefan Sicklinger
	 ***********/
	virtual ~BoundaryCondition(void) {};
	/***********************************************************************************************
	* \brief Compute distributing coupling loads
	* \param[in] handle _referenceNode labels
	* \param[in] handle _couplingNodes labels
	* \param[in] handle _loadVector load on reference nodes (i.e. 6Dofs
	* \param[in,out] handle to rhs: rhs beeing a vector of type T and length totalNumDofs times numRhs
	* \author Stefan Sicklinger
	***********/
	void computeDistributingCouplingLoad(std::vector<int> &_referenceNode, std::vector<int> &_couplingNodes, std::vector<T> &_loadVector, std::vector<T> &_rhs) {
		int numCouplingNodes = _couplingNodes.size();
		double weightingFactor = 1.0 / (double)numCouplingNodes;
		double xMean[] = { 0, 0, 0 };

		std::string analysisType = MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin()->TYPE()->data();

		std::vector<double> forceVector;
		std::vector<double> forceVectorIm;
		std::vector<double> momentVector;
		std::vector<double> momentVectorIm;

		//real branch
		if (std::is_same<T, double>::value) {
			for (int i = 0; i < 3; i++) {
				forceVector.push_back(_loadVector[i]);
				momentVector.push_back(_loadVector[i + 3]);
			}
		}
		//complex branch
		if (std::is_same<T, STACCATOComplexDouble>::value) {
			for (int i = 0; i < 3; i++) {
				forceVector.push_back(_loadVector[i].real());
				momentVector.push_back(_loadVector[i + 3].real());
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
		//complex branch
		if (std::is_same<T, STACCATOComplexDouble>::value) {
			momentRefereceDIm = MathLibrary::computeVectorCrossProduct(referencePositionVector, forceVectorIm);
			MathLibrary::computeDenseVectorAddition(&momentVector[0], &momentRefereceDIm[0], 1, 3);
		}

		std::vector<double> r(numCouplingNodes * 3, 0);
		std::vector<double> T(9, 0);
		std::vector<std::complex<double>> RForces(numCouplingNodes * 3, 0);

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
		for (int m = 0; m < 3; m++) {
			for (int n = 0; n < 3; n++) {
				for (int i = 0; i < numCouplingNodes; i++) {
					T[m * 3 + n] += -weightingFactor*r[i * 3 + m] * r[i * 3 + n];
					if (m == n) {		// Diagonal Index Check
						for (int j = 0; j < 3; j++) {
							// Diagonals - Identity Summation
							T[m * 3 + n] += weightingFactor*r[i * 3 + j] * r[i * 3 + j];
						}
					}
				}
			}
		}

		std::vector<double> temp = MathLibrary::solve3x3LinearSystem(T, momentRefereceD, 1e-14);
		std::vector<double> tempIm;
		//complex branch
		if (std::is_same<T, STACCATOComplexDouble>::value) {
			tempIm = MathLibrary::solve3x3LinearSystem(T, momentRefereceDIm, 1e-14);
		}

		for (int i = 0; i < numCouplingNodes; i++) {
			std::vector<double> rCurrent = { r[i * 3 + 0] , r[i * 3 + 1],r[i * 3 + 2] };
			std::vector<double> temp2 = MathLibrary::computeVectorCrossProduct(temp, rCurrent);
			std::vector<double> temp2Im;
			//complex branch
			if (std::is_same<T, STACCATOComplexDouble>::value) {
				temp2Im = MathLibrary::computeVectorCrossProduct(tempIm, rCurrent);
			}
			//real branch
			if (std::is_same<T, double>::value) {
				for (int j = 0; j < 3; j++) {
					_rhs[i * 3 + j](weightingFactor*(forceVector[j] + temp2[j]));
				}
			}
			//complex branch
			if (std::is_same<T, STACCATOComplexDouble>::value) {
				for (int j = 0; j < 3; j++) {
					_rhs[i * 3 + j].real(weightingFactor*(forceVector[j] + temp2[j]));
					_rhs[i * 3 + j].imag(weightingFactor*(forceVectorIm[j] + temp2Im[j]));
				}
			}
		}
	}
	/***********************************************************************************************
	* \brief addConcentratedForceContribution
	* \param[in] handle _nodeList
	* \param[in,out] handle to rhs: rhs beeing a vector of type T and length totalNumDofs times numRhs
	* \author Stefan Sicklinger
	***********/
	void addConcentratedForceContribution(std::vector<int> &_nodeList, std::vector<T> &_rhs) {
		bool flagLabel = true;
		for (int m = 0; m < _nodeList.size(); m++) {
			int numDoFsPerNode = myHMesh->getNumDoFsPerNode(myHMesh->convertNodeLabelToNodeIndex(nodeSet[m]));
			for (int l = 0; l < numDoFsPerNode; l++) {
				if (myHMesh->getNodeLabels()[myHMesh->convertNodeLabelToNodeIndex(nodeSet[m])] == _nodeList[m]) {
					flagLabel = false;

					//real branch
					if (std::is_same<T, double>::value) {
						int dofIndex = myHMesh->getNodeIndexToDoFIndices()[myHMesh->convertNodeLabelToNodeIndex(nodeSet[m])][l];
						switch (l) {
						case 0:
							_rhs[dofIndex] += std::atof(iLoads->LOAD()[k].REAL().begin()->X()->data());
							break;
						case 1:
							_rhs[dofIndex] += std::atof(iLoads->LOAD()[k].REAL().begin()->Y()->data());
							break;
						case 2:
							_rhs[dofIndex] += std::atof(iLoads->LOAD()[k].REAL().begin()->Z()->data());
							break;
						default:
							break;
						}
					}
					//complex branch
					if (std::is_same<T, STACCATOComplexDouble>::value) {
						int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
						switch (l) {
						case 0:
							_rhs[dofIndex].real += std::atof(iLoads->LOAD()[k].REAL().begin()->X()->data());
							_rhs[dofIndex].imag += std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->X()->data());
							break;
						case 1:
							_rhs[dofIndex].real += iLoads->LOAD()[k].REAL().begin()->Y()->data());
							_rhs[dofIndex].imag += std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->Y()->data());
							break;
						case 2:
							_rhs[dofIndex].real += temp_Fz(std::atof(iLoads->LOAD()[k].REAL().begin()->Z()->data());
							_rhs[dofIndex].imag += std::atof(iLoads->LOAD()[k].IMAGINARY().begin()->Z()->data());
							break;
						default:
							break;
						}
					}
				}
			}
		}
		if (flagLabel)
			std::cerr << ">> Error while Loading: NODE of NODESET " << std::string(iLoads->LOAD()[k].NODESET().begin()->Name()->c_str()) << " not found.\n";
	}
private:
	/// HMesh object 
	HMesh *myHMesh;
};

#endif /* BOUNDARYCONDITION_H_ */
