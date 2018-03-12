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
/*************************************************************************************************
* \file BoundaryCondition.h
* This file holds the class BoundaryCondition
* \date 2/2/2018
**************************************************************************************************/

#pragma once

#include <string>
#include <vector>
#include <assert.h>
#include "MathLibrary.h"
#include <HMesh.h>
#include <complex>
using namespace std::complex_literals;

#include <iostream>
#include "Timer.h"

#include <MetaDatabase.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <type_traits>

/**********
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
	void computeDistributingCouplingLoad(std::vector<int> &_referenceNode, std::vector<int> &_couplingNodes, std::vector<std::complex<double>> &_loadVector, std::vector<T> &_rhs) {
		myCaseType = STACCATO_Case_Load;

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
				forceVector.push_back(_loadVector[i].real());
				momentVector.push_back(_loadVector[i + 3].real());
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
		std::vector<double> TVector(9, 0);
		RForces.clear();
		RForces.resize(numCouplingNodes * 3, 0);

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
					TVector[m * 3 + n] += -weightingFactor*r[i * 3 + m] * r[i * 3 + n];
					if (m == n) {		// Diagonal Index Check
						for (int j = 0; j < 3; j++) {
							// Diagonals - Identity Summation
							TVector[m * 3 + n] += weightingFactor*r[i * 3 + j] * r[i * 3 + j];
						}
					}
				}
			}
		}

		std::vector<double> temp = MathLibrary::solve3x3LinearSystem(TVector, momentRefereceD, 1e-14);
		std::vector<double> tempIm;
		//complex branch
		if (std::is_same<T, STACCATOComplexDouble>::value) {
			tempIm = MathLibrary::solve3x3LinearSystem(TVector, momentRefereceDIm, 1e-14);
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
	}
	/***********************************************************************************************
	* \brief addConcentratedForceContribution
	* \param[in] handle _nodeList
	* \param[in,out] handle to rhs: rhs beeing a vector of type T and length totalNumDofs times numRhs
	* \author Stefan Sicklinger
	***********/
	void addConcentratedForceContribution(std::vector<int> &_nodeList, std::vector<double> &_loadVector, std::vector<T> &_rhs) {
		std::vector<std::complex<double>> loadVector(3);
		loadVector[0] = { _loadVector[0], _loadVector[3] };
		loadVector[1] = { _loadVector[1], _loadVector[4] };
		loadVector[2] = { _loadVector[2], _loadVector[5] };

		bool flagLabel = true;
		for (int m = 0; m < _nodeList.size(); m++) {
			int numDoFsPerNode = myHMesh->getNumDoFsPerNode(myHMesh->convertNodeLabelToNodeIndex(_nodeList[m]));
			for (int l = 0; l < numDoFsPerNode; l++) {
				if (myHMesh->getNodeLabels()[myHMesh->convertNodeLabelToNodeIndex(_nodeList[m])] == _nodeList[m]) {
					flagLabel = false;
					int dofIndex = myHMesh->getNodeIndexToDoFIndices()[myHMesh->convertNodeLabelToNodeIndex(_nodeList[m])][l];

					storeConcentratedLoadRHS(std::is_floating_point<T>{}, dofIndex, loadVector[l], _rhs);
				}
			}
		}
		if (flagLabel)
			std::cerr << ">> Error while Loading: NODE of _nodeList not found.\n";
	}
	/***********************************************************************************************
	* \brief addRotatingForceContribution
	* \param[in] handle _nodeList
	* \param[in,out] handle to rhs: rhs beeing a vector of type T and length totalNumDofs times numRhs
	* \author Harikrishnan Sreekumar
	***********/
	void addRotatingForceContribution(std::vector<int> &_refNode, std::vector<int> &_couplingNodes, std::vector<double> &_loadVector, std::vector<T> &_rhs) {
		myCaseType = STACCATO_Case_Load;

		std::cout << ">> Looking for Rotating Distributed Coupling ...\n";

		// Resize RHS vector
		int newSize = getBCCaseDescription().size()*_rhs.size();
		_rhs.resize(newSize);

		int i = 0;
		do {
			std::cout << ">> Computing for theta = " << getBCCaseDescription()[i] << " ...\n";
			double loadRe[] = { _loadVector[0] , _loadVector[1] , _loadVector[2] };
			double loadIm[] = { _loadVector[3] , _loadVector[4] , _loadVector[5] };
			double loadMagnitudeRe = MathLibrary::computeDenseEuclideanNorm(loadRe, 3);
			double loadMagnitudeIm = MathLibrary::computeDenseEuclideanNorm(loadIm, 3);

			std::vector<std::complex<double>> loadVector(6);
			loadVector[0] = { sin(getBCCaseDescription()[i] * M_PI / 180)*loadMagnitudeRe, sin(getBCCaseDescription()[i] * M_PI / 180)*loadMagnitudeIm };
			loadVector[1] = { _loadVector[1], _loadVector[4] };
			loadVector[2] = { cos(getBCCaseDescription()[i] * M_PI / 180)*loadMagnitudeRe, cos(getBCCaseDescription()[i] * M_PI / 180)*loadMagnitudeIm };
			loadVector[3] = 0;
			loadVector[4] = 0;
			loadVector[5] = 0;

			std::cout << ">> Load at theta " << getBCCaseDescription()[i] << " is " << loadVector[0] << " : " << loadVector[1] << " : " << loadVector[2] << std::endl;

			if (_refNode.size() == 1 && _couplingNodes.size() != 0)
				computeDistributingCouplingLoad(_refNode, _couplingNodes, loadVector, _rhs);
			else
				std::cerr << ">> Error in DistributingCouplingForce Input.\n" << std::endl;

			bool flagLabel = false;
			for (int j = 0; j < _couplingNodes.size(); j++)
			{
				int numDoFsPerNode = myHMesh->getNumDoFsPerNode(myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[j]));
				for (int l = 0; l < numDoFsPerNode; l++) {
					if (myHMesh->getNodeLabels()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[j])] == _couplingNodes[j]) {
						flagLabel = true;
						int dofIndex = myHMesh->getNodeIndexToDoFIndices()[myHMesh->convertNodeLabelToNodeIndex(_couplingNodes[j])][l];

						storeConcentratedLoadRHS(std::is_floating_point<T>{}, i*myHMesh->getTotalNumOfDoFsRaw() + dofIndex, RForces[j * 3 + l], _rhs);
					}
				}
			}

			if (flagLabel)
				std::cout << ">> Building RHS with DistributedCouplingForce Finished." << std::endl;
			else
				std::cerr << ">> Error in building RHS with DistributedCouplingForce." << std::endl;
			i++;
		} while (i < getBCCaseDescription().size());
	}
	/***********************************************************************************************
	* \brief Add Boundary Influenced Case Description
	* \author Harikrishnan Sreekumar
	***********/
	void addBCCaseDescription(double _caseDescription) {
		boundaryCaseDescription.push_back(_caseDescription);
	}
	/***********************************************************************************************
	* \brief Get Boundary Influenced Case Description
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<double>& getBCCaseDescription() {
		return boundaryCaseDescription;
	}
	/***********************************************************************************************
	* \brief Get Number of RHS that is to be solved with BoundaryCondition
	* \author Harikrishnan Sreekumar
	***********/
	int getNumberOfTotalCases() {
		switch (myCaseType)
		{
		case STACCATO_Case_Load:
			return getBCCaseDescription().size();
		default:
			return 1;
		}
	}

	void storeConcentratedLoadRHS(std::true_type, int _targetIndex, std::complex<double> _source, std::vector<T> &_target) {
		_target[_targetIndex] += _source.real();
	}

	void storeConcentratedLoadRHS(std::false_type, int _targetIndex, std::complex<double> _source, std::vector<T> &_target) {
		_target[_targetIndex].real += _source.real();
		_target[_targetIndex].imag += _source.imag();
	}

	std::vector<double>& getRealPartOfVector(std::true_type, std::vector<T> &_vector) {
		return _vector;
	}

	std::vector<double>& getRealPartOfVector(std::false_type, std::vector<T> &_vector) {
		std::vector<double> realVector;
		for (int i = 0; i < _vector.size(); i++)		{
			realVector.push_back(_vector[i].real);
		}
		return realVector;
	}

	std::vector<double>& getImagPartOfVector(std::true_type, std::vector<T> &_vector) {
		return std::vector<double>(_vector.size(), 0);			// Imaginary part of Double Template Routine is by principle zero
	}

	std::vector<double>& getImagPartOfVector(std::false_type, std::vector<T> &_vector) {
		std::vector<double> imagVector;
		for (int i = 0; i < _vector.size(); i++) {
			imagVector.push_back(_vector[i].imag);
		}
		return imagVector;
	}


private:
	/// HMesh object 
	HMesh *myHMesh;
	/// BC vector case description
	std::vector<double> boundaryCaseDescription;
	/// Computing Force Vector
	std::vector<std::complex<double>> RForces;
public:
	STACCATO_ResultsCase_type myCaseType;
};