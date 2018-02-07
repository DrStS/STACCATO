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
class BoundaryCondition {
public:
    /***********************************************************************************************
     * \brief Constructor
     * \author Stefan Sicklinger
     ***********/
	BoundaryCondition(HMesh& _hMesh);
    /***********************************************************************************************
     * \brief Destructor
     *
     * \author Stefan Sicklinger
     ***********/
	virtual ~BoundaryCondition(void);
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	void addConcentratedForce(std::vector<double> &_rhsReal);
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	void addConcentratedForce(std::vector<MKL_Complex16> &_rhsComplex);
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	std::vector<std::vector<double>> computeDistributingCouplingLoad(std::vector<int> &_referenceNode, std::vector<int> &_couplingNodes, std::vector<double> &_loadVector);
	/***********************************************************************************************
	* \brief Computes the Cross Product of two vectors
	* \param[in] Vector 1
	* \param[in] Vector 2
	* \param[out] Cross Product
	* \author Stefan Sicklinger
	***********/
	std::vector<double> computeVectorCrossProduct(std::vector<double> &_v1, std::vector<double> &_v2);
	/***********************************************************************************************
	* \brief Computes the Sum of two vectors
	* \param[in] Vector 1
	* \param[in] Vector 2
	* \param[out] Sum Vector
	* \author Stefan Sicklinger
	***********/
	std::vector<double> computeVectorAddition(std::vector<double> &_v1, std::vector<double> &_v2);
	/***********************************************************************************************
	* \brief Computes the Difference of two vectors
	* \param[in] Vector 1
	* \param[in] Vector 2
	* \param[out] Difference Vector = Vector 1 - Vector 2
	* \author Stefan Sicklinger
	***********/
	std::vector<double> computeVectorSubstraction(std::vector<double> &_v1, std::vector<double> &_v2);
	/***********************************************************************************************
	* \brief Computes the Scalar Multiplication
	* \param[in] Vector
	* \param[in] Scalar
	* \param[out] Resulting Vector
	* \author Stefan Sicklinger
	***********/
	std::vector<double> computeVectorScalarMultiplication(std::vector<double> &_v, int _a);

	std::vector<double> BoundaryCondition::solve3x3LinearSystem(std::vector<std::vector<double>>& _A, std::vector<double>& _b, double _EPS);
	double BoundaryCondition::det3x3(std::vector<std::vector<double>>& _A);
private:
	/// HMesh object 
	HMesh *myHMesh;

};

#endif /* BOUNDARYCONDITION_H_ */
