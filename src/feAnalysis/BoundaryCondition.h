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
private:
	/// HMesh object 
	HMesh *myHMesh;

};

#endif /* BOUNDARYCONDITION_H_ */
