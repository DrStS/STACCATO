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
/***********************************************************************************************//**
* \file FeUmaElement.h
* This file holds the class of Abaqus SIM Element
* \date 1/31/2018
**************************************************************************************************/

#ifndef FEUMAELEMENT_H_
#define FEUMAELEMENT_H_

#include <cstddef>
#include <assert.h>
#include <math.h>
#include <vector>
#include <FeElement.h>

/********//**
* \brief Class FeUmaElement
**************************************************************************************************/
class FeUmaElement : public FeElement {
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Harikrishnan Sreekumar
	***********/
	FeUmaElement(Material *_material);
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	virtual ~FeUmaElement(void);
	/***********************************************************************************************
	* \brief Compute stiffness, mass and damping matrices
	* \param[in] _eleCoords Element cooord vector
	* \author Harikrishnan Sreekumar
	***********/
	void computeElementMatrix(const double* _eleCoords);
	/***********************************************************************************************
	* \brief Return pointer to double array
	* \author Harikrishnan Sreekumar
	***********/
	const std::vector<double> &  getStiffnessMatrix(void) const { return myKe; }
	/***********************************************************************************************
	* \brief Return pointer to double array
	* \author Harikrishnan Sreekumar
	***********/
	const std::vector<double> & getMassMatrix(void) const { return myMe; }
};


#endif /* FEUMAELEMENT_H_ */