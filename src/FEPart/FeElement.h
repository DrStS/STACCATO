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
 * \file FeElement.h
 * This file holds the class FeElement; Base class for a Fe element
 * \date 8/28/2017
 **************************************************************************************************/

#ifndef FEELEMENT_H_
#define FEELEMENT_H_

#include <cstddef>
#include <assert.h>
#include <math.h>
#include <vector>
class Material;
/********//**
* \brief Class FeElement 
 **************************************************************************************************/
class FeElement{
public:
	/***********************************************************************************************
	 * \brief Constructor
	 * \author Stefan Sicklinger
	 ***********/
	FeElement(Material *_material);
	/***********************************************************************************************
	 * \brief Destructor
	 * \author Stefan Sicklinger
	 ***********/
	virtual ~FeElement(void);
	/***********************************************************************************************
	* \brief  Compute stiffness, mass and damping matrices  (interface)
	* \param[in] _eleCoords Element cooord vector
	* \author Stefan Sicklinger
	***********/
	virtual void computeElementMatrix(const double* _eleCoords) = 0;
	/***********************************************************************************************
	* \brief Return pointer to double array (interface)
	* \author Stefan Sicklinger
	***********/
	virtual const std::vector<double> &  getStiffnessMatrix(void) const = 0;
	/***********************************************************************************************
	* \brief Return pointer to double array (interface)
	* \author Stefan Sicklinger
	***********/
	virtual const std::vector<double> & getMassMatrix(void) const = 0;
	/***********************************************************************************************
	* \brief Return pointer to Element Material
	* \author Harikrishnan Sreekumar
	***********/
	virtual const Material* getMaterial(void) { return myMaterial; }
protected:
	/// Stiffness matrix
	std::vector<double> myKe;
	/// Mass matrix
	std::vector<double> myMe;
	/// Material
	Material * myMaterial;
};


#endif /* FEELEMENT_H_ */