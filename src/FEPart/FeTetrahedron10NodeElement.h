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
 * \file FeTetrahedron10NodeElement.h
 * This file holds the class of quadratic tetrahedron element
 * \date 8/28/2017
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <assert.h>
#include <math.h>
#include <vector>
#include <FeElement.h>

/********//**
* \brief Class FeTetrahedron10NodeElement 
 **************************************************************************************************/
class FeTetrahedron10NodeElement : public FeElement {
public:
	/***********************************************************************************************
	 * \brief Constructor
	 * \author Stefan Sicklinger
	 ***********/
	FeTetrahedron10NodeElement(Material *_material);
	/***********************************************************************************************
	 * \brief Destructor
	 * \author Stefan Sicklinger
	 ***********/
	virtual ~FeTetrahedron10NodeElement(void);
	/***********************************************************************************************
	* \brief Compute stiffness, mass and damping matrices
	* \param[in] _eleCoords Element cooord vector
	* \author Stefan Sicklinger
	***********/
	void computeElementMatrix(const double* _eleCoords);
	/***********************************************************************************************
	* \brief Return pointer to double array
	* \author Stefan Sicklinger
	***********/
	 const std::vector<double> &  getStiffnessMatrix(void) const  { return myKe; }
	/***********************************************************************************************
	* \brief Return pointer to double array
	* \author Stefan Sicklinger
	***********/
	 const std::vector<double> & getMassMatrix(void) const  { return myMe; }
private:
	/***********************************************************************************************
	* \brief Evalute derivative of local shape functions for 10 node ted element
	* \param[in] _eleCoords
	* \param[in] _xi1
	* \param[in] _xi2
	* \param[in] _xi3
	* \param[out] _N
	* \param[out] _dNx
	* \param[out] _dNy
	* \param[out] _dNz
	* \param[out] _Jdet
	* \author Stefan Sicklinger
	***********/
	void evalTet10IsoPShapeFunDer(const double* _eleCoords, const double _xi1, const double _xi2, const double _xi3, const double _xi4, double *_N, double *_dNx, double *_dNy, double *_dNz, double &_Jdet);
};