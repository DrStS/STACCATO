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
 * \file FePlainStress4NodeElement.h
 * This file holds the class of linear four node plane stress element
 * \date 8/28/2017
 **************************************************************************************************/

#ifndef FEPLAINSTRESS4NODEELEMENT_H_
#define FEPLAINSTRESS4NODEELEMENT_H_

#include <cstddef>
#include <assert.h>
#include <math.h>
#include <vector>
#include <FeElement.h>

/********//**
* \brief Class FePlainStress4NodeElement 
 **************************************************************************************************/
class FePlainStress4NodeElement : public FeElement {
public:
	/***********************************************************************************************
	 * \brief Constructor
	 * \author Stefan Sicklinger
	 ***********/
	FePlainStress4NodeElement(Material *_material);
	/***********************************************************************************************
	 * \brief Destructor
	 *
	 * \author Stefan Sicklinger
	 ***********/
	virtual ~FePlainStress4NodeElement(void);
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
	* \brief Evalute derivative of local shape functions for bi-linear element
	* \param[in] _xi
	* \param[in] _eta
	* \param[in] _eleCoords
	* \param[out] _dNx
	* \param[out] _dNy
	* \param[out] _Jdet
	* \author Stefan Sicklinger
	***********/
	void evalQuad4IsoPShapeFunDer(const double* _eleCoords, const double _xi, const double _eta, double *_N, double *_dNx, double *_dNy, double &_Jdet);
};


#endif /* FEPLAINSTRESS4NODEELEMENT_H_ */