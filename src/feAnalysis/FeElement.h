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

#include <assert.h>
#include <math.h>

/********//**
 *
 **************************************************************************************************/
class FeElement{
public:
	/***********************************************************************************************
	 * \brief Constructor
	 * \author Stefan Sicklinger
	 ***********/
	FeElement(void);
	/***********************************************************************************************
	 * \brief Destructor
	 *
	 * \author Stefan Sicklinger
	 ***********/
	virtual ~FeElement(void);
	/***********************************************************************************************
	* \brief Evalute derivative of local shape functions for bi-linear element
	* \param[in] xi
	* \param[in] eta
	* \param[in] eleCoords
	* \param[out] d_N_d_xi_eta
	* \author Stefan Sicklinger
	***********/
	void evalQuad4IsoPShapeFunDer(const double* eleCoords, const double xi, const double eta, double *N, double *dNx, double *dNy, double &Jdet);
private:
};


#endif /* FEELEMENT_H_ */