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
 * \file Material.h
 * This file holds the class Material; Base class for a material
 * \date 8/28/2017
 **************************************************************************************************/

#pragma once

#include <cstddef>
#include <assert.h>
#include <math.h>
#include <string>

/********//**
* \brief Class Material 
 **************************************************************************************************/
class Material {
public:
	/***********************************************************************************************
	 * \brief Constructor
	 * \author Stefan Sicklinger
	 ***********/
	Material(void);
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	Material(std::string _materialName, int _partId);
	/***********************************************************************************************
	 * \brief Destructor
	 *
	 * \author Stefan Sicklinger
	 ***********/
	virtual ~Material(void);
	/***********************************************************************************************
	* \brief Return youngs modulus
	* \author Stefan Sicklinger
	***********/
	double getYoungsModulus(void) const  { return myYoungsModulus; }
	/***********************************************************************************************
	* \brief Return Poisson's ratio
	* \author Stefan Sicklinger
	***********/
	 double getPoissonsRatio(void) const  { return myPoissonsRatio; }
	 /***********************************************************************************************
	 * \brief Return density
	 * \author Stefan Sicklinger
	 ***********/
	 double getDensity(void) const { return myDensity; }
	 /***********************************************************************************************
	 * \brief Return density
	 * \author Stefan Sicklinger
	 ***********/
	 double getDampingParameter(void) const { return myDampingParameter; }
private:
    ///
	double myYoungsModulus;
	double myPoissonsRatio;
	double myDensity;
	double myDampingParameter;
};