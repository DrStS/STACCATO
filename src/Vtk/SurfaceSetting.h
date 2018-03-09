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
/*************************************************************************************************
* \file SurfaceSetting.h
* This file holds the class of SurfaceSetting.
* \date 3/2/2018
**************************************************************************************************/
#ifndef _SURFACESETTING_H_
#define _SURFACESETTING_H_

#include <FieldDataSetting.h>

class SurfaceSetting: public FieldDataSetting
{
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Harikrishnan Sreekumar
	***********/
	SurfaceSetting();
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	~SurfaceSetting();
};


#endif // _SURFACESETTING_H_
