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
* \file SurfaceWithEdgesSetting.h
* This file holds the class of SurfaceWithEdgesSetting.
* \date 3/2/2018
**************************************************************************************************/
#ifndef _SURFACEWITHEDGESSETTING_H_
#define _SURFACEWITHEDGESSETTING_H_

#include <FieldDataSetting.h>

class SurfaceWithEdgesSetting : public FieldDataSetting
{
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Harikrishnan Sreekumar
	***********/
	SurfaceWithEdgesSetting();
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	~SurfaceWithEdgesSetting();
};


#endif // _SURFACEWITHEDGESSETTING_H_
