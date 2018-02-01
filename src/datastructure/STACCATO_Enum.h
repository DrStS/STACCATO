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
 * \file STACCATO_Enum.h
 * This file holds all enumerations (specifying types) of the program
 * \date 27/1/2017
 **************************************************************************************************/
#ifndef STACCATO_ENUM_H_
#define STACCATO_ENUM_H_

enum STACCATO_Element_type {
	STACCATO_Mass,
	STACCATO_Spring,
	STACCATO_Truss2D,
	STACCATO_Truss3D,
	STACCATO_PlainStrain4Node2D,
	STACCATO_PlainStress4Node2D,
	STACCATO_PlainStrain3Node2D,
	STACCATO_Tetrahedron10Node3D,
	STACCATO_UmaElement
};

enum STACCATO_Result_type {
	STACCATO_Ux_Re,
	STACCATO_Uy_Re,
	STACCATO_Uz_Re,
	STACCATO_Magnitude_Re,
	STACCATO_Ux_Im,
	STACCATO_Uy_Im,
	STACCATO_Uz_Im,
	STACCATO_Magnitude_Im,
};

enum STACCATO_Picker_type {
	STACCATO_Picker_None,
	STACCATO_Picker_Node,
	STACCATO_Picker_Element
};

#endif /* STACCATO_ENUM_H_ */
