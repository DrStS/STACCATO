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
#pragma once

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

enum STACCATO_ScalarField_components {
	STACCATO_Scalar_Re,
	STACCATO_Scalar_Im
};

enum STACCATO_VectorField_components {
	STACCATO_x_Re,
	STACCATO_y_Re,
	STACCATO_z_Re,
	STACCATO_Magnitude_Re,
	STACCATO_x_Im,
	STACCATO_y_Im,
	STACCATO_z_Im,
	STACCATO_Magnitude_Im,
};

enum STACCATO_Results_type {
	STACCATO_Result_Displacement
};

enum STACCATO_Analysis_type {
	STACCATO_Analysis_Static,
	STACCATO_Analysis_DynamicReal,
	STACCATO_Analysis_Dynamic
};

enum STACCATO_ResultsCase_type {
	STACCATO_Case_None,
	STACCATO_Case_Load
};

enum STACCATO_ResultsEvaluation_type {
	STACCATO_Evaluation_Nodal,
	STACCATO_Evaluation_Elemental
};

enum STACCATO_Picker_type {
	STACCATO_Picker_None,
	STACCATO_Picker_Node,
	STACCATO_Picker_Element
};

