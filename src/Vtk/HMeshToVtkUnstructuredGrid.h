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
* \file HMeshToVtkUnstructuredGrid.h
* This file holds the class of HMeshToVtkUnstructuredGrid.
* \date 9/12/2017
 **************************************************************************************************/
#ifndef HMESHTOVTKUNSTRUCTUREDGRID
#define HMESHTOVTKUNSTRUCTUREDGRID

//VTK
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
class HMesh;

class HMeshToVtkUnstructuredGrid
{

public:
  /***********************************************************************************************
  * \brief Constructor
  * \author Stefan Sicklinger
  ***********/
  HMeshToVtkUnstructuredGrid(HMesh& _HMesh);


protected:


private:
	vtkSmartPointer<vtkUnstructuredGrid> myVtkUnstructuredGrid;

};





#endif // HMESHTOVTKUNSTRUCTUREDGRID
