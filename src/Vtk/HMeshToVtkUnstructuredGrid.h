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
#pragma once

#include <vector>

//VTK
#include <vtkSmartPointer.h>

class HMesh;
class vtkUnstructuredGrid;

class HMeshToVtkUnstructuredGrid
{

public:
  /***********************************************************************************************
  * \brief Constructor
  * \author Stefan Sicklinger
  ***********/
  HMeshToVtkUnstructuredGrid(HMesh& _HMesh);
  /***********************************************************************************************
  * \brief Return reference to smart pointer
  * \author Stefan Sicklinger
  ***********/
  vtkSmartPointer<vtkUnstructuredGrid> &  getVtkUnstructuredGrid(void) { return myVtkUnstructuredGrid; }
  /***********************************************************************************************
  * \brief Set scalar field for dispay
  * \author Stefan Sicklinger
  ***********/
  void setScalarFieldAtNodes(std::vector<double> _scalarField);
  void setVectorFieldAtNodes(std::vector<double> _x, std::vector<double> _y, std::vector<double> _z);

protected:


private:
	vtkSmartPointer<vtkUnstructuredGrid> myVtkUnstructuredGrid;

};