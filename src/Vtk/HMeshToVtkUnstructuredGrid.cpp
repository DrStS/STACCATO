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
#include <HMeshToVtkUnstructuredGrid.h>
#include <HMesh.h>
#include <Message.h>
#include <STACCATO_Enum.h>

//VTK
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkQuad.h>


//================================================================
// Function : Constructor
// Purpose  :
//================================================================
HMeshToVtkUnstructuredGrid::HMeshToVtkUnstructuredGrid(HMesh& _HMesh)
{
		int numNodes = _HMesh.getNumNodes();
		vtkSmartPointer< vtkPoints > myNodes =vtkSmartPointer< vtkPoints > ::New();

		// Node loop
		for (int i = 0; i < numNodes; i++)
		{
			myNodes->InsertNextPoint(_HMesh.getNodeCoords()[(i * 3) + 0],
				                    _HMesh.getNodeCoords()[(i * 3) + 1],
				                    _HMesh.getNodeCoords()[(i * 3) + 2]);
		}

		// Element loop
		int numElements = _HMesh.getNumElements();
		vtkSmartPointer<vtkQuad> aLinearQuad = vtkSmartPointer<vtkQuad>::New();
		vtkSmartPointer<vtkCellArray> cellArray = vtkSmartPointer<vtkCellArray>::New();
		int index = 0;
		for (int i = 0; i < numElements; i++)
		{
			if (_HMesh.getElementTypes()[i] == STACCATO_PlainStrain4Node2D || _HMesh.getElementTypes()[i] == STACCATO_PlainStress4Node2D) {

				aLinearQuad->GetPointIds()->SetId(0, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 0]));
				aLinearQuad->GetPointIds()->SetId(1, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 1]));
				aLinearQuad->GetPointIds()->SetId(2, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 2]));
				aLinearQuad->GetPointIds()->SetId(3, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 3]));
				cellArray->InsertNextCell(aLinearQuad);
				index = index + 4;
			}
		}

		 myVtkUnstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
		 myVtkUnstructuredGrid->SetPoints(myNodes);
		 myVtkUnstructuredGrid->SetCells(VTK_QUAD, cellArray);

}

