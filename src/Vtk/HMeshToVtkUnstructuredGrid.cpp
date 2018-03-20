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
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkQuad.h>
#include <vtkQuadraticTetra.h>
#include <vtkFloatArray.h>
#include <vtkWarpVector.h>


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
		vtkSmartPointer<vtkCellArray> cellArrayLinearQuad = vtkSmartPointer<vtkCellArray>::New();
		vtkSmartPointer<vtkQuadraticTetra> aQuadTet = vtkSmartPointer<vtkQuadraticTetra>::New();
		vtkSmartPointer<vtkCellArray> cellArrayQuadTed = vtkSmartPointer<vtkCellArray>::New();
		int index = 0;
		for (int i = 0; i < numElements; i++)
		{
			if (_HMesh.getElementTypes()[i] == STACCATO_PlainStrain4Node2D || _HMesh.getElementTypes()[i] == STACCATO_PlainStress4Node2D) {
				aLinearQuad->GetPointIds()->SetId(0, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 0]));
				aLinearQuad->GetPointIds()->SetId(1, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 1]));
				aLinearQuad->GetPointIds()->SetId(2, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 2]));
				aLinearQuad->GetPointIds()->SetId(3, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 3]));
				cellArrayLinearQuad->InsertNextCell(aLinearQuad);
				index = index + 4;
			}
			if (_HMesh.getElementTypes()[i] == STACCATO_Tetrahedron10Node3D) {
				aQuadTet->GetPointIds()->SetId(0, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 0]));
				aQuadTet->GetPointIds()->SetId(1, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 1]));
				aQuadTet->GetPointIds()->SetId(2, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 2]));
				aQuadTet->GetPointIds()->SetId(3, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 3]));
				aQuadTet->GetPointIds()->SetId(4, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 4]));
				aQuadTet->GetPointIds()->SetId(5, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 5]));
				aQuadTet->GetPointIds()->SetId(6, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 6]));
				aQuadTet->GetPointIds()->SetId(7, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 7]));
				aQuadTet->GetPointIds()->SetId(8, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 8]));
				aQuadTet->GetPointIds()->SetId(9, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 9]));
				cellArrayQuadTed->InsertNextCell(aQuadTet);
				index = index + 10;
			}
		}

		 myVtkUnstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
		 myVtkUnstructuredGrid->SetPoints(myNodes);
		 if (cellArrayLinearQuad->GetNumberOfCells() != 0) {
			 myVtkUnstructuredGrid->SetCells(VTK_QUAD, cellArrayLinearQuad);
		 }
		 if (cellArrayQuadTed->GetNumberOfCells() != 0) {
			 myVtkUnstructuredGrid->SetCells(VTK_QUADRATIC_TETRA, cellArrayQuadTed);
		 }

}

void HMeshToVtkUnstructuredGrid::setScalarFieldAtNodes(std::vector<double> _scalarField) {
	int numPts = myVtkUnstructuredGrid->GetPoints()->GetNumberOfPoints();
	vtkSmartPointer<vtkFloatArray> scalarField = vtkSmartPointer<vtkFloatArray>::New();
	scalarField->SetNumberOfValues(numPts);
	for (int i = 0; i < numPts; ++i)
	{
		scalarField->SetValue(i, _scalarField[i]);
	}
	myVtkUnstructuredGrid->GetPointData()->SetScalars(scalarField);
}

void HMeshToVtkUnstructuredGrid::setVectorFieldAtNodes(std::vector<double> _x, std::vector<double> _y, std::vector<double> _z) {

	int numPts = myVtkUnstructuredGrid->GetPoints()->GetNumberOfPoints();
	vtkSmartPointer<vtkFloatArray> vectorField = vtkSmartPointer<vtkFloatArray>::New();
	vectorField->SetNumberOfComponents(3);
	vectorField->SetName("warpData");

	float vec[3] = { 0.0, 0.0, 0.0 };
	for (int i = 0; i < numPts; ++i)
	{
		vec[0] = _x[i];
		vec[1] = _y[i];
		vec[2] = _z[i];
		//std::cout << "DefoWeg " << vec[0] << " : "<< vec[1] << " : " << vec[2] << std::endl;
		vectorField->InsertNextTuple(vec);
	}

	myVtkUnstructuredGrid->GetPointData()->AddArray(vectorField);
	myVtkUnstructuredGrid->GetPointData()->SetActiveVectors(vectorField->GetName());
}