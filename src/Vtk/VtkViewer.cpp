/*  Copyright &copy; 2016, Stefan Sicklinger, Munich
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
#include <VtkViewer.h>

//VTK
#include <vtkCamera.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkDataSetMapper.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkAxesActor.h>
#include <vtkRenderer.h>
#include <vtkCellPicker.h>
#include <vtkActor.h>
#include <vtkIdTypeArray.h>
#include <vtkSelectionNode.h>
#include <vtkSelection.h>
#include <vtkExtractSelection.h>
#include <vtkUnstructuredGrid.h>
//QT5
#include <QInputEvent>

VtkViewer::VtkViewer(QWidget* parent): QVTKOpenGLWidget(parent){
	
	vtkNew<vtkGenericOpenGLRenderWindow> window;
	SetRenderWindow(window.Get());

	// Camera
	vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
	camera->SetViewUp(0, 1, 0);
	camera->SetPosition(0, 0, 10);
	camera->SetFocalPoint(0, 0, 0);

	// Renderer
	myRenderer = vtkSmartPointer<vtkRenderer>::New();
	myRenderer->SetActiveCamera(camera);

	// Setup the background gradient
	myRenderer->GradientBackgroundOn();
	myBGColor = QColor(255, 235, 100);
	setBackgroundGradient(myBGColor.red(), myBGColor.green(), myBGColor.blue());
	GetRenderWindow()->AddRenderer(myRenderer);

	//Draw compass
	displayCompass();

	// Some members
	mySelectedMapper = vtkSmartPointer<vtkDataSetMapper>::New();
	mySelectedActor = vtkSmartPointer<vtkActor>::New();
	//mySelectedProperty = vtkSmartPointer<vtkProperty>::New();
}

void VtkViewer::zoomToExtent()
{
	// Zoom to extent of last added actor
	vtkSmartPointer<vtkActor> actor = myRenderer->GetActors()->GetLastActor();
	if (actor != nullptr)
	{
		myRenderer->ResetCamera(actor->GetBounds());
	}

}

void VtkViewer::setBackgroundGradient(int r, int g, int b)
{
		float R1 = r / 255.;
		float G1 = g / 255.;
		float B1 = b / 255.;
		float fu = 2.;
		float fd = 0.2;

		myRenderer->SetBackground(R1*fd > 1 ? 1. : R1*fd, G1*fd > 1 ? 1. : G1*fd, B1*fd > 1 ? 1. : B1*fd);
		myRenderer->SetBackground2(R1*fu > 1 ? 1. : R1*fu, G1*fu > 1 ? 1. : G1*fu, B1*fu > 1 ? 1. : B1*fu);
}



void VtkViewer::displayCompass(void) {
	vtkSmartPointer<vtkAxesActor> axes =
		vtkSmartPointer<vtkAxesActor>::New();
	myOrientationMarkerWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
	myOrientationMarkerWidget->SetOutlineColor(0.9300, 0.5700, 0.1300);
	myOrientationMarkerWidget->SetOrientationMarker(axes);
	myOrientationMarkerWidget->SetInteractor(GetRenderWindow()->GetInteractor());
	myOrientationMarkerWidget->SetViewport(0.0, 0.0, 0.4, 0.4);
	myOrientationMarkerWidget->SetEnabled(1);
	myOrientationMarkerWidget->InteractiveOff();
	myRenderer->ResetCamera();
	myRenderer->GetRenderWindow()->Render();
}


void VtkViewer::mousePressEvent(QMouseEvent * 	_event) {

	// The button mappings can be used as a mask. This code prevents conflicts
	// when more than one button pressed simultaneously.

	// Remove selectionActor
	static bool mySelectedActorActive = false;
	if (mySelectedActorActive) {
		myRenderer->RemoveActor(mySelectedActor);
	}

	if (_event->button() & Qt::LeftButton) {

		// Get the location of the click (in window coordinates)
		int* pos = GetRenderWindow()->GetInteractor()->GetEventPosition();

		vtkSmartPointer<vtkCellPicker> picker =
			vtkSmartPointer<vtkCellPicker>::New();
		picker->SetTolerance(0.005);

		// Pick from this location.
		picker->Pick(pos[0], pos[1], 0, myRenderer);

		double* worldPosition = picker->GetPickPosition();
		std::cout << "Element id is: " << picker->GetCellId() << std::endl;
		std::cout << "Node id is: " << picker->GetPointId() << std::endl;

		if (picker->GetCellId() != -1)
		{

			std::cout << "Pick position is: " << worldPosition[0] << " " << worldPosition[1]
				<< " " << worldPosition[2] << endl;

			vtkSmartPointer<vtkIdTypeArray> ids =
				vtkSmartPointer<vtkIdTypeArray>::New();
			ids->SetNumberOfComponents(1);

			if (myCurrentPickerType == STACCATO_Picker_Element) {
				ids->InsertNextValue(picker->GetCellId());
			}
			else if (myCurrentPickerType == STACCATO_Picker_Node) {
				ids->InsertNextValue(picker->GetPointId());
			}
			
			vtkSmartPointer<vtkSelectionNode> selectionNode =
				vtkSmartPointer<vtkSelectionNode>::New();
			if (myCurrentPickerType == STACCATO_Picker_Element) {
				selectionNode->SetFieldType(vtkSelectionNode::CELL);
			}
			else if (myCurrentPickerType == STACCATO_Picker_Node) {
				selectionNode->SetFieldType(vtkSelectionNode::POINT);
			}
			selectionNode->SetContentType(vtkSelectionNode::INDICES);
			selectionNode->SetSelectionList(ids);

			vtkSmartPointer<vtkSelection> selection =
				vtkSmartPointer<vtkSelection>::New();
			selection->AddNode(selectionNode);

			vtkSmartPointer<vtkExtractSelection> extractSelection =
				vtkSmartPointer<vtkExtractSelection>::New();

			extractSelection->SetInputData(0, myRenderer->GetActors()->GetLastActor()->GetMapper()->GetInput());
			extractSelection->SetInputData(1, selection);
			extractSelection->Update();

			// In selection
			vtkSmartPointer<vtkUnstructuredGrid> selected =
				vtkSmartPointer<vtkUnstructuredGrid>::New();
			selected->ShallowCopy(extractSelection->GetOutput());

			std::cout << "There are " << selected->GetNumberOfPoints()
				<< " points in the selection." << std::endl;
			std::cout << "There are " << selected->GetNumberOfCells()
				<< " cells in the selection." << std::endl;

			mySelectedMapper->SetInputData(selected);
			mySelectedActor->SetMapper(mySelectedMapper);
			if (myCurrentPickerType == STACCATO_Picker_Element) {
				mySelectedActor->GetProperty()->SetColor(0.5, 0.5, 0.5);
				mySelectedActor->GetProperty()->EdgeVisibilityOn();
				mySelectedActor->GetProperty()->SetEdgeColor(0, 0, 1);
				mySelectedActor->GetProperty()->SetLineWidth(3);
				mySelectedMapper->ScalarVisibilityOff();
			}
			else if (myCurrentPickerType == STACCATO_Picker_Node) {
				mySelectedActor->GetProperty()->SetColor(0, 0, 1);
				mySelectedActor->GetProperty()->SetPointSize(8.0);
				mySelectedMapper->ScalarVisibilityOff();
			}
			myRenderer->AddActor(mySelectedActor);
			mySelectedActorActive = true;
			myRenderer->GetRenderWindow()->Render();
		}


	}
	else if (_event->button() & Qt::RightButton) {

	}
	else if (_event->button() & Qt::MidButton) {

	}
	QVTKOpenGLWidget::mouseReleaseEvent(_event);
	
}

void VtkViewer::setPickerMode(STACCATO_Picker_type _currentPickerType) {
	myCurrentPickerType = _currentPickerType;
}