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

#include "Timer.h"
#include "MemWatcher.h"

#include "VisualizerWindow.h"
#include "HMesh.h"

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
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkScalarBarActor.h>
#include <vtkScalarBarWidget.h>
#include <vtkLookupTable.h>
#include <vtkRenderWindow.h>
#include <vtkWarpVector.h>
#include <vtkPolyDataMapper.h>
#include <vtkExtractEdges.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkPointPicker.h>
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
	myScalarBarWidget = vtkSmartPointer<vtkScalarBarWidget>::New();
	selectedPickActor = vtkActor::New();
	//mySelectedProperty = vtkSmartPointer<vtkProperty>::New();
	myEdgeActor = vtkActor::New();

	//Properties
	myEdgeVisibility = false;
	mySurfaceVisiiblity = true;
	myScalarBarVisibility = true;
	myInteractivePickerVM = false;
	myTitle = "u_x";
	myScaleFactor = 1;
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
		if (_event->button() & Qt::LeftButton && !myRotateMode) {
			// Some Picker VtkObjects
			vtkSmartPointer<vtkDataSetMapper> selectedPickMapper = vtkDataSetMapper::New();

			// Remove selectionPickActor
			static bool mySelectedActorActive = false;
			if (mySelectedActorActive) {
				myRenderer->RemoveActor(selectedPickActor);
				myRenderer->GetRenderWindow()->Render();
			}

			// Get the location of the click (in window coordinates)
			int* pos = GetRenderWindow()->GetInteractor()->GetEventPosition();

			vtkSmartPointer<vtkPointPicker> pointPicker = vtkSmartPointer<vtkPointPicker>::New();
			vtkSmartPointer<vtkCellPicker> cellPicker = vtkSmartPointer<vtkCellPicker>::New();
			pointPicker->SetTolerance(0.05);
			cellPicker->SetTolerance(1e-6);

			// Dettach Renderer from other actors other than Surface
			// Remove edgeActor
			if (myEdgeVisibility)
				myRenderer->RemoveActor(myEdgeActor);
			// Enable Surface Actor
			myRenderer->AddActor(mySelectedActor);

			// Pick from this location.
			pointPicker->Pick(pos[0], pos[1], 0, myRenderer);
			cellPicker->Pick(pos[0], pos[1], 0, myRenderer);

			double* worldPosition = pointPicker->GetPickPosition();
			std::cout << "Element id is: " << cellPicker->GetCellId() << std::endl;
			std::cout << "Node id is: " << pointPicker->GetPointId() << std::endl;

			// Store the Pick
			mySelectedNodes.push_back(pointPicker->GetPointId());
			mySelectedElements.push_back(cellPicker->GetCellId());

			if (myCurrentPickerType!=STACCATO_Picker_None) {

				if (pointPicker->GetPointId() != -1 && cellPicker->GetCellId() != -1)
				{

					std::cout << "Pick position is: " << worldPosition[0] << " " << worldPosition[1]
						<< " " << worldPosition[2] << endl;

					vtkSmartPointer<vtkIdTypeArray> ids =
						vtkSmartPointer<vtkIdTypeArray>::New();
					ids->SetNumberOfComponents(1);

					if (myCurrentPickerType == STACCATO_Picker_Element) {
						ids->InsertNextValue(cellPicker->GetCellId());
						vtkSmartPointer<vtkIdList> ids_points = vtkIdList::New();
					}
					else if (myCurrentPickerType == STACCATO_Picker_Node) {
						ids->InsertNextValue(pointPicker->GetPointId());
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

					selectedPickMapper->SetInputData(selected);
					selectedPickActor->SetMapper(selectedPickMapper);
					if (myCurrentPickerType == STACCATO_Picker_Element) {
						selectedPickActor->GetProperty()->SetColor(1, 0, 0);
						selectedPickActor->GetProperty()->SetPointSize(8.0);
						selectedPickActor->GetProperty()->EdgeVisibilityOff();
						selectedPickActor->GetProperty()->SetEdgeColor(0, 0, 1);
						selectedPickActor->GetProperty()->SetLineWidth(4.0);
						selectedPickMapper->ScalarVisibilityOff();
					}
					else if (myCurrentPickerType == STACCATO_Picker_Node) {
						selectedPickActor->GetProperty()->SetColor(0, 0, 1);
						selectedPickActor->GetProperty()->SetPointSize(10.0);
						selectedPickMapper->ScalarVisibilityOff();
					}
					myRenderer->AddActor(selectedPickActor);

					mySelectedActorActive = true;	

					if(myInteractivePickerVM)
						myUpdateVisualizerWindow();
				}
			}
			// Attach Model View Properties to Renderer
			if (mySurfaceVisiiblity) {
				mySelectedActor->SetMapper(mySelectedMapper);
				myRenderer->AddActor(mySelectedActor);
			}
			else
				myRenderer->RemoveActor(mySelectedActor);

			if (myEdgeVisibility)
				myRenderer->AddActor(myEdgeActor);
			myRenderer->GetRenderWindow()->Render();
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

void VtkViewer::plotVectorField(vtkSmartPointer<vtkUnstructuredGrid>& _vtkUnstructuredGrid) {
	// Reset the Renderer
	myRenderer->RemoveAllViewProps();

	mySelectedMapper->SetInputData(_vtkUnstructuredGrid);

	mySelectedMapper->ScalarVisibilityOn();
	mySelectedMapper->SetScalarModeToUsePointData();
	mySelectedMapper->SetColorModeToMapScalars();

	mySelectedActor->SetMapper(mySelectedMapper);

	double scalarRange[2];
	_vtkUnstructuredGrid->GetPointData()->GetScalars()->GetRange(scalarRange);

	// Set the color for edges of the sphere
	mySelectedActor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0); //(R,G,B)
	mySelectedActor->GetProperty()->EdgeVisibilityOff();

	vtkSmartPointer<vtkWarpVector> warpFilter = vtkWarpVector::New();
	warpFilter->SetInputData(_vtkUnstructuredGrid);
	warpFilter->SetScaleFactor(myScaleFactor);
	warpFilter->Update();
	mySelectedMapper->SetInputData(warpFilter->GetUnstructuredGridOutput());

	mySelectedMapper->UseLookupTableScalarRangeOn();
	// Create a lookup table to share between the mapper and the scalarbar
	vtkSmartPointer<vtkLookupTable> hueLut = vtkLookupTable::New();
	hueLut->SetTableRange(scalarRange[0], scalarRange[1]);
	hueLut->SetHueRange(0.667, 0.0);
	hueLut->SetValueRange(1, 1);
	hueLut->Build();

	mySelectedMapper->SetLookupTable(hueLut);

	myScalarBarWidget->SetResizable(true);
	myScalarBarWidget->SetInteractor(GetRenderWindow()->GetInteractor());
	myScalarBarWidget->GetScalarBarActor()->SetTitle(myTitle);
	myScalarBarWidget->GetScalarBarActor()->SetNumberOfLabels(4);
	myScalarBarWidget->GetScalarBarActor()->SetLookupTable(hueLut);
	myScalarBarWidget->EnabledOn();

	this->getRenderer()->AddActor(myEdgeActor);

	if (myEdgeVisibility) {
		//Edge vis
		vtkSmartPointer<vtkExtractEdges> edgeExtractor = vtkExtractEdges::New();
		edgeExtractor->SetInputData(warpFilter->GetUnstructuredGridOutput());
		vtkSmartPointer<vtkPolyDataMapper> edgeMapper = vtkPolyDataMapper::New();
		edgeMapper->SetInputConnection(edgeExtractor->GetOutputPort());
		myEdgeActor->SetMapper(edgeMapper);
		myEdgeActor->GetProperty()->SetColor(0., 0., 0.);
		myEdgeActor->GetProperty()->SetLineWidth(3);
		edgeMapper->ScalarVisibilityOff();
	}
	else
		myRenderer->RemoveActor(myEdgeActor);
		
	if (!mySurfaceVisiiblity)
		this->getRenderer()->RemoveActor(mySelectedActor);
	else
		this->getRenderer()->AddActor(mySelectedActor);
	
	if (myScalarBarVisibility)
		this->getRenderer()->AddActor2D(myScalarBarWidget->GetScalarBarActor());
	else
		this->getRenderer()->RemoveActor2D(myScalarBarWidget->GetScalarBarActor());

	static bool resetCamera = true;
	if (resetCamera) {
		this->getRenderer()->ResetCamera();
		resetCamera = false;
	}
	this->GetRenderWindow()->Render();

	vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkXMLUnstructuredGridWriter::New();
	writer->SetFileName("3dTestGrid");
	writer->SetInputData(_vtkUnstructuredGrid);
	writer->Write();
	anaysisTimer01.stop();
	//debugOut << "Duration for display Hmesh and results: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	//debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
}

void VtkViewer::setDisplayProperties(STACCATO_Result_type _type, bool _edge, bool _surface, bool _scalarBar) {
	if (_type == STACCATO_Ux_Re) {
		this->myTitle = "Ux_Re";
	}
	else if (_type == STACCATO_Uy_Re) {
		this->myTitle = "Uy_Re";
	}
	else if (_type == STACCATO_Uz_Re) {
		this->myTitle = "Uz_Re";
	}
	else if (_type == STACCATO_Ux_Im) {
		this->myTitle = "Ux_Im";
	}
	else if (_type == STACCATO_Uy_Im) {
		this->myTitle = "Uy_Im";
	}
	else if (_type == STACCATO_Uz_Im) {
		this->myTitle = "Uz_Im";
	}
	else if (_type == STACCATO_Magnitude_Re) {
		this->myTitle = "UMag_Re";
	}
	else if (_type == STACCATO_Magnitude_Im) {
		this->myTitle = "UMag_Im";
	}

	this->myEdgeVisibility = _edge;
	this->mySurfaceVisiiblity = _surface;
	this->myScalarBarVisibility = _scalarBar;
}

void VtkViewer::setScalingFactor(double _factor) {
	this->myScaleFactor = _factor;
	cout << "Scaling Factor changed to " << this->myScaleFactor << endl;
}

void VtkViewer::setViewMode(bool _rotate) {
	this->myPickMode = !_rotate;
	this->myRotateMode = _rotate;
}

std::vector<int> VtkViewer::getSelection(STACCATO_Picker_type _type) {
	if (_type == STACCATO_Picker_Node)
		return mySelectedNodes;
	else if (_type == STACCATO_Picker_Element)
		return mySelectedElements;
}

void VtkViewer::my2dVisualizerInterface(HMesh& _hMesh) {
	VW = new VisualizerWindow(_hMesh);
	VW->show();
	myInteractivePickerVM=true;
}

void VtkViewer::myUpdateVisualizerWindow(){
	if (myCurrentPickerType == STACCATO_Picker_Node)
		VW->setSelection(mySelectedNodes);
	else if (myCurrentPickerType == STACCATO_Picker_Element)
		VW->setSelection(mySelectedElements);
}
