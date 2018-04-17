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
#include "FieldDataVisualizer.h"
#include "OutputDatabase.h"
#include "HMesh.h"
#include "VtkAnimator.h"
#include "VtkViewer.h"
#include "VisualizerSetting.h"
#include "VectorFieldResults.h"

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
#include <vtkAnimationScene.h>
#include <vtkAnimationCue.h>
#include <vtkCommand.h>

//QT5
#include <QInputEvent>

#define _USE_MATH_DEFINES
#include <math.h>

FieldDataVisualizer::FieldDataVisualizer(QWidget* parent): QVTKOpenGLWidget(parent){
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
	edgeExtractor = vtkExtractEdges::New();
	edgeMapper = vtkPolyDataMapper::New();
	warpFilter = vtkWarpVector::New();
	hueLut = vtkLookupTable::New();

	//Properties
	isAnimationSceneInstantiated = false;

	myVtkViewer = new VtkViewer(*this);

	setPickerModeNone();
	setViewMode(true);
	myHarmonicScale = 1;


	myVtkAnimatorActive = false;
}

void FieldDataVisualizer::myHMeshToVtkUnstructuredGridInitializer(){
	myHMeshToVtkUnstructuredGrid = new HMeshToVtkUnstructuredGrid(*myHMesh);
}

void FieldDataVisualizer::myHMeshToVtkUnstructuredGridSetScalar(STACCATO_VectorField_components _type, int _index) {
	myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(_type, _index));
}

void FieldDataVisualizer::myHMeshToVtkUnstructuredGridSetScaledScalar(STACCATO_VectorField_components _type, int _index) {
	myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScaledScalarFieldAtNodes(_type, _index, myHarmonicScale));
}

void FieldDataVisualizer::myHMeshToVtkUnstructuredGridSetVector(int _index) {
	myHMeshToVtkUnstructuredGrid->setVectorFieldAtNodes(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(STACCATO_x_Re, _index), myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultScalarFieldAtNodes(STACCATO_y_Re, _index), myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultScalarFieldAtNodes(STACCATO_z_Re, _index));
}

void FieldDataVisualizer::plotVectorField() {
	myVtkViewer->plotVectorField();
}

void FieldDataVisualizer::zoomToExtent()
{
	// Zoom to extent of last added actor
	vtkSmartPointer<vtkActor> actor = myRenderer->GetActors()->GetLastActor();
	if (actor != nullptr)
	{
		myRenderer->ResetCamera(actor->GetBounds());
	}

}

void FieldDataVisualizer::setBackgroundGradient(int r, int g, int b)
{
		float R1 = r / 255.;
		float G1 = g / 255.;
		float B1 = b / 255.;
		float fu = 2.;
		float fd = 0.2;

		myRenderer->SetBackground(R1*fd > 1 ? 1. : R1*fd, G1*fd > 1 ? 1. : G1*fd, B1*fd > 1 ? 1. : B1*fd);
		myRenderer->SetBackground2(R1*fu > 1 ? 1. : R1*fu, G1*fu > 1 ? 1. : G1*fu, B1*fu > 1 ? 1. : B1*fu);
}

void FieldDataVisualizer::displayCompass(void) {
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


void FieldDataVisualizer::mousePressEvent(QMouseEvent * 	_event) {
	
		// The button mappings can be used as a mask. This code prevents conflicts
		// when more than one button pressed simultaneously.
		if (_event->button() & Qt::LeftButton) {
			if ((!myRotateMode && myCurrentPickerType != STACCATO_Picker_None) && myVisualizerSetting->PROPERTY_RESULTS_AVALABLE) {
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
				if (myVisualizerSetting->myFieldDataSetting->getEdgeProperty())
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

				if (myCurrentPickerType != STACCATO_Picker_None) {

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

						if (observers.size() != 0) {
							this->notify();
						}
					}
				}
				// Attach Model View Properties to Renderer
				if (myVisualizerSetting->myFieldDataSetting->getSurfaceProperty()) {
					mySelectedActor->SetMapper(mySelectedMapper);
					myRenderer->AddActor(mySelectedActor);
				}
				else
					myRenderer->RemoveActor(mySelectedActor);

				if (myVisualizerSetting->myFieldDataSetting->getEdgeProperty())
					myRenderer->AddActor(myEdgeActor);

				myRenderer->GetRenderWindow()->Render();
			}			
		}
		else if (_event->button() & Qt::RightButton) {

		}
		else if (_event->button() & Qt::MidButton) {

		}
		QVTKOpenGLWidget::mouseReleaseEvent(_event);
	}


void FieldDataVisualizer::setPickerMode(STACCATO_Picker_type _currentPickerType) {
	myCurrentPickerType = _currentPickerType;
}

void FieldDataVisualizer::setViewMode(bool _rotate) {
	this->myPickMode = !_rotate;
	this->myRotateMode = _rotate;
}

std::vector<int> FieldDataVisualizer::getSelection(STACCATO_Picker_type _type) {
	if (_type == STACCATO_Picker_Node)
		return mySelectedNodes;
	else if (_type == STACCATO_Picker_Element)
		return mySelectedElements;
}

void FieldDataVisualizer::animate(STACCATO_VectorField_components _type, std::vector<int>& _animationIndices, bool _isHarmonic) {
	if (!myVtkAnimatorActive) {
		myVtkAnimator = new VtkAnimator(*this);
		myVtkAnimatorActive = true;
	}
	myVtkAnimator->bufferAnimationFrames(_type, _animationIndices, _isHarmonic);
	isAnimationSceneInstantiated = false;
}

void FieldDataVisualizer::plotVectorFieldAtIndex(int _index) {
	myVtkAnimator->plotVectorFieldAtIndex(_index);
	getRenderer()->Render();
	GetRenderWindow()->Render();
}

void FieldDataVisualizer::myAnimationScenePlayProc(int _duration, int _loops) {
	if (!isAnimationSceneInstantiated) {
		myVtkAnimator->instantiateAnimationScene(_duration, _loops);
		isAnimationSceneInstantiated = true;
	}
	if (myVtkAnimator->myFrameID.size() > 0)
		myVtkAnimator->playAnimationScene(_duration, _loops);
}

void FieldDataVisualizer::myAnimationSceneStopProc(){
	if (myVtkAnimatorActive)
		myVtkAnimator->stopAnimationScene();
}

void FieldDataVisualizer::setActiveMapper(vtkSmartPointer<vtkDataSetMapper>& _mapper) {
	_mapper->SetInputData(myHMeshToVtkUnstructuredGrid->getVtkUnstructuredGrid());
	_mapper->ScalarVisibilityOn();
	_mapper->SetScalarModeToUsePointData();
	_mapper->SetColorModeToMapScalars();
}

void FieldDataVisualizer::setActiveActor(vtkSmartPointer<vtkActor>& _actor) {
	_actor->SetMapper(mySelectedMapper);
	// Set the color for edges of the actor
	_actor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0); //(R,G,B)
	_actor->GetProperty()->EdgeVisibilityOff();
}

void FieldDataVisualizer::setActiveWarpFilter(vtkSmartPointer<vtkWarpVector>& _warpFilter) {
	_warpFilter->SetInputData(myHMeshToVtkUnstructuredGrid->getVtkUnstructuredGrid());
	_warpFilter->SetScaleFactor(myVisualizerSetting->PROPERTY_SCALING_FACTOR * myHarmonicScale);
	_warpFilter->Update();
	mySelectedMapper->SetInputData(_warpFilter->GetUnstructuredGridOutput());
}

void FieldDataVisualizer::setActiveHueLut(vtkSmartPointer<vtkLookupTable>& _hueLut) {
	double scalarRange[2];
	myHMeshToVtkUnstructuredGrid->getVtkUnstructuredGrid()->GetPointData()->GetScalars()->GetRange(scalarRange);
	mySelectedMapper->UseLookupTableScalarRangeOn();
	// Create a lookup table to share between the mapper and the scalarbar
	_hueLut->SetTableRange(scalarRange[0], scalarRange[1]);
	_hueLut->SetHueRange(0.667, 0.0);
	_hueLut->SetValueRange(1, 1);
	_hueLut->Build();

	mySelectedMapper->SetLookupTable(_hueLut);
}

void FieldDataVisualizer::setActiveScalarBarWidget() {
	myScalarBarWidget->SetResizable(true);
	myScalarBarWidget->SetInteractor(GetRenderWindow()->GetInteractor());
	myScalarBarWidget->GetScalarBarActor()->SetTitle(myVisualizerSetting->PROPERTY_SCALARBAR_TITLE);
	myScalarBarWidget->GetScalarBarActor()->SetMaximumWidthInPixels(100);
	myScalarBarWidget->GetScalarBarActor()->SetNumberOfLabels(4);
	myScalarBarWidget->GetScalarBarActor()->SetLookupTable(hueLut);
	myScalarBarWidget->EnabledOn();
}

void FieldDataVisualizer::setActiveEdgeActor(vtkSmartPointer<vtkActor>& _edgeActor) {
	//Edge vis
	vtkSmartPointer<vtkExtractEdges> edgeExtractorTemp = vtkExtractEdges::New();
	edgeExtractorTemp->SetInputData(warpFilter->GetUnstructuredGridOutput());
	vtkSmartPointer<vtkPolyDataMapper> edgeMapperTemp = vtkPolyDataMapper::New();
	edgeMapperTemp->SetInputConnection(edgeExtractorTemp->GetOutputPort());
	_edgeActor->SetMapper(edgeMapperTemp);
	_edgeActor->GetProperty()->SetColor(0., 0., 0.);
	_edgeActor->GetProperty()->SetLineWidth(3);
	edgeMapperTemp->ScalarVisibilityOff();
}

void FieldDataVisualizer::enableActiveEdgeActor(bool _enable) {
	if (_enable) {
		getRenderer()->AddActor(myEdgeActor);
	}
	else
		getRenderer()->RemoveActor(myEdgeActor);
}

void FieldDataVisualizer::enableActiveActor(bool _enable) {
	if (_enable)
		getRenderer()->AddActor(mySelectedActor);
	else
		getRenderer()->RemoveActor(mySelectedActor);
}

void FieldDataVisualizer::enableActiveScalarBarWidget(bool _enable) {
	if (_enable)
		getRenderer()->AddActor2D(myScalarBarWidget->GetScalarBarActor());
	else
		getRenderer()->RemoveActor2D(myScalarBarWidget->GetScalarBarActor());
}

void FieldDataVisualizer::setNewMapper(vtkSmartPointer<vtkDataSetMapper>& _newMapper) {
	// Set New Mapper to Actor
	mySelectedActor->SetMapper(_newMapper);

	// Assuming New WarpFilter is already Assigned for Update
	mySelectedMapper->SetInputData(warpFilter->GetUnstructuredGridOutput());
	mySelectedMapper->UseLookupTableScalarRangeOn();
	// Assuming New HueLut is already Assigned for Update
	mySelectedMapper->SetLookupTable(hueLut);
}

void FieldDataVisualizer::connectVisualizerSetting(VisualizerSetting* _setting) {
	myVisualizerSetting = _setting;
}

void FieldDataVisualizer::setHarmonicScaleAtStep(STACCATO_VectorField_components _type, int _step, int _totalSteps) {
	if (_type == STACCATO_x_Re || _type == STACCATO_y_Re || _type == STACCATO_z_Re || _type == STACCATO_Magnitude_Re)		// Real Part
		myHarmonicScale = cos(_step * 2 * M_PI / (_totalSteps - 1));
	else if (_type == STACCATO_x_Im || _type == STACCATO_y_Im || _type == STACCATO_z_Im || _type == STACCATO_Magnitude_Im)	// Imaginary Part
		myHarmonicScale = sin(_step * 2 * M_PI / (_totalSteps - 1));
	std::cout << ">> Scaling Factor: " << myHarmonicScale << std::endl;
}