/*  Copyright &copy; 2018, Stefan Sicklinger, Munich
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

#include "VtkAnimator.h"
#include "AnimationCueObserver.h"
#include "CueAnimator.h"

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

#include <VisualizerSetting.h>

#include <qprogressdialog.h>

#define _USE_MATH_DEFINES
#include <math.h>

VtkAnimator::VtkAnimator(FieldDataVisualizer& _fieldDataVisualizer) : myFieldDataVisualizer(&_fieldDataVisualizer)
{

	// Create an Animation Field
	myAnimationCue = vtkSmartPointer<vtkAnimationCue>::New();
	myCueAnimator = new CueAnimator(*myFieldDataVisualizer->myHMeshToVtkUnstructuredGrid, *myFieldDataVisualizer->getHMesh(), *this);
	myAnimationCueObserver = vtkSmartPointer<AnimationCueObserver>::New();
	myAnimationScene = vtkSmartPointer<vtkAnimationScene>::New();
}

VtkAnimator ::~VtkAnimator()
{
}

void VtkAnimator::bufferAnimationFrames(STACCATO_VectorField_components _type, std::vector<int>& _frameID) {
	myFrameID = _frameID;

	int sizeOfArray = myFrameID.size();
	myArrayActor = new vtkSmartPointer<vtkActor>[sizeOfArray];
	myArrayMapper = new vtkSmartPointer<vtkDataSetMapper>[sizeOfArray];
	warpFilterArray = new vtkSmartPointer<vtkWarpVector>[sizeOfArray];
	hueLutArray = new vtkSmartPointer<vtkLookupTable>[sizeOfArray];
	myArrayEdgeActor = new vtkSmartPointer<vtkActor>[sizeOfArray];

	// Array Initialization
	for (int i = 0; i < sizeOfArray; i++) {
		myArrayActor[i] = vtkSmartPointer<vtkActor>::New();
		myArrayMapper[i] = vtkSmartPointer<vtkDataSetMapper>::New();
		warpFilterArray[i] = vtkSmartPointer<vtkWarpVector>::New();
		hueLutArray[i] = vtkSmartPointer<vtkLookupTable>::New();
		myArrayEdgeActor[i] = vtkSmartPointer<vtkActor>::New();
	}

	std::cout << ">> Generating "<< myFrameID.size() <<" frames and Buffering...\n";

	std::cout << "Animation: ";
	for (int i = 0; i < _frameID.size(); i++) {
		std::cout << _frameID[i] << " << ";
	}
	std::cout << std::endl;

	QProgressDialog animationProgessDialog("Generating Animation...", "Cancel", 0, sizeOfArray);
	animationProgessDialog.setWindowModality(Qt::WindowModal);

	clearAnimationScene();

	for (int i = 0; i <  sizeOfArray; i++)
	{
		animationProgessDialog.setValue(i);

		myFieldDataVisualizer->myHMeshToVtkUnstructuredGridSetScalar(_type, myFrameID[i]);
		myFieldDataVisualizer->myHMeshToVtkUnstructuredGridSetVector(myFrameID[i]);

		// Assign Properties
		myFieldDataVisualizer->setActiveMapper(myArrayMapper[i]);
		myFieldDataVisualizer->setActiveActor(myArrayActor[i]);
		myFieldDataVisualizer->setActiveWarpFilter(warpFilterArray[i]);
		myFieldDataVisualizer->warpFilter = warpFilterArray[i];
		myFieldDataVisualizer->setActiveHueLut(hueLutArray[i]);
		myFieldDataVisualizer->setActiveEdgeActor(myArrayEdgeActor[i]);
		
		// Buffer
		plotVectorFieldAtIndex(i);

		if (animationProgessDialog.wasCanceled())
			break;
	}
	animationProgessDialog.setValue(sizeOfArray);
}

void VtkAnimator::bufferHarmonicAnimationFrames(STACCATO_VectorField_components _type, std::vector<int>& _frameID) {
	myFrameID = _frameID;

	int sizeOfArray = _frameID.size();
	myArrayActor = new vtkSmartPointer<vtkActor>[sizeOfArray];
	myArrayMapper = new vtkSmartPointer<vtkDataSetMapper>[sizeOfArray];
	warpFilterArray = new vtkSmartPointer<vtkWarpVector>[sizeOfArray];
	hueLutArray = new vtkSmartPointer<vtkLookupTable>[sizeOfArray];
	myArrayEdgeActor = new vtkSmartPointer<vtkActor>[sizeOfArray];

	// Array Initialization
	for (int i = 0; i < sizeOfArray; i++) {
		myArrayActor[i] = vtkSmartPointer<vtkActor>::New();
		myArrayMapper[i] = vtkSmartPointer<vtkDataSetMapper>::New();
		warpFilterArray[i] = vtkSmartPointer<vtkWarpVector>::New();
		hueLutArray[i] = vtkSmartPointer<vtkLookupTable>::New();
		myArrayEdgeActor[i] = vtkSmartPointer<vtkActor>::New();
	}

	std::cout << ">> Generating " << sizeOfArray << " frames and Buffering...\n";

	QProgressDialog animationProgessDialog("Generating Animation...", "Cancel", 0, sizeOfArray);
	animationProgessDialog.setWindowModality(Qt::WindowModal);

	clearAnimationScene();

	for (int i = 0; i < sizeOfArray; i++)
	{
		animationProgessDialog.setValue(i);

		if(_type == STACCATO_x_Re || _type == STACCATO_y_Re || _type == STACCATO_z_Re || _type == STACCATO_Magnitude_Re)  // Real Part
			myFieldDataVisualizer->myHarmonicScale = cos(i * 2 * M_PI / (sizeOfArray - 1));
		else if(_type == STACCATO_x_Im || _type == STACCATO_y_Im || _type == STACCATO_z_Im || _type == STACCATO_Magnitude_Im)
			myFieldDataVisualizer->myHarmonicScale = sin(i * 2 * M_PI / (sizeOfArray - 1));
		std::cout << ">> Scaling Factor: " << myFieldDataVisualizer->myHarmonicScale<< std::endl;

		myFieldDataVisualizer->myHMeshToVtkUnstructuredGridSetScaledScalar(_type, myFieldDataVisualizer->myVisualizerSetting->PROPERTY_FRAMEID);
		myFieldDataVisualizer->myHMeshToVtkUnstructuredGridSetVector(myFieldDataVisualizer->myVisualizerSetting->PROPERTY_FRAMEID);

		// Assign Properties
		myFieldDataVisualizer->setActiveMapper(myArrayMapper[i]);
		myFieldDataVisualizer->setActiveActor(myArrayActor[i]);

		myFieldDataVisualizer->setActiveWarpFilter(warpFilterArray[i]);
		myFieldDataVisualizer->warpFilter = warpFilterArray[i];
		myFieldDataVisualizer->setActiveHueLut(hueLutArray[i]);
		myFieldDataVisualizer->setActiveEdgeActor(myArrayEdgeActor[i]);

		// Buffer
		plotVectorFieldAtIndex(i);

		if (animationProgessDialog.wasCanceled())
			break;
	}
	animationProgessDialog.setValue(sizeOfArray);
	myFieldDataVisualizer->myHarmonicScale = 1;
}

void VtkAnimator::plotVectorFieldAtIndex(int _index) {
	int index = _index;

	myFieldDataVisualizer->getRenderer()->RemoveActor(myFieldDataVisualizer->mySelectedActor);

	// Reset the Renderer
	myFieldDataVisualizer->getRenderer()->RemoveAllViewProps();

	myFieldDataVisualizer->mySelectedActor = myArrayActor[index];
	myFieldDataVisualizer->mySelectedMapper = myArrayMapper[index];
	myFieldDataVisualizer->myEdgeActor = myArrayEdgeActor[index];
	myFieldDataVisualizer->warpFilter = warpFilterArray[index];
	myFieldDataVisualizer->hueLut = hueLutArray[index];

	myFieldDataVisualizer->setNewMapper(myArrayMapper[index]);
	myFieldDataVisualizer->setActiveScalarBarWidget();

	myFieldDataVisualizer->enableActiveEdgeActor(myFieldDataVisualizer->myVisualizerSetting->myFieldDataSetting->getEdgeProperty());
	myFieldDataVisualizer->enableActiveActor(myFieldDataVisualizer->myVisualizerSetting->myFieldDataSetting->getSurfaceProperty());
	myFieldDataVisualizer->enableActiveScalarBarWidget(myFieldDataVisualizer->myVisualizerSetting->PROPERTY_SCALARBAR_VISIBILITY);

	myFieldDataVisualizer->getRenderer()->Render();
}

void VtkAnimator::instantiateAnimationScene(int _duration, int _loops) {
	// Reset the Renderer
	myFieldDataVisualizer->getRenderer()->RemoveAllViewProps();
	myFieldDataVisualizer->getRenderer()->GetRenderWindow()->SetMultiSamples(0);
	myFieldDataVisualizer->getRenderer()->GetRenderWindow()->GetInteractor()->SetRenderWindow(myFieldDataVisualizer->getRenderer()->GetRenderWindow());
	myFieldDataVisualizer->getRenderer()->GetRenderWindow()->AddRenderer(myFieldDataVisualizer->getRenderer());
	myFieldDataVisualizer->getRenderer()->GetRenderWindow()->Render();

	clearAnimationScene();

	myAnimationScene->SetModeToRealTime();

	myAnimationScene->SetLoop(_loops);
	myAnimationScene->SetFrameRate(10);
	myAnimationScene->SetStartTime(0);
	myAnimationScene->SetEndTime(_duration);

	myAnimationCue->SetStartTime(0);
	myAnimationCue->SetEndTime(_duration);
	myAnimationScene->AddCue(myAnimationCue);

	myAnimationCueObserver->Renderer = myFieldDataVisualizer->getRenderer();
	myAnimationCueObserver->Animator = myCueAnimator;
	myAnimationCueObserver->RenWin = myFieldDataVisualizer->getRenderer()->GetRenderWindow();

	myAnimationCue->AddObserver(vtkCommand::StartAnimationCueEvent, myAnimationCueObserver);
	myAnimationCue->AddObserver(vtkCommand::EndAnimationCueEvent, myAnimationCueObserver);
	myAnimationCue->AddObserver(vtkCommand::AnimationCueTickEvent, myAnimationCueObserver);
}

void VtkAnimator::playAnimationScene(int _duration, int _loops) {
	myAnimationScene->Stop();
	myAnimationScene->SetStartTime(0);
	myAnimationScene->SetLoop(_loops);
	myAnimationScene->SetEndTime(_duration);
	myAnimationCue->SetStartTime(0);
	myAnimationCue->SetEndTime(_duration);
	myAnimationScene->SetAnimationTime(0);

	myAnimationScene->Play();
}

void VtkAnimator::stopAnimationScene() {
	if (myAnimationScene) {
		myAnimationScene->Stop();
	}
}

void VtkAnimator::clearAnimationScene(){
	stopAnimationScene();
	myAnimationScene->RemoveAllCues();
	myAnimationScene->RemoveAllObservers();
	myAnimationCue->RemoveAllObservers();
}

bool VtkAnimator::isAnimationScenePlaying() {
	return myAnimationScene->IsInPlay() == 1 ? true : false;
}