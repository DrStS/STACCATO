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

VtkAnimator::VtkAnimator(FieldDataVisualizer& _fieldDataVisualizer) : myFieldDataVisualizer(&_fieldDataVisualizer)
{
	int sizeOfArray = myFieldDataVisualizer->getHMesh()->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().size();
	myArrayActor = new vtkSmartPointer<vtkActor>[sizeOfArray];
	myArrayMapper = new vtkSmartPointer<vtkDataSetMapper>[sizeOfArray];
	warpFilterArray = new vtkSmartPointer<vtkWarpVector>[sizeOfArray];
	hueLutArray = new vtkSmartPointer<vtkLookupTable>[sizeOfArray];
	myArrayEdgeActor = new vtkSmartPointer<vtkActor>[sizeOfArray];

	// Array Initialization
	for (int i = 0; i <  myFieldDataVisualizer->getHMesh()->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().size(); i++) {
		myArrayActor[i] = vtkSmartPointer<vtkActor>::New(); 
		myArrayMapper[i] = vtkSmartPointer<vtkDataSetMapper>::New();
		warpFilterArray[i] = vtkSmartPointer<vtkWarpVector>::New();
		hueLutArray[i] = vtkSmartPointer<vtkLookupTable>::New();
		myArrayEdgeActor[i] = vtkSmartPointer<vtkActor>::New();
	}

	// Create an Animation Field
	myAnimationCue = vtkSmartPointer<vtkAnimationCue>::New();
	myCueAnimator = new CueAnimator(*myFieldDataVisualizer->myHMeshToVtkUnstructuredGrid, *myFieldDataVisualizer->getHMesh(), *this);
	myAnimationCueObserver = vtkSmartPointer<AnimationCueObserver>::New();
	myAnimationScene = vtkSmartPointer<vtkAnimationScene>::New();
}

VtkAnimator ::~VtkAnimator()
{
}

void VtkAnimator::bufferAnimationFrames(STACCATO_VectorField_components _type) {
	std::cout << ">> Generating frames and Buffering...\n";
	clearAnimationScene();

	for (int i = 0; i <  myFieldDataVisualizer->getHMesh()->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().size(); i++)
	{
		int resultIndex = i*myFieldDataVisualizer->getHMesh()->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsTimeDescription().size() + 0;

		myFieldDataVisualizer->myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myFieldDataVisualizer->getHMesh()->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultScalarFieldAtNodes(_type, resultIndex));
		myFieldDataVisualizer->myHMeshToVtkUnstructuredGrid->setVectorFieldAtNodes(myFieldDataVisualizer->getHMesh()->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultScalarFieldAtNodes(STACCATO_x_Re, resultIndex), myFieldDataVisualizer->getHMesh()->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultScalarFieldAtNodes(STACCATO_y_Re, resultIndex), myFieldDataVisualizer->getHMesh()->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultScalarFieldAtNodes(STACCATO_z_Re, resultIndex));

		// Assign Properties
		myFieldDataVisualizer->setActiveMapper(myArrayMapper[i]);
		myFieldDataVisualizer->setActiveActor(myArrayActor[i]);
		myFieldDataVisualizer->setActiveWarpFilter(warpFilterArray[i]);
		myFieldDataVisualizer->warpFilter = warpFilterArray[i];
		myFieldDataVisualizer->setActiveHueLut(hueLutArray[i]);
		myFieldDataVisualizer->setActiveEdgeActor(myArrayEdgeActor[i]);
		
		// Buffer
		myFieldDataVisualizer->plotVectorFieldAtIndex(i);
	}
}

void VtkAnimator::plotVectorFieldAtIndex(int _index) {
	myFieldDataVisualizer->getRenderer()->RemoveActor(myFieldDataVisualizer->mySelectedActor);

	// Reset the Renderer
	myFieldDataVisualizer->getRenderer()->RemoveAllViewProps();

	myFieldDataVisualizer->mySelectedActor = myArrayActor[_index];
	myFieldDataVisualizer->mySelectedMapper = myArrayMapper[_index];
	myFieldDataVisualizer->myEdgeActor = myArrayEdgeActor[_index];
	myFieldDataVisualizer->warpFilter = warpFilterArray[_index];
	myFieldDataVisualizer->hueLut = hueLutArray[_index];

	myFieldDataVisualizer->setNewMapper(myArrayMapper[_index]);
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