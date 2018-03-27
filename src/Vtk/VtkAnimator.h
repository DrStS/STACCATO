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
/*************************************************************************************************
* \file VtkAnimator.h
* This file holds the class VtkAnimator
* \date 2/19/2018
**************************************************************************************************/
#pragma once

#include "FieldDataVisualizer.h"
#include "HMesh.h"
#include "HMeshToVtkUnstructuredGrid.h"

#include <STACCATO_Enum.h>

#include <vtkSmartPointer.h>
#include <vtkAnimationScene.h>
#include <vtkUnstructuredGrid.h>

class vtkActor;
class vtkDataSetMapper;
class vtkWarpVector;
class vtkLookupTable;
class AnimationCueObserver;
class vtkAnimationCue;
class CueAnimator;

/*************************************************************************************************
* \brief Class VtkAnimator
**************************************************************************************************/

class VtkAnimator {
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Harikrishnan Sreekumar
	***********/
	VtkAnimator(FieldDataVisualizer& _fieldDataVisualizer);
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	~VtkAnimator();
	/***********************************************************************************************
	* \brief Buffer Animation Frames
	* \param[in] _type Result Type, for which the frames are buffered into memory
	* \author Harikrishnan Sreekumar
	***********/
	void bufferAnimationFrames(STACCATO_VectorField_components _type, std::vector<int> &_frameID, bool _isHarmonic);
	/***********************************************************************************************
	* \brief Plot Vector Field for an Index
	* \param[in] _index Index of Frame
	* \author Harikrishnan Sreekumar
	***********/
	void plotVectorFieldAtIndex(int _index);
	/***********************************************************************************************
	* \brief Play/Resume the Animation Scene
	* \param[in] _duration Duration of Scene
	* \param[in] _loops Enable (1)/Disable (0) Loop
	* \author Harikrishnan Sreekumar
	***********/
	void playAnimationScene(int _duration, int _loops);
	/***********************************************************************************************
	* \brief Instantiate the First Animation Scene
	* \param[in] _duration Duration of Scene
	* \param[in] _loops Enable (1)/Disable (0) Loop
	* \author Harikrishnan Sreekumar
	***********/
	void instantiateAnimationScene(int _duration, int _loops);
	/***********************************************************************************************
	* \brief Stop Animation Scene
	* \author Harikrishnan Sreekumar
	***********/
	void stopAnimationScene();
	/***********************************************************************************************
	* \brief Clear Animation Scene
	* \author Harikrishnan Sreekumar
	***********/
	void clearAnimationScene();
	/***********************************************************************************************
	* \brief Animation Scene Status of Playing or Stopped
	* \author Harikrishnan Sreekumar
	***********/
	bool isAnimationScenePlaying();

private:
	// Handle to Field Visualizer
	FieldDataVisualizer* myFieldDataVisualizer;

	// Animation Members
	vtkSmartPointer<vtkAnimationScene> myAnimationScene;
	vtkSmartPointer<vtkAnimationCue> myAnimationCue;
	vtkSmartPointer<AnimationCueObserver> myAnimationCueObserver;
	CueAnimator* myCueAnimator;

	// Array
	vtkSmartPointer<vtkActor> *myArrayActor;
	vtkSmartPointer<vtkActor> *myArrayEdgeActor;
	vtkSmartPointer<vtkDataSetMapper> *myArrayMapper;
	vtkSmartPointer<vtkWarpVector>* warpFilterArray;
	vtkSmartPointer<vtkLookupTable>* hueLutArray;
public:
	std::vector<int> myFrameID;
};