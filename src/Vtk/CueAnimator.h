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
* \file CueAnimator.h
* This file holds the class CueAnimator
* \date 2/20/2018
**************************************************************************************************/

#ifndef CUEANIMATOR_H_
#define CUEANIMATOR_H_

#include <vtkSmartPointer.h>
#include <vtkAnimationCue.h>
#include <vtkCommand.h>
#include <vtkUnstructuredGrid.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include "HMesh.h"
#include "HMeshToVtkUnstructuredGrid.h"

#include "VtkAnimator.h"
#include <QApplication>

/*************************************************************************************************
* \brief Class CueAnimator
**************************************************************************************************/

class CueAnimator
{
public:
	CueAnimator(HMeshToVtkUnstructuredGrid& _vtkUnstructuredGrid, HMesh& _hMesh, VtkAnimator& _vtkViewer) : myHMeshToVtkUnstructuredGrid(&_vtkUnstructuredGrid), myHMesh(&_hMesh), myVtkAnimator(&_vtkViewer)
	{
		frameIndex = 0;
	}

	~CueAnimator()
	{
		this->Cleanup();
	}

	void StartCue(vtkAnimationCue::AnimationCueInfo *vtkNotUsed(info),
		vtkRenderer *ren)
	{
		//cout << "*** IN StartCue " << endl;
	}

	void Tick(vtkAnimationCue::AnimationCueInfo *info,
		vtkRenderer *ren)
	{
		double refreshRate = (static_cast<double>(info->EndTime - info->StartTime)) / (static_cast<double>(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().size() - 1));
		if (static_cast<double>(info->AnimationTime) >= refreshRate*frameIndex && frameIndex < myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().size() - 1) {
			myVtkAnimator->plotVectorFieldAtIndex(frameIndex);
			frameIndex++;
		}
		//https://stackoverflow.com/questions/41685872/how-to-render-programmatically-a-vtk-item-in-qml
		QApplication::processEvents();
	}

	void EndCue(vtkAnimationCue::AnimationCueInfo *vtkNotUsed(info),
		vtkRenderer *ren)
	{
		myVtkAnimator->plotVectorFieldAtIndex(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().size() - 1);
		frameIndex = 0;
		(void)ren;
		// don't remove the actor for the regression image.
		//      ren->RemoveActor(this->Actor);
		this->Cleanup();
	}

protected:
	vtkSmartPointer<vtkUnstructuredGrid>* myVtkUnstructuredGrid;
	HMeshToVtkUnstructuredGrid* myHMeshToVtkUnstructuredGrid;

	VtkAnimator* myVtkAnimator;

	int frameIndex;

	/// HMesh object 
	HMesh *myHMesh;

	void Cleanup()
	{
	}
};

#endif /* CUEANIMATOR_H_ */