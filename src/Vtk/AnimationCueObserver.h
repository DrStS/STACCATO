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
* \file AnimationCueObserver.h
* This file holds the class AnimationCueObserver
* \date 2/20/2018
**************************************************************************************************/
#pragma once

#include <vtkSmartPointer.h>
#include <vtkAnimationCue.h>
#include <vtkCommand.h>
#include <vtkUnstructuredGrid.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include "HMesh.h"
#include "HMeshToVtkUnstructuredGrid.h"

#include "VtkAnimator.h"
#include "CueAnimator.h"
#include <QApplication>

/*************************************************************************************************
* \brief Class AnimationCueObserver
**************************************************************************************************/

class AnimationCueObserver : public vtkCommand
{
public:
	static AnimationCueObserver *New()
	{
		return new AnimationCueObserver;
	}

	virtual void Execute(vtkObject *vtkNotUsed(caller),
		unsigned long event,
		void *calldata)
	{
		if (this->Animator != 0 && this->Renderer != 0)
		{
			vtkAnimationCue::AnimationCueInfo *info =
				static_cast<vtkAnimationCue::AnimationCueInfo *>(calldata);
			switch (event)
			{
			case vtkCommand::StartAnimationCueEvent:
				this->Animator->StartCue(info, this->Renderer);
				break;
			case vtkCommand::EndAnimationCueEvent:
				this->Animator->EndCue(info, this->Renderer);
				break;
			case vtkCommand::AnimationCueTickEvent:
				this->Animator->Tick(info, this->Renderer);
				break;
			}
		}
		if (this->RenWin != 0)
		{
			this->RenWin->Render();
		}
	}

	vtkRenderer *Renderer;
	vtkRenderWindow *RenWin;
	CueAnimator *Animator;
protected:
	AnimationCueObserver()
	{
		this->Renderer = 0;
		this->Animator = 0;
		this->RenWin = 0;
	}
};