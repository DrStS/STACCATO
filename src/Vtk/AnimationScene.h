#include <vtkSmartPointer.h>
#include <vtkAnimationCue.h>
#include <vtkCommand.h>
#include <vtkUnstructuredGrid.h>
#include "HMesh.h"
#include "HMeshToVtkUnstructuredGrid.h"

#include "vtkViewer.h"
#include <QApplication>
#pragma once


class CueAnimator
{
public:
	CueAnimator(HMeshToVtkUnstructuredGrid& _vtkUnstructuredGrid, HMesh& _hMesh, VtkViewer& _vtkViewer) : myHMeshToVtkUnstructuredGrid(&_vtkUnstructuredGrid), myHMesh(&_hMesh), myVtkViewer(&_vtkViewer)
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
		double refreshRate = (static_cast<double>(info->EndTime - info->StartTime)) / (static_cast<double>(myHMesh->getResultsSubFrameDescription().size()-1));
		if (static_cast<double>(info->AnimationTime) >= refreshRate*frameIndex && frameIndex < myHMesh->getResultsSubFrameDescription().size()-1) {
			myVtkViewer->plotVectorFieldAtIndex(frameIndex);
			frameIndex++;
		}
		//https://stackoverflow.com/questions/41685872/how-to-render-programmatically-a-vtk-item-in-qml
		QApplication::processEvents();
	}

	void EndCue(vtkAnimationCue::AnimationCueInfo *vtkNotUsed(info),
		vtkRenderer *ren)
	{
		myVtkViewer->plotVectorFieldAtIndex(myHMesh->getResultsSubFrameDescription().size() - 1);
		frameIndex = 0;
		(void)ren;
		// don't remove the actor for the regression image.
		//      ren->RemoveActor(this->Actor);
		this->Cleanup();
	}

protected:
	vtkSmartPointer<vtkUnstructuredGrid>* myVtkUnstructuredGrid;
	HMeshToVtkUnstructuredGrid* myHMeshToVtkUnstructuredGrid;

	VtkViewer* myVtkViewer;

	int frameIndex;

	/// HMesh object 
	HMesh *myHMesh;

	void Cleanup()
	{
	}
};

class vtkAnimationCueObserver : public vtkCommand
{
public:
	static vtkAnimationCueObserver *New()
	{
		return new vtkAnimationCueObserver;
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
	vtkAnimationCueObserver()
	{
		this->Renderer = 0;
		this->Animator = 0;
		this->RenWin = 0;
	}
};

class AnimationScene
{
public:
	AnimationScene(HMeshToVtkUnstructuredGrid& _vtkUnstructuredGrid, HMesh& _hMesh);
	~AnimationScene();

private:

};
