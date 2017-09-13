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

#include <vtkCamera.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkDataSetMapper.h>

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkSphereSource.h>

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

void VtkViewer::demo(void) {
	// Create a sphere
	vtkSmartPointer<vtkSphereSource> sphereSource =
		vtkSmartPointer<vtkSphereSource>::New();
	sphereSource->Update();

	// Create a mapper and actor
	vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(sphereSource->GetOutputPort());

	vtkSmartPointer<vtkActor> actor =
		vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	myRenderer->AddActor(actor);

}
