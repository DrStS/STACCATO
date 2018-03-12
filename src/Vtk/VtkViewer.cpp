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

#include "VtkViewer.h"

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

#include "Timer.h"

VtkViewer::VtkViewer(FieldDataVisualizer& _fieldDataVisualizer): myFieldDataVisualizer(&_fieldDataVisualizer)
{
}

VtkViewer::~VtkViewer()
{
}

void VtkViewer::plotVectorField() {
	// Reset the Renderer
	myFieldDataVisualizer->getRenderer()->RemoveAllViewProps();
	
	myFieldDataVisualizer->setActiveMapper(myFieldDataVisualizer->mySelectedMapper);
	myFieldDataVisualizer->setActiveActor(myFieldDataVisualizer->mySelectedActor);

	myFieldDataVisualizer->setActiveWarpFilter(myFieldDataVisualizer->warpFilter);
	myFieldDataVisualizer->setActiveHueLut(myFieldDataVisualizer->hueLut);
	myFieldDataVisualizer->setActiveScalarBarWidget();
	myFieldDataVisualizer->setActiveEdgeActor(myFieldDataVisualizer->myEdgeActor);

	myFieldDataVisualizer->enableActiveEdgeActor(myFieldDataVisualizer->myVisualizerSetting->myFieldDataSetting->getEdgeProperty());
	myFieldDataVisualizer->enableActiveActor(myFieldDataVisualizer->myVisualizerSetting->myFieldDataSetting->getSurfaceProperty());
	myFieldDataVisualizer->enableActiveScalarBarWidget(myFieldDataVisualizer->myVisualizerSetting->PROPERTY_SCALARBAR_VISIBILITY);

	static bool resetCamera = true;
	if (resetCamera) {
		myFieldDataVisualizer->getRenderer()->ResetCamera();
		resetCamera = false;
	}
	myFieldDataVisualizer->GetRenderWindow()->Render();
}