/*  Copyright &copy; 2017, Stefan Sicklinger, Munich
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
/***********************************************************************************************//**
* \file FieldDataVisualizer.h
* This file holds the class of FieldDataVisualizer.
* \date 9/11/2017
**************************************************************************************************/
#pragma once

#include <STACCATO_Enum.h>
#include "HMesh.h"
//VTK
#include <QVTKOpenGLWidget.h>
#include <vtkSmartPointer.h>

#include <vtkAnimationScene.h>
#include <DiscreteVisualizer.h>
#include <HMeshToVtkUnstructuredGrid.h>
#include "VisualizerSettingSubject.h"

class QVTKOpenGLWidget;
class vtkOrientationMarkerWidget;
class vtkRenderer;
class vtkDataSetMapper;
class vtkActor;
class QColor;
class vtkScalarBarWidget;
class vtkUnstructuredGrid;
class vtkWarpVector;
class vtkLookupTable;
class vtkExtractEdges;
class vtkPolyDataMapper;
class VtkAnimator;
class VtkViewer;
class VisualizerSetting;

class FieldDataVisualizer : public QVTKOpenGLWidget, public DiscreteVisualizer, public VisualizerSettingSubject
{
	Q_OBJECT
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	explicit FieldDataVisualizer(QWidget* parent);
	/***********************************************************************************************
	* \brief Return reference to smart pointer
	* \author Stefan Sicklinger
	***********/
	vtkSmartPointer<vtkRenderer> &  getRenderer(void) { return myRenderer; }

private:
	/***********************************************************************************************
	* \brief Show compass
	* \author Stefan Sicklinger
	***********/
	void displayCompass(void);
	/***********************************************************************************************
	* \brief Set background color gradient
	* \author Stefan Sicklinger
	***********/
	void setBackgroundGradient(int r, int g, int b);
	/***********************************************************************************************
	* \brief Set picker mode (adapter for public slots)
	* \author Stefan Sicklinger
	***********/
	void setPickerMode(STACCATO_Picker_type _currentPickerType);
public:
	/***********************************************************************************************
	* \brief Set VTK Viewer with Vector Field
	* \author Harikrishnan Sreekumar
	***********/
	void plotVectorField();
	/***********************************************************************************************
	* \brief Set Properties for Rotating and Picking
	* \author Harikrishnan Sreekumar
	***********/
	void setViewMode(bool);
	/***********************************************************************************************
	* \brief Get Selected Nodes and Elements
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<int> getSelection(STACCATO_Picker_type);
	/***********************************************************************************************
	* \brief Make object of HMeshToVtkUnstructuredGrid
	* \author Harikrishnan Sreekumar
	***********/
	void myHMeshToVtkUnstructuredGridInitializer();
	/***********************************************************************************************
	* \brief Set Scalar Field for HMeshToVtkUnstructuredGrid
	* \author Harikrishnan Sreekumar
	***********/
	void myHMeshToVtkUnstructuredGridSetScalar(STACCATO_VectorField_components _type, int _index);
	/***********************************************************************************************
	* \brief Set Scaled Scalar Field for HMeshToVtkUnstructuredGrid
	* \author Harikrishnan Sreekumar
	***********/
	void myHMeshToVtkUnstructuredGridSetScaledScalar(STACCATO_VectorField_components _type, int _index);
	/***********************************************************************************************
	* \brief Set Vector Field for HMeshToVtkUnstructuredGrid
	* \author Harikrishnan Sreekumar
	***********/
	void myHMeshToVtkUnstructuredGridSetVector(int _index);
	/***********************************************************************************************
	* \brief Set VTK Viewer with Selected Mapper
	* \author Harikrishnan Sreekumar
	***********/
	void setActiveMapper(vtkSmartPointer<vtkDataSetMapper>& _mapper);
	/***********************************************************************************************
	* \brief Set VTK Viewer with Selected Actor
	* \author Harikrishnan Sreekumar
	***********/
	void setActiveActor(vtkSmartPointer<vtkActor>& _actor);
	/***********************************************************************************************
	* \brief Set VTK Viewer with Selected EdgeActor
	* \author Harikrishnan Sreekumar
	***********/
	void setActiveEdgeActor(vtkSmartPointer<vtkActor>& _edgeActor);
	/***********************************************************************************************
	* \brief Set VTK Viewer with Active Warp Filter with Scalar Range
	* \author Harikrishnan Sreekumar
	***********/
	void setActiveWarpFilter(vtkSmartPointer<vtkWarpVector>& _warpFilter);
	/***********************************************************************************************
	* \brief Set VTK Viewer with Active HueLut
	* \author Harikrishnan Sreekumar
	***********/
	void setActiveHueLut(vtkSmartPointer<vtkLookupTable>& _hueLut);
	/***********************************************************************************************
	* \brief Set VTK Viewer with Active ScalarBarWidget
	* \author Harikrishnan Sreekumar
	***********/
	void setActiveScalarBarWidget();
	/***********************************************************************************************
	* \brief Toggle Edge Visibility - EdgeActor
	* \author Harikrishnan Sreekumar
	***********/
	void enableActiveEdgeActor(bool _enable);
	/***********************************************************************************************
	* \brief Toggle Surface Visibility - SelectedActor
	* \author Harikrishnan Sreekumar
	***********/
	void enableActiveActor(bool _enable);
	/***********************************************************************************************
	* \brief Toggle ScalarBarWidget Visibility - ScalarBarWidgetActor
	* \author Harikrishnan Sreekumar
	***********/
	void enableActiveScalarBarWidget(bool _enable);
	/***********************************************************************************************
	* \brief Set VTK Viewer with New Mapper
	* \author Harikrishnan Sreekumar
	***********/
	void setNewMapper(vtkSmartPointer<vtkDataSetMapper>& _newMapper);

	void connectVisualizerSetting(VisualizerSetting* _setting);

protected:
	/***********************************************************************************************
	* \brief Custom mouse press event
	* \author Stefan Sicklinger
	***********/
	virtual void mousePressEvent(QMouseEvent * _event);
private:
	vtkSmartPointer<vtkRenderer> myRenderer;

	// Animator
	VtkAnimator* myVtkAnimator;
	// Viewer
	VtkViewer* myVtkViewer;

public:
	HMeshToVtkUnstructuredGrid* myHMeshToVtkUnstructuredGrid;

	vtkSmartPointer<vtkOrientationMarkerWidget> myOrientationMarkerWidget;
	QColor myBGColor;
	vtkSmartPointer<vtkActor> mySelectedActor;
	//vtkSmartPointer<vtkProperty> mySelectedProperty;
	vtkSmartPointer<vtkDataSetMapper> mySelectedMapper;
	vtkSmartPointer<vtkActor> myEdgeActor;
	vtkSmartPointer<vtkScalarBarWidget> myScalarBarWidget;
	vtkSmartPointer<vtkActor> selectedPickActor;
	vtkSmartPointer<vtkExtractEdges> edgeExtractor;
	vtkSmartPointer<vtkPolyDataMapper> edgeMapper;
	vtkSmartPointer<vtkWarpVector> warpFilter;
	vtkSmartPointer<vtkLookupTable> hueLut;

	STACCATO_Picker_type myCurrentPickerType;

	///Display Properties
	VisualizerSetting *myVisualizerSetting;

	bool myRotateMode;
	bool myPickMode;
	bool isAnimationSceneInstantiated;

	std::vector<int> mySelectedNodes;
	std::vector<int> mySelectedElements;

	double myHarmonicScale;

public slots:
	//! Zoom to the extent of the data set in the scene
	void zoomToExtent();
	void setPickerModeNone() { setPickerMode(STACCATO_Picker_None); }
	void setPickerModeNode() { setPickerMode(STACCATO_Picker_Node); }
	void setPickerModeElement() { setPickerMode(STACCATO_Picker_Element); }
	void animate(STACCATO_VectorField_components _type, std::vector<int>& _animationIndices);
	void animateHarmonics(STACCATO_VectorField_components _type, std::vector<int>& _animationIndices);
	void plotVectorFieldAtIndex(int _index);
	void myAnimationScenePlayProc(int _duration, int _loops);
	void myAnimationSceneStopProc();
};

Q_DECLARE_METATYPE(FieldDataVisualizer*)
