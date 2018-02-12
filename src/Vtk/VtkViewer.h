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
* \file VtkViewer.h
* This file holds the class of VtkViewer.
* \date 9/11/2017
**************************************************************************************************/
#ifndef _VTKVIEWER_H_
#define _VTKVIEWER_H_

#include <STACCATO_Enum.h>
#include "HMesh.h"
#include "VisualizerWindow.h"
//VTK
#include <QVTKOpenGLWidget.h>
#include <vtkSmartPointer.h>

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


class VtkViewer : public QVTKOpenGLWidget
{
	Q_OBJECT
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	explicit VtkViewer(QWidget* parent);
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
	/***********************************************************************************************
	* \brief Update the Visualizer Window for update in picking
	* \author Harikrishnan Sreekumar
	***********/
	void myUpdateVisualizerWindow();
public:
	/***********************************************************************************************
	* \brief Set VTK Viewer with Vector Field
	* \author Harikrishnan Sreekumar
	***********/
	void plotVectorField(vtkSmartPointer<vtkUnstructuredGrid>&);
	/***********************************************************************************************
	* \brief Set Properties for Rendering
	* \author Harikrishnan Sreekumar
	***********/
	void setDisplayProperties(STACCATO_Result_type, bool, bool, bool);
	/***********************************************************************************************
	* \brief Set Properties for Scaling
	* \author Harikrishnan Sreekumar
	***********/
	void setScalingFactor(double);
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
	* \brief Create the Visualizer Window
	* \author Harikrishnan Sreekumar
	***********/
	void my2dVisualizerInterface(HMesh& _hMesh);
protected:
	/***********************************************************************************************
	* \brief Custom mouse press event
	* \author Stefan Sicklinger
	***********/
	virtual void mousePressEvent(QMouseEvent * _event);
private:
	vtkSmartPointer<vtkRenderer> myRenderer;
	vtkSmartPointer<vtkOrientationMarkerWidget> myOrientationMarkerWidget;
	QColor myBGColor;
	vtkSmartPointer<vtkActor> mySelectedActor;
	//vtkSmartPointer<vtkProperty> mySelectedProperty;
	vtkSmartPointer<vtkDataSetMapper> mySelectedMapper;
	vtkSmartPointer<vtkActor> myEdgeActor;
	vtkSmartPointer<vtkScalarBarWidget> myScalarBarWidget;
	vtkSmartPointer<vtkActor> selectedPickActor;

	// Array
	vtkSmartPointer<vtkActor> *myArrayActor;
	vtkSmartPointer<vtkDataSetMapper> *myArrayMapper;
	vtkSmartPointer<vtkWarpVector>* warpFilterArray;
	vtkSmartPointer<vtkLookupTable>* hueLutArray;

	STACCATO_Picker_type myCurrentPickerType;

	///Display Properties
	bool myEdgeVisibility;
	bool mySurfaceVisiiblity;
	bool myScalarBarVisibility;
	bool myRotateMode;
	bool myPickMode;
	bool myInteractivePickerVM;

	char* myTitle;
	double myScaleFactor;	

	std::vector<int> mySelectedNodes;
	std::vector<int> mySelectedElements;

	VisualizerWindow* VW;

public slots:
	//! Zoom to the extent of the data set in the scene
	void zoomToExtent();
	void setPickerModeNone() { setPickerMode(STACCATO_Picker_None); }
	void setPickerModeNode() { setPickerMode(STACCATO_Picker_Node); }
	void setPickerModeElement() { setPickerMode(STACCATO_Picker_Element); }
	void animate(HMeshToVtkUnstructuredGrid& _vtkUnstructuredGrid, HMesh &_hMesh);
	void plotVectorFieldAtIndex(int _index);
};

Q_DECLARE_METATYPE(VtkViewer*)

#endif // _VTKVIEWER_H_
