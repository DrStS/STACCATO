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
//VTK
#include <QVTKOpenGLWidget.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
//QT5
#include <Quantity_Color.hxx>

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
	* \brief Return reference to samrt pointer
	* \author Stefan Sicklinger
	***********/
	vtkSmartPointer<vtkRenderer> &  getRenderer(void) { return myRenderer; }
	/***********************************************************************************************
	* \brief Set background color gradient
	* \author Stefan Sicklinger
	***********/
	void setBackgroundGradient(int r, int g, int b);

	void demo(void);

private:
	vtkSmartPointer<vtkRenderer> myRenderer;
	QColor                        myBGColor;

public slots:
	//! Zoom to the extent of the data set in the scene
	void zoomToExtent();
};

Q_DECLARE_METATYPE(VtkViewer*)

#endif // _VTKVIEWER_H_
