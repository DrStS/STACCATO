/*  Copyright &copy; 2016, Stefan Sicklinger, Munich
*  
*  All rights reserved.
*
*  This file is part of STACCATO.
*
*  EMPIRE is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  EMPIRE is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with STACCATO.  If not, see http://www.gnu.org/licenses/.
*/
/***********************************************************************************************//**
* \file OccView.h
* This file holds the class of OccView.
* \date 9/15/2016
**************************************************************************************************/
#ifndef _OCCVIEWER_H_
#define _OCCVIEWER_H_
//Qt5
#include <QWidget>
#include <QMetaType>
#include <Quantity_Color.hxx>
#include <Standard_Version.hxx>
//OCC7
#include <AIS_InteractiveContext.hxx>
#include <V3d_View.hxx>
#include <AIS_Shape.hxx>
#include <AIS_RubberBand.hxx>

#define Handle_AIS_RubberBand Handle(AIS_RubberBand)
/** the key for multi selection */
#define MULTISELECTIONKEY  Qt::ShiftModifier   
/** The key for shortcut ( use to activate dynamic rotation, panning ) */
#define CASCADESHORTCUTKEY Qt::ControlModifier 
#define VALZWMIN 1 /** For elastic bean selection */

class TopoDS_Shape;
class gp_Pnt;
class gp_Vec;

class OccViewer : public QWidget
{
	Q_OBJECT

public:

	enum CurrentAction3d
	{
		CurAction3d_Undefined,
		CurAction3d_Nothing,
		CurAction3d_Picking,
		CurAction3d_DynamicZooming,
		CurAction3d_WindowZooming,
		CurAction3d_DynamicPanning,
		CurAction3d_GlobalPanning,
		CurAction3d_DynamicRotation
	};

public:
	/***********************************************************************************************
	* \brief Constructor of OccViewer
	* \author Stefan Sicklinger
	***********/
	OccViewer(QWidget*);
	/***********************************************************************************************
	* \brief Distructor of OccViewer
	* \author Stefan Sicklinger
	***********/
	~OccViewer();

	Handle_V3d_View                  getView(void)    { return myView; }
	Handle_AIS_InteractiveContext    getContext(void)    { return myContext; }

	//Overrides
	QPaintEngine*   paintEngine() const;
	class QToolBar* myToolBar;
	void redraw(bool isPainting = false);

signals:

	void initialized();
	void selectionChanged();
	void mouseMoved(V3d_Coordinate X, V3d_Coordinate Y, V3d_Coordinate Z);
	void pointClicked(V3d_Coordinate X, V3d_Coordinate Y, V3d_Coordinate Z);
	void sendStatus(const QString aMessage);

	void error(int errorCode, QString& errorDescription);

	public slots:

	void idle();
	void fitExtents();
	void fitAll();
	void fitArea();
	void zoom();
	void zoomIn();
	void zoomOut();
	void pan();
	void globalPan();
	void rotation();
	void selecting();
	void hiddenLineOn();
	void hiddenLineOff();
	void setBackgroundGradient(int r, int g, int b);
	void setBackgroundColor(int r, int g, int b);
	void setBGImage(const QString&);
	void viewFront();
	void viewBack();
	void viewTop();
	void viewBottom();
	void viewLeft();
	void viewRight();
	void viewAxo();
	void viewTopFront();
	void viewGrid();
	void viewReset();
	void setReset();
	void eraseSelected();
	void setTransparency();
	void setTransparency(int);
	void setObjectsWireframe();
	void setObjectsShading();
	void setObjectsColor();
	void setObjectsMaterial();
	bool makeScreenshot(const QString& filename, bool whiteBGEnabled = true, int width = 0, int height = 0, int quality = 90);
	void showGrid(Standard_Boolean show);

protected: // methods

	virtual void paintEvent(QPaintEvent* e);
	virtual void resizeEvent(QResizeEvent* e);
	virtual void mousePressEvent(QMouseEvent* e);
	virtual void mouseReleaseEvent(QMouseEvent* e);
	virtual void mouseMoveEvent(QMouseEvent* e);
	virtual void wheelEvent(QWheelEvent* e);
	virtual void keyPressEvent(QKeyEvent* e);

	virtual void leaveEvent(QEvent *);

private: // members
	Handle_V3d_View                 myView;
	Handle_V3d_Viewer               myViewer;
	Handle_AIS_InteractiveContext   myContext;
	Handle_AIS_RubberBand           whiteRect, blackRect;

	Standard_Boolean                myViewResized;
	Standard_Boolean                myViewInitialized;
	CurrentAction3d                 myMode;
	Quantity_Factor                 myCurZoom;
	Standard_Boolean                myGridSnap;
	AIS_StatusOfDetection           myDetection;

	V3d_Coordinate                  myV3dX,
		myV3dY,
		myV3dZ;

	QPoint                          myStartPoint;
	QPoint                          myCurrentPoint;

	Standard_Real                   myPrecision;
	Standard_Real                   myViewPrecision;
	Standard_Boolean                myMapIsValid;
	Qt::KeyboardModifiers           myKeyboardFlags;
	Qt::MouseButton                 myButtonFlags;
	QCursor                         myCrossCursor;
	QColor                          myBGColor;

private: // methods
	void initOccViewer();
	Handle(V3d_Viewer) viewer(const Standard_ExtString theName,
		const Standard_CString theDomain,
		const Standard_Real theViewSize,
		const V3d_TypeOfOrientation theViewProj,
		const Standard_Boolean theComputedMode,
		const Standard_Boolean theDefaultComputedMode);
	

	void onLeftButtonDown(Qt::KeyboardModifiers nFlags, const QPoint point);
	void onMiddleButtonDown(Qt::KeyboardModifiers nFlags, const QPoint point);
	void onRightButtonDown(Qt::KeyboardModifiers nFlags, const QPoint point);
	void onLeftButtonUp(Qt::KeyboardModifiers nFlags, const QPoint point);
	void onMiddleButtonUp(Qt::KeyboardModifiers nFlags, const QPoint point);
	void onRightButtonUp(Qt::KeyboardModifiers nFlags, const QPoint point);

	void onMouseMove(Qt::MouseButtons buttons,
		Qt::KeyboardModifiers nFlags, const QPoint point);

	AIS_StatusOfPick        dragEvent(const QPoint startPoint, const QPoint endPoint, const bool multi = false);
	AIS_StatusOfPick        inputEvent(const bool multi = false);
	AIS_StatusOfDetection   moveEvent(const QPoint point);

	void setMode(const CurrentAction3d mode);

	Standard_Real precision(Standard_Real aReal);
	Standard_Real viewPrecision(bool resized = false);

	void drawRubberBand(const QPoint origin, const QPoint position);
	void hideRubberBand(void);

	Standard_Boolean convertToPlane(Standard_Integer Xs,
		Standard_Integer Ys,
		Standard_Real& X,
		Standard_Real& Y,
		Standard_Real& Z);


};

Q_DECLARE_METATYPE(OccViewer*)

#endif // _OCCVIEWER_H_
