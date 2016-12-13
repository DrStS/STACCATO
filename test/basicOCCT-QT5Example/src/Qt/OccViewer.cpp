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
#include <OpenGl_GraphicDriver.hxx>
#undef Bool
#undef CursorShape
#undef None
#undef KeyPress
#undef KeyRelease
#undef FocusIn
#undef FocusOut
#undef FontChange
#undef Expose
#include <OccViewer.h>
#include <OcctWindow.h>

//QT5
#include <QApplication>
#include <QBitmap>
#include <QPainter>
#include <QInputEvent>
#include <QColorDialog>
#include <QMessageBox>
#include <QInputDialog>

//OCC 7
#include <Aspect_DisplayConnection.hxx>
#include <AIS_InteractiveObject.hxx>
#include <Graphic3d_NameOfMaterial.hxx>
#include <TCollection_AsciiString.hxx>
#include <Aspect_Grid.hxx>
#include <IntAna_IntConicQuad.hxx>

#include <BRepPrimAPI_MakeWedge.hxx>

#if defined _WIN32 || defined __WIN32__
  #include <WNT_Window.hxx>
  #include <gl/GL.h>
  #include <gl/GLU.h>
#elif defined __APPLE__
  #include <Cocoa_Window.hxx>
  #include <OpenGL/gl.h>
  #include <OpenGL/glu.h>
#else
  #include <Xw_Window.hxx>
  #include <GL/gl.h>
  #include <GL/glu.h>
#endif

// the key for multi selection :
#define MULTISELECTIONKEY Qt::ShiftModifier
// the key for shortcut ( use to activate dynamic rotation, panning )
#define CASCADESHORTCUTKEY Qt::ControlModifier
// for elastic bean selection
#define ValZWMin 1
#define STACCATO_ZOOM_STEP 1.10

#define SIGN(a) ( ( (a) < 0 )  ?  -1   : ( (a) > 0 ) )


OccViewer::OccViewer(QWidget* parent )
: QWidget(parent), 
	myView(NULL),
	myViewer(NULL),
	whiteRect(NULL),
	blackRect(NULL),
	myViewResized(Standard_False),
	myViewInitialized(Standard_False),
	myMode(CurAction3d_Undefined),
	myGridSnap(Standard_True),
	myDetection(AIS_SOD_Nothing),
	myPrecision(0.001),
	myViewPrecision(0.0),
	myKeyboardFlags(Qt::NoModifier),
	myButtonFlags(Qt::NoButton)
{
	initOccView();

}


void OccViewer::initOccView(){

	/// Create 3D Viewer
	TCollection_ExtendedString a3DName("STACCATO");
	myViewer = createViewer(a3DName.ToExtString(), "", 1000.0, V3d_XposYnegZpos, Standard_True, Standard_True);
	myViewer->SetDefaultLights();
	// activates all the lights defined in this viewer
	myViewer->SetLightOn();
	// set background color to black
	myViewer->SetDefaultBackgroundColor(Quantity_NOC_BLACK);

	///Setup QT5
	setAttribute(Qt::WA_PaintOnScreen);

	/// Create 3D View
	if (myView.IsNull()){
		myView = myViewer->CreateView();
	}
	Handle_Aspect_Window myWindow;
#if OCC_VERSION_HEX >= 0x070000 
	myWindow = new OcctWindow( this );
#else
#if defined _WIN32 || defined __WIN32__
	myWindow = new WNT_Window((Aspect_Handle)winId());
#elif defined __APPLE__
	myWindow = new Cocoa_Window((NSView *)winId());
#else
	Aspect_Handle windowHandle = (Aspect_Handle)winId();
	myWindow = new Xw_Window(myContext->CurrentViewer()->Driver()->GetDisplayConnection(),windowHandle);
#endif
#endif // OCC_VERSION_HEX >= 0x070000

	

	myView->SetWindow(myWindow);
	if (!myWindow->IsMapped())
	{
		myWindow->Map();
	}
	myView->SetBackgroundColor(Quantity_NOC_BLACK);
	myView->MustBeResized();
	myView->ChangeRenderingParams().Method = Graphic3d_RM_RAYTRACING;

	/// Create an interactive context
	myContext = new AIS_InteractiveContext(myViewer);   


	//Test
	TopoDS_Shape aShape = BRepPrimAPI_MakeWedge(100, 100, 200, 2).Solid();
	Handle(AIS_Shape) anAISShape = new AIS_Shape(aShape);
	myContext->Display(anAISShape);

}


Handle(V3d_Viewer) OccViewer::createViewer(const Standard_ExtString theName,
	const Standard_CString theDomain,
	const Standard_Real theViewSize,
	const V3d_TypeOfOrientation theViewProj,
	const Standard_Boolean theComputedMode,
	const Standard_Boolean theDefaultComputedMode)
{
	static Handle(OpenGl_GraphicDriver) aGraphicDriver;

	if (aGraphicDriver.IsNull())
	{
		Handle(Aspect_DisplayConnection) aDisplayConnection;
#if !defined(_WIN32) && !defined(__WIN32__) && (!defined(__APPLE__) || defined(MACOSX_USE_GLX))
		aDisplayConnection = new Aspect_DisplayConnection (qgetenv ("DISPLAY").constData());
#endif
		aGraphicDriver = new OpenGl_GraphicDriver(aDisplayConnection);
	}

	Handle(V3d_Viewer) aViewer = new V3d_Viewer(aGraphicDriver);
	aViewer->SetDefaultViewSize(theViewSize);
	aViewer->SetDefaultViewProj(theViewProj);
	aViewer->SetComputedMode(theComputedMode);
	aViewer->SetDefaultComputedMode(theDefaultComputedMode);
	return aViewer;
}

OccViewer::~OccViewer()
{
}

QPaintEngine* OccViewer::paintEngine() const
{
	return NULL;
}

void OccViewer::paintEvent(QPaintEvent *)
{
	myView->Redraw();
}

void OccViewer::resizeEvent(QResizeEvent *)
{
	if (!myView.IsNull())
	{
		myView->MustBeResized();
	}
}