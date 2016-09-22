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
	myGridSnap(Standard_False),
	myDetection(AIS_SOD_Nothing),
	myPrecision(0.001),
	myViewPrecision(0.0),
	myKeyboardFlags(Qt::NoModifier),
	myButtonFlags(Qt::NoButton)
{
	whiteRect = new AIS_RubberBand(Quantity_Color(Quantity_NOC_WHITE), Aspect_TOL_DOT, 1.0);
	blackRect = new AIS_RubberBand(Quantity_Color(Quantity_NOC_BLACK), Aspect_TOL_DOT, 1.0);

	myMode = CurAction3d_Nothing;
	setMouseTracking(true);
	setAutoFillBackground(false);
	setAttribute(Qt::WA_NoSystemBackground);
	setAttribute(Qt::WA_PaintOnScreen);
	// Here's a modified pick point cursor from AutoQ3D
	QBitmap curb1(48, 48);
	QBitmap curb2(48, 48);
	curb1.fill(QColor(255, 255, 255));
	curb2.fill(QColor(255, 255, 255));
	QPainter p;

	p.begin(&curb1);
	p.drawLine(24, 0, 24, 47);
	p.drawLine(0, 24, 47, 24);
	p.setBrush(Qt::NoBrush);
	p.drawRect(18, 18, 12, 12);
	p.end();
	myCrossCursor = QCursor(curb2, curb1, 24, 24);

	// create win id, this is required on qt 5
	// without winid, the view can't be properly initialized
	winId();

}


Handle(V3d_Viewer) OccViewer::viewer(const Standard_ExtString theName,
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
		aDisplayConnection = new Aspect_DisplayConnection(qgetenv("DISPLAY").constData());
#endif
		aGraphicDriver = new OpenGl_GraphicDriver(aDisplayConnection);
	}

	return new V3d_Viewer(aGraphicDriver,
		theName,
		theDomain,
		theViewSize,
		theViewProj,
		Quantity_NOC_GRAY30,
		V3d_ZBUFFER,
		V3d_GOURAUD,
		V3d_WAIT,
		theComputedMode,
		theDefaultComputedMode,
		V3d_TEX_NONE);
}


void OccViewer::initOccViewer(){
	myBGColor = QColor(255, 235, 100);
	TCollection_ExtendedString a3DName("STACCATO");
	myViewer = viewer(a3DName.ToExtString(), "", 1000.0, V3d_XposYnegZpos, Standard_True, Standard_True);
	myContext = new AIS_InteractiveContext(myViewer);
	if (myView.IsNull()){
		myView = myContext->CurrentViewer()->CreateView();
	}

        Handle_Aspect_Window myWindow;
        
        #if OCC_VERSION_HEX >= 0x070800
        myWindow = new OcctWindow(this);
        #else
        #if defined _WIN32 || defined __WIN32__
        myWindow = new WNT_Window((Aspect_Handle)winId());
        #elif defined __APPLE__
        myWindow = new Cocoa_Window((NSView *)winId());
        #else
        Aspect_Handle windowHandle = (Aspect_Handle)winId();
        myWindow = new Xw_Window(myContext->CurrentViewer()->Driver()->GetDisplayConnection(),
                                 windowHandle);
        #endif
        #endif // OCC_VERSION_HEX >= 0x060800
        
        
    
 	myView->SetWindow(myWindow);
	if (!myWindow->IsMapped())
	{
		myWindow->Map();
	}
	myView->SetScale(2);
	myView->SetBackgroundColor(Quantity_NOC_BLACK);
	myView->TriedronDisplay(Aspect_TOTP_LEFT_LOWER, Quantity_NOC_GOLD, 0.08, V3d_ZBUFFER);
	myView->SetAntialiasingOn();

	myViewer->SetDefaultLights();
	myViewer->SetLightOn();

	// Force a redraw to the new window on next paint event
	myViewResized = Standard_True;
	// Set default cursor as a cross
	setMode(CurAction3d_Nothing);
	setBackgroundGradient(myBGColor.red(), myBGColor.green(), myBGColor.blue());
	myContext->SetDisplayMode(AIS_Shaded);
	myViewInitialized = Standard_True;
        
}

OccViewer::~OccViewer()
{
}

QPaintEngine* OccViewer::paintEngine() const
{
	return NULL;
}


void OccViewer::paintEvent(QPaintEvent * /* e */)
{
	if (!myViewInitialized) {
		if (winId()) {
			initOccViewer();
		}
	}
	if (!myViewer.IsNull()) {
		redraw(true);
	}
}



void OccViewer::resizeEvent(QResizeEvent * /* e */)
{
	myViewResized = Standard_True;
}


void OccViewer::mousePressEvent(QMouseEvent* e)
{
	myButtonFlags = e->button();

	// Cache the keyboard flags for the whole gesture
	myKeyboardFlags = e->modifiers();

	// The button mappings can be used as a mask. This code prevents conflicts
	// when more than one button pressed simultaneously.
	if (e->button() & Qt::LeftButton) {
		onLeftButtonDown(myKeyboardFlags, e->pos());
	}
	else if (e->button() & Qt::RightButton) {
		onRightButtonDown(myKeyboardFlags, e->pos());
	}
	else if (e->button() & Qt::MidButton) {
		onMiddleButtonDown(myKeyboardFlags, e->pos());
	}
}



void OccViewer::mouseReleaseEvent(QMouseEvent* e)
{
	myButtonFlags = Qt::NoButton;
	redraw();                            // Clears up screen when menu selected but not used.
	hideRubberBand();
	if (e->button() & Qt::LeftButton) {
		onLeftButtonUp(myKeyboardFlags, e->pos());
	}
	else if (e->button() & Qt::RightButton) {
		onRightButtonUp(myKeyboardFlags, e->pos());
	}
	else if (e->button() & Qt::MidButton) {
		onMiddleButtonUp(myKeyboardFlags, e->pos());
	}
}



void OccViewer::mouseMoveEvent(QMouseEvent* e)
{
	Standard_Real X, Y, Z;

	myCurrentPoint = e->pos();
	//Check if the grid is active and that we're snapping to it
	if (myContext->CurrentViewer()->Grid()->IsActive() && myGridSnap) {
		myView->ConvertToGrid(myCurrentPoint.x(),
			myCurrentPoint.y(),
			myV3dX,
			myV3dY,
			myV3dZ);
		emit mouseMoved(myV3dX, myV3dY, myV3dZ);
	}
	else {
		bool success = convertToPlane(myCurrentPoint.x(),
			myCurrentPoint.y(),
			X, Y, Z);
		//    this is the standard case
		if (success) {

			myV3dX = precision(X);
			myV3dY = precision(Y);
			myV3dZ = precision(Z);
			emit mouseMoved(myV3dX, myV3dY, myV3dZ);
		}
		else {
			emit sendStatus(tr("Indeterminate Point"));
		}
	}

	onMouseMove(e->buttons(), myKeyboardFlags, e->pos());
}



void OccViewer::leaveEvent(QEvent* /* e */)
{
	myButtonFlags = Qt::NoButton;
}


void OccViewer::wheelEvent(QWheelEvent* e)
{
	if (!myView.IsNull()) {
		Standard_Real currentScale = myView->Scale();
		if (e->delta() > 0) {
			currentScale *= STACCATO_ZOOM_STEP; // +10%
		}
		else {
			currentScale /= STACCATO_ZOOM_STEP; // -10%
		}
		myView->SetScale(currentScale);
	}
	else {
		e->ignore();
	}
}

void OccViewer::keyPressEvent(QKeyEvent* e)
{
	if (e->key() == Qt::Key_Delete) {
		eraseSelected();
	}
	else {
		QWidget::keyPressEvent(e);
	}
}


void OccViewer::idle()
{
	setMode(CurAction3d_Nothing);
}


void OccViewer::redraw(bool isPainting)
{
	if (!myView.IsNull()) {
		// Defensive test.
		if (myViewResized) {
			myView->MustBeResized();
			viewPrecision(true);
		}
		else {
			// Don't repaint if we are already redrawing
			// elsewhere due to a keypress or mouse gesture
			if (!isPainting || (isPainting && myButtonFlags == Qt::NoButton)) {
				myView->Redraw();
			}
		}
	}
	myViewResized = Standard_False;
}



void OccViewer::fitExtents(void)
{
	if (!myView.IsNull()) {
		myView->FitAll();
		viewPrecision(true);
	}
}



void OccViewer::fitAll(void)
{
	if (!myView.IsNull()) {
		myView->ZFitAll();
		myView->FitAll();
		viewPrecision(true);
	}
}



void OccViewer::fitArea(void)
{
	setMode(CurAction3d_WindowZooming);
}



void OccViewer::zoom(void)
{
	setMode(CurAction3d_DynamicZooming);
}

void OccViewer::zoomIn()
{
	if (!myView.IsNull()){
		myView->SetScale(myView->Scale() * STACCATO_ZOOM_STEP);
	}
}

void OccViewer::zoomOut()
{
	if (!myView.IsNull()) {
		myView->SetScale(myView->Scale() / STACCATO_ZOOM_STEP);
	}
}

void OccViewer::pan(void)
{
	setMode(CurAction3d_DynamicPanning);
}



void OccViewer::rotation(void)
{
	setMode(CurAction3d_DynamicRotation);
}


void OccViewer::selecting(void)
{
	setMode(CurAction3d_Nothing);
}


void OccViewer::globalPan()
{
	if (!myView.IsNull()) {
		// save the current zoom value
		myCurZoom = myView->Scale();
		// Do a Global Zoom
		myView->FitAll();
		viewPrecision(true);
		// Set the mode
		setMode(CurAction3d_GlobalPanning);
	}
}



void OccViewer::viewGrid()
{
	if (!myView.IsNull()) {
		myView->SetFront();
	}
}



void OccViewer::viewFront()
{
	if (!myView.IsNull()) {
		myView->SetProj(V3d_Xneg);
	}
}


void OccViewer::viewBack()
{
	if (!myView.IsNull()) {
		myView->SetProj(V3d_Xpos);
	}
}



void OccViewer::viewTop()
{
	if (!myView.IsNull()) {
		myView->SetProj(V3d_Zpos);
	}
}



void OccViewer::viewBottom()
{
	if (!myView.IsNull()) {
		myView->SetProj(V3d_Zneg);
	}
}


void OccViewer::viewLeft()
{
	if (!myView.IsNull()) {
		myView->SetProj(V3d_Yneg);
	}
}


void OccViewer::viewRight()
{
	if (!myView.IsNull()) {
		myView->SetProj(V3d_Ypos);
	}
}



void OccViewer::viewAxo()
{
	if (!myView.IsNull()) {
		myView->SetProj(V3d_XnegYnegZpos);
	}
}


void OccViewer::viewTopFront()
{
	if (!myView.IsNull()) {
		myView->SetProj(V3d_YnegZpos);
	}
}


void OccViewer::viewReset()
{
	if (!myView.IsNull()) {
		myView->Reset();
	}
}

void OccViewer::hiddenLineOff()
{
	if (!myView.IsNull()) {
		QApplication::setOverrideCursor(Qt::WaitCursor);
		myView->SetComputedMode(Standard_False);
		QApplication::restoreOverrideCursor();
	}
}

void OccViewer::hiddenLineOn()
{
	if (!myView.IsNull()) {
		QApplication::setOverrideCursor(Qt::WaitCursor);
		myView->SetComputedMode(Standard_True);
		QApplication::restoreOverrideCursor();
	}
}

void OccViewer::setBackgroundGradient(int r, int g, int b)
{
	myBGColor = QColor(r, g, b);
	if (!myView.IsNull()) {
		Standard_Real R1 = r / 255.;
		Standard_Real G1 = g / 255.;
		Standard_Real B1 = b / 255.;

		// Disable provious gradient
		myView->SetBgGradientColors(Quantity_NOC_BLACK, Quantity_NOC_BLACK, Aspect_GFM_NONE, Standard_False);

		Standard_Real fu = 2.;
		Standard_Real fd = 0.2;

		Quantity_Color up(R1*fu > 1 ? 1. : R1*fu, G1*fu > 1 ? 1. : G1*fu, B1*fu > 1 ? 1. : B1*fu, Quantity_TOC_RGB);
		Quantity_Color down(R1*fd > 1 ? 1. : R1*fd, G1*fd > 1 ? 1. : G1*fd, B1*fd > 1 ? 1. : B1*fd, Quantity_TOC_RGB);

		myView->SetBgGradientColors(up, down, Aspect_GFM_VER, Standard_False);
	}
	redraw();
}

void OccViewer::setBackgroundColor(int r, int g, int b)
{
	if (!myView.IsNull()) {
		// Disable provious gradient
		myView->SetBgGradientColors(Quantity_NOC_BLACK, Quantity_NOC_BLACK, Aspect_GFM_NONE, Standard_False);
		myView->SetBackgroundColor(Quantity_TOC_RGB, r / 255., g / 255., b / 255.);
		redraw();
	}
}

void OccViewer::setReset()
{
	if (!myView.IsNull()) {
		myView->SetViewOrientationDefault();
		viewPrecision(true);
	}
}

void OccViewer::setBGImage(const QString& filename)
{
	if (!myView.IsNull()) {
		myView->SetBackgroundImage(filename.toStdString().c_str(), Aspect_FM_CENTERED, Standard_False);
		redraw();
	}
}

void OccViewer::eraseSelected()
{
	if (!myView.IsNull()) {
		myContext->EraseSelected(Standard_False);

		myContext->ClearCurrents();
		myContext->UpdateCurrentViewer();
	}
}


void OccViewer::setTransparency()
{
	bool ok;
	int transparency;

	QInputDialog *dialog = new QInputDialog(this);
	dialog->setInputMode(QInputDialog::IntInput);
	connect(dialog, SIGNAL(intValueChanged(int)), this, SLOT(setTransparency(int)));
	transparency = dialog->getInt(this, tr("Select transparency level"), tr("Transparency:"), 25, 0, 100, 1, &ok);
	setTransparency(transparency);
}

void OccViewer::setTransparency(int tr)
{
	Standard_Real transparency = 0.1;
	if ((tr < 0) || (tr > 100)) {
		return;
	}
	transparency = tr * 0.01;

	if (!myView.IsNull()) {
		for (myContext->InitCurrent(); myContext->MoreCurrent(); myContext->NextCurrent()) {
			myContext->SetTransparency(myContext->Current(), transparency, Standard_True);
		}
	}
}

void OccViewer::setObjectsWireframe()
{
	if (!myView.IsNull()) {
		for (myContext->InitCurrent(); myContext->MoreCurrent(); myContext->NextCurrent()) {
			myContext->SetDisplayMode(myContext->Current(), 0);
		}
	}
}

void OccViewer::setObjectsShading()
{
	if (!myView.IsNull()) {
		for (myContext->InitCurrent(); myContext->MoreCurrent(); myContext->NextCurrent()) {
			myContext->SetDisplayMode(myContext->Current(), 1);
		}
	}
}

void OccViewer::setObjectsColor()
{
	QColor color = QColorDialog::getColor(Qt::green, this);
	if (color.isValid()) {
		for (myContext->InitCurrent(); myContext->MoreCurrent(); myContext->NextCurrent()) {
			myContext->SetColor(myContext->Current(), Quantity_Color(color.red() / 255., color.green() / 255., color.blue() / 255., Quantity_TOC_RGB));
		}
	}
}

void OccViewer::setObjectsMaterial()
{
	if (myContext.IsNull()) {
		return;
	}

	bool ok;
	QStringList items;

	QMap<QString, Graphic3d_NameOfMaterial> materialMap;
	materialMap["Brass"] = Graphic3d_NOM_BRASS;
	materialMap["Bronze"] = Graphic3d_NOM_BRONZE;
	materialMap["Copper"] = Graphic3d_NOM_COPPER;
	materialMap["Gold"] = Graphic3d_NOM_GOLD;
	materialMap["Pewter"] = Graphic3d_NOM_PEWTER;
	materialMap["Plaster"] = Graphic3d_NOM_PLASTER;
	materialMap["Plastic"] = Graphic3d_NOM_PLASTIC;
	materialMap["Silver"] = Graphic3d_NOM_SILVER;
	materialMap["Steel"] = Graphic3d_NOM_STEEL;
	materialMap["Stone"] = Graphic3d_NOM_STONE;
	materialMap["Shiny Plastic"] = Graphic3d_NOM_SHINY_PLASTIC;
	materialMap["Satin"] = Graphic3d_NOM_SATIN;
	materialMap["Metalized"] = Graphic3d_NOM_METALIZED;
	materialMap["Neon GNC"] = Graphic3d_NOM_NEON_GNC;
	materialMap["Chrome"] = Graphic3d_NOM_CHROME;
	materialMap["Aluminium"] = Graphic3d_NOM_ALUMINIUM;
	materialMap["Obsidian"] = Graphic3d_NOM_OBSIDIAN;
	materialMap["Neon PHC"] = Graphic3d_NOM_NEON_PHC;
	materialMap["Jade"] = Graphic3d_NOM_JADE;
	materialMap["Default"] = Graphic3d_NOM_DEFAULT;

	QMapIterator<QString, Graphic3d_NameOfMaterial> i(materialMap);
	while (i.hasNext()) {
		i.next();
		items << i.key();
	}
	QString item = QInputDialog::getItem(this, tr("Select Material"), tr("Material:"), items, 0, false, &ok);
	if (ok && !item.isEmpty()) {
		for (myContext->InitCurrent(); myContext->MoreCurrent(); myContext->NextCurrent()) {
			myContext->SetMaterial(myContext->Current(), materialMap[item]);
		}
	}
}


void OccViewer::onLeftButtonDown(Qt::KeyboardModifiers nFlags, const QPoint point)
{
	setFocus(Qt::MouseFocusReason);
	myStartPoint = point;
	if (nFlags & CASCADESHORTCUTKEY) {
		setMode(CurAction3d_DynamicZooming);
	}
	else {
		switch (myMode) {
		case CurAction3d_Nothing:
			break;

		case CurAction3d_Picking:
			break;

		case CurAction3d_DynamicZooming:
			break;

		case CurAction3d_WindowZooming:
			break;

		case CurAction3d_DynamicPanning:
			break;

		case CurAction3d_GlobalPanning:
			break;

		case CurAction3d_DynamicRotation:
			myView->StartRotation(myStartPoint.x(), myStartPoint.y());
			break;

		default:
			Standard_Failure::Raise("Incompatible Current Mode");
			break;
		}
	}
}


void OccViewer::onMiddleButtonDown(Qt::KeyboardModifiers nFlags, const QPoint point)
{
	myStartPoint = point;
	if (nFlags & CASCADESHORTCUTKEY) {
		setMode(CurAction3d_DynamicPanning);
	}
	else {
		setMode(CurAction3d_DynamicRotation);
		myView->StartRotation(myStartPoint.x(), myStartPoint.y());
	}
}



void OccViewer::onRightButtonDown(Qt::KeyboardModifiers nFlags, const QPoint point)
{
	myStartPoint = point;
	//    if ( nFlags & CASCADESHORTCUTKEY )
	//    {
	//        setMode( CurAction3d_DynamicRotation );
	//        myView->StartRotation( point.x(), point.y() );
	//    }
	//    else
	//    {
	//        emit popupMenu ( this, point );
	//    }
}



void OccViewer::onLeftButtonUp(Qt::KeyboardModifiers nFlags, const QPoint point)
{
	myCurrentPoint = point;
	if (nFlags & CASCADESHORTCUTKEY) {
		// Deactivates dynamic zooming
		setMode(CurAction3d_Nothing);
	}
	else {
		switch (myMode) {
		case CurAction3d_Nothing:
			if (myCurrentPoint == myStartPoint) {
				inputEvent(nFlags & MULTISELECTIONKEY);
			}
			else {
				dragEvent(myStartPoint,
					myCurrentPoint,
					nFlags & MULTISELECTIONKEY);
			}
			break;

		case CurAction3d_DynamicZooming:
			viewPrecision(true);
			break;

		case CurAction3d_WindowZooming:
			if ((abs(myCurrentPoint.x() - myStartPoint.x()) > VALZWMIN) ||
				(abs(myCurrentPoint.y() - myStartPoint.y()) > VALZWMIN)) {

				myView->WindowFitAll(myStartPoint.x(),
					myStartPoint.y(),
					myCurrentPoint.x(),
					myCurrentPoint.y());
			}
			viewPrecision(true);
			break;

		case CurAction3d_DynamicPanning:
			break;

		case CurAction3d_GlobalPanning:
			myView->Place(myCurrentPoint.x(), myCurrentPoint.y(), myCurZoom);
			break;

		case CurAction3d_DynamicRotation:
			break;

		default:
			Standard_Failure::Raise(" Incompatible Current Mode ");
			break;
		}
	}
	emit selectionChanged();
}



void OccViewer::onMiddleButtonUp(Qt::KeyboardModifiers /* nFlags */, const QPoint /* point */)
{
	setMode(CurAction3d_Nothing);
}



void OccViewer::onRightButtonUp(Qt::KeyboardModifiers nFlags, const QPoint point)
{
	myCurrentPoint = point;
	if (nFlags & CASCADESHORTCUTKEY) {
		setMode(CurAction3d_Nothing);
	}
	else {
		if (myMode == CurAction3d_Nothing) {
			//            emit popupMenu ( this, point );
		}
		else {
			setMode(CurAction3d_Nothing);
		}
	}
}


void OccViewer::onMouseMove(Qt::MouseButtons buttons,
	Qt::KeyboardModifiers nFlags,
	const QPoint point)
{
	if (myView.IsNull()) {
		return;
	}
	myCurrentPoint = point;

	if (buttons & Qt::LeftButton || buttons & Qt::RightButton || buttons & Qt::MidButton) {
		switch (myMode) {
		case CurAction3d_Nothing:
			drawRubberBand(myStartPoint, myCurrentPoint);
			dragEvent(myStartPoint, myCurrentPoint, nFlags & MULTISELECTIONKEY);
			break;

		case CurAction3d_DynamicZooming:
			myView->Zoom(myStartPoint.x(),
				myStartPoint.y(),
				myCurrentPoint.x(),
				myCurrentPoint.y());
			viewPrecision(true);
			myStartPoint = myCurrentPoint;
			break;

		case CurAction3d_WindowZooming:
			drawRubberBand(myStartPoint, myCurrentPoint);
			break;

		case CurAction3d_DynamicPanning:
			myView->Pan(myCurrentPoint.x() - myStartPoint.x(),
				myStartPoint.y() - myCurrentPoint.y());
			myStartPoint = myCurrentPoint;
			break;

		case CurAction3d_GlobalPanning:
			break;

		case CurAction3d_DynamicRotation:
			myView->Rotation(myCurrentPoint.x(), myCurrentPoint.y());
			break;

		default:
			Standard_Failure::Raise("Incompatible Current Mode");
			break;
		}
	}
	else {
		moveEvent(myCurrentPoint);
	}
}


AIS_StatusOfDetection OccViewer::moveEvent(QPoint point)
{
	AIS_StatusOfDetection status = AIS_SOD_Error;
	if (myContext.IsNull()) {
		return status;
	}

	status = myContext->MoveTo(point.x(), point.y(), myView);
	return status;
}



AIS_StatusOfPick OccViewer::dragEvent(const QPoint startPoint, const QPoint endPoint, const bool multi)
{
	using namespace std;
	AIS_StatusOfPick pick = AIS_SOP_NothingSelected;
	if (myContext.IsNull()) {
		return pick;
	}

	if (multi) {
		pick = myContext->ShiftSelect(min(startPoint.x(), endPoint.x()),
			min(startPoint.y(), endPoint.y()),
			max(startPoint.x(), endPoint.x()),
			max(startPoint.y(), endPoint.y()),
			myView);
	}
	else {
		pick = myContext->Select(min(startPoint.x(), endPoint.x()),
			min(startPoint.y(), endPoint.y()),
			max(startPoint.x(), endPoint.x()),
			max(startPoint.y(), endPoint.y()),
			myView);
	}
	emit selectionChanged();
	return pick;
}



AIS_StatusOfPick OccViewer::inputEvent(bool multi)
{
	AIS_StatusOfPick pick = AIS_SOP_NothingSelected;

	if (myContext.IsNull()) {
		return pick;
	}

	if (multi) {
		pick = myContext->ShiftSelect();
	}
	else {
		pick = myContext->Select();
	}
	if (pick != AIS_SOP_NothingSelected) {
		emit selectionChanged();
	}
	return pick;
}

void OccViewer::setMode(const CurrentAction3d mode)
{
	if (mode != myMode) {
		switch (mode) {
		case CurAction3d_DynamicPanning:
			setCursor(Qt::SizeAllCursor);
			break;
		case CurAction3d_DynamicZooming:
			setCursor(Qt::PointingHandCursor);
			break;
		case CurAction3d_DynamicRotation:
			setCursor(Qt::CrossCursor);
			break;
		case CurAction3d_GlobalPanning:
			setCursor(Qt::CrossCursor);
			break;
		case CurAction3d_WindowZooming:
			setCursor(Qt::PointingHandCursor);
			break;
		case CurAction3d_Nothing:
			//setCursor( myCrossCursor );
			setCursor(Qt::CrossCursor);
			break;
		default:
			setCursor(Qt::ArrowCursor);
			break;
		}
		myMode = mode;
	}
}



Standard_Real OccViewer::precision(Standard_Real aReal)
{
	using namespace std;
	Standard_Real preciseReal;
	Standard_Real thePrecision = max(myPrecision, viewPrecision());

	if (myPrecision != 0.0) {
		preciseReal = SIGN(aReal) * floor((std::abs(aReal) + thePrecision * 0.5) / thePrecision) * thePrecision;
	}
	else {
		preciseReal = aReal;
	}
	return preciseReal;
}


Standard_Boolean OccViewer::convertToPlane(Standard_Integer Xs,
	Standard_Integer Ys,
	Standard_Real& X,
	Standard_Real& Y,
	Standard_Real& Z)
{
	Standard_Real Xv, Yv, Zv;
	Standard_Real Vx, Vy, Vz;

	if (myView.IsNull() || myViewer.IsNull()) {
		return Standard_False;
	}

	gp_Pln aPlane(myViewer->PrivilegedPlane());

	myView->Convert(Xs, Ys, Xv, Yv, Zv);

	myView->Proj(Vx, Vy, Vz);
	gp_Lin aLine(gp_Pnt(Xv, Yv, Zv), gp_Dir(Vx, Vy, Vz));
	IntAna_IntConicQuad theIntersection(aLine, aPlane, Precision::Angular());
	if (theIntersection.IsDone()) {
		if (!theIntersection.IsParallel()) {
			if (theIntersection.NbPoints() > 0) {
				gp_Pnt theSolution(theIntersection.Point(1));
				X = theSolution.X();
				Y = theSolution.Y();
				Z = theSolution.Z();
				return Standard_True;
			}
		}
	}
	return Standard_False;
}


void OccViewer::drawRubberBand(const QPoint origin, const QPoint position)
{
	double left = origin.x();
	double right = position.x();
	double top = origin.y();
	double bottom = position.y();

	if (!myContext.IsNull()) {
		// Draw black-white dotted, imitate a shadowy look
		// This makes it possible to draw even on white or
		// black backgrounds
		whiteRect->SetRectangle(left, height() - bottom, right, height() - top);
		blackRect->SetRectangle(left + 1, height() - bottom - 1, right + 1, height() - top - 1);

		if (!myContext->IsDisplayed(whiteRect)) {
			myContext->Display(whiteRect, Standard_False);
			myContext->Display(blackRect, Standard_False);
		}
		else {
			myContext->Redisplay(whiteRect, Standard_False);
			myContext->Redisplay(blackRect, Standard_False);
		}

		myContext->CurrentViewer()->RedrawImmediate();
	}
}


void OccViewer::hideRubberBand(void)
{
	if (!myContext.IsNull()) {
		myContext->Remove(whiteRect, Standard_False);
		myContext->Remove(blackRect, Standard_False);
		myContext->CurrentViewer()->RedrawImmediate();
	}
}


Standard_Real OccViewer::viewPrecision(bool resized)
{

	Standard_Real X1, Y1, Z1;
	Standard_Real X2, Y2, Z2;

	if (resized || myViewPrecision == 0.0) {
		myView->Convert(0, 0, X1, Y1, Z1);
		myView->Convert(1, 0, X2, Y2, Z2);
		Standard_Real pixWidth = X2 - X1;
		if (pixWidth != 0.0) {
			// Return the precision as the next highest decade above the pixel width
			myViewPrecision = std::pow(10.0, std::floor(std::log10(pixWidth) + 1.0));
		}
		else {
			// Return the user precision if window not defined
			myViewPrecision = myPrecision;
		}
	}
	return myViewPrecision;
}

/**
* Note:This function with custom width and height seems to be working only on windows
* If it fails, we are switching back to the current screen resolution.
*/
bool OccViewer::makeScreenshot(const QString& filename, bool whiteBGEnabled, int width, int height, int quality)
{
	if (!myView) {
		return false;
	}

	if (whiteBGEnabled) {
		myView->SetBgGradientColors(Quantity_NOC_BLACK, Quantity_NOC_BLACK, Aspect_GFM_NONE, Standard_False);
		myView->SetBackgroundColor(Quantity_NOC_WHITE);
		myView->Redraw();
	}

	// get window size
	// we could also use a higher resolution if we want
	if (width == 0 || height == 0) {
		myView->Window()->Size(width, height);
	}

	// write screenshot to pixmap
	Image_PixMap pixmap;
	bool success = myView->ToPixMap(pixmap, width, height, Graphic3d_BT_RGB);
	if (!success) {
		// use size of the window
		cout << "Error changing image size to " << width << "x" << height
			<< " for screenshot. Using current resolution.";
		myView->Window()->Size(width, height);
		success = myView->ToPixMap(pixmap, width, height, Graphic3d_BT_RGB);
	}

	if (whiteBGEnabled) {
		// reset color
		setBackgroundGradient(myBGColor.red(), myBGColor.green(), myBGColor.blue());
	}

	// copy to qimage which supports a variety of file formats
	QImage img(QSize((int)pixmap.Width(), (int)pixmap.Height()), QImage::Format_RGB888);
	for (unsigned int aRow = 0; aRow < pixmap.Height(); ++aRow) {
		for (unsigned int aCol = 0; aCol < pixmap.Width(); ++aCol) {
			// extremely SLOW but universal (implemented for all supported pixel formats)
			Quantity_Color aColor = pixmap.PixelColor((Standard_Integer)aCol, (Standard_Integer)aRow);
			QColor qcol(aColor.Red()*255., aColor.Green() * 255, aColor.Blue() * 255);
			img.setPixel(aCol, aRow, qcol.rgb());
		}
	}

	if (!img.save(filename, NULL, quality)) {
		cout << "Unable to save screenshot to file '" + filename.toStdString() + "'";
		return false;
	}
	else {
		return true;
	}

}

void OccViewer::showGrid(Standard_Boolean show){
	//GRID
	if (show){
	myViewer->ActivateGrid(Aspect_GT_Rectangular, Aspect_GDM_Lines);
	myViewer->SetGridEcho(Standard_True);
	myViewer->Grid()->SetColors(Quantity_NOC_RED4, Quantity_NOC_GRAY90);
	gp_Ax3 aPlane(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.));
	myViewer->SetPrivilegedPlane(aPlane);
	}
	else{
	myViewer->DeactivateGrid();
	myViewer->SetGridEcho(Standard_False);
	}
}
