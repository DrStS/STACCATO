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
#include "StartWindow.h"
#include "ui_StartWindow.h"
#include "OccViewer.h"
#include "VtkViewer.h"
#include "OcctQtProcessIndicator.h"
#include "STLVRML_DataSource.h"
#include "AuxiliaryParameters.h"
#include "Message.h"
#include "SimuliaODB.h"
#include "HMeshToMeshVS_DataSource.h"
#include "FeMetaDatabase.h"
#include "FeAnalysis.h"
#include "Timer.h"
#include "MemWatcher.h"
#include "qnemainwindow.h"
#include "HMesh.h"

//QT5
#include <QToolBar>
#include <QTreeView>
#include <QMessageBox>
#include <QDockWidget>
#include <QtWidgets>

//OCC 7
#include <StlMesh_Mesh.hxx> 
#include <MeshVS_Mesh.hxx>
#include <MeshVS_MeshPrsBuilder.hxx>
#include <MeshVS_Drawer.hxx>
#include <RWStl.hxx>
#include <MeshVS_DrawerAttribute.hxx>
#include <Graphic3d_MaterialAspect.hxx>
#include <OSD_Path.hxx>
#include <Geom_CartesianPoint.hxx>
#include <AIS_Line.hxx>
#include <AIS_Point.hxx>
#include <TopoDS_Vertex.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <Prs3d_PointAspect.hxx>
#include <GC_MakeSegment.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <TopoDS_Edge.hxx>
#include <AIS_InteractiveContext.hxx>
#include <BRep_Tool.hxx>
#include <TopoDS.hxx>
#include <Geom2d_CartesianPoint.hxx>
#include <ElCLib.hxx>
#include <MeshVS_SelectionModeFlags.hxx>
#include <TColStd_HPackedMapOfInteger.hxx>
#include <Select3D_SensitiveTriangle.hxx>
#include <MeshVS_MeshEntityOwner.hxx>
#include <Select3D_SensitiveTriangulation.hxx>
#include <Select3D_SensitiveFace.hxx>
#include <MeshVS_CommonSensitiveEntity.hxx>
#include <MeshVS_Buffer.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPConstruct.hxx>
#include <IGESControl_Reader.hxx>
#include <Geom_Plane.hxx>
#include <AIS_Plane.hxx>
#include <Prs3d_PlaneAspect.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <MeshVS_NodalColorPrsBuilder.hxx>
#include <AIS_ColorScale.hxx>

StartWindow::StartWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::StartWindow)
{
	ui->setupUi(this);
	setWindowIcon(QIcon(":/Qt/resources/STACCATO.png"));
	setWindowTitle("STACCATO" + QString::fromStdString(STACCATO::AuxiliaryParameters::gitTAG));
	//myOccViewer = new OccViewer(this);
	myVtkViewer = new VtkViewer(this);
	setCentralWidget(myVtkViewer);
	createActions();
	createMenus();
	createToolBars();
	createDockWindows();
	myVtkViewer->demo();
	resize(QDesktopWidget().availableGeometry(this).size() * 0.8);
}

StartWindow::~StartWindow()
{
	//delete myOccViewer;
	delete myVtkViewer;
}

void StartWindow::openDataFlowWindow(void){
	newWin = new QNEMainWindow();
	newWin->show();
}

void StartWindow::createActions(void)
{
	// File actions
	mExitAction = new QAction(tr("Exit"), this);
	mExitAction->setShortcut(tr("Ctrl+Q"));
	mExitAction->setIcon(QIcon(":/Qt/resources/closeDoc.png"));
	mExitAction->setStatusTip(tr("Exit the application"));
	connect(mExitAction, SIGNAL(triggered()), this, SLOT(close()));

	mReadOBDFileAction = new QAction(tr("Open OBD file"), this);
	mReadOBDFileAction->setStatusTip(tr("Read Abaqus OBD file"));
	connect(mReadOBDFileAction, SIGNAL(triggered()), this, SLOT(openOBDFile()));

	mReadFileAction = new QAction(tr("Import file"), this);
	mReadFileAction->setIcon(QIcon(":/Qt/resources/openDoc.png"));
	mReadFileAction->setStatusTip(tr("Import 3D file"));
	connect(mReadFileAction, SIGNAL(triggered()), this, SLOT(importFile()));

/*
	// View actions
	mPanAction = new QAction(tr("Pan"), this);
	mPanAction->setIcon(QIcon(":/Qt/resources/pan.png"));
	mPanAction->setStatusTip(tr("Panning the view"));
	connect(mPanAction, SIGNAL(triggered()), myOccViewer, SLOT(pan()));

	mZoomAction = new QAction(tr("Zoom"), this);
	mZoomAction->setIcon(QIcon(":/Qt/resources/zoom.png"));
	mZoomAction->setStatusTip(tr("Zooming the view"));
	connect(mZoomAction, SIGNAL(triggered()), myOccViewer, SLOT(zoom()));

	mFitAllAction = new QAction(tr("Zoom fit all"), this);
	mFitAllAction->setIcon(QIcon(":/Qt/resources/fitAll.png"));
	mFitAllAction->setStatusTip(tr("Fit the view to show all"));
	connect(mFitAllAction, SIGNAL(triggered()), myOccViewer, SLOT(fitAll()));

	mRotateAction = new QAction(tr("Rotate"), this);
	mRotateAction->setIcon(QIcon(":/Qt/resources/rotate.png"));
	mRotateAction->setStatusTip(tr("Rotate the view"));
	connect(mRotateAction, SIGNAL(triggered()), myOccViewer, SLOT(rotation()));

	connect(myOccViewer, SIGNAL(selectionChanged()), this, SLOT(handleSelectionChanged()));
*/

	// Create actions
	mDrawCantileverAction = new QAction(tr("Draw Cantilever"), this);
	mDrawCantileverAction->setIcon(QIcon(":/Qt/resources/torus.png"));
	mDrawCantileverAction->setStatusTip(tr("Draw Cantilever"));
	connect(mDrawCantileverAction, SIGNAL(triggered()), this, SLOT(drawCantilever()));

	mDataFlowAction = new QAction(tr("Dataflow manager"), this);
	mDataFlowAction->setIcon(QIcon(":/Qt/resources/dataflow.png"));
	mDataFlowAction->setStatusTip(tr("Open dataflow manager"));
	connect(mDataFlowAction, SIGNAL(triggered()), this, SLOT(openDataFlowWindow()));

	mAnimationAction = new QAction(tr("Animate object"), this);
	mAnimationAction->setStatusTip(tr("Animate object"));
	connect(mAnimationAction, SIGNAL(triggered()), this, SLOT(animateObject()));

	//Help actions
	mAboutAction = new QAction(tr("About"), this);
	mAboutAction->setStatusTip(tr("About the application"));
	mAboutAction->setIcon(QIcon(":/Qt/resources/about.png"));
	connect(mAboutAction, SIGNAL(triggered()), this, SLOT(about()));

	

}

void StartWindow::createMenus(void)
{
	mFileMenu = menuBar()->addMenu(tr("&File"));
	mFileMenu->addAction(mReadFileAction);
	mFileMenu->addAction(mReadOBDFileAction);
	mFileMenu->addAction(mExitAction);

	mCreateMenu = menuBar()->addMenu(tr("Create"));
	mCreateMenu->addAction(mDataFlowAction);
	mCreateMenu->addAction(mDrawCantileverAction);
	mCreateMenu->addAction(mAnimationAction);

	mHelpMenu = menuBar()->addMenu(tr("&Help"));
	mHelpMenu->addAction(mAboutAction);
}

void StartWindow::createToolBars(void)
{
	mFileToolBar = addToolBar(tr("&File"));
	mFileToolBar->addAction(mReadFileAction);
	mCreateToolBar = addToolBar(tr("Create"));
	mCreateToolBar->addAction(mDataFlowAction);
	mViewToolBar = addToolBar(tr("View"));
	mViewToolBar->addAction(mPanAction);
	mViewToolBar->addAction(mZoomAction);
	mViewToolBar->addAction(mFitAllAction);
	mViewToolBar->addAction(mRotateAction);
	mHelpToolBar = addToolBar(tr("Help"));
	mHelpToolBar->addAction(mAboutAction);
}

void StartWindow::createDockWindows()
{
	QDockWidget *dock = new QDockWidget(tr("Output"), this);
	dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
	
	//Create dockable widget
	textOutput = new QTextEdit(dock);
	QPalette pal;
	pal.setColor(QPalette::Base, Qt::gray);
	pal.setColor(QPalette::WindowText, Qt::green);
	pal.setColor(QPalette::Text, Qt::green);
	textOutput->setPalette(pal);
	//textOutput->setEnabled(false);
	dock->setWidget(textOutput);
	addDockWidget(Qt::BottomDockWidgetArea, dock);


	//Set textOutput to message streams 
	debugOut.setQTextEditReference(textOutput);
	infoOut.setQTextEditReference(textOutput);
	warningOut.setQTextEditReference(textOutput);
	errorOut.setQTextEditReference(textOutput);


	infoOut << "Hello STACCATO is fired up!" << std::endl;
	debugOut << "GIT: " << STACCATO::AuxiliaryParameters::gitSHA1 << std::endl;

	//debugOut << "debugOut" << std::endl;
	//infoOut << "infoOut" << std::endl;
	//warningOut << "warningOut" << std::endl;
	//errorOut << "errorOut" << std::endl;
	//infoOut << QThread::currentThread() << std::endl;

}


void StartWindow::about()
{
	myOccViewer->showGrid(Standard_True);
	myOccViewer->viewTop();
	myOccViewer->fitAll();
	myOccViewer->viewGrid();
	QMessageBox::about(this, tr("About STACCATO"),
		tr("<h2>STACCATO: STefAn's Computational vibroaCoustics Analysis TOol</h2>"
		"<p>Copyright &copy; 2017 "
		"<p>STACCATO is using Qt and OpenCASCADE."));
}


void StartWindow::openOBDFile(void){
	
	QString myWorkingFolder = "";
	QString fileType;
	QFileInfo fileInfo;

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open odb file"), myWorkingFolder, tr("ODB (*.odb)"));

	fileInfo.setFile(fileName);
	fileType = fileInfo.suffix();
	if (!fileName.isEmpty() && !fileName.isNull()){
		if (fileType.toLower() == tr("odb") ) {
			infoOut << "ODB file: " << fileName.toStdString() << std::endl;
		}
	}

	SimuliaODB myOBD =  SimuliaODB();
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	anaysisTimer01.start();
	anaysisTimer03.start();
	myOBD.openODBFile(fileName.toStdString());
	anaysisTimer01.stop();
	debugOut << "Duration for reading odb file: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory()/1000000 << " Mb" << std::endl;
	anaysisTimer01.start();
	Handle(MeshVS_DataSource) aDataSource = new HMeshToMeshVS_DataSource(*myOBD.getHMeshHandle());
	anaysisTimer01.stop();
	debugOut << "Duration for reading HMeshToMeshVS_DataSource " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	anaysisTimer01.start();
	Handle(MeshVS_Mesh) aMesh = new MeshVS_Mesh();
	aMesh->SetDataSource(aDataSource);
	
	/*aMesh->AddBuilder(new MeshVS_MeshPrsBuilder(aMesh), Standard_True);//False -> No selection
	aMesh->GetDrawer()->SetBoolean(MeshVS_DA_DisplayNodes, Standard_False); //MeshVS_DrawerAttribute
	aMesh->GetDrawer()->SetBoolean(MeshVS_DA_ShowEdges, Standard_False);
	aMesh->GetDrawer()->SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_NOM_BRASS);
	aMesh->SetColor(Quantity_NOC_AZURE);
	aMesh->SetDisplayMode(MeshVS_DMF_Shading); // Mode as defaut
	aMesh->SetHilightMode(MeshVS_DMF_WireFrame); // Wireframe as default hilight mode*/


	// assign nodal builder to the mesh
	Handle(MeshVS_NodalColorPrsBuilder) aBuilder = new MeshVS_NodalColorPrsBuilder(aMesh, MeshVS_DMF_NodalColorDataPrs | MeshVS_DMF_OCCMask); 
	aBuilder->UseTexture(Standard_True);
	// prepare color map
	Aspect_SequenceOfColor  aColorMap;
	aColorMap.Append((Quantity_NameOfColor)Quantity_NOC_RED);
	aColorMap.Append((Quantity_NameOfColor)Quantity_NOC_BLUE1);
	// assign color scale map  values (0..1) to nodes
	TColStd_DataMapOfIntegerReal  aScaleMap;
	// iterate through the  nodes and add an node id and an appropriate value to the map

	Handle(TColStd_HPackedMapOfInteger) aNodes = new TColStd_HPackedMapOfInteger();
	Standard_Integer aLen = (myOBD.getHMeshHandle())->getNumNodes();
	for (Standard_Integer anIndex = 1; anIndex <= aLen; anIndex++) {
		double someNumber = (double)rand() / (RAND_MAX);
		aScaleMap.Bind(anIndex, someNumber);
	}

	// pass color map and color scale values to the builder
	aBuilder->SetColorMap(aColorMap);
	aBuilder->SetInvalidColor(Quantity_NOC_BLACK);
	aBuilder->SetTextureCoords(aScaleMap);
	aMesh->AddBuilder(aBuilder, Standard_True);
	aMesh->SetDisplayMode(MeshVS_DMF_NodalColorDataPrs); // Mode as defaut

	Handle(AIS_ColorScale) aCS = new AIS_ColorScale();
	// configuring
	Standard_Integer aWidth, aHeight;
	myOccViewer->getView()->Window()->Size(aWidth, aHeight);
	aCS->SetSize(aWidth, aHeight);
	aCS->SetRange(0.0, 10.0);
	aCS->SetNumberOfIntervals(10);
	// displaying
	aCS->SetZLayer(Graphic3d_ZLayerId_TopOSD);
	aCS->SetTransformPersistence(Graphic3d_TMF_2d, gp_Pnt(-1, -1, 0));
	aCS->SetToUpdate();
	myOccViewer->getContext()->Display(aCS);

	// Hide all nodes by default
	/*Handle(TColStd_HPackedMapOfInteger) aNodes = new TColStd_HPackedMapOfInteger();
	Standard_Integer aLen = (myOBD.getHMeshHandle())->getNumNodes();
	for (Standard_Integer anIndex = 1; anIndex <= aLen; anIndex++){
		aNodes->ChangeMap().Add(anIndex);
	}*/

	//aMesh->SetHiddenNodes(aNodes);
	//aMesh->SetSelectableNodes(aNodes);
	myOccViewer->getContext()->Display(aMesh);
	myOccViewer->getContext()->Deactivate(aMesh);
	myOccViewer->getContext()->Load(aMesh, -1, Standard_True);
	//myOccViewer->getContext()->Activate(aMesh, 1); // Node selection
	myOccViewer->getContext()->Activate(aMesh, 8); // Element selection
	anaysisTimer01.stop();
	debugOut << "Duration for build and display Hmesh: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	//Run FE Analysis
	FeMetaDatabase *mFeMetaDatabase = new FeMetaDatabase();
	FeAnalysis *mFeAnalysis = new FeAnalysis(*myOBD.getHMeshHandle(), *mFeMetaDatabase);

	anaysisTimer03.stop();
	debugOut << "Duration for STACCATO Finite Element run: " << anaysisTimer03.getDurationSec() << " sec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

}

void StartWindow::importFile(void)
{

	QString myWorkingFolder = "";
	QString fileType;
	QFileInfo fileInfo;

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Import file"), myWorkingFolder, tr(
		"STEP (*.step *.stp);;"
		"IGES (*.iges *.igs);;"
		"STL  (*.stl)"));

	fileInfo.setFile(fileName);
	fileType = fileInfo.suffix();
	if (!fileName.isEmpty() && !fileName.isNull()){
		if (fileType.toLower() == tr("step") || fileType.toLower() == tr("stp")) {
			readSTEP(fileName);
		}
		if (fileType.toLower() == tr("iges") || fileType.toLower() == tr("igs")) {
			readIGES(fileName);
		}
		if (fileType.toLower() == tr("stl")) {
			readSTL(fileName);
		}

	}
}

void StartWindow::readSTEP(QString fileName){
	// create additional log file
	STEPControl_Reader aReader;
	IFSelect_ReturnStatus status = aReader.ReadFile(fileName.toUtf8().constData());
	if (status != IFSelect_RetDone){
		return;
	}

	Standard_Boolean failsonly = Standard_False;
	aReader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity);

	// Root transfers
	Standard_Integer nbr = aReader.NbRootsForTransfer();
	aReader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity);
	for (Standard_Integer n = 1; n <= nbr; n++) {
		/*Standard_Boolean ok =*/ aReader.TransferRoot(n);
	}

	// Collecting resulting entities
	Standard_Integer nbs = aReader.NbShapes();
	if (nbs == 0) {
		return;
	}
	for (Standard_Integer i = 1; i <= nbs; i++) {
		Handle(AIS_Shape) aisShape = new AIS_Shape(aReader.Shape(i));
		myOccViewer->getContext()->Display(aisShape);
	}

}

void StartWindow::readIGES(QString fileName){

	IGESControl_Reader Reader;

	Standard_Integer status = Reader.ReadFile(fileName.toUtf8().constData());

	if (status != IFSelect_RetDone) return;
	Reader.TransferRoots();
	Handle(AIS_Shape) aisShape = new AIS_Shape(Reader.OneShape());
	myOccViewer->getContext()->Display(aisShape);

}


void StartWindow::readSTL(QString fileName){
	Handle(Message_ProgressIndicator) aIndicator = new OcctQtProcessIndicator(this);
	aIndicator->SetRange(0, 100);
	OSD_Path aFile(fileName.toUtf8().constData());
	Handle(StlMesh_Mesh) aSTLMesh = RWStl::ReadFile(aFile, aIndicator);
	Handle(MeshVS_Mesh) aMesh = new MeshVS_Mesh();
	Handle(STLVRML_DataSource) aDS = new STLVRML_DataSource(aSTLMesh);
	aMesh->SetDataSource(aDS);
	aMesh->AddBuilder(new MeshVS_MeshPrsBuilder(aMesh), Standard_True);//False -> No selection
	aMesh->GetDrawer()->SetBoolean(MeshVS_DA_DisplayNodes, Standard_False); //MeshVS_DrawerAttribute
	aMesh->GetDrawer()->SetBoolean(MeshVS_DA_ShowEdges, Standard_False);
	aMesh->GetDrawer()->SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_NOM_BRASS);
	aMesh->SetColor(Quantity_NOC_AZURE);
	aMesh->SetDisplayMode(MeshVS_DMF_Shading); // Mode as defaut
	aMesh->SetHilightMode(MeshVS_DMF_WireFrame); // Wireframe as default hilight mode
	aMesh->GetDrawer()->SetColor(MeshVS_DA_EdgeColor, Quantity_NOC_YELLOW);

	// Hide all nodes by default
	Handle(TColStd_HPackedMapOfInteger) aNodes = new TColStd_HPackedMapOfInteger();
	Standard_Integer aLen = aSTLMesh->Vertices().Length();
	for (Standard_Integer anIndex = 1; anIndex <= aLen; anIndex++){
		aNodes->ChangeMap().Add(anIndex);
	}
	aMesh->SetHiddenNodes(aNodes);
	aMesh->SetSelectableNodes(aNodes);
	myOccViewer->getContext()->Display(aMesh);
	myOccViewer->getContext()->Deactivate(aMesh);
	myOccViewer->getContext()->Load(aMesh, -1, Standard_True);
	myOccViewer->getContext()->Activate(aMesh, 1); // Node selection
	//myOccViewer->getContext()->Activate(aMesh, 8); // Element selection
}

//void StartWindow::drawInt2DLine(void){
//}

void StartWindow::drawCantilever(void){
	//3D cartesian point
	gp_Pnt mGp_Pnt_Start = gp_Pnt(0., 0., 0.);
	gp_Pnt mGp_Pnt_End = gp_Pnt(0., 100., 100.);

	//Geom_CartesianPoint
	Handle(Geom_CartesianPoint)  start = new Geom_CartesianPoint(mGp_Pnt_Start);
	Handle(Geom_CartesianPoint)    end = new Geom_CartesianPoint(mGp_Pnt_End);

	Handle(AIS_Line) aisSegmentA = new AIS_Line(start, end);
	aisSegmentA->SetColor(Quantity_NOC_GREEN);
	aisSegmentA->SetWidth(2.);
	//	myOccViewer->getContext()->Display(aisSegmentA);


	Handle(Geom_TrimmedCurve) aTrimmedCurve = GC_MakeSegment(mGp_Pnt_Start, mGp_Pnt_End);
	TopoDS_Edge mTopoEdge = BRepBuilderAPI_MakeEdge(aTrimmedCurve);

	Handle(AIS_Shape) aisSegmentB = new AIS_Shape(mTopoEdge);
	aisSegmentB->SetColor(Quantity_NOC_RED);
	aisSegmentB->SetWidth(2.);
	//myOccViewer->getContext()->Display(aisSegmentB,1,1);


	//Draw vertex
	Handle(AIS_Point) aPointA = new AIS_Point(start);
	// Set the vertex shape, color, and size
	Handle_Prs3d_PointAspect myPointAspectA = new Prs3d_PointAspect(Aspect_TOM_O, Quantity_NOC_RED, 2);
	aPointA->Attributes()->SetPointAspect(myPointAspectA);
	//myOccViewer->getContext()->Display(aPointA);

	// Create the AIS_Shape
	TopoDS_Vertex V1 = BRepBuilderAPI_MakeVertex(mGp_Pnt_End);
	Handle(AIS_Shape) aPointB = new AIS_Shape(V1);
	Handle_Prs3d_PointAspect myPointAspectB = new Prs3d_PointAspect(Aspect_TOM_O, Quantity_NOC_GREEN, 2);
	aPointB->Attributes()->SetPointAspect(myPointAspectB);
	myOccViewer->getContext()->Display(aPointB);



	//============ 2D Stuff
	gp_Pnt2d mGp_Pnt_Start_2D = gp_Pnt2d(11., 10.);
	Handle(Geom2d_CartesianPoint) myGeom2d_Point = new Geom2d_CartesianPoint(mGp_Pnt_Start_2D);
	gp_Ax3	curCoordinateSystem = gp_Ax3();
	Handle(Geom_CartesianPoint) myGeom_Point = new Geom_CartesianPoint(ElCLib::To3d(curCoordinateSystem.Ax2(), mGp_Pnt_Start_2D));
	Handle(AIS_Point) myAIS_Point = new AIS_Point(myGeom_Point);
	myOccViewer->getContext()->Display(myAIS_Point);
}

void StartWindow::animateObject(void){

	gp_Pnt mGp_Pnt_Start = gp_Pnt(-150., -150., -0.0001);
	gp_Pnt mGp_Pnt_End = gp_Pnt(150., 150., 0.);
	TopoDS_Solid theBox = BRepPrimAPI_MakeBox(mGp_Pnt_Start, mGp_Pnt_End);
	Handle(AIS_Shape) aisBox = new AIS_Shape(theBox);

	Quantity_Color TUMblue(0, 0.3960, 0.7411, Quantity_TOC_RGB);

	aisBox->SetColor(TUMblue);
	myOccViewer->getContext()->Display(aisBox);
	myOccViewer->getContext()->SetDeviationCoefficient(0.0001);
	Handle(AIS_Shape) aisShape;

	STEPControl_Reader aReader;
	IFSelect_ReturnStatus status = aReader.ReadFile("KreiselTrafo.stp");
	if (status != IFSelect_RetDone){
		return;
	}

	Standard_Boolean failsonly = Standard_False;
	aReader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity);

	// Root transfers
	Standard_Integer nbr = aReader.NbRootsForTransfer();
	aReader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity);
	for (Standard_Integer n = 1; n <= nbr; n++) {
		 aReader.TransferRoot(n);
	}

	// Collecting resulting entities
	Standard_Integer nbs = aReader.NbShapes();
	if (nbs == 0) {
		return;
	}
	for (Standard_Integer i = 1; i <= nbs; i++) {
		aisShape = new AIS_Shape(aReader.Shape(i));
		myOccViewer->getContext()->Display(aisShape);
	}

	int numSteps = 2000;

	Standard_Real  a11;
	Standard_Real  a12;
	Standard_Real  a13;
	Standard_Real  a14 = 0.0;
	Standard_Real  a21;
	Standard_Real  a22;
	Standard_Real  a23;
	Standard_Real  a24 = 0.0;
	Standard_Real  a31;
	Standard_Real  a32;
	Standard_Real  a33;
	Standard_Real  a34 = 0.0;
	Standard_Real  phi;
	Standard_Real  alpha;
	Standard_Real  gamma=asin(5.0/12.0);

	for (int i = 0; i < numSteps; i++){
		gp_Trsf myTrafo;
		phi = ((double)i / numSteps) * 3600 * (M_PI / 180);
		alpha = (12 / (5*cos(gamma)))*phi;

		a11 = cos(phi)*cos(-gamma);
		a12 = -sin(phi)*cos(-alpha) + cos(phi)*sin(-gamma)*sin(-alpha);
		a13 = sin(phi)*sin(-alpha) + cos(phi)*sin(-gamma)*cos(-alpha);;
		a14 = 0.0;
		a21 = sin(phi)*cos(-gamma);
		a22 = cos(phi)*cos(-alpha) + sin(phi)*sin(-gamma)*sin(-alpha);
		a23 = -cos(phi)*sin(-alpha) + sin(phi)*sin(-gamma)*cos(-alpha);
		a24 = 0.0;
		a31 = -sin(-gamma);
		a32 = cos(-gamma)*sin(-alpha);
		a33 = cos(-gamma)*cos(-alpha);
		a34 = 0.0;
		myTrafo.SetValues(a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34);
		myOccViewer->getContext()->SetLocation(aisShape, myTrafo);
		myOccViewer->getContext()->UpdateCurrentViewer();
		QCoreApplication::processEvents();

	}

}


void StartWindow::handleSelectionChanged(void){

	for (myOccViewer->getContext()->InitSelected(); myOccViewer->getContext()->MoreSelected(); myOccViewer->getContext()->NextSelected())
	{
		Handle(AIS_InteractiveObject) anIO = myOccViewer->getContext()->SelectedInteractive();
		infoOut << "anIO->Type() : " << anIO->Type() << std::endl;

		if (anIO->Type() == AIS_KOI_None){
				Handle(SelectMgr_Selection) aSelection = anIO->CurrentSelection();
				Handle(SelectMgr_EntityOwner) aEntOwn = myOccViewer->getContext()->SelectedOwner();

				// If statement to check for valid for DownCast
				Handle_MeshVS_MeshEntityOwner owner = Handle_MeshVS_MeshEntityOwner::DownCast(aEntOwn);
				Handle(MeshVS_Mesh) aisMesh = Handle(MeshVS_Mesh)::DownCast(anIO);
				Handle_MeshVS_DataSource source = aisMesh->GetDataSource();
				Handle_MeshVS_Drawer drawer = aisMesh->GetDrawer();

				infoOut << "AIS_KOI_None -> owner->Type(): " << owner->Type() << std::endl;

				if (owner->Type() == MeshVS_ET_Face)
				{
					int maxFaceNodes;
					if (drawer->GetInteger(MeshVS_DA_MaxFaceNodes, maxFaceNodes) && maxFaceNodes > 0)
					{
						MeshVS_Buffer coordsBuf(3 * maxFaceNodes * sizeof(Standard_Real));
						TColStd_Array1OfReal coords(coordsBuf, 1, 3 * maxFaceNodes);

						int nbNodes = 0;
						MeshVS_EntityType entityType;
						infoOut << "A Element" << std::endl;
						if (source->GetGeom(owner->ID(), Standard_True, coords, nbNodes, entityType))
						{
							if (nbNodes >= 3)
							{
								infoOut << "==========" << std::endl;
								infoOut << "ID: " << owner->ID() << std::endl;
								int i;
								for (i = 0; i < nbNodes; i++)
								{
									infoOut << "Node: " << i << std::endl;
									gp_Pnt p = gp_Pnt(coords((i * 3) + 1), coords((i * 3) + 2), coords((i * 3) + 3));
									infoOut << "X: " << p.X() << std::endl;
									infoOut << "Y: " << p.Y() << std::endl;
									infoOut << "Z: " << p.Z() << std::endl;
								}
								infoOut << "==========" << std::endl;
							}
						}
					}

				}
				else if (owner->Type() == MeshVS_ET_Node){
					infoOut << "A Node" << endl;
					int maxNodes = 1;
					MeshVS_Buffer coordsBuf(3 * maxNodes * sizeof(Standard_Real));
					TColStd_Array1OfReal coords(coordsBuf, maxNodes, 3);
					int nbNodes = 0;
					MeshVS_EntityType entityType;
					if (source->GetGeom(owner->ID(), Standard_False, coords, nbNodes, entityType))
					{
						if (nbNodes == 1)
						{
							gp_Pnt p1 = gp_Pnt(coords(1), coords(2), coords(3));
							infoOut << "ID: " << owner->ID() << std::endl;
							infoOut << "==========" << std::endl;
							infoOut << "X: " << p1.X() << std::endl;
							infoOut << "Y: " << p1.Y() << std::endl;
							infoOut << "Z: " << p1.Z() << std::endl;
							infoOut << "==========" << std::endl;
						}
					}
				}

		}
		else if (anIO->Type() == AIS_KOI_Datum){
			infoOut << "anIO: " << anIO->Signature() << endl;
			if (anIO->Signature() == 1){//datum point
				Handle(AIS_Point) aAISPoint = Handle(AIS_Point)::DownCast(anIO);
				TopoDS_Vertex vertex = aAISPoint->Vertex();
				gp_Pnt myPoint = BRep_Tool::Pnt(TopoDS::Vertex(vertex));
				infoOut << "==========" << std::endl;
				infoOut << "X: " << myPoint.X() << std::endl;
				infoOut << "Y: " << myPoint.Y() << std::endl;
				infoOut << "Z: " << myPoint.Z() << std::endl;
				infoOut << "==========" << std::endl;
			}
			else if (anIO->Signature() == 1){//datum axis

			}
		}
		else if (anIO->Type() == AIS_KOI_Shape){
			TopoDS_Shape vertexShape = Handle(AIS_Shape)::DownCast(anIO)->Shape();
			infoOut << "TopoDS_Shape: " << vertexShape.ShapeType() << endl;
			if (TopAbs_VERTEX == vertexShape.ShapeType())
			{
				gp_Pnt myPoint = BRep_Tool::Pnt(TopoDS::Vertex(vertexShape));
				infoOut << "==========" << std::endl;
				infoOut << "X: " << myPoint.X() << std::endl;
				infoOut << "Y: " << myPoint.Y() << std::endl;
				infoOut << "Z: " << myPoint.Z() << std::endl;
				infoOut << "==========" << std::endl;
			}

		}
		else if (anIO->Type() == AIS_KOI_Object){

		}
		else if (anIO->Type() == AIS_KOI_Relation){

		}
		else if (anIO->Type() == AIS_KOI_Dimension){

		}





	}

}
