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
#include <StartWindow.h>
#include <ui_StartWindow.h>
#include <OccViewer.h>
#include <QtProcessIndicator.h>
#include <STLVRML_DataSource.h>
#include <AuxiliaryParameters.h>
#include <Message.h>
#include <qnemainwindow.h>


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

StartWindow::StartWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::StartWindow)
{
	ui->setupUi(this);
	setWindowIcon(QIcon(":/Qt/resources/STACCATO.png"));
	setWindowTitle("STACCATO" + QString::fromStdString(STACCATO::AuxiliaryParameters::gitTAG));
	myOccViewer = new OccViewer(this);
	setCentralWidget(myOccViewer);
	createActions();
	createMenus();
	createToolBars();
	createDockWindows();
}

StartWindow::~StartWindow()
{
	delete myOccViewer;
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

	mReadFileAction = new QAction(tr("Import file"), this);
	mReadFileAction->setIcon(QIcon(":/Qt/resources/openDoc.png"));
	mReadFileAction->setStatusTip(tr("Import 3D file"));
	connect(mReadFileAction, SIGNAL(triggered()), this, SLOT(importFile()));

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


	// Create actions
	mDrawCantileverAction = new QAction(tr("Draw Cantilever"), this);
	mDrawCantileverAction->setIcon(QIcon(":/Qt/resources/torus.png"));
	mDrawCantileverAction->setStatusTip(tr("Draw Cantilever"));
	connect(mDrawCantileverAction, SIGNAL(triggered()), this, SLOT(drawCantilever()));

	mDataFlowAction = new QAction(tr("Dataflow manager"), this);
	mDataFlowAction->setIcon(QIcon(":/Qt/resources/dataflow.png"));
	mDataFlowAction->setStatusTip(tr("Open dataflow manager"));
	connect(mDataFlowAction, SIGNAL(triggered()), this, SLOT(openDataFlowWindow()));

	//Help actions
	mAboutAction = new QAction(tr("About"), this);
	mAboutAction->setStatusTip(tr("About the application"));
	mAboutAction->setIcon(QIcon(":/Qt/resources/about.png"));
	connect(mAboutAction, SIGNAL(triggered()), this, SLOT(about()));

	connect(myOccViewer, SIGNAL(selectionChanged()), this, SLOT(handleSelectionChanged()));

}

void StartWindow::createMenus(void)
{
	mFileMenu = menuBar()->addMenu(tr("&File"));
	mFileMenu->addAction(mReadFileAction);
	mFileMenu->addAction(mExitAction);

	mCreateMenu = menuBar()->addMenu(tr("Create"));
	mCreateMenu->addAction(mDataFlowAction);
	mCreateMenu->addAction(mDrawCantileverAction);

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
	textOutput->setEnabled(false);
	dock->setWidget(textOutput);
	addDockWidget(Qt::BottomDockWidgetArea, dock);


	//Set textOutput to message streams 
	debugOut.setQTextEditReference(textOutput);
	infoOut.setQTextEditReference(textOutput);
	warningOut.setQTextEditReference(textOutput);
	errorOut.setQTextEditReference(textOutput);


	infoOut << "Hello STACCATO is fired up!" << std::endl;
	debugOut << "GIT: " << STACCATO::AuxiliaryParameters::gitSHA1 << std::endl;

	debugOut << "debugOut" << std::endl;
	infoOut << "infoOut" << std::endl;
	warningOut << "warningOut" << std::endl;
	errorOut << "errorOut" << std::endl;

}


void StartWindow::about()
{
	myOccViewer->showGrid(Standard_True);
	myOccViewer->viewTop();
	myOccViewer->fitAll();
	myOccViewer->viewGrid();
	QMessageBox::about(this, tr("About STACCATO"),
		tr("<h2>STACCATO: STefAn's Computational vibroaCoustics Analysis TOol</h2>"
		"<p>Copyright &copy; 2016 "
		"<p>STACCATO is using Qt and OpenCASCADE."));
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
	Handle(Message_ProgressIndicator) aIndicator = new QtProcessIndicator(this);
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
	//myOccViewer->getContext()->Activate(aMesh, 1); // Node selection
	myOccViewer->getContext()->Activate(aMesh, 8); // Element selection
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

void StartWindow::handleSelectionChanged(void){

	for (myOccViewer->getContext()->InitSelected(); myOccViewer->getContext()->MoreSelected(); myOccViewer->getContext()->NextSelected())
	{
		Handle(AIS_InteractiveObject) anIO = myOccViewer->getContext()->SelectedInteractive();
		cout << "anIO->Type() : " << anIO->Type() << endl;

		if (anIO->Type() == AIS_KOI_None){
				Handle(SelectMgr_Selection) aSelection = anIO->CurrentSelection();
				Handle(SelectMgr_EntityOwner) aEntOwn = myOccViewer->getContext()->SelectedOwner();

				// If statement to check for valid for DownCast
				Handle_MeshVS_MeshEntityOwner owner = Handle_MeshVS_MeshEntityOwner::DownCast(aEntOwn);
				Handle(MeshVS_Mesh) aisMesh = Handle(MeshVS_Mesh)::DownCast(anIO);
				Handle_MeshVS_DataSource source = aisMesh->GetDataSource();
				Handle_MeshVS_Drawer drawer = aisMesh->GetDrawer();


				if (owner->Type() == MeshVS_ET_Face)
				{
					int maxFaceNodes;
					if (drawer->GetInteger(MeshVS_DA_MaxFaceNodes, maxFaceNodes) && maxFaceNodes > 0)
					{
						MeshVS_Buffer coordsBuf(3 * maxFaceNodes * sizeof(Standard_Real));
						TColStd_Array1OfReal coords(coordsBuf, 1, 3 * maxFaceNodes);

						int nbNodes = 0;
						MeshVS_EntityType entityType;
						if (source->GetGeom(owner->ID(), Standard_True, coords, nbNodes, entityType))
						{
							if (nbNodes >= 3)
							{
								gp_Pnt p1 = gp_Pnt(coords(1), coords(2), coords(3));
								gp_Pnt p2 = gp_Pnt(coords(4), coords(5), coords(6));
								gp_Pnt p3 = gp_Pnt(coords(7), coords(8), coords(9));


								infoOut << "==========" << std::endl;
								infoOut << "X: " << p1.X() << std::endl;
								infoOut << "Y: " << p2.Y() << std::endl;
								infoOut << "Z: " << p3.Z() << std::endl;
								infoOut << "==========" << std::endl;

								// do something with p1, p2 and p3
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
			cout << "anIO: " << anIO->Signature() << endl;
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
			cout << "TopoDS_Shape: " << vertexShape.ShapeType() << endl;
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
