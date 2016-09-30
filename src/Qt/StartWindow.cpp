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

//QT5
#include <QToolBar>
#include <QTreeView>
#include <QMessageBox>
#include <QDockWidget>
#include <QtWidgets>

//OCC 7
#include <StlMesh_Mesh.hxx> 
#include <MeshVS_Mesh.hxx>
#include <XSDRAWSTLVRML_DataSource.hxx>
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
StartWindow::StartWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::StartWindow)
{
	ui->setupUi(this);
	setWindowIcon(QIcon(":/Qt/resources/FitAll.png"));
	myOccViewer = new OccViewer(this);
	setCentralWidget(myOccViewer);
	createActions();
	createMenus();
	createToolBars();
	createDockWindows();
	setAttribute(Qt::WA_QuitOnClose, Standard_False);
}

StartWindow::~StartWindow()
{
}

void StartWindow::createActions(void)
{
	mExitAction = new QAction(tr("Exit"), this);
	mExitAction->setShortcut(tr("Ctrl+Q"));
	mExitAction->setIcon(QIcon(":/Qt/resources/close.png"));
	mExitAction->setStatusTip(tr("Exit the application"));
	connect(mExitAction, SIGNAL(triggered()), this, SLOT(close()));

	mReadSTLAction = new QAction(tr("Read STL file"), this);
	mReadSTLAction->setShortcut(tr("Ctrl+R"));
	mReadSTLAction->setIcon(QIcon(":/Qt/resources/close.png"));
	mReadSTLAction->setStatusTip(tr("Read STL file"));
	connect(mReadSTLAction, SIGNAL(triggered()), this, SLOT(readSTL()));

	mDrawCantileverAction = new QAction(tr("Draw Cantilever"), this);
	mDrawCantileverAction->setIcon(QIcon(":/Qt/resources/torus.png"));
	mDrawCantileverAction->setStatusTip(tr("Draw Cantilever"));
	connect(mDrawCantileverAction, SIGNAL(triggered()), this, SLOT(drawCantilever()));

	mAboutAction = new QAction(tr("About"), this);
	mAboutAction->setStatusTip(tr("About the application"));
	mAboutAction->setIcon(QIcon(":/Qt/resources/lamp.png"));
	connect(mAboutAction, SIGNAL(triggered()), this, SLOT(about()));

	connect(myOccViewer, SIGNAL(selectionChanged()), this, SLOT(handleSelectionChanged()));

}

void StartWindow::createMenus(void)
{
	mFileMenu = menuBar()->addMenu(tr("&File"));
	mFileMenu->addAction(mExitAction);

	mCreateMenu = menuBar()->addMenu(tr("Create"));
	mCreateMenu->addAction(mDrawCantileverAction);

	mFileMenu->addAction(mReadSTLAction);
	mHelpMenu = menuBar()->addMenu(tr("&Help"));
	mHelpMenu->addAction(mAboutAction);
}

void StartWindow::createToolBars(void)
{
	mFileToolBar = addToolBar(tr("&File"));
	mFileToolBar->addAction(mReadSTLAction);
	mHelpToolBar = addToolBar(tr("Help"));
	mHelpToolBar->addAction(mAboutAction);
}


void StartWindow::about()
{
	myOccViewer->showGrid(Standard_True);
	myOccViewer->viewTop();
	myOccViewer->viewGrid();
	QMessageBox::about(this, tr("About STACCATO"),
		tr("<h2>STACCATO: STefAn's Computational vibroaCoustics Analysis TOol</h2>"
		"<p>Copyright &copy; 2016 "
		"<p>STACCATO is using Qt and OpenCASCADE."));
}

void StartWindow::readSTL(void)
{

	QString fileNameSTL = QFileDialog::getOpenFileName(this,
		tr("Import STL File"), "", tr("STL Files (*.stl)"));

	if (!fileNameSTL.isEmpty() && !fileNameSTL.isNull()){
		OSD_Path aFile(fileNameSTL.toUtf8().constData());
		Handle(StlMesh_Mesh) aSTLMesh = RWStl::ReadFile(aFile);
		Handle(MeshVS_Mesh) aMesh = new MeshVS_Mesh();
		Handle(XSDRAWSTLVRML_DataSource) aDS = new XSDRAWSTLVRML_DataSource(aSTLMesh);
		aMesh->SetDataSource(aDS);
		aMesh->AddBuilder(new MeshVS_MeshPrsBuilder(aMesh), Standard_True);//False -> No selection
		aMesh->GetDrawer()->SetBoolean(MeshVS_DA_DisplayNodes, Standard_False); //MeshVS_DrawerAttribute
		aMesh->GetDrawer()->SetBoolean(MeshVS_DA_ShowEdges, Standard_False);
		aMesh->GetDrawer()->SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_NOM_BRASS);
		aMesh->SetColor(Quantity_NOC_AZURE);
		aMesh->SetDisplayMode(MeshVS_DMF_Shading); // Mode as defaut
		aMesh->SetHilightMode(MeshVS_DMF_WireFrame); // Wireframe as default hilight mode
		myOccViewer->getContext()->Display(aMesh);
	}

}

void StartWindow::createDockWindows()
{
	QDockWidget *dock = new QDockWidget(tr("Output"), this);
	dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
	textOutput = new QTextEdit(dock);
	textOutput->setText("STACCATO");
	dock->setWidget(textOutput);
	addDockWidget(Qt::BottomDockWidgetArea, dock);

	//connect(textOutput, SIGNAL(currentTextChanged(QString)),this, SLOT(insertCustomer(QString)));

}

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
	//myOccViewer->getContext()->Display(aPointB);

	//============ 2D Stuff
	gp_Pnt2d mGp_Pnt_Start_2D = gp_Pnt2d(0., 0.);
	Handle(Geom2d_CartesianPoint) myGeom2d_Point = new Geom2d_CartesianPoint(mGp_Pnt_Start_2D);
	gp_Ax3	curCoordinateSystem = gp_Ax3();
	Handle(Geom_CartesianPoint) myGeom_Point = new Geom_CartesianPoint(ElCLib::To3d(curCoordinateSystem.Ax2(), mGp_Pnt_Start_2D));
	Handle(AIS_Point) myAIS_Point = new AIS_Point(myGeom_Point);	
	myOccViewer->getContext()->Display(myAIS_Point);
}

void StartWindow::handleSelectionChanged(void){


	bool aHasSelected = false;
	for (myOccViewer->getContext()->InitSelected(); myOccViewer->getContext()->MoreSelected() && !aHasSelected; myOccViewer->getContext()->NextSelected())
	{
		Handle(AIS_InteractiveObject) anIO = myOccViewer->getContext()->SelectedInteractive();
		//TopoDS_Shape vertexShape = Handle(AIS_Shape)::DownCast(anIO)->Shape();

		cout << "anIO: " << anIO->Signature() << endl;
		/*cout << "TopoDS_Shape: " << vertexShape.ShapeType() << endl;
		if (TopAbs_VERTEX == vertexShape.ShapeType())
		{
			gp_Pnt myPoint = BRep_Tool::Pnt(TopoDS::Vertex(vertexShape));
			cout << "=========="<< endl;
			cout << "X: " << myPoint.X() << endl;
			cout << "Y: " << myPoint.Y() << endl;
			cout << "Z: " << myPoint.Z() << endl;
			cout << "==========" << endl;
		}*/
	}

}