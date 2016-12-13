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
}

StartWindow::~StartWindow()
{
	delete myOccViewer;
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

void StartWindow::readSTL(void)
{

	QString fileNameSTL = QFileDialog::getOpenFileName(this,
		tr("Import STL File"), "", tr("STL Files (*.stl)"));

	if (!fileNameSTL.isEmpty() && !fileNameSTL.isNull()){
		Handle(Message_ProgressIndicator) aIndicator = new QtProcessIndicator(this);
		aIndicator->SetRange(0, 100);
		OSD_Path aFile(fileNameSTL.toUtf8().constData());
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

	for (myOccViewer->getContext()->InitSelected(); myOccViewer->getContext()->MoreSelected(); myOccViewer->getContext()->NextSelected())
	{
		Handle(AIS_InteractiveObject) anIO = myOccViewer->getContext()->SelectedInteractive();
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

						cout << "==========" << endl;
						cout << "X: " << p1.X() << endl;
						cout << "Y: " << p2.Y() << endl;
						cout << "Z: " << p3.Z() << endl;
						cout << "==========" << endl;

						// do something with p1, p2 and p3
					}
				}
			}

		}
		else if (owner->Type() == MeshVS_ET_Node){
			cout << "A Node" << endl;
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
					cout << "==========" << endl;
					cout << "X: " << p1.X() << endl;
					cout << "Y: " << p1.Y() << endl;
					cout << "Z: " << p1.Z() << endl;
					cout << "==========" << endl;
				}
			}
		}
		//=====




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
