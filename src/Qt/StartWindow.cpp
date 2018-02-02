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
#include "SimuliaUMA.h"
#include "ui_StartWindow.h"
#include "OccViewer.h"
#include "VtkViewer.h"
#include "OcctQtProcessIndicator.h"
#include "STLVRML_DataSource.h"
#include "AuxiliaryParameters.h"
#include "Message.h"
#include "HMeshToMeshVS_DataSource.h"
#include "FeAnalysis.h"
#include "Timer.h"
#include "MemWatcher.h"
#include "qnemainwindow.h"
#include "HMesh.h"
#include "HMeshToVtkUnstructuredGrid.h"
#include "VisualizerWindow.h"
#include "Reader.h"

//Q5
#include <QToolBar>
#include <QTreeView>
#include <QMessageBox>
#include <QDockWidget>
#include <QtWidgets>
#include <QLabel>
#include <QDesktopWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QStringList>
#include <QComboBox>
#include <QCheckBox>
#include <QGroupBox>
#include <QFormLayout>
#include <QSpinBox>
#include <QtCharts/QLineSeries>
//#include <QtCharts/QChart>
#include <QtCharts/QChartView>

QT_CHARTS_USE_NAMESPACE

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
#include <IVtkOCC_Shape.hxx>
#include <IVtkTools_ShapeDataSource.hxx>

//VTK
#include <vtkDataSetMapper.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkScalarBarActor.h>
#include <vtkLookupTable.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWarpVector.h>
#include <vtkProperty.h>
#include <vtkPolyDataMapper.h>
#include <vtkExtractEdges.h>
#include <vtkXMLUnstructuredGridWriter.h>

//XML
#include "MetaDatabase.h"


StartWindow::StartWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::StartWindow)
{
	ui->setupUi(this);
	setWindowIcon(QIcon(":/Qt/resources/STACCATO.png"));
	setWindowTitle("STACCATO" + QString::fromStdString(STACCATO::AuxiliaryParameters::gitTAG));
	//myOccViewer = new OccViewer(this);

	myScalingFactorValue = 1;

	myVtkViewer = new VtkViewer(this);

	setCentralWidget(myVtkViewer);
	createActions();
	createMenus();
	createToolBars();
	createDockWindows();

	// Works for one instance only
	myHMesh = new HMesh("default");
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
	myExitAction = new QAction(tr("Exit"), this);
	myExitAction->setShortcut(tr("Ctrl+Q"));
	myExitAction->setIcon(QIcon(":/Qt/resources/closeDoc.png"));
	myExitAction->setStatusTip(tr("Exit the application"));
	connect(myExitAction, SIGNAL(triggered()), this, SLOT(close()));

	myReadOBDFileAction = new QAction(tr("Open OBD file"), this);
	myReadOBDFileAction->setStatusTip(tr("Read Abaqus OBD file"));
	connect(myReadOBDFileAction, SIGNAL(triggered()), this, SLOT(openOBDFile()));

	myImportXMLFileAction = new QAction(tr("Import XML file"), this);
	myImportXMLFileAction->setStatusTip(tr("Import STACCATO XML file"));
	connect(myImportXMLFileAction, SIGNAL(triggered()), this, SLOT(importXMLFile()));

	myReadFileAction = new QAction(tr("Import file"), this);
	myReadFileAction->setIcon(QIcon(":/Qt/resources/openDoc.png"));
	myReadFileAction->setStatusTip(tr("Import 3D file"));
	connect(myReadFileAction, SIGNAL(triggered()), this, SLOT(importFile()));

	// Time Step
	myTimeStepLabel = new QLabel(tr("Frequency (in Hz):"), this);
	myTimeStepLessAction = new QPushButton(tr("<"), this);
	myTimeStepLessAction->setFixedWidth(40);
	myTimeStepLessAction->setStatusTip(tr("Previous Frequency"));
	connect(myTimeStepLessAction, SIGNAL(clicked()), this, SLOT(myTimeStepLessProc()));

	myTimeStepAddAction = new QPushButton(tr(">"), this);
	myTimeStepAddAction->setFixedWidth(40);
	myTimeStepAddAction->setStatusTip(tr("Next Frequency"));
	connect(myTimeStepAddAction, SIGNAL(clicked()), this, SLOT(myTimeStepAddProc()));

	myTimeStepText = new QLineEdit(tr("0 Hz"),this);
	myTimeStepText->setFixedWidth(50);
	myTimeStepText->setAlignment(Qt::AlignHCenter);
	myTimeStepText->setReadOnly(true);
	myFreqIndex = 0;

	// Solution Selector
	allDispSolutionTypes.push_back("Displacement");
	
	allDispVectorComponents.push_back("Ux_Re");
	allDispVectorComponents.push_back("Uy_Re");
	allDispVectorComponents.push_back("Uz_Re");
	allDispVectorComponents.push_back("Magnitude_Re");
	allDispVectorComponents.push_back("Ux_Im");
	allDispVectorComponents.push_back("Uy_Im");
	allDispVectorComponents.push_back("Uz_Im");
	allDispVectorComponents.push_back("Magnitude_Im");

	allViewModes.push_back("Surface");
	allViewModes.push_back("Surface with Edges");
	allViewModes.push_back("Wireframe");

	mySolutionSelector = new QComboBox();
	mySolutionSelector->setStatusTip(tr("Select Solution Type"));
	for (int i = 0; i < allDispSolutionTypes.size(); i++)
		mySolutionSelector->addItem(QString::fromStdString(allDispSolutionTypes[i]));
	connect(mySolutionSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));
	mySolutionSelector->setEnabled(false);

	myComponentSelector = new QComboBox();
	myComponentSelector->setStatusTip(tr("Select Vector Component"));
	for (int i = 0; i < allDispVectorComponents.size() ; i++)
		myComponentSelector->addItem(QString::fromStdString(allDispVectorComponents[i]));
	connect(myComponentSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));
	myComponentSelector->setEnabled(false);

	myViewModeSelector = new QComboBox();
	myViewModeSelector->setStatusTip(tr("Select Viewing Mode"));
	for (int i = 0; i < allViewModes.size() ; i++)
		myViewModeSelector->addItem(QString::fromStdString(allViewModes[i]));
	connect(myViewModeSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));
	myViewModeSelector->setEnabled(false);

	// 

	// Picker Widgets
	myPickerModeNone = new QPushButton(this);
	myPickerModeNone->setIcon(QIcon(":/Qt/resources/none.ico"));
	myPickerModeNone->setStatusTip(tr("Select Node"));
	myPickerModeNone->setCheckable(true);
	myPickerModeNone->setFlat(true);
	myPickerModeNone->setChecked(true);
	connect(myPickerModeNone, SIGNAL(clicked()), myVtkViewer, SLOT(setPickerModeNone()));

	myPickerModeNode = new QPushButton(this);
	myPickerModeNode->setIcon(QIcon(":/Qt/resources/add.ico"));
	myPickerModeNode->setStatusTip(tr("Select Node"));
	myPickerModeNode->setCheckable(true);
	myPickerModeNode->setFlat(true);
	connect(myPickerModeNode, SIGNAL(clicked()), myVtkViewer, SLOT(setPickerModeNode()));

	myPickerModeElement = new QPushButton(this);
	myPickerModeElement->setIcon(QIcon(":/Qt/resources/selectElement.ico"));
	myPickerModeElement->setStatusTip(tr("Select Element"));
	myPickerModeElement->setCheckable(true);
	myPickerModeElement->setFlat(true);
	connect(myPickerModeElement, SIGNAL(clicked()), myVtkViewer, SLOT(setPickerModeElement()));

	myPickerButtonGroup = new QButtonGroup(this);
	myPickerButtonGroup->addButton(myPickerModeNone);
	myPickerButtonGroup->addButton(myPickerModeNode);
	myPickerButtonGroup->addButton(myPickerModeElement);

	// View actions
	myPickModeButton = new QPushButton(this);
	myPickModeButton->setIcon(QIcon(":/Qt/resources/cursor.ico"));
	myPickModeButton->setStatusTip(tr("Picking Mode"));
	myPickModeButton->setStatusTip(tr("Panning the view"));
	myPickModeButton->setCheckable(true);
	myPickModeButton->setChecked(true);
	myPickModeButton->setFlat(true);
	connect(myPickModeButton, SIGNAL(clicked()), this, SLOT(myViewModeTriggered()));

	myRotateModeButton = new QPushButton( this);
	myRotateModeButton->setStatusTip(tr("Rotation Mode"));
	myRotateModeButton->setIcon(QIcon(":/Qt/resources/rotate.png"));
	myRotateModeButton->setStatusTip(tr("Rotate the view"));
	myRotateModeButton->setCheckable(true);
	myRotateModeButton->setFlat(true);
	connect(myRotateModeButton, SIGNAL(clicked()), this, SLOT(myViewModeTriggered()));

	myPickerButtonGroup = new QButtonGroup(this);
	myPickerButtonGroup->addButton(myPickModeButton);
	myPickerButtonGroup->addButton(myRotateModeButton);

	/*
	// View actions
	myPanAction = new QAction(tr("Pan"), this);
	myPanAction->setIcon(QIcon(":/Qt/resources/pan.png"));
	myPanAction->setStatusTip(tr("Panning the view"));
	connect(myPanAction, SIGNAL(triggered()), myOccViewer, SLOT(pan()));

	myZoomAction = new QAction(tr("Zoom"), this);
	myZoomAction->setIcon(QIcon(":/Qt/resources/zoom.png"));
	myZoomAction->setStatusTip(tr("Zooming the view"));
	connect(myZoomAction, SIGNAL(triggered()), myOccViewer, SLOT(zoom()));

	myFitAllAction = new QAction(tr("Zoom fit all"), this);
	myFitAllAction->setIcon(QIcon(":/Qt/resources/fitAll.png"));
	myFitAllAction->setStatusTip(tr("Fit the view to show all"));
	connect(myFitAllAction, SIGNAL(triggered()), myOccViewer, SLOT(fitAll()));

	myRotateAction = new QAction(tr("Rotate"), this);
	myRotateAction->setIcon(QIcon(":/Qt/resources/rotate.png"));
	myRotateAction->setStatusTip(tr("Rotate the view"));
	connect(myRotateAction, SIGNAL(triggered()), myOccViewer, SLOT(rotation()));
	
	connect(myOccViewer, SIGNAL(selectionChanged()), this, SLOT(handleSelectionChanged()));
	*/

	// View Control ToolBar
	myScalarBarVisibility = new QPushButton(this);
	myScalarBarVisibility->setIcon(QIcon(":/Qt/resources/scalarBar.ico"));
	myScalarBarVisibility->setStatusTip(tr("Enable ScalarBar"));
	myScalarBarVisibility->setCheckable(true);
	myScalarBarVisibility->setChecked(true);
	myScalarBarVisibility->setFlat(true);
	connect(myScalarBarVisibility, SIGNAL(clicked()), this, SLOT(myViewPropertyUpdate()));
	myScalarBarVisibility->setEnabled(false);

	// View Control ToolBar
	myWarpVectorVisibility = new QPushButton(this);
	myWarpVectorVisibility->setIcon(QIcon(":/Qt/resources/scale.ico"));
	myWarpVectorVisibility->setStatusTip(tr("Enable Warp Vector"));
	myWarpVectorVisibility->setCheckable(true);
	myWarpVectorVisibility->setFlat(true);
	connect(myWarpVectorVisibility, SIGNAL(clicked()), this, SLOT(myWarpVectorTriggered()));
	myWarpVectorVisibility->setEnabled(false);

	my2dVisualizerVisibility = new QPushButton(this);
	my2dVisualizerVisibility->setIcon(QIcon(":/Qt/resources/2dPlot.ico"));
	my2dVisualizerVisibility->setStatusTip(tr("Enable 2D Visualizer"));
	my2dVisualizerVisibility->setFlat(true);
	connect(my2dVisualizerVisibility, SIGNAL(clicked()), this, SLOT(my2dVisualizerInterface()));
	my2dVisualizerVisibility->setEnabled(false);


	// Create actions
	myDrawCantileverAction = new QAction(tr("Draw Cantilever"), this);
	myDrawCantileverAction->setIcon(QIcon(":/Qt/resources/torus.png"));
	myDrawCantileverAction->setStatusTip(tr("Draw Cantilever"));
	connect(myDrawCantileverAction, SIGNAL(triggered()), this, SLOT(drawCantilever()));

	myDataFlowAction = new QAction(tr("Dataflow manager"), this);
	myDataFlowAction->setIcon(QIcon(":/Qt/resources/dataflow.png"));
	myDataFlowAction->setStatusTip(tr("Open dataflow manager"));
	connect(myDataFlowAction, SIGNAL(triggered()), this, SLOT(openDataFlowWindow()));

	myAnimationAction = new QAction(tr("Animate object"), this);
	myAnimationAction->setStatusTip(tr("Animate object"));
	connect(myAnimationAction, SIGNAL(triggered()), this, SLOT(animateObject()));

	// Selection actions
	mySetSelectionModeNoneAction = new QAction(tr("Reset selection"), this);
	mySetSelectionModeNoneAction->setStatusTip(tr("Reset selection"));
	connect(mySetSelectionModeNoneAction, SIGNAL(triggered()), myVtkViewer, SLOT(setPickerModeNone()));

	mySetSelectionModeNodeAction = new QAction(tr("Select a node"), this);
	mySetSelectionModeNodeAction->setStatusTip(tr("Select a node"));
	connect(mySetSelectionModeNodeAction, SIGNAL(triggered()), myVtkViewer, SLOT(setPickerModeNode()));

	mySetSelectionModeElementAction = new QAction(tr("Select an element"), this);
	mySetSelectionModeElementAction->setStatusTip(tr("Select an element"));
	connect(mySetSelectionModeElementAction, SIGNAL(triggered()), myVtkViewer, SLOT(setPickerModeElement()));

	// Export Actions
	myUMAAction = new QAction(tr("SIM via UMA"), this);
	myUMAAction->setStatusTip(tr("SIM via UMA"));
	myUMAAction->setCheckable(true);
	myUMAAction->setChecked(false);
	connect(myUMAAction, SIGNAL(triggered()), this, SLOT(myUMATriggered()));

	// Layout Actions
	myResetLayoutAction = new QAction(tr("Reset Layout"), this);
	myResetLayoutAction->setStatusTip(tr("Reset Layout"));

	myDockWarpVectorAction = new QAction(tr("Warp Vector"), this);
	myDockWarpVectorAction->setStatusTip(tr("Warp Vector"));
	myDockWarpVectorAction->setCheckable(true);
	myDockWarpVectorAction->setChecked(false);
	connect(myDockWarpVectorAction, SIGNAL(triggered()), this, SLOT(myWarpVectorTriggered()));

	my2dVisualizerAction = new QAction(tr("2D Visualizer"));
	my2dVisualizerAction->setStatusTip(tr("2D Visualizer"));
	my2dVisualizerAction->setCheckable(true);
	my2dVisualizerAction->setChecked(false);
	connect(my2dVisualizerAction, SIGNAL(triggered()), this, SLOT(my2dVisualizerInterface()));

	//Help actions
	myAboutAction = new QAction(tr("About"), this);
	myAboutAction->setStatusTip(tr("About the application"));
	myAboutAction->setIcon(QIcon(":/Qt/resources/about.png"));
	connect(myAboutAction, SIGNAL(triggered()), this, SLOT(about()));

}

void StartWindow::myViewPropertyUpdate(void) {
	static bool edge = false;
	static bool surface = true;	

	// Update View Mode
	if (myViewModeSelector->currentText().toStdString() == allViewModes[0]) {			// Surface With Edges
		edge = false;
		surface = true;
	}
	else if (myViewModeSelector->currentText().toStdString() == allViewModes[1]) {			// Surface With Edges
		edge = true;
		surface = true;
	}
	else if (myViewModeSelector->currentText().toStdString() == allViewModes[2]) {		// Wireframe
		edge = true;
		surface = false;
	}

	// Update Solution
	if (myComponentSelector->currentText().toStdString() == allDispVectorComponents[0]) {	//u_x_Re
		myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Ux_Re, myFreqIndex));
		myVtkViewer->setDisplayProperties(STACCATO_Ux_Re, edge, surface, myScalarBarVisibility->isChecked());
	} else if (myComponentSelector->currentText().toStdString() == allDispVectorComponents[1]) {	//u_y_Re
		myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Uy_Re, myFreqIndex));
		myVtkViewer->setDisplayProperties(STACCATO_Uy_Re, edge, surface, myScalarBarVisibility->isChecked());
	} else if (myComponentSelector->currentText().toStdString() == allDispVectorComponents[2]) {	//u_z_Re
		myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Uz_Re, myFreqIndex));
		myVtkViewer->setDisplayProperties(STACCATO_Uz_Re, edge, surface, myScalarBarVisibility->isChecked());
	} else if (myComponentSelector->currentText().toStdString() == allDispVectorComponents[3]) {	//u_Mag_Re
		myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Magnitude_Re, myFreqIndex));
		myVtkViewer->setDisplayProperties(STACCATO_Magnitude_Re, edge, surface, myScalarBarVisibility->isChecked());
	} else if (myComponentSelector->currentText().toStdString() == allDispVectorComponents[4]) {	//u_x_Im
		myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Ux_Im, myFreqIndex));
		myVtkViewer->setDisplayProperties(STACCATO_Ux_Im, edge, surface, myScalarBarVisibility->isChecked());
	} else if (myComponentSelector->currentText().toStdString() == allDispVectorComponents[5]) {	//u_y_Im
		myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Uy_Im, myFreqIndex));
		myVtkViewer->setDisplayProperties(STACCATO_Uy_Im, edge, surface, myScalarBarVisibility->isChecked());
	} else if (myComponentSelector->currentText().toStdString() == allDispVectorComponents[6]) {	//u_z_Im
		myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Uz_Im, myFreqIndex));
		myVtkViewer->setDisplayProperties(STACCATO_Uz_Im, edge, surface, myScalarBarVisibility->isChecked());
	}
	else if (myComponentSelector->currentText().toStdString() == allDispVectorComponents[7]) {	//u_Mag_Im
		myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Magnitude_Im, myFreqIndex));
		myVtkViewer->setDisplayProperties(STACCATO_Magnitude_Im, edge, surface, myScalarBarVisibility->isChecked());
	}
	myVtkViewer->plotVectorField(myHMeshToVtkUnstructuredGrid->getVtkUnstructuredGrid());		// Update Plot
}

void StartWindow::myTimeStepLessProc(void) {
	if (myFreqIndex > 0) {
		myFreqIndex--;
		myTimeStepText->setText(QString::fromStdString(std::to_string(std::stoi(myHMesh->getResultsTimeDescription()[myFreqIndex])) + " Hz"));		// Update Slider
		myViewPropertyUpdate();
	}
}

void StartWindow::myTimeStepAddProc(void) {
	if (myFreqIndex < myHMesh->getResultsTimeDescription().size()-1) {
		myFreqIndex++;
		myTimeStepText->setText(QString::fromStdString(std::to_string(std::stoi(myHMesh->getResultsTimeDescription()[myFreqIndex])) + " Hz"));		// Update Slider
		myViewPropertyUpdate();
	}
}

void StartWindow::createMenus(void)
{
	myFileMenu = menuBar()->addMenu(tr("&File"));
	myFileMenu->addAction(myReadFileAction);
	//myFileMenu->addAction(myReadOBDFileAction);
	myFileMenu->addAction(myImportXMLFileAction);
	myFileMenu->addAction(myExitAction);

	myCreateMenu = menuBar()->addMenu(tr("Create"));
	myCreateMenu->addAction(myDataFlowAction);
	myCreateMenu->addAction(myDrawCantileverAction);
	myCreateMenu->addAction(myAnimationAction);

	mySelectionMenu = menuBar()->addMenu(tr("Selection"));
	mySelectionMenu->addAction(mySetSelectionModeNoneAction);
	mySelectionMenu->addAction(mySetSelectionModeNodeAction);
	mySelectionMenu->addAction(mySetSelectionModeElementAction);

	myImportMenu = menuBar()->addMenu(tr("Import"));
	myImportMenu->addAction(myUMAAction);

	myLayoutMenu = menuBar()->addMenu(tr("Layout"));
	myLayoutMenu->addAction(myResetLayoutAction);

	myViewToolbarSubMenu = myLayoutMenu->addMenu(tr("View Toolbars"));

	myViewDockSubMenu = myLayoutMenu->addMenu(tr("View Dock Windows"));

	myViewDockSubMenu->addAction(myDockWarpVectorAction);

	myHelpMenu = menuBar()->addMenu(tr("&Help"));
	myHelpMenu->addAction(myAboutAction);
}

void StartWindow::createToolBars(void)
{
	myFileToolBar = addToolBar(tr("&File"));
	myFileToolBar->addAction(myReadFileAction);
	myCreateToolBar = addToolBar(tr("Create"));
	myCreateToolBar->addAction(myDataFlowAction);

	myViewToolBar = addToolBar(tr("View"));
	myViewToolBar->addWidget(myPickModeButton);
	myViewToolBar->addWidget(myRotateModeButton);

	/*
	myViewToolBar = addToolBar(tr("View"));
	myViewToolBar->addAction(myPanAction);
	myViewToolBar->addAction(myZoomAction);
	myViewToolBar->addAction(myFitAllAction);
	myViewToolBar->addAction(myRotateAction);
	*/

	myTimeToolBar = addToolBar(tr("Time Step"));
	myTimeToolBar->addWidget(myTimeStepLabel);
	myTimeToolBar->addWidget(myTimeStepLessAction);
	myTimeToolBar->addWidget(myTimeStepText);
	myTimeToolBar->addWidget(myTimeStepAddAction);

	mySolutionToolBar = addToolBar(tr("View Solution"));
	mySolutionToolBar->addWidget(mySolutionSelector);
	mySolutionToolBar->addSeparator();
	mySolutionToolBar->addWidget(myComponentSelector);
	mySolutionToolBar->addSeparator();
	mySolutionToolBar->addWidget(myViewModeSelector);

	myPickerViewToolBar = addToolBar(tr("Pick Finite Nodes/Elements"));
	myPickerViewToolBar->addWidget(myPickerModeNone);
	myPickerViewToolBar->addWidget(myPickerModeNode);
	myPickerViewToolBar->addWidget(myPickerModeElement);

	myHelpToolBar = addToolBar(tr("Help"));
	myHelpToolBar->addAction(myAboutAction);

	myDisplayControlToolBar = addToolBar(tr("Control Diplay within View Region"));
	addToolBar(Qt::RightToolBarArea, myDisplayControlToolBar);
	myDisplayControlToolBar->addWidget(myScalarBarVisibility);
	myDisplayControlToolBar->addWidget(myWarpVectorVisibility);
	myDisplayControlToolBar->addWidget(my2dVisualizerVisibility);
	myDisplayControlToolBar->setOrientation(Qt::Vertical);
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


void StartWindow::about(){
	myOccViewer->showGrid(Standard_True);
	myOccViewer->viewTop();
	myOccViewer->fitAll();
	myOccViewer->viewGrid();
	QMessageBox::about(this, tr("About STACCATO"),
		tr("<h2>STACCATO: STefAn's Computational vibroaCoustics Analysis TOol</h2>"
		"<p>Copyright &copy; 2017 "
		"<p>STACCATO is using Qt and OpenCASCADE."));
}

void StartWindow::importXMLFile(void) {
	QString myWorkingFolder = "";
	QString fileType;
	QFileInfo fileInfo;

	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open xml file"), myWorkingFolder, tr("XML (*.xml)"));

	fileInfo.setFile(fileName);
	fileType = fileInfo.suffix();
	if (!fileName.isEmpty() && !fileName.isNull()) {
		if (fileType.toLower() == tr("xml")) {
			infoOut << "XML file: " << fileName.toStdString() << std::endl;
		}
	}

	// Intialize XML Parsing
	MetaDatabase::init(fileName.toStdString());

	int numParts = MetaDatabase::getInstance()->xmlHandle->FILEIMPORT().size();
	std::cout << "There are " << MetaDatabase::getInstance()->xmlHandle->FILEIMPORT().size() << " models.\n";

	anaysisTimer01.start();
	anaysisTimer03.start();
	std::vector<Reader*> allReader(numParts);
	int i = 0;
	for (STACCATO_XML::FILEIMPORT_const_iterator iFileImport(MetaDatabase::getInstance()->xmlHandle->FILEIMPORT().begin());
		iFileImport != MetaDatabase::getInstance()->xmlHandle->FILEIMPORT().end();
		++iFileImport, i++) {
		std::string filePath = "C:/software/repos/STACCATO/model/";
		filePath += std::string(iFileImport->FILE()->data());
		if (std::string(iFileImport->Type()->data()) == "AbqODB") {
			allReader[i] = new SimuliaODB(filePath, *myHMesh);
		}
		else if (std::string(iFileImport->Type()->data()) == "AbqSIM") {
			allReader[i] = new SimuliaUMA(filePath, *myHMesh);
		}
		else {
			std::cerr << ">> XML Error: Unidentified FileImport type " << iFileImport->Type()->data() << std::endl;
		}
	}
	anaysisTimer01.stop();
	debugOut << "Duration for reading all file imports: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

	anaysisTimer01.start();

	//Run FE Analysis
	FeAnalysis *mFeAnalysis = new FeAnalysis(*myHMesh);
	anaysisTimer03.stop();
	debugOut << "Duration for STACCATO Finite Element run: " << anaysisTimer03.getDurationSec() << " sec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

	// Enable Tools
	mySolutionSelector->setEnabled(true);
	myComponentSelector->setEnabled(true);
	myViewModeSelector->setEnabled(true);
	myScalarBarVisibility->setEnabled(true);
	myWarpVectorVisibility->setEnabled(true);
	my2dVisualizerVisibility->setEnabled(true);

	anaysisTimer01.start();
	myHMeshToVtkUnstructuredGrid = new HMeshToVtkUnstructuredGrid(*myHMesh);
	anaysisTimer01.stop();
	debugOut << "Duration for reading HMeshToVtkUnstructuredGrid " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

	// Update Slider
	myTimeStepText->setText(QString::fromStdString(std::to_string(std::stoi(myHMesh->getResultsTimeDescription()[myFreqIndex])) + " Hz"));

	myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Ux_Re, myFreqIndex));
	myHMeshToVtkUnstructuredGrid->setVectorFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Ux_Re, myFreqIndex), myHMesh->getResultScalarFieldAtNodes(STACCATO_Uy_Re, myFreqIndex), myHMesh->getResultScalarFieldAtNodes(STACCATO_Uz_Re, myFreqIndex));

	// Plot Vector Field
	myVtkViewer->plotVectorField(myHMeshToVtkUnstructuredGrid->getVtkUnstructuredGrid());
}


void StartWindow::openOBDFile(void){
	// Intialize XML Parsing
	std::string inputFileName = "C:/software/repos/STACCATO/xsd/IP_STACCATO_XML.xml";
	MetaDatabase::init(inputFileName);

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

	SimuliaODB myOBD = SimuliaODB(fileName.toStdString(), *myHMesh);
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	anaysisTimer01.start();
	anaysisTimer03.start();
	myOBD.openFile();
	anaysisTimer01.stop();
	debugOut << "Duration for reading odb file: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory()/1000000 << " Mb" << std::endl;

	//Run FE Analysis
	FeAnalysis *mFeAnalysis = new FeAnalysis(*myHMesh);
	anaysisTimer03.stop();
	debugOut << "Duration for STACCATO Finite Element run: " << anaysisTimer03.getDurationSec() << " sec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	
	// Enable Tools
	mySolutionSelector->setEnabled(true);
	myComponentSelector->setEnabled(true);
	myViewModeSelector->setEnabled(true);
	myScalarBarVisibility->setEnabled(true);
	myWarpVectorVisibility->setEnabled(true);
	my2dVisualizerVisibility->setEnabled(true);

	anaysisTimer01.start();
	myHMeshToVtkUnstructuredGrid = new HMeshToVtkUnstructuredGrid(*myHMesh);
	anaysisTimer01.stop();
	debugOut << "Duration for reading HMeshToVtkUnstructuredGrid " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

	// Update Slider
	myTimeStepText->setText(QString::fromStdString(std::to_string(std::stoi(myHMesh->getResultsTimeDescription()[myFreqIndex])) + " Hz"));

	myHMeshToVtkUnstructuredGrid->setScalarFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Ux_Re, myFreqIndex));
	myHMeshToVtkUnstructuredGrid->setVectorFieldAtNodes(myHMesh->getResultScalarFieldAtNodes(STACCATO_Ux_Re, myFreqIndex), myHMesh->getResultScalarFieldAtNodes(STACCATO_Uy_Re, myFreqIndex), myHMesh->getResultScalarFieldAtNodes(STACCATO_Uz_Re, myFreqIndex));
	
	// Plot Vector Field
	myVtkViewer->plotVectorField(myHMeshToVtkUnstructuredGrid->getVtkUnstructuredGrid());
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

void StartWindow::myViewModeTriggered() {
	myVtkViewer->setViewMode(myRotateModeButton->isChecked());
}

void StartWindow::myWarpVectorTriggered() {
	if (myDockWarpVectorAction->isChecked() || myWarpVectorVisibility->isChecked()) {
		myWarpDock = new QDockWidget(tr("Warp Vector"), this);
		myWarpDock->setAllowedAreas(Qt::LeftDockWidgetArea);

		myAutoScaling = new QCheckBox(tr("Default Scaling"), this);
		myAutoScaling->setChecked(true);
		connect(myAutoScaling, SIGNAL(clicked()), this, SLOT(myAutoScalingState()));

		myScalingFactorLabel = new QLabel(tr("Scaling Factor"), this);

		myScalingFactor = new QSpinBox;
		myScalingFactor->setRange(0, 10000000000);
		myScalingFactor->setValue(myScalingFactorValue);
		myScalingFactor->setSingleStep(1);
		myScalingFactor->setEnabled(false);
		connect(myScalingFactor, SIGNAL(editingFinished()), this, SLOT(myScalingFactorState()));

		QFormLayout *layout = new QFormLayout;
		layout->addRow(myAutoScaling);
		layout->addRow(myScalingFactorLabel, myScalingFactor);
		QWidget* temp = new QWidget(this);
		temp->setLayout(layout);

		myWarpDock->setWidget(temp);
		myWarpDock->show();

		addDockWidget(Qt::LeftDockWidgetArea, myWarpDock);
	}
	else {
		removeDockWidget(myWarpDock);
	}
}

void StartWindow::myAutoScalingState() {
	if (myAutoScaling->isChecked()) {
		myScalingFactorValue = 1;
		myScalingFactor->setValue(myScalingFactorValue);
		myScalingFactor->setEnabled(false);
		myViewPropertyUpdate();
	}
	else{
		myScalingFactor->setValue(myScalingFactorValue);
		myScalingFactor->setEnabled(true);
	}
}

void StartWindow::myScalingFactorState() {
	double factor = QString(myScalingFactor->text()).toDouble();
	myVtkViewer->setScalingFactor(factor);
	myScalingFactorValue = factor;
	myViewPropertyUpdate();
}

void StartWindow::my2dVisualizerInterface() {
	myVtkViewer->my2dVisualizerInterface(*myHMesh);
}

void StartWindow::myUMATriggered() {
	if (myUMAAction->isChecked()) {
		myUMADock = new QDockWidget(tr("SIM via UMA"), this);
		myUMADock->setAllowedAreas(Qt::LeftDockWidgetArea);

		mySIMFileName = new QLineEdit(this);
		mySIMFileName->setText("C:/software/repos/staccato/sim/B31_fe_X1.sim");

		myUMAInterfaceButton = new QPushButton(tr("Parse SIM File"), this);
		connect(myUMAInterfaceButton, SIGNAL(clicked()), this, SLOT(myUMAImport()));

		mySIMImportButton = new QPushButton(tr("Import SIM to HMesh"), this);
		connect(mySIMImportButton, SIGNAL(clicked()), this, SLOT(myUMAHMesh()));

		mySIMImportLabel = new QLabel(tr("File: "), this);

		QFormLayout *layout = new QFormLayout;
		layout->addRow(mySIMImportLabel, mySIMFileName);
		layout->addRow(myUMAInterfaceButton);
		layout->addRow(mySIMImportButton);
		QWidget* temp = new QWidget(this);
		temp->setLayout(layout);

		myUMADock->setWidget(temp);
		myUMADock->show();

		addDockWidget(Qt::LeftDockWidgetArea, myUMADock);
	}
	else {
		removeDockWidget(myUMADock);
	}
}

void StartWindow::myUMAImport() {
	std::cout << ">> Trying to start UMA Interface ... " << std::endl;
	//myUMA->openSIM(mySIMFileName->text().toLocal8Bit().data());

}

void StartWindow::myUMAHMesh() {
	// Intialize XML Parsing
	std::string inputFileName = "C:/software/repos/STACCATO/xsd/IP_STACCATO_XML_B31_fe.xml";
	MetaDatabase::init(inputFileName);

	// Check for Imports
	bool flagSIM = false;
	for (STACCATO_XML::FILEIMPORT_const_iterator iFileImport(MetaDatabase::getInstance()->xmlHandle->FILEIMPORT().begin());
		iFileImport != MetaDatabase::getInstance()->xmlHandle->FILEIMPORT().end();
		++iFileImport)
	{
		if (std::string(iFileImport->Type()->c_str()) == "AbqSIM") {
			flagSIM = true;
		}
	}
	if (flagSIM) {
		std::cout << ">> SIM Import from XML Detected.\n";
		//myUMA->importToHMesh(*myHMesh);
		//Run FE Analysis
		myHMesh->isSIM = true;
		FeAnalysis *mFeAnalysis = new FeAnalysis(*myHMesh);
	}
	else {
		std::cerr << "FILEIMPORT Error: AbqSIM not found in XML.\n";
	}
}