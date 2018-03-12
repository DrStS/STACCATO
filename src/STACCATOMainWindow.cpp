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
#include "STACCATOMainWindow.h"
#include "SimuliaUMA.h"
#include "ui_STACCATOMainWindow.h"
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
#include "Reader.h"
#include "FieldDataVisualizer.h"

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
#include <QSlider>

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

STACCATOMainWindow::STACCATOMainWindow(QWidget *parent) : QMainWindow(parent), myGui(new Ui::STACCATOMainWindow)
{
	myGui->setupUi(this);
	setWindowIcon(QIcon(":/Qt/resources/STACCATO.png"));
	setWindowTitle("STACCATO" + QString::fromStdString(STACCATO::AuxiliaryParameters::gitTAG));
	//myOccViewer = new OccViewer(this);

	// Works for one instance only
	myHMesh = new HMesh("default");

	myFieldDataVisualizer = new FieldDataVisualizer(this);
	myFieldDataVisualizer->setHMesh(*myHMesh);

	myVisualizerSetting = new VisualizerSetting();
	myVisualizerSetting->setCommuniationToFieldDataVisualizer(*myFieldDataVisualizer);

	isSubFrame = false;

	setCentralWidget(myFieldDataVisualizer);
	createActions();
	createMenus();
	createToolBars();
	createDockWindows();
	createAnimationOptionsDock();
}

STACCATOMainWindow::~STACCATOMainWindow()
{
	//delete myOccViewer;
	delete myFieldDataVisualizer;
}


void STACCATOMainWindow::openDataFlowWindow(void) {
	newWin = new QNEMainWindow();
	newWin->show();
}

void STACCATOMainWindow::createActions(void)
{
	// File actions
	myExitAction = new QAction(tr("Exit"), this);
	myExitAction->setShortcut(tr("Ctrl+Q"));
	myExitAction->setIcon(QIcon(":/Qt/resources/closeDoc.png"));
	myExitAction->setStatusTip(tr("Exit the application"));
	connect(myExitAction, SIGNAL(triggered()), this, SLOT(close()));

	myImportXMLFileAction = new QAction(tr("Import XML file"), this);
	myImportXMLFileAction->setIcon(QIcon(":/Qt/resources/xmlFile.ico"));
	myImportXMLFileAction->setStatusTip(tr("Import STACCATO XML file"));
	connect(myImportXMLFileAction, SIGNAL(triggered()), this, SLOT(importXMLFile()));

	myReadFileAction = new QAction(tr("Import file"), this);
	myReadFileAction->setIcon(QIcon(":/Qt/resources/openDoc.png"));
	myReadFileAction->setStatusTip(tr("Import 3D file"));
	connect(myReadFileAction, SIGNAL(triggered()), this, SLOT(importFile()));

	// Time Step
	myTimeStepLabel = new QLabel(tr("Analysis:"), this);

	myTimeStepLessAction = new QPushButton(tr("<"), this);
	myTimeStepLessAction->setFixedWidth(40);
	myTimeStepLessAction->setStatusTip(tr("Previous Frequency"));
	connect(myTimeStepLessAction, SIGNAL(clicked()), this, SLOT(myTimeStepLessProc()));

	myTimeStepAddAction = new QPushButton(tr(">"), this);
	myTimeStepAddAction->setFixedWidth(40);
	myTimeStepAddAction->setStatusTip(tr("Next Frequency"));
	connect(myTimeStepAddAction, SIGNAL(clicked()), this, SLOT(myTimeStepAddProc()));

	myTimeStepText = new QLineEdit(this);
	myTimeStepText->setText("-");
	myTimeStepText->setFixedWidth(50);
	myTimeStepText->setAlignment(Qt::AlignHCenter);
	myTimeStepText->setReadOnly(true);

	myResultCaseSelector = new QComboBox();
	myResultCaseSelector->addItem("Case..");
	QStandardItemModel* model = qobject_cast<QStandardItemModel*>(myResultCaseSelector->model());
	QStandardItem* item = model->item(0);
	item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
	myResultCaseSelector->setStatusTip(tr("Select Case Type"));
	connect(myResultCaseSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myResultCaseChanged()));
	myResultCaseSelector->setEnabled(true);

	myAnimatePrevFrameButton = new QPushButton(tr("<"), this);
	connect(myAnimatePrevFrameButton, SIGNAL(clicked()), this, SLOT(myCaseStepLessProc()));
	myAnimatePrevFrameButton->setEnabled(false);
	myAnimatePrevFrameButton->setFixedWidth(40);
	myAnimatePrevFrameButton->setEnabled(true);

	myAnimateNextFrameButton = new QPushButton(tr(">"), this);
	connect(myAnimateNextFrameButton, SIGNAL(clicked()), this, SLOT(myCaseStepAddProc()));
	myAnimateNextFrameButton->setEnabled(false);
	myAnimateNextFrameButton->setFixedWidth(40);
	myAnimateNextFrameButton->setEnabled(true);

	myCaseStepText = new QLineEdit(tr("-"), this);
	myCaseStepText->setFixedWidth(50);
	myCaseStepText->setAlignment(Qt::AlignHCenter);
	myCaseStepText->setReadOnly(true);
	myCaseStepText->setEnabled(true);

	mySolutionSelector = new QComboBox();
	mySolutionSelector->addItem("Solution..");
	model = qobject_cast<QStandardItemModel*>(mySolutionSelector->model());
	item = model->item(0);
	item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
	mySolutionSelector->setStatusTip(tr("Select Solution Type"));
	connect(mySolutionSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));

	myComponentSelector = new QComboBox();
	myComponentSelector->addItem("Component..");
	model = qobject_cast<QStandardItemModel*>(myComponentSelector->model());
	item = model->item(0);
	item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
	myComponentSelector->setStatusTip(tr("Select Vector Component"));
	connect(myComponentSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));

	myViewModeSelector = new QComboBox();
	myViewModeSelector->addItem("View Mode..");
	model = qobject_cast<QStandardItemModel*>(myViewModeSelector->model());
	item = model->item(0);
	item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
	myViewModeSelector->setStatusTip(tr("Select Viewing Mode"));
	connect(myViewModeSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));

	// 

	// Picker Widgets
	myPickerModeNone = new QPushButton(this);
	myPickerModeNone->setIcon(QIcon(":/Qt/resources/none.ico"));
	myPickerModeNone->setStatusTip(tr("Select Node"));
	myPickerModeNone->setCheckable(true);
	myPickerModeNone->setFlat(true);
	myPickerModeNone->setChecked(true);
	connect(myPickerModeNone, SIGNAL(clicked()), myFieldDataVisualizer, SLOT(setPickerModeNone()));

	myPickerModeNode = new QPushButton(this);
	myPickerModeNode->setIcon(QIcon(":/Qt/resources/add.ico"));
	myPickerModeNode->setStatusTip(tr("Select Node"));
	myPickerModeNode->setCheckable(true);
	myPickerModeNode->setFlat(true);
	connect(myPickerModeNode, SIGNAL(clicked()), myFieldDataVisualizer, SLOT(setPickerModeNode()));

	myPickerModeElement = new QPushButton(this);
	myPickerModeElement->setIcon(QIcon(":/Qt/resources/selectElement.ico"));
	myPickerModeElement->setStatusTip(tr("Select Element"));
	myPickerModeElement->setCheckable(true);
	myPickerModeElement->setFlat(true);
	connect(myPickerModeElement, SIGNAL(clicked()), myFieldDataVisualizer, SLOT(setPickerModeElement()));

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
	myPickModeButton->setChecked(false);
	myPickModeButton->setFlat(true);
	connect(myPickModeButton, SIGNAL(clicked()), this, SLOT(myViewModeTriggered()));

	myRotateModeButton = new QPushButton(this);
	myRotateModeButton->setStatusTip(tr("Rotation Mode"));
	myRotateModeButton->setIcon(QIcon(":/Qt/resources/rotate.png"));
	myRotateModeButton->setStatusTip(tr("Rotate the view"));
	myRotateModeButton->setCheckable(true);
	myRotateModeButton->setFlat(true);
	myRotateModeButton->setChecked(true);
	connect(myRotateModeButton, SIGNAL(clicked()), this, SLOT(myViewModeTriggered()));

	myViewButtonGroup = new QButtonGroup(this);
	myViewButtonGroup->addButton(myPickModeButton);
	myViewButtonGroup->addButton(myRotateModeButton);

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

	mySignalDataVisualizerVisiblility = new QPushButton(this);
	mySignalDataVisualizerVisiblility->setIcon(QIcon(":/Qt/resources/2dPlot.ico"));
	mySignalDataVisualizerVisiblility->setStatusTip(tr("Enable Signal Data Visualizer"));
	mySignalDataVisualizerVisiblility->setFlat(true);
	connect(mySignalDataVisualizerVisiblility, SIGNAL(clicked()), this, SLOT(mySignalDataVisualizerInterface()));
	mySignalDataVisualizerVisiblility->setEnabled(false);

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
	connect(mySetSelectionModeNoneAction, SIGNAL(triggered()), myFieldDataVisualizer, SLOT(setPickerModeNone()));

	mySetSelectionModeNodeAction = new QAction(tr("Select a node"), this);
	mySetSelectionModeNodeAction->setStatusTip(tr("Select a node"));
	connect(mySetSelectionModeNodeAction, SIGNAL(triggered()), myFieldDataVisualizer, SLOT(setPickerModeNode()));

	mySetSelectionModeElementAction = new QAction(tr("Select an element"), this);
	mySetSelectionModeElementAction->setStatusTip(tr("Select an element"));
	connect(mySetSelectionModeElementAction, SIGNAL(triggered()), myFieldDataVisualizer, SLOT(setPickerModeElement()));

	// Export Actions
	myUMAAction = new QAction(tr("SIM via UMA"), this);
	myUMAAction->setStatusTip(tr("SIM via UMA"));
	myUMAAction->setCheckable(true);
	myUMAAction->setChecked(false);
	connect(myUMAAction, SIGNAL(triggered()), this, SLOT(myUMATriggered()));

	// ToolBar Actions
	mySubFrameAnimateAction = new QAction(tr("Animate Sub Frame"), this);
	mySubFrameAnimateAction->setStatusTip(tr("Animate sub Frame"));
	mySubFrameAnimateAction->setCheckable(true);
	mySubFrameAnimateAction->setChecked(false);
	connect(mySubFrameAnimateAction, SIGNAL(triggered()), this, SLOT(mySubFrameAnimate()));

	// Animate Actions
	myAnimationOptionsDockAction = new QAction(tr("Options"), this);
	myAnimationOptionsDockAction->setStatusTip(tr("Animation Options"));
	myAnimationOptionsDockAction->setCheckable(false);
	connect(myAnimationOptionsDockAction, SIGNAL(triggered()), this, SLOT(myAnimationOptionsTriggered()));

	// Layout Actions
	myResetLayoutAction = new QAction(tr("Reset Layout"), this);
	myResetLayoutAction->setStatusTip(tr("Reset Layout"));

	myDockWarpVectorAction = new QAction(tr("Warp Vector"), this);
	myDockWarpVectorAction->setStatusTip(tr("Warp Vector"));
	myDockWarpVectorAction->setCheckable(true);
	myDockWarpVectorAction->setChecked(false);
	connect(myDockWarpVectorAction, SIGNAL(triggered()), this, SLOT(myWarpVectorTriggered()));

	myViewPropertyAction = new QAction(tr("View Property"), this);
	myViewPropertyAction->setStatusTip(tr("View Property"));
	myViewPropertyAction->setCheckable(true);
	myViewPropertyAction->setChecked(false);
	connect(myViewPropertyAction, SIGNAL(triggered()), this, SLOT(myViewPropertyDockTriggered()));

	my2dVisualizerAction = new QAction(tr("Signal Data Visualizer"));
	my2dVisualizerAction->setStatusTip(tr("Signal Data Visualizer"));
	my2dVisualizerAction->setCheckable(true);
	my2dVisualizerAction->setChecked(false);
	connect(my2dVisualizerAction, SIGNAL(triggered()), this, SLOT(mySignalDataVisualizerInterface()));

	//Help actions
	myAboutAction = new QAction(tr("About"), this);
	myAboutAction->setStatusTip(tr("About the application"));
	myAboutAction->setIcon(QIcon(":/Qt/resources/about.png"));
	connect(myAboutAction, SIGNAL(triggered()), this, SLOT(about()));

	// Animate Widgets
	myAnimationButton = new QAction(this);
	myAnimationButton->setIcon(QIcon(":/Qt/resources/animate.ico"));
	myAnimationButton->setStatusTip(tr("Enable Animation"));
	myAnimationButton->setCheckable(false);
	myAnimationButton->setChecked(false);
	connect(myAnimationButton, SIGNAL(triggered()), this, SLOT(myGenerateAnimationFramesProc()));
	myAnimationButton->setEnabled(false);

	myAnimationDurationLabel = new QLabel(tr("Duration (sec):"), this);

	myAnimationDuration = new QLineEdit(this);
	myAnimationDuration->setText("5");
	myAnimationDuration->setAlignment(Qt::AlignHCenter);
	myAnimationDuration->setFixedWidth(30);

	myAnimateRepeatButton = new QAction(this);
	myAnimateRepeatButton->setIcon(QIcon(":/Qt/resources/repeat.ico"));
	myAnimateRepeatButton->setCheckable(true);
	myAnimateRepeatButton->setChecked(true);
	myAnimateRepeatButton->setEnabled(false);

	myAnimatePlayButton = new QAction(this);
	myAnimatePlayButton->setIcon(QIcon(":/Qt/resources/play.ico"));
	connect(myAnimatePlayButton, SIGNAL(triggered()), this, SLOT(myAnimationScenePlayProc()));
	myAnimatePlayButton->setEnabled(false);

	myAnimateStopButton = new QAction(this);
	myAnimateStopButton->setIcon(QIcon(":/Qt/resources/stop.ico"));
	connect(myAnimateStopButton, SIGNAL(triggered()), this, SLOT(myAnimationSceneStopProc()));
	myAnimateStopButton->setEnabled(false);

	myHorizontalSlider = new QSlider(Qt::Horizontal);
	myHorizontalSlider->setEnabled(false);
	myHorizontalSlider->setFixedWidth(60);

	myResetFrameAnimationButton = new QAction(this);
	myResetFrameAnimationButton->setIcon(QIcon(":/Qt/resources/delete.ico"));
	connect(myResetFrameAnimationButton, SIGNAL(triggered()), this, SLOT(myAnimationResetProc()));
	myResetFrameAnimationButton->setEnabled(false);
}

void STACCATOMainWindow::fillFEResultInGUI() {
	myVisualizerSetting->setResultAvailable(true);

	mySolutionSelector->disconnect();		// Disconnect the ComboBox first to avoid error during dynamic updating
	for (int i = 0; i < myHMesh->myOutputDatabase->getVectorFieldFromDatabase().size(); i++) 				// Adding Vector Field
		mySolutionSelector->addItem(QString::fromStdString(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[i].myLabel));
	mySolutionSelector->setCurrentIndex(1);
	connect(mySolutionSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));

	myResultCaseSelector->disconnect();		// Disconnect the ComboBox first to avoid error during dynamic updating
	for (std::map<std::string, STACCATO_ResultsCase_type>::iterator it = myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].myResultCaseLabelMap.begin(); it != myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].myResultCaseLabelMap.end(); ++it) {
		myResultCaseSelector->addItem(QString::fromStdString(it->first));
	}
	myResultCaseSelector->setCurrentIndex(1);
	myResultCaseSelector->setFixedWidth(80);
	connect(myResultCaseSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myResultCaseChanged()));

	myComponentSelector->disconnect();		// Disconnect the ComboBox first to avoid error during dynamic updating
	for (std::map<std::string, STACCATO_VectorField_components>::iterator it = myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].myResultLabelMap.begin(); it != myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].myResultLabelMap.end(); ++it) {
		myComponentSelector->addItem(QString::fromStdString(it->first));
	}
	myComponentSelector->setCurrentIndex(1);
	connect(myComponentSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));

	myViewModeSelector->disconnect();		// Disconnect the ComboBox first to avoid error during dynamic updating
	for (std::map<std::string, STACCATO_FieldProperty_type>::iterator it = myVisualizerSetting->myViewModeLabelMap.begin(); it != myVisualizerSetting->myViewModeLabelMap.end(); ++it)
		myViewModeSelector->addItem(QString::fromStdString(it->first));
	myViewModeSelector->setFixedWidth(120);
	myViewModeSelector->setCurrentIndex(1);
	connect(myViewModeSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));

	// Fill Animation Options
	// - Analysis
	myAnalysisSelector->disconnect();
	for (int i = 0; i < myHMesh->myOutputDatabase->getVectorFieldAnalysisDectription().size(); i++)	{
		myAnalysisSelector->addItem(QString::fromStdString(myHMesh->myOutputDatabase->getVectorFieldAnalysisDectription()[i]));
	}
	myAnalysisSelector->setCurrentIndex(1);

	// Setting Iterators
	myFreqIndex = myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsTimeDescription().begin();
	myCaseIndex = myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().begin();

	std::cout << "Frequency Description\n";
	for (std::map<int, std::string>::iterator it = myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsTimeDescription().begin(); it != myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsTimeDescription().end(); ++it) {
		std::cout << "@ Index " << it->first << " Description: " << it->second << std::endl;
	}

	std::cout << "Case Description\n";
	for (std::map<int, std::string>::iterator it = myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().begin(); it != myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().end(); ++it) {
		std::cout << "@ Index " << it->first << " Description: " << it->second << std::endl;
	}
}

void STACCATOMainWindow::myViewPropertyUpdate(void) {
	myAnimationResetProc();

	myTimeStepText->setText(QString::fromStdString(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsTimeDescription()[myFreqIndex->first] + myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].myTimeUnit));		// Update Slider
	myCaseStepText->setText(QString::fromStdString(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription()[myCaseIndex->first] + myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].myCaseUnit));		// Update Slider
																																																			// Change in Properties
	myVisualizerSetting->commitCurrentFrame(myFreqIndex->first + myCaseIndex->first);
	myVisualizerSetting->commitVectorFieldComponent(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].myResultLabelMap[myComponentSelector->currentText().toStdString()]);	// Result Component
	myVisualizerSetting->commitViewSetting(myVisualizerSetting->myViewModeLabelMap[myViewModeSelector->currentText().toStdString()]);						// View Mode
	myVisualizerSetting->setScalarbarTitle(myComponentSelector->currentText().toStdString());																// Scalarbar Title
	myVisualizerSetting->commitScalarBar(myScalarBarVisibility->isChecked());																				// Scalarbar Visibility

	myVisualizerSetting->updateSetting();	// Visualize Frame

}

void STACCATOMainWindow::myTimeStepLessProc(void) {
	isSubFrame = false;
	if (myFreqIndex != myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsTimeDescription().begin())
		myFreqIndex--;
	myViewPropertyUpdate();
}

void STACCATOMainWindow::myTimeStepAddProc(void) {
	isSubFrame = false;
	if (myFreqIndex != --myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsTimeDescription().end())
		myFreqIndex++;
	myViewPropertyUpdate();
}


void STACCATOMainWindow::myCaseStepLessProc(void) {
	isSubFrame = true;
	if (myCaseIndex != myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().begin())
		--myCaseIndex;
	myViewPropertyUpdate();
}

void STACCATOMainWindow::myCaseStepAddProc(void) {
	isSubFrame = true;
	if (myCaseIndex != --myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().end()) {
		++myCaseIndex;
	}
	myViewPropertyUpdate();
}

void STACCATOMainWindow::createMenus(void)
{
	myFileMenu = menuBar()->addMenu(tr("&File"));
	myFileMenu->addAction(myReadFileAction);
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

	myAnimateMenu = menuBar()->addMenu(tr("Animate"));
	myAnimateMenu->addAction(myAnimationOptionsDockAction);

	myLayoutMenu = menuBar()->addMenu(tr("Layout"));
	myLayoutMenu->addAction(myResetLayoutAction);

	myViewToolbarSubMenu = myLayoutMenu->addMenu(tr("View Toolbars"));
	myViewToolbarSubMenu->addAction(mySubFrameAnimateAction);

	myViewDockSubMenu = myLayoutMenu->addMenu(tr("View Dock Windows"));

	myViewDockSubMenu->addAction(myDockWarpVectorAction);
	myViewDockSubMenu->addAction(myViewPropertyAction);

	myHelpMenu = menuBar()->addMenu(tr("&Help"));
	myHelpMenu->addAction(myAboutAction);
}

void STACCATOMainWindow::createToolBars(void)
{
	myFileToolBar = addToolBar(tr("&File"));
	myFileToolBar->addAction(myReadFileAction);
	myFileToolBar->addAction(myImportXMLFileAction);
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

	myTimeToolBar->addSeparator();
	myTimeToolBar->addWidget(myResultCaseSelector);
	myTimeToolBar->addWidget(myAnimatePrevFrameButton);
	myTimeToolBar->addWidget(myCaseStepText);
	myTimeToolBar->addWidget(myAnimateNextFrameButton);

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
	myDisplayControlToolBar->addWidget(mySignalDataVisualizerVisiblility);
	myDisplayControlToolBar->setOrientation(Qt::Vertical);

	mySubFrameAnimatorToolBar = addToolBar(tr("Animate"));
	mySubFrameAnimatorToolBar->addAction(myAnimationButton);
	mySubFrameAnimatorToolBar->addSeparator();
	mySubFrameAnimatorToolBar->addWidget(myAnimationDurationLabel);
	mySubFrameAnimatorToolBar->addWidget(myAnimationDuration);
	mySubFrameAnimatorToolBar->addSeparator();
	mySubFrameAnimatorToolBar->addAction(myAnimateRepeatButton);
	mySubFrameAnimatorToolBar->addSeparator();
	mySubFrameAnimatorToolBar->addAction(myAnimatePlayButton);
	mySubFrameAnimatorToolBar->addAction(myAnimateStopButton);
	mySubFrameAnimatorToolBar->addAction(myResetFrameAnimationButton);
	mySubFrameAnimatorToolBar->addSeparator();
	mySubFrameAnimatorToolBar->addWidget(myHorizontalSlider);
}

void STACCATOMainWindow::createDockWindows()
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

void STACCATOMainWindow::createAnimationOptionsDock() {
	/// Visualizer Info Dock Widget
	myAnalysisSelectorLabel = new QLabel(tr("Analysis:"));

	myAnalysisSelector = new QComboBox();
	myAnalysisSelector->addItem("Select..");
	QStandardItemModel* model = qobject_cast<QStandardItemModel*>(myAnalysisSelector->model());
	QStandardItem* item = model->item(0);
	item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
	myAnalysisSelector->setStatusTip(tr("Select Analysis"));
	//connect(myAnalysisSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myResultCaseChanged()));
	myAnalysisSelector->setEnabled(true);

	myAnimationAnalysisTree = new QTreeWidget;
	myAnimationAnalysisTree->setColumnCount(3);
	myAnimationAnalysisTree->header()->close();
	myAnimationAnalysisTree->setContextMenuPolicy(Qt::CustomContextMenu);
	myAnimationAnalysisTree->setMinimumHeight(100);

	myAnimationLabel = new QLabel(tr("Animate:"), this);

	myFrequencyAnimationRadio = new QRadioButton(tr("&Frequency"));
	myFrequencyAnimationRadio->setChecked(true);
	//connect(myFrequencyAnimationRadio, SIGNAL(clicked()), this, SLOT(updateAnalysisTree()));
	myHarmonicAnimationRadio = new QRadioButton(tr("&Harmonic"));
	myCaseAnimationRadio = new QRadioButton(tr("&Case(s): "));

	myResultsCaseSelector_dummy = new QComboBox();
	myResultsCaseSelector_dummy->addItem("Case..");
	model = qobject_cast<QStandardItemModel*>(myResultsCaseSelector_dummy->model());
	item = model->item(0);
	item->setFlags(item->flags() & ~Qt::ItemIsEnabled);
	myResultsCaseSelector_dummy->setStatusTip(tr("Select Case Type"));
	//connect(myResultsCaseSelector_dummy, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myResultCaseChanged()));
	myResultsCaseSelector_dummy->setEnabled(true);

	myAnimationButtonGroup = new QButtonGroup(this);
	myAnimationButtonGroup->addButton(myFrequencyAnimationRadio);
	myAnimationButtonGroup->addButton(myHarmonicAnimationRadio);
	myAnimationButtonGroup->addButton(myCaseAnimationRadio);

	myAnimationOptionPreview = new QCheckBox(tr("Preview"), this);
	myAnimationOptionPreview->setChecked(false);

	myAnimationOptionApplyButton = new QPushButton(tr("Apply"));
	connect(myAnimationOptionApplyButton, SIGNAL(clicked()), this, SLOT(myGenerateAnimationFramesProc()));
	myAnimationSetDefaultButton = new QPushButton(tr("Set Default"));

	myAnimationOptionsDock = new QDockWidget(tr("Options"), this);
	myAnimationOptionsDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);

	QGridLayout *layout = new QGridLayout;
	layout->addWidget(myAnalysisSelectorLabel, 1, 0);
	layout->addWidget(myAnalysisSelector, 1, 1, 1, -1);
	layout->addWidget(myAnimationLabel, 2, 0);
	layout->addWidget(myFrequencyAnimationRadio, 2, 1);
	layout->addWidget(myCaseAnimationRadio, 3, 1);
	layout->addWidget(myResultsCaseSelector_dummy, 3, 2, 1, -1);
	layout->addWidget(myHarmonicAnimationRadio, 4, 1);
	layout->addWidget(myAnimationAnalysisTree, 5, 0, 1, -1);
	layout->addWidget(myAnimationOptionPreview, 6, 0);
	layout->addWidget(myAnimationOptionApplyButton, 6, 1);
	layout->addWidget(myAnimationSetDefaultButton, 6, 2);

	QWidget* temp = new QWidget(this);
	temp->setLayout(layout);

	myAnimationOptionsDock->setWidget(temp);

	addDockWidget(Qt::LeftDockWidgetArea, myAnimationOptionsDock);
	myAnimationOptionsDock->setFloating(true);
	myAnimationOptionsDock->hide();
	//myAnimationOptionsDock->setFixedHeight(myAnimationOptionsDock->rect().height());

	myAnimationOptionsDock->move(QApplication::desktop()->screen()->rect().center() - myAnimationOptionsDock->rect().center());
}

void STACCATOMainWindow::about() {
	myOccViewer->showGrid(Standard_True);
	myOccViewer->viewTop();
	myOccViewer->fitAll();
	myOccViewer->viewGrid();
	QMessageBox::about(this, tr("About STACCATO"),
		tr("<h2>STACCATO: STefAn's Computational vibroaCoustics Analysis TOol</h2>"
			"<p>Copyright &copy; 2017 "
			"<p>STACCATO is using Qt and OpenCASCADE."));
}

void STACCATOMainWindow::importXMLFile(void) {
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

		//Run FE Analysis
		FeAnalysis *mFeAnalysis = new FeAnalysis(*myHMesh);
		anaysisTimer03.stop();
		debugOut << "Duration for STACCATO Finite Element run: " << anaysisTimer03.getDurationSec() << " sec" << std::endl;
		debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

		std::cout << ">> FeAnalysis Finished."<< std::endl;
		fillFEResultInGUI();

		// Enable Tools
		myScalarBarVisibility->setEnabled(true);
		myWarpVectorVisibility->setEnabled(true);
		mySignalDataVisualizerVisiblility->setEnabled(true);
		myAnimationButton->setEnabled(true);

		anaysisTimer01.start();
		myFieldDataVisualizer->myHMeshToVtkUnstructuredGridInitializer();
		anaysisTimer01.stop();
		debugOut << "Duration for reading HMeshToVtkUnstructuredGrid " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
		debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

		myFieldDataVisualizer->myHMeshToVtkUnstructuredGridSetScalar(STACCATO_x_Re, myFreqIndex->first);
		myFieldDataVisualizer->myHMeshToVtkUnstructuredGridSetVector(myFreqIndex->first);

		myViewPropertyUpdate();
	}
}

void STACCATOMainWindow::importFile(void)
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
	if (!fileName.isEmpty() && !fileName.isNull()) {
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

void STACCATOMainWindow::readSTEP(QString fileName) {
	// create additional log file
	STEPControl_Reader aReader;
	IFSelect_ReturnStatus status = aReader.ReadFile(fileName.toUtf8().constData());
	if (status != IFSelect_RetDone) {
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

void STACCATOMainWindow::readIGES(QString fileName) {

	IGESControl_Reader Reader;

	Standard_Integer status = Reader.ReadFile(fileName.toUtf8().constData());

	if (status != IFSelect_RetDone) return;
	Reader.TransferRoots();
	Handle(AIS_Shape) aisShape = new AIS_Shape(Reader.OneShape());
	myOccViewer->getContext()->Display(aisShape);

}


void STACCATOMainWindow::readSTL(QString fileName) {
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
	for (Standard_Integer anIndex = 1; anIndex <= aLen; anIndex++) {
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

//void STACCATOMainWindow::drawInt2DLine(void){
//}

void STACCATOMainWindow::drawCantilever(void) {
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

void STACCATOMainWindow::animateObject(void) {

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
	if (status != IFSelect_RetDone) {
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
	Standard_Real  gamma = asin(5.0 / 12.0);

	for (int i = 0; i < numSteps; i++) {
		gp_Trsf myTrafo;
		phi = ((double)i / numSteps) * 3600 * (M_PI / 180);
		alpha = (12 / (5 * cos(gamma)))*phi;

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


void STACCATOMainWindow::handleSelectionChanged(void) {

	for (myOccViewer->getContext()->InitSelected(); myOccViewer->getContext()->MoreSelected(); myOccViewer->getContext()->NextSelected())
	{
		Handle(AIS_InteractiveObject) anIO = myOccViewer->getContext()->SelectedInteractive();
		infoOut << "anIO->Type() : " << anIO->Type() << std::endl;

		if (anIO->Type() == AIS_KOI_None) {
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
			else if (owner->Type() == MeshVS_ET_Node) {
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
		else if (anIO->Type() == AIS_KOI_Datum) {
			infoOut << "anIO: " << anIO->Signature() << endl;
			if (anIO->Signature() == 1) {//datum point
				Handle(AIS_Point) aAISPoint = Handle(AIS_Point)::DownCast(anIO);
				TopoDS_Vertex vertex = aAISPoint->Vertex();
				gp_Pnt myPoint = BRep_Tool::Pnt(TopoDS::Vertex(vertex));
				infoOut << "==========" << std::endl;
				infoOut << "X: " << myPoint.X() << std::endl;
				infoOut << "Y: " << myPoint.Y() << std::endl;
				infoOut << "Z: " << myPoint.Z() << std::endl;
				infoOut << "==========" << std::endl;
			}
			else if (anIO->Signature() == 1) {//datum axis

			}
		}
		else if (anIO->Type() == AIS_KOI_Shape) {
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
		else if (anIO->Type() == AIS_KOI_Object) {

		}
		else if (anIO->Type() == AIS_KOI_Relation) {

		}
		else if (anIO->Type() == AIS_KOI_Dimension) {

		}





	}

}

void STACCATOMainWindow::myViewModeTriggered() {
	myFieldDataVisualizer->setViewMode(myRotateModeButton->isChecked());
}

void STACCATOMainWindow::myWarpVectorTriggered() {
	if (myDockWarpVectorAction->isChecked() || myWarpVectorVisibility->isChecked()) {
		myWarpDock = new QDockWidget(tr("Warp Vector"), this);
		myWarpDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);

		myAutoScaling = new QCheckBox(tr("Default Scaling"), this);
		myAutoScaling->setChecked(true);
		connect(myAutoScaling, SIGNAL(clicked()), this, SLOT(myAutoScalingState()));

		myScalingFactorLabel = new QLabel(tr("Scaling Factor"), this);

		myScalingFactor = new QSpinBox;
		myScalingFactor->setRange(0, 10000000000);
		myScalingFactor->setValue(myVisualizerSetting->PROPERTY_SCALING_FACTOR);
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

void STACCATOMainWindow::myViewPropertyDockTriggered() {
	if (myViewPropertyAction->isChecked()) {
		myViewPropertyDock = new QDockWidget(tr("View Property"), this);
		myViewPropertyDock->setAllowedAreas(Qt::LeftDockWidgetArea);

		myReferenceNode = new QCheckBox(tr("Reference Node"), this);
		myReferenceNode->setChecked(false);
		myReferenceNode->setEnabled(false);
		connect(myReferenceNode, SIGNAL(clicked()), this, SLOT(myReferenceNodeTriggered()));

		QFormLayout *layout = new QFormLayout;
		layout->addRow(myReferenceNode);
		QWidget* temp = new QWidget(this);
		temp->setLayout(layout);

		myViewPropertyDock->setWidget(temp);
		myViewPropertyDock->show();

		addDockWidget(Qt::LeftDockWidgetArea, myViewPropertyDock);
	}
	else {
		removeDockWidget(myViewPropertyDock);
	}
}


void STACCATOMainWindow::myAutoScalingState() {
	if (myAutoScaling->isChecked()) {
		myScalingFactor->setValue(1);
		myScalingFactor->setEnabled(false);
	}
	else {
		myScalingFactor->setValue(myVisualizerSetting->PROPERTY_SCALING_FACTOR);
		myScalingFactor->setEnabled(true);
	}
	myScalingFactorState();
}

void STACCATOMainWindow::myScalingFactorState() {
	myVisualizerSetting->commitScalingFactor(QString(myScalingFactor->text()).toDouble());
	myViewPropertyUpdate();
}

void STACCATOMainWindow::mySignalDataVisualizerInterface() {
	static bool mySignalDataVisualizerActive = false;
	if (!mySignalDataVisualizerActive) {
		mySignalDataVisualizer = new SignalDataVisualizer();
		mySignalDataVisualizer->setHMesh(*myHMesh);
		mySignalDataVisualizer->attachSubject(myFieldDataVisualizer);
		mySignalDataVisualizer->initiate();
		mySignalDataVisualizerActive = true;
	}
	mySignalDataVisualizer->show();
}

void STACCATOMainWindow::myUMATriggered() {
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

void STACCATOMainWindow::myUMAImport() {
	std::cout << ">> Trying to start UMA Interface ... " << std::endl;
	//myUMA->openSIM(mySIMFileName->text().toLocal8Bit().data());

}

void STACCATOMainWindow::myUMAHMesh() {
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

void STACCATOMainWindow::myReferenceNodeTriggered() {
	if (myHMesh->referenceNodeLabel.size() != 0)
		myReferenceNode->setEnabled(true);
}

void STACCATOMainWindow::mySubFrameAnimate() {
	if (mySubFrameAnimateAction->isChecked()) {
		mySubFrameAnimatorToolBar->show();
	}
	else {
		mySubFrameAnimatorToolBar->hide();
	}
}

void STACCATOMainWindow::myUpdateAnimationSlider(int _currentIndex) {
	isSubFrame = true;
	myCaseIndex = myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().find(_currentIndex - 1);
	myCaseStepText->setText(QString::fromStdString(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription()[myCaseIndex->first]));		// Update Slider
	myFieldDataVisualizer->plotVectorFieldAtIndex(myCaseIndex->first);
}

void STACCATOMainWindow::myGenerateAnimationFramesProc(void) {
	anaysisTimer01.start();
	std::cout << ">> Animation Data ... " << std::endl;
	std::cout << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

	myVisualizerSetting->generateAnimation();

	myCaseStepText->setEnabled(true);
	myHorizontalSlider->setEnabled(true);
	myAnimatePlayButton->setEnabled(true);
	myAnimateStopButton->setEnabled(true);
	myAnimateRepeatButton->setEnabled(true);
	myResetFrameAnimationButton->setEnabled(true);

	myHorizontalSlider->setFocusPolicy(Qt::StrongFocus);
	myHorizontalSlider->setTickPosition(QSlider::TicksBothSides);
	myHorizontalSlider->setTickInterval(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().size() - 1);
	myHorizontalSlider->setSingleStep(1);
	myHorizontalSlider->setMinimum(1);
	myHorizontalSlider->setMaximum(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[0].getResultsCaseDescription().size());
	myHorizontalSlider->connect(myHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(myUpdateAnimationSlider(int)));

	anaysisTimer01.stop();
	std::cout << "Duration for Frame Generation: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	std::cout << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
}

void STACCATOMainWindow::myAnimationResetProc() {
	myHorizontalSlider->setEnabled(false);
	myAnimatePlayButton->setEnabled(false);
	myAnimateStopButton->setEnabled(false);
	myAnimateRepeatButton->setEnabled(false);
	myResetFrameAnimationButton->setEnabled(false);

	myHorizontalSlider->setValue(1);
	myVisualizerSetting->stopAnimation();
}

void STACCATOMainWindow::myAnimationScenePlayProc() {
	myAnimatePlayButton->setEnabled(false);

	int duration = std::stoi(myAnimationDuration->text().toStdString());
	int repeat = (myAnimateRepeatButton->isChecked()) ? 1 : 0;

	myVisualizerSetting->visualizeAnimationFrames(duration, repeat);
	
	myAnimatePlayButton->setEnabled(true);
}

void STACCATOMainWindow::myAnimationSceneStopProc() {
	myVisualizerSetting->stopAnimation();
}

void STACCATOMainWindow::myResultCaseChanged() {
}

void STACCATOMainWindow::myAnimationOptionsTriggered() {
	myAnimationOptionsDock->show();
}

QTreeWidgetItem* STACCATOMainWindow::addRootToTree(QTreeWidget* _tree, QString _name, bool _checkable) {
	QTreeWidgetItem* item = new QTreeWidgetItem(_tree);
	item->setText(0, _name);
	item->setFlags(item->flags() & ~Qt::ItemIsSelectable);
	if (_checkable) {
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(0, Qt::Unchecked);
	}
	_tree->addTopLevelItem(item);
	return item;
}

void STACCATOMainWindow::addChildToTree(QTreeWidgetItem* _parent, QString _name, bool _checkable) {
	QTreeWidgetItem* item = new QTreeWidgetItem();
	item->setText(0, _name);
	if (_checkable) {
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(0, Qt::Unchecked);
	}
	_parent->addChild(item);
}