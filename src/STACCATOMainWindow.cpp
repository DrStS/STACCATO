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
#include "STACCATOComputeEngine.h"
#include "ui_STACCATOMainWindow.h"
#include "AuxiliaryParameters.h"
#include "Message.h"
#include "Timer.h"
#include "MemWatcher.h"
#include "OutputDatabase.h"
#include "VectorFieldResults.h"
#include "VisualizerSetting.h"
#include "SignalDataVisualizer.h"
#include "FieldDataVisualizer.h"

#include "qnemainwindow.h"

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
#include <QRadioButton>

#include <map>

QT_CHARTS_USE_NAMESPACE


STACCATOMainWindow::STACCATOMainWindow(QWidget *parent) : QMainWindow(parent), myGui(new Ui::STACCATOMainWindow)
{
	myGui->setupUi(this);
	setWindowIcon(QIcon(":/Qt/resources/STACCATO.png"));
	setWindowTitle("STACCATO" + QString::fromStdString(STACCATO::AuxiliaryParameters::gitTAG));

	myFieldDataVisualizer = new FieldDataVisualizer(this);

	myVisualizerSetting = new VisualizerSetting();
	myVisualizerSetting->setCommuniationToFieldDataVisualizer(*myFieldDataVisualizer);

	setCentralWidget(myFieldDataVisualizer);
	createActions();
	createMenus();
	createToolBars();
	createDockWindows();
	createAnimationOptionsDock();
}

STACCATOMainWindow::~STACCATOMainWindow()
{
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

	// Time Step
	myTimeStepLessAction = new QPushButton(tr("<"), this);
	myTimeStepLessAction->setFixedWidth(40);
	myTimeStepLessAction->setStatusTip(tr("Previous Frequency"));
	myTimeStepLessAction->setEnabled(false);
	connect(myTimeStepLessAction, SIGNAL(clicked()), this, SLOT(myTimeStepLessProc()));

	myTimeStepAddAction = new QPushButton(tr(">"), this);
	myTimeStepAddAction->setFixedWidth(40);
	myTimeStepAddAction->setStatusTip(tr("Next Frequency"));
	myTimeStepAddAction->setEnabled(false);
	connect(myTimeStepAddAction, SIGNAL(clicked()), this, SLOT(myTimeStepAddProc()));

	myTimeStepText = new QLineEdit(this);
	myTimeStepText->setText("-");
	myTimeStepText->setFixedWidth(50);
	myTimeStepText->setAlignment(Qt::AlignHCenter);
	myTimeStepText->setReadOnly(true);

	mySolutionSelector = new QComboBox();
	mySolutionSelector->addItem("Solution..");
	QStandardItemModel* model = qobject_cast<QStandardItemModel*>(mySolutionSelector->model());
	QStandardItem* item = model->item(0);
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
	myDataFlowAction = new QAction(tr("Dataflow manager"), this);
	myDataFlowAction->setIcon(QIcon(":/Qt/resources/dataflow.png"));
	myDataFlowAction->setStatusTip(tr("Open dataflow manager"));
	connect(myDataFlowAction, SIGNAL(triggered()), this, SLOT(openDataFlowWindow()));

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
	connect(myAnimationButton, SIGNAL(triggered()), this, SLOT(myAnimationOptionsTriggered()));
	myAnimationButton->setEnabled(false);

	myAnimationOptions = new QAction(tr("Options"), this);
	myAnimationOptions->setIcon(QIcon(":/Qt/resources/setting.ico"));
	myAnimationOptions->setStatusTip(tr("Open Animation Options"));
	myAnimationOptions->setCheckable(false);
	myAnimationOptions->setChecked(false);
	connect(myAnimationOptions, SIGNAL(triggered()), this, SLOT(myAnimationOptionsTriggered()));
	//myAnimationOptions->setEnabled(false);

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

	// Fill Animation Options
	// - Analysis
	myAnalysisSelector->disconnect();
	for (int i = 0; i < myOutputDatabase->getAnalysisDescription().size(); i++) {
		myAnalysisSelector->addItem(QString::fromStdString(myOutputDatabase->getAnalysisDescription()[i]));
	}
	myAnalysisSelector->setCurrentIndex(1);
	connect(myAnalysisSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myAnalysisChangeProc()));

	myAnalysisTreeUpdate();

	mySolutionSelector->disconnect();
	for (int i = 0; i < myOutputDatabase->getVectorFieldFromDatabase().size(); i++)
		mySolutionSelector->addItem(QString::fromStdString(myOutputDatabase->getVectorFieldFromDatabase()[i].myLabel));
	mySolutionSelector->setCurrentIndex(1);
	connect(mySolutionSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));

	myComponentSelector->disconnect();
	for (std::map<std::string, STACCATO_VectorField_components>::iterator it = myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].myResultLabelMap.begin(); it != myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].myResultLabelMap.end(); ++it) {
		myComponentSelector->addItem(QString::fromStdString(it->first));
	}
	myComponentSelector->setCurrentIndex(1);
	connect(myComponentSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));

	myViewModeSelector->disconnect();
	for (std::map<std::string, STACCATO_FieldProperty_type>::iterator it = myVisualizerSetting->myViewModeLabelMap.begin(); it != myVisualizerSetting->myViewModeLabelMap.end(); ++it)
		myViewModeSelector->addItem(QString::fromStdString(it->first));
	myViewModeSelector->setFixedWidth(120);
	myViewModeSelector->setCurrentIndex(1);
	connect(myViewModeSelector, SIGNAL(currentTextChanged(const QString&)), this, SLOT(myViewPropertyUpdate()));
}

void STACCATOMainWindow::myViewPropertyUpdate(void) {
	myAnimationResetProc();

	myVisualizerSetting->commitVectorFieldComponent(myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].myResultLabelMap[myComponentSelector->currentText().toStdString()]);	// Result Component
	myVisualizerSetting->commitViewSetting(myVisualizerSetting->myViewModeLabelMap[myViewModeSelector->currentText().toStdString()]);						// View Mode
	myVisualizerSetting->setScalarbarTitle(myComponentSelector->currentText().toStdString());																// Scalarbar Title
	myVisualizerSetting->commitScalarBar(myScalarBarVisibility->isChecked());																				// Scalarbar Visibility

	myVisualizerSetting->updateSetting();	// Visualize Frame

}

void STACCATOMainWindow::myTimeStepLessProc(void) {
	myVisualizerSetting->commitToPerviousTimeStep();
	myTimeStepText->setText(QString::fromStdString(myOutputDatabase->getTimeDescription(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX)[myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX] + myOutputDatabase->getUnit(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)));
	myViewPropertyUpdate();
}

void STACCATOMainWindow::myTimeStepAddProc(void) {
	myVisualizerSetting->commitToNextTimeStep();
	myTimeStepText->setText(QString::fromStdString(myOutputDatabase->getTimeDescription(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX)[myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX] + myOutputDatabase->getUnit(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)));
	myViewPropertyUpdate();
}

void STACCATOMainWindow::createMenus(void)
{
	myFileMenu = menuBar()->addMenu(tr("&File"));
	myFileMenu->addAction(myImportXMLFileAction);
	myFileMenu->addAction(myExitAction);

	myCreateMenu = menuBar()->addMenu(tr("Create"));
	myCreateMenu->addAction(myDataFlowAction);

	mySelectionMenu = menuBar()->addMenu(tr("Selection"));
	mySelectionMenu->addAction(mySetSelectionModeNoneAction);
	mySelectionMenu->addAction(mySetSelectionModeNodeAction);
	mySelectionMenu->addAction(mySetSelectionModeElementAction);

	myImportMenu = menuBar()->addMenu(tr("Import"));
	myImportMenu->addAction(myUMAAction);

	myAnimateMenu = menuBar()->addMenu(tr("Animate"));
	myAnimateMenu->addAction(myAnimationOptions);

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
	myFileToolBar->addAction(myImportXMLFileAction);
	myCreateToolBar = addToolBar(tr("Create"));
	myCreateToolBar->addAction(myDataFlowAction);

	myViewToolBar = addToolBar(tr("View"));
	myViewToolBar->addWidget(myPickModeButton);
	myViewToolBar->addWidget(myRotateModeButton);

	myTimeToolBar = addToolBar(tr("Time Step"));
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
	myAnalysisSelector->setEnabled(true);

	myAnimationAnalysisTreeLabel = new QLabel(tr("Select Analysis Tree:"));

	myAnimationAnalysisTree = new QTreeWidget;
	myAnimationAnalysisTree->setColumnCount(2);
	myAnimationAnalysisTree->setColumnWidth(0, 250);
	myAnimationAnalysisTree->setContextMenuPolicy(Qt::CustomContextMenu);
	myAnimationAnalysisTree->setMinimumHeight(100);
	myAnimationAnalysisTree->setHeaderLabels(QStringList() << "Frame" << "Description");
	connect(myAnimationAnalysisTree, SIGNAL(itemClicked(QTreeWidgetItem *, int)), this, SLOT(myAnimationOptionAnalysisItemSelected(QTreeWidgetItem *)));

	myAnimationCaseTreeLabel = new QLabel(tr("Select Frame(s) for Animation:"));

	myAnimationCaseTreeSelectAllButton = new QPushButton(tr("Select All"));
	connect(myAnimationCaseTreeSelectAllButton, SIGNAL(clicked()), this, SLOT(myCaseTreeSelectAll()));

	myAnimationCaseTreeDeselectAllButton = new QPushButton(tr("Deselect All"));
	connect(myAnimationCaseTreeDeselectAllButton, SIGNAL(clicked()), this, SLOT(myCaseTreeDeselectAll()));

	myAnimationCaseTree = new QTreeWidget;
	myAnimationCaseTree->setColumnCount(2);
	myAnimationCaseTree->setColumnWidth(0, 250);
	myAnimationCaseTree->setContextMenuPolicy(Qt::CustomContextMenu);
	myAnimationCaseTree->setMinimumHeight(100);
	myAnimationCaseTree->setHeaderLabels(QStringList() << "Frame" << "Description");
	myAnimationCaseTree->setSelectionMode(QAbstractItemView::ExtendedSelection);
	connect(myAnimationCaseTree, SIGNAL(itemClicked(QTreeWidgetItem *, int)), this, SLOT(myAnimationOptionCaseItemSelected(QTreeWidgetItem *)));

	myAnimationLabel = new QLabel(tr("Animate:"), this);

	myFrequencyAnimationRadio = new QRadioButton(tr("&Time Steps"));
	myHarmonicAnimationRadio = new QRadioButton(tr("&Harmonic"));
	myCaseAnimationRadio = new QRadioButton(tr("&LoadCases"));
	myCaseAnimationRadio->setChecked(true);

	myAnimationButtonGroup = new QButtonGroup(this);
	myAnimationButtonGroup->addButton(myFrequencyAnimationRadio);
	myAnimationButtonGroup->addButton(myHarmonicAnimationRadio);
	myAnimationButtonGroup->addButton(myCaseAnimationRadio);

	myAnimationOptionPreview = new QCheckBox(tr("Preview"), this);
	myAnimationOptionPreview->setChecked(true);

	myAnimationOptionApplyButton = new QPushButton(tr("Apply"));
	connect(myAnimationOptionApplyButton, SIGNAL(clicked()), this, SLOT(myGenerateAnimationFramesProc()));
	myAnimationSetDefaultButton = new QPushButton(tr("List Pos"));
	connect(myAnimationSetDefaultButton, SIGNAL(clicked()), this, SLOT(mySetDefaultProc()));

	myAnimationOptionsDock = new QDockWidget(tr("Options"), this);
	myAnimationOptionsDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);

	QGridLayout *layout = new QGridLayout;
	layout->addWidget(myAnalysisSelectorLabel, 1, 0);
	layout->addWidget(myAnalysisSelector, 1, 1, 1, -1);
	layout->addWidget(myAnimationLabel, 2, 0);
	layout->addWidget(myFrequencyAnimationRadio, 2, 1);
	layout->addWidget(myCaseAnimationRadio, 3, 1);
	layout->addWidget(myHarmonicAnimationRadio, 4, 1);
	layout->addWidget(myAnimationAnalysisTreeLabel, 5, 0);
	layout->addWidget(myAnimationAnalysisTree, 6, 0, 1, -1);
	layout->addWidget(myAnimationCaseTreeLabel, 7, 0);
	layout->addWidget(myAnimationCaseTreeSelectAllButton, 7, 1);
	layout->addWidget(myAnimationCaseTreeDeselectAllButton, 7, 2);
	layout->addWidget(myAnimationCaseTree, 8, 0, 1, -1);
	layout->addWidget(myAnimationOptionPreview, 9, 0);
	layout->addWidget(myAnimationOptionApplyButton, 9, 1);
	layout->addWidget(myAnimationSetDefaultButton, 9, 2);

	QWidget* temp = new QWidget(this);
	temp->setLayout(layout);

	myAnimationOptionsDock->setWidget(temp);

	addDockWidget(Qt::LeftDockWidgetArea, myAnimationOptionsDock);
	myAnimationOptionsDock->setFloating(true);
	myAnimationOptionsDock->hide();
	myAnimationOptionsDock->setMinimumWidth(500);

	myAnimationOptionsDock->move(QApplication::desktop()->screen()->rect().center() - myAnimationOptionsDock->rect().center());
}

void STACCATOMainWindow::about() {
	QMessageBox::about(this, tr("About STACCATO"),
		tr("<h2>STACCATO: STefAn's Computational vibroaCoustics Analysis TOol</h2>"
			"<p>Copyright &copy; 2018 "
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

		myComputeEngine = new STACCATOComputeEngine(fileName.toStdString());
		myComputeEngine->prepare();
		myComputeEngine->compute();
		myComputeEngine->clean();

		myOutputDatabase = (myComputeEngine->getOutputDatabase());
		if (!myComputeEngine->getHMesh().isKROM) {
			myFieldDataVisualizer->setHMesh(myComputeEngine->getHMesh());
			notifyAnalysisCompleteSuccessfully();
			anaysisTimer01.start();
			myFieldDataVisualizer->myHMeshToVtkUnstructuredGridInitializer();
			anaysisTimer01.stop();
			debugOut << "Duration for reading HMeshToVtkUnstructuredGrid " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
			myViewPropertyUpdate();
		} else
			mySignalDataVisualizerVisiblility->setEnabled(true);
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
	myScalingFactor->clearFocus();
	myVisualizerSetting->commitScalingFactor(QString(myScalingFactor->text()).toDouble());
	myViewPropertyUpdate();
}

void STACCATOMainWindow::mySignalDataVisualizerInterface() {
	static bool mySignalDataVisualizerActive = false;
	if (!mySignalDataVisualizerActive) {
		mySignalDataVisualizer = new SignalDataVisualizer();
		myVisualizerSetting->setCommuniationToSignalDataVisualizer(*mySignalDataVisualizer);
		mySignalDataVisualizer->setHMesh(myComputeEngine->getHMesh());
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
		//connect(myUMAInterfaceButton, SIGNAL(clicked()), this, SLOT(myUMAImport()));

		mySIMImportButton = new QPushButton(tr("Import SIM to HMesh"), this);
		//connect(mySIMImportButton, SIGNAL(clicked()), this, SLOT(myUMAHMesh()));

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

void STACCATOMainWindow::mySubFrameAnimate() {
	if (mySubFrameAnimateAction->isChecked()) {
		mySubFrameAnimatorToolBar->show();
	}
	else {
		mySubFrameAnimatorToolBar->hide();
	}
}

void STACCATOMainWindow::myUpdateAnimationSlider(int _currentIndex) {
	myFieldDataVisualizer->plotVectorFieldAtIndex(_currentIndex);
}

void STACCATOMainWindow::myGenerateAnimationFramesProc(void) {
	myAnimationOptionsDock->hide();
	myAnimationResetProc();

	anaysisTimer01.start();
	std::cout << ">> Animation Data ... " << std::endl;
	std::cout << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

	int sliderSteps = 0;

	std::vector<int> animationIndices;
	if (myFrequencyAnimationRadio->isChecked()) {
		std::cout << ">> Frequency Step Animation running...\n";
		// Generate Index
		for (int it = 0; it < myOutputDatabase->getTimeDescription(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX).size(); it++) {
			animationIndices.push_back(myOutputDatabase->myAnalyses[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].timeSteps[it].startIndex);
		}
		sliderSteps = animationIndices.size() - 1;
		myVisualizerSetting->generateCaseAnimation(animationIndices);
		myEnableAnimationWidgets();
	}
	else if (myHarmonicAnimationRadio->isChecked()) {
		std::cout << ">> Harmonic Animation running...\n";
		int numberOfFrames = 20;
		// Generate Index
		for (int it = 0; it < numberOfFrames; it++) {
			animationIndices.push_back(it);
		}
		sliderSteps = animationIndices.size() - 1;
		myVisualizerSetting->generateHarmonicAnimation(animationIndices);
		myEnableAnimationWidgets();
	}
	else if (myCaseAnimationRadio->isChecked()) {
		std::cout << ">> Load Case Animation running...\n";
		std::cout << ">> Selected Number of Frames: " << myAnimationCaseTree->selectedItems().size() << std::endl;
		// Generate Index
		if (myAnimationCaseTree->selectedItems().size() != 0) {
			for (int it = 0; it < myAnimationCaseTree->selectedItems().size(); it++) {
				int itemIndex = std::stoi(myAnimationCaseTree->selectedItems()[it]->text(0).toStdString());
				std::cout << ">> Selected: " << it << " = " << itemIndex << "Frame: " << myOutputDatabase->myAnalyses[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].timeSteps[myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX].caseList[itemIndex].startIndex << std::endl;

				animationIndices.push_back(myOutputDatabase->myAnalyses[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].timeSteps[myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX].caseList[itemIndex].startIndex);
			}
			sliderSteps = animationIndices.size() - 1;

			myVisualizerSetting->generateCaseAnimation(animationIndices);
			myEnableAnimationWidgets();
		}
	}

	myHorizontalSlider->setFocusPolicy(Qt::StrongFocus);
	if (sliderSteps > 0)
		myHorizontalSlider->setTickPosition(QSlider::TicksBothSides);
	else
		myHorizontalSlider->setTickPosition(QSlider::TicksLeft);
	myHorizontalSlider->setTickInterval(sliderSteps);
	myHorizontalSlider->setSingleStep(1);
	myHorizontalSlider->setMinimum(0);
	myHorizontalSlider->setMaximum(animationIndices.size() - 1);
	myHorizontalSlider->setValue(0);
	connect(myHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(myUpdateAnimationSlider(int)));

	anaysisTimer01.stop();
	std::cout << "Duration for Frame Generation: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	std::cout << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
}

void STACCATOMainWindow::myEnableAnimationWidgets() {
	myHorizontalSlider->setEnabled(true);
	myAnimatePlayButton->setEnabled(true);
	myAnimateStopButton->setEnabled(true);
	myAnimateRepeatButton->setEnabled(true);
	myResetFrameAnimationButton->setEnabled(true);
}

void STACCATOMainWindow::myAnimationResetProc() {
	myHorizontalSlider->setEnabled(false);
	myAnimatePlayButton->setEnabled(false);
	myAnimateStopButton->setEnabled(false);
	myAnimateRepeatButton->setEnabled(false);
	myResetFrameAnimationButton->setEnabled(false);

	myHorizontalSlider->setValue(0);
	myVisualizerSetting->stopAnimation();
}

void STACCATOMainWindow::myAnimationScenePlayProc() {
	myRotateModeButton->setChecked(true);
	myViewModeTriggered();

	myAnimatePlayButton->setEnabled(false);

	int duration = std::stoi(myAnimationDuration->text().toStdString());
	int repeat = (myAnimateRepeatButton->isChecked()) ? 1 : 0;

	myVisualizerSetting->visualizeAnimationFrames(duration, repeat);

	myAnimatePlayButton->setEnabled(true);
}

void STACCATOMainWindow::myAnimationSceneStopProc() {
	myVisualizerSetting->stopAnimation();
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
	item->setExpanded(true);
	_tree->addTopLevelItem(item);
	return item;
}

void STACCATOMainWindow::addChildToTree(QTreeWidgetItem* _parent, QString _name1, QString _name2, bool _checkable) {
	QTreeWidgetItem* item = new QTreeWidgetItem();
	item->setText(0, _name1);
	item->setText(1, _name2);
	if (_checkable) {
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(0, Qt::Unchecked);
	}
	item->setExpanded(true);
	_parent->addChild(item);
}

void STACCATOMainWindow::myAnalysisTreeUpdate() {
	// Clear All Trees
	myAnimationAnalysisTree->clear();
	myAnimationCaseTree->clear();

	// Fill Animation Tree
	myVisualizerSetting->setCurrentAnalysis(myAnalysisSelector->currentText().toStdString());
	QTreeWidgetItem* AnalysisRoot;
	AnalysisRoot = addRootToTree(myAnimationAnalysisTree, QString::fromStdString(myOutputDatabase->myAnalyses[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].name), false);
	for (int it = 0; it < myOutputDatabase->getTimeDescription(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX).size(); it++)
	{
		int index = myOutputDatabase->myAnalyses[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].timeSteps[it].startIndex;
		addChildToTree(AnalysisRoot, QString::fromStdString(std::to_string(it)), QString::fromStdString(myOutputDatabase->getTimeDescription(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX)[it] + myOutputDatabase->getUnit(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX, it)), false);
	}
	myTimeStepText->setText(QString::fromStdString(myOutputDatabase->getTimeDescription(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX)[0] + myOutputDatabase->getUnit(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX, 0)));

	// Fill LoadCase Tree & LoadCase Selector
	QTreeWidgetItem* CaseRoot;
	CaseRoot = addRootToTree(myAnimationCaseTree, QString::fromStdString("Load Cases"), false);
	for (int iCase = 0; iCase < myOutputDatabase->getNumberOfLoadCases(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX); iCase++) {
		// Tree
		OutputDatabase::LoadCase loadCase = myOutputDatabase->myAnalyses[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].timeSteps[myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX].caseList[iCase];
		addChildToTree(CaseRoot, QString::fromStdString(std::to_string(iCase)), QString::fromStdString(loadCase.name), false);
	}
}

void STACCATOMainWindow::myAnimationOptionAnalysisItemSelected(QTreeWidgetItem* _item) {
	// Clear Case Tree
	myAnimationCaseTree->clear();

	// Find Index
	int itemIndex = 0;
	if (_item->parent()) {
		itemIndex = std::stoi(_item->text(0).toStdString());
	}

	// Preview
	if (myAnimationOptionPreview->isChecked()) {
		myVisualizerSetting->commitTimeStepIndex(itemIndex);
		std::cout << ">> Frequency Changed for Case Frame: " << itemIndex << std::endl;

		myViewPropertyUpdate();
		myVisualizerSetting->listProperties();
		myTimeStepText->setText(QString::fromStdString(myOutputDatabase->getTimeDescription(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX)[myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX] + myOutputDatabase->getUnit(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)));
	}

	// Rebuild Case Tree
	QTreeWidgetItem* CaseRoot;
	CaseRoot = addRootToTree(myAnimationCaseTree, QString::fromStdString("Load Cases"), false);
	for (int iCase = 0; iCase < myOutputDatabase->getNumberOfLoadCases(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX, itemIndex); iCase++)
	{
		OutputDatabase::LoadCase loadCase = myOutputDatabase->myAnalyses[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].timeSteps[itemIndex].caseList[iCase];
		addChildToTree(CaseRoot, QString::fromStdString(std::to_string(iCase)), QString::fromStdString(loadCase.name), false);
	}
}

void STACCATOMainWindow::myAnimationOptionCaseItemSelected(QTreeWidgetItem* _item) {
	int index;

	if (myAnimationOptionPreview->isChecked() && _item->parent()) {
		index = std::stoi(_item->text(0).toStdString());
		myVisualizerSetting->commitLoadCaseIndex(index);

		std::cout << ">> Case Changed for Frequency Frame: " << index << std::endl;
		myVisualizerSetting->listProperties();
		myViewPropertyUpdate();
	}
}

void STACCATOMainWindow::myCaseTreeSelectAll() {
	myAnimationCaseTree->selectAll();
}

void STACCATOMainWindow::myCaseTreeDeselectAll() {
	myAnimationCaseTree->clearSelection();
}

void STACCATOMainWindow::mySetDefaultProc() {
	myVisualizerSetting->listProperties();
}

void STACCATOMainWindow::myAnalysisChangeProc() {
	myAnalysisTreeUpdate();
	myViewPropertyUpdate();
}

void STACCATOMainWindow::notifyAnalysisCompleteSuccessfully() {
	// Fill GUI
	fillFEResultInGUI();

	// Enable Tools
	myTimeStepLessAction->setEnabled(true);
	myTimeStepAddAction->setEnabled(true);
	myScalarBarVisibility->setEnabled(true);
	myWarpVectorVisibility->setEnabled(true);
	mySignalDataVisualizerVisiblility->setEnabled(true);
	myAnimationButton->setEnabled(true);
}

void STACCATOMainWindow::closeEvent(QCloseEvent *_event) {
	std::cout << ">> Closing STACCATO." << std::endl;
	// Animation process in loop has to be stopped, if running.
	myAnimationResetProc();
}
