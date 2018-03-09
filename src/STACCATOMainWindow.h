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
/***********************************************************************************************//**
* \file STACCATOMainWindow.h
* This file holds the class of StartWindow.
* \date 9/16/2016
**************************************************************************************************/
#pragma once

#include "HMeshToVtkUnstructuredGrid.h"

// QT5
#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
// OCC
#include <AIS_InteractiveContext.hxx>
// SimuliaOBD
#include "SimuliaODB.h"
/// Visualizer
#include "SignalDataVisualizer.h"
#include "VisualizerSetting.h"

//Enums
#include "STACCATO_Enum.h"

// forward declaration
class OccViewer;
class VtkViewer;
class UMA_AccessSparse;
class QTextEdit;
class QCheckBox;
class QGroupBox;
class QSpinBox;
class QFormLayout;
class QSlider;

namespace Ui {
	class STACCATOMainWindow;
}
/********//**
* \brief Class STACCATOMainWindow the core of the GUI
***********/
class STACCATOMainWindow : public QMainWindow {
	Q_OBJECT

public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	explicit STACCATOMainWindow(QWidget *parent = 0);
	/***********************************************************************************************
	* \brief Destructor
	* \author Stefan Sicklinger
	***********/
	~STACCATOMainWindow(); 
protected:
	/***********************************************************************************************
	* \brief Creat all Actions of Qt
	* \author Stefan Sicklinger
	***********/
	void createActions(void);
	void createMenus(void);
	void createToolBars(void);
	void createDockWindows(void);
	void readSTEP(QString);
	void readIGES(QString);
	void readSTL(QString);
	void fillFEResultInGUI();

	private slots:
	void about(void);
	void importFile(void);
	void drawCantilever(void);
	void handleSelectionChanged(void);
	void openDataFlowWindow(void);
	void animateObject(void);
	void myTimeStepLessProc(void);
	void myTimeStepAddProc(void);
	void myViewPropertyUpdate(void);
	void myWarpVectorTriggered(void);
	void myAutoScalingState(void);
	void myScalingFactorState(void);
	void mySignalDataVisualizerInterface(void);
	void myViewModeTriggered(void);
	void myUMATriggered(void);
	void myUMAImport(void);
	void myUMAHMesh(void);
	void importXMLFile(void);
	void myViewPropertyDockTriggered(void);
	void myReferenceNodeTriggered(void);
	void mySubFrameAnimate(void);
	void myCaseStepLessProc(void);
	void myCaseStepAddProc(void);
	void myGenerateAnimationFramesProc(void);
	void myAnimationResetProc(void);
	void myUpdateAnimationSlider(int);
	void myAnimationScenePlayProc(void);
	void myAnimationSceneStopProc(void);
	void myResultCaseChanged(void);

private:
	std::vector<std::string> allDispSolutionTypes;
	std::vector<std::string> allDispVectorComponents;
	std::vector<std::string> allViewModes;

	HMeshToVtkUnstructuredGrid* myHMeshToVtkUnstructuredGrid;

	Ui::STACCATOMainWindow *myGui;
	/// File action.
	QAction* myExitAction;
	QAction* myReadFileAction;
	QAction* myImportXMLFileAction;
	/// Buttons
	QPushButton* myTimeStepLessAction;
	QPushButton* myTimeStepAddAction;
	/// Create action.
	QAction* myDrawCantileverAction;
	QAction* myDataFlowAction;
	QAction* myAnimationAction;
	/// View action
	QAction* myPanAction;
	QAction* myZoomAction;
	QAction* myFitAllAction;
	QAction* myRotateAction;
	/// Selection action
	QAction* mySetSelectionModeNoneAction;
	QAction* mySetSelectionModeNodeAction;
	QAction* mySetSelectionModeElementAction;
	/// Export action
	QAction* myUMAAction;
	/// Layout action
	QAction* myResetLayoutAction;
	QAction* myDockWarpVectorAction;
	QAction* myViewPropertyAction;
	QAction* my2dVisualizerAction;
	/// Help action.
	QAction* myAboutAction;
	/// Menus.
	QMenu* myFileMenu;
	QMenu* myCreateMenu;
	QMenu* mySelectionMenu;
	QMenu* myImportMenu;
	QMenu* myLayoutMenu;
	QMenu* myHelpMenu;
	/// SubMenus
	QMenu* myViewToolbarSubMenu;
	QMenu* myViewDockSubMenu;
	/// Toolbars
	QToolBar* myFileToolBar;
	QToolBar* myViewToolBar;
	QToolBar* myCreateToolBar;
	QToolBar* myHelpToolBar;
	QToolBar* myTimeToolBar;
	QToolBar* mySolutionToolBar;
	QToolBar* myDisplayControlToolBar;
	QToolBar* myPickerViewToolBar;
	QToolBar* mySubFrameAnimatorToolBar;
	/// Selectors
	QComboBox* mySolutionSelector;
	QComboBox* myComponentSelector;
	QComboBox* myViewModeSelector;
	QComboBox* myAnimationSolutionSelector;
	QComboBox* myResultCaseSelector;

	/// wrapped the widget for occ.
	OccViewer* myOccViewer;

	FieldDataVisualizer* myFieldDataVisualizer;
	SignalDataVisualizer* mySignalDataVisualizer;
	VisualizerSetting* myVisualizerSetting;

	/// the dockable widgets
	QTextEdit* textOutput;
	QLineEdit* myTimeStepText;

	/// Spin Boxes
	QSpinBox* myScalingFactor;

	/// Labels
	QLabel* myTimeStepLabel;
	QLabel* myScalingFactorLabel;
	QLabel* myAnimateSolutionTypeLabel;

	/// Check Boxes
	QCheckBox* myAutoScaling;
	QCheckBox* myReferenceNode;

	/// View Control ToolBar Buttons
	QPushButton* myScalarBarVisibility;
	QPushButton* myWarpVectorVisibility;
	QPushButton* myUMAImportVisibility;
	QPushButton* mySignalDataVisualizerVisiblility;
	QPushButton* myRotateModeButton;
	QPushButton* myPickModeButton;

	/// Picker View ToolBar Buttons
	QPushButton* myPickerModeNone;
	QPushButton* myPickerModeNode;
	QPushButton* myPickerModeElement;

	/// Button Groups
	QButtonGroup* myPickerButtonGroup;
	QButtonGroup* myViewButtonGroup;

	QMainWindow* newWin;

	/// Dock Windows
	QDockWidget *myWarpDock;
	QDockWidget *myUMADock;
	QDockWidget *myViewPropertyDock;

	/// UMA Widgets
	QPushButton* myUMAInterfaceButton;
	QPushButton* mySIMImportButton;
	QLabel* mySIMImportLabel;
	QLineEdit* mySIMFileName;

	/// Layouts
	QGroupBox* myWarpVectorLayout;
	QFormLayout* myVisualizerDockLayout;

	/// Widgets
	QWidget* myVisualizerDockWidgets;

	/// Sub Frame Animator Widgets
	QAction* myAnimationButton;
	QAction* mySubFrameAnimateAction;
	QAction* myCreateFrameAnimationButton;
	QAction* myResetFrameAnimationButton;
	QAction* myAnimatePlayButton;
	QPushButton* myAnimateNextFrameButton;
	QPushButton* myAnimatePrevFrameButton;
	QAction* myAnimateStopButton;
	QLineEdit* myCaseStepText;
	QLineEdit* myAnimationDuration;
	QAction* myAnimateRepeatButton;
	QSlider *myHorizontalSlider;
	QLabel* myAnimationDataLabel;
	QLabel* myManualFrameControlLabel;
	QLabel* myAnimationControlLabel;
	QLabel* myAnimationDurationLabel;

	std::map<int, std::string>::iterator myFreqIndex;
	std::map<int, std::string>::iterator myCaseIndex;

private:
	/// HMesh object 
	HMesh *myHMesh;

public:
	int globalFrame;
	bool isSubFrame;
};