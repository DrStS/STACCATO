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
* \file StartWindow.h
* This file holds the class of StartWindow.
* \date 9/16/2016
**************************************************************************************************/
#ifndef STARTWINDOW_H
#define STARTWINDOW_H

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
#include "VisualizerWindow.h"

//#include <python.h>

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
	class StartWindow;
}
/********//**
* \brief Class StartWindow the core of the GUI
***********/
class StartWindow : public QMainWindow {
	Q_OBJECT

public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	explicit StartWindow(QWidget *parent = 0);
	/***********************************************************************************************
	* \brief Destructor
	* \author Stefan Sicklinger
	***********/
	~StartWindow(); 
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
	void my2dVisualizerInterface(void);
	void myViewModeTriggered(void);
	void myUMATriggered(void);
	void myUMAImport(void);
	void myUMAHMesh(void);
	void importXMLFile(void);
	void myViewPropertyDockTriggered(void);
	void myReferenceNodeTriggered(void);
	void mySubFrameAnimate(void);
	void mySubFramePrevProc(void);
	void mySubFrameNextProc(void);
	void mySubFramePlayProv(void);
	void myAnimationDockTriggered(void);

private:
	std::vector<std::string> allDispSolutionTypes;
	std::vector<std::string> allDispVectorComponents;
	std::vector<std::string> allViewModes;

	HMeshToVtkUnstructuredGrid* myHMeshToVtkUnstructuredGrid;

	Ui::StartWindow *ui;
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

	/// wrapped the widget for occ.
	OccViewer* myOccViewer;

	VtkViewer* myVtkViewer;
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
	QPushButton* my2dVisualizerVisibility;
	QPushButton* myRotateModeButton;
	QPushButton* myPickModeButton;
	QPushButton* myAnimationDockButton;

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
	QDockWidget *myAnimationDock;

	/// QLineEdits
	QLineEdit* myAnimateScalingFactor;

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
	QAction* mySubFrameAnimateAction;
	QPushButton* myPreviousFrameButton;
	QPushButton* myNextFrameButton;
	QPushButton* mySubFrameAnimateButton;
	QLineEdit* mySubFrameText;
	QPushButton* myCreateFrameAnimationButton;
	QPushButton* myResetFrameAnimationButton;
	QPushButton* myAnimatePlayPauseButton;
	QPushButton* myAnimateNextFrameButton;
	QPushButton* myAnimatePrevFrameButton;
	QCheckBox* myAnimateRepeat;
	QSlider *myHorizontalSlider;

	int myFreqIndex;
	int mySubFrameIndex;
	int myScalingFactorValue;

private:
	/// HMesh object 
	HMesh *myHMesh;

public:
	bool isSubFrame;
};

#endif /* STARTWINDOW_H */
