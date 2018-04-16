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
																								 //std
#include <future>
#include <thread>
																								 //STACCATO
#include "STACCATO_Enum.h"
																								 // QT5
#include <QMainWindow>

																								 // forward declaration
class VtkViewer;
class STACCATOComputeEngine;
class OutputDatabase;
class SignalDataVisualizer;
class VisualizerSetting;
class FieldDataVisualizer;
class STACCATOComputeEngine;
class OutputDatabase;
class QTreeWidget;
class QTreeWidgetItem;
class QRadioButton;
class QLabel;
class QTreeWidgetItem;
class QTreeWidget;
class QPushButton;
class QComboBox;
class QLineEdit;
class QTextEdit;
class QCheckBox;
class QGroupBox;
class QSpinBox;
class QFormLayout;
class QSlider;
class QButtonGroup;

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
	void fillFEResultInGUI();
	void createAnimationOptionsDock(void);
	QTreeWidgetItem* addRootToTree(QTreeWidget*, QString, bool);
	void addChildToTree(QTreeWidgetItem*, QString, QString, bool);

	private slots:
	void about(void);
	void openDataFlowWindow(void);
	void myTimeStepLessProc(void);
	void myTimeStepAddProc(void);
	void myViewPropertyUpdate(void);
	void myWarpVectorTriggered(void);
	void myAutoScalingState(void);
	void myScalingFactorState(void);
	void mySignalDataVisualizerInterface(void);
	void myViewModeTriggered(void);
	void myUMATriggered(void);
	void importXMLFile(void);
	void myViewPropertyDockTriggered(void);
	void mySubFrameAnimate(void);
	void myGenerateAnimationFramesProc(void);
	void myAnimationResetProc(void);
	void myUpdateAnimationSlider(int);
	void myAnimationScenePlayProc(void);
	void myAnimationSceneStopProc(void);
	void myAnimationOptionsTriggered(void);
	void myAnimationOptionAnalysisItemSelected(QTreeWidgetItem* _item);
	void myAnimationOptionCaseItemSelected(QTreeWidgetItem* _item);
	void myAnalysisTreeUpdate(void);
	void myCaseTreeSelectAll(void);
	void myCaseTreeDeselectAll(void);
	void mySetDefaultProc(void);
	void myAnalysisChangeProc(void);
	void notifyAnalysisCompleteSuccessfully(void);

private:
	std::vector<std::string> allDispSolutionTypes;
	std::vector<std::string> allDispVectorComponents;
	std::vector<std::string> allViewModes;

	Ui::STACCATOMainWindow *myGui;

	/// File action.
	QAction* myExitAction;
	QAction* myImportXMLFileAction;
	/// Buttons
	QPushButton* myTimeStepLessAction;
	QPushButton* myTimeStepAddAction;
	/// Create action.
	QAction* myDataFlowAction;
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
	QMenu* myAnimateMenu;
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

	FieldDataVisualizer* myFieldDataVisualizer;
	SignalDataVisualizer* mySignalDataVisualizer;
	VisualizerSetting* myVisualizerSetting;

	/// the dockable widgets
	QTextEdit* textOutput;
	QLineEdit* myTimeStepText;

	/// Spin Boxes
	QSpinBox* myScalingFactor;

	/// Labels
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
	QDockWidget *myAnimationOptionsDock;

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
	QAction* myAnimationOptions;
	QAction* mySubFrameAnimateAction;
	QAction* myCreateFrameAnimationButton;
	QAction* myResetFrameAnimationButton;
	QAction* myAnimatePlayButton;
	QAction* myAnimateStopButton;
	QLineEdit* myAnimationDuration;
	QAction* myAnimateRepeatButton;
	QSlider *myHorizontalSlider;
	QLabel* myAnimationDataLabel;
	QLabel* myManualFrameControlLabel;
	QLabel* myAnimationControlLabel;
	QLabel* myAnimationDurationLabel;

	/// Visualizer Info Dock Widgets
	///Labels
	QLabel* myAnimationLabel;
	QLabel* myAnimationAnalysisTreeLabel;
	QLabel* myAnimationCaseTreeLabel;
	QLabel* myAnalysisSelectorLabel;
	///Radio Buttons
	QRadioButton* myHarmonicAnimationRadio;
	QRadioButton* myCaseAnimationRadio;
	QRadioButton* myFrequencyAnimationRadio;
	QButtonGroup* myAnimationButtonGroup;
	///ComboBox
	QComboBox* myAnalysisSelector;
	///Tree
	QTreeWidget* myAnimationAnalysisTree;
	QTreeWidget* myAnimationCaseTree;
	///PushButtons
	QPushButton* myAnimationCaseTreeSelectAllButton;
	QPushButton* myAnimationCaseTreeDeselectAllButton;
	QPushButton* myAnimationOptionApplyButton;
	QPushButton* myAnimationSetDefaultButton;
	///CheckBox
	QCheckBox* myAnimationOptionPreview;

	OutputDatabase* myOutputDatabase;
	STACCATOComputeEngine * myComputeEngine;

	//Parallel Execution 
	std::future<void> myFuture;
};