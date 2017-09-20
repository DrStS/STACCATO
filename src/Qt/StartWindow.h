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

// QT5
#include <QMainWindow>
// OCC
#include <AIS_InteractiveContext.hxx>

// forward declaration
class OccViewer;
class VtkViewer;
class QTextEdit;


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
	void openOBDFile(void);
	void animateObject(void);

private:
	Ui::StartWindow *ui;
	/// File action.
	QAction* myExitAction;
    QAction* myReadFileAction;
	QAction* myReadOBDFileAction;
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
	/// Help action.
	QAction* myAboutAction;
	/// Menus.
	QMenu* myFileMenu;
	QMenu* myCreateMenu;
	QMenu* mySelectionMenu;
	QMenu* myHelpMenu;
	/// Toolbars
	QToolBar* myFileToolBar;
	QToolBar* myViewToolBar;
	QToolBar* myCreateToolBar;
	QToolBar* myHelpToolBar;

	/// wrapped the widget for occ.
	OccViewer* myOccViewer;

	VtkViewer* myVtkViewer;
	/// the dockable widgets
	QTextEdit* textOutput;

	QMainWindow* newWin;

};

#endif /* STARTWINDOW_H */
