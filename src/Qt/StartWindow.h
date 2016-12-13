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
class QTextEdit;


namespace Ui {
	class StartWindow;
}

class StartWindow : public QMainWindow {
	Q_OBJECT

public:
	explicit StartWindow(QWidget *parent = 0);
	~StartWindow();

protected:
	void createActions(void);
	void createMenus(void);
	void createToolBars(void);
	void createDockWindows(void);

private slots:
	void about(void);
	void readSTL(void);
	void drawCantilever(void);
	void handleSelectionChanged(void);

private:
	Ui::StartWindow *ui;
	//! the exit action.
	QAction* mExitAction;

	//! show the about info action.
	QAction* mAboutAction;
	QAction* mReadSTLAction;
	QAction* mDrawCantileverAction;

	//! the menus of the application.
	QMenu* mFileMenu;
	QMenu* mCreateMenu;
	QMenu* mHelpMenu;

	QToolBar* mFileToolBar;
	QToolBar* mHelpToolBar;

	// wrapped the widget for occ.
	OccViewer* myOccViewer;

	//! the dockable widgets
	QTextEdit* textOutput;

};

#endif // STARTWINDOW_H 
