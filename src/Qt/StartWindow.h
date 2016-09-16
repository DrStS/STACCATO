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
	//! create all the actions.
	void createActions(void);

	//! create all the menus.
	void createMenus(void);

	//! create the toolbar.
	void createToolBars(void);

	private slots:

	//! show about box.
	void about(void);

	void readSTL(void);

private:
	Ui::StartWindow *ui;


private:
	//! the exit action.
	QAction* mExitAction;

	//! show the about info action.
	QAction* mAboutAction;

	QAction* mReadSTLAction;

	//! the menus of the application.
	QMenu* mFileMenu;
	QMenu* mHelpMenu;

	QToolBar* mFileToolBar;
	QToolBar* mHelpToolBar;

	// wrapped the widget for occ.
	OccViewer* myOccViewer;

};

#endif // STARTWINDOW_H 
