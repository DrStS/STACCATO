#ifndef MAINWINDOW_H
#define MAINWINDOW_H

//QT5
#include <QMainWindow>
// OCC
#include <AIS_InteractiveContext.hxx>

// forward declaration
class OccView;

namespace Ui {
class MainWindow;
}

class MainWindow: public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();

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

	//! make box test.
	void makeBox(void);

	//! make cone test.
	void makeCone(void);

	//! make sphere test.
	void makeSphere(void);

	//! make cylinder test.
	void makeCylinder(void);

	//! make torus test.
	void makeTorus(void);

	//! fillet test.
	void makeFillet(void);

	//! chamfer test.
	void makeChamfer(void);

	//! test extrude algorithm.
	void makeExtrude(void);

	//! test revol algorithm.
	void makeRevol(void);

	//! test loft algorithm.
	void makeLoft(void);

	//! test boolean operation cut.
	void testCut(void);

	//! test boolean operation fuse.
	void testFuse(void);

	//! test boolean operation common.
	void testCommon(void);

	//! test helix shapes.
	void testHelix(void);

private:
	Ui::MainWindow *ui;

	//! make cylindrical helix.
	void makeCylindricalHelix(void);

	//! make conical helix.
	void makeConicalHelix(void);

	//! make toroidal helix.
	void makeToroidalHelix(void);

private:
	//! the exit action.
	QAction* mExitAction;

	//! the actions for the view: pan, reset, fitall.
	QAction* mViewZoomAction;
	QAction* mViewPanAction;
	QAction* mViewRotateAction;
	QAction* mViewResetAction;
	QAction* mViewFitallAction;

	//! the actions to test the OpenCASCADE modeling algorithms.
	QAction* mMakeBoxAction;
	QAction* mMakeConeAction;
	QAction* mMakeSphereAction;
	QAction* mMakeCylinderAction;
	QAction* mMakeTorusAction;

	//! make a fillet box.
	QAction* mFilletAction;
	QAction* mChamferAction;
	QAction* mExtrudeAction;
	QAction* mRevolveAction;
	QAction* mLoftAction;

	//! boolean operations.
	QAction* mCutAction;
	QAction* mFuseAction;
	QAction* mCommonAction;

	//! helix shapes.
	QAction* myHelixAction;

	//! show the about info action.
	QAction* mAboutAction;

	//! the menus of the application.
	QMenu* mFileMenu;
	QMenu* mViewMenu;
	QMenu* mPrimitiveMenu;
	QMenu* mModelingMenu;
	QMenu* mHelpMenu;

	//! the toolbars of the application.
	QToolBar* mViewToolBar;
	QToolBar* mNavigateToolBar;
	QToolBar* mPrimitiveToolBar;
	QToolBar* mModelingToolBar;
	QToolBar* mHelpToolBar;

	// wrapped the widget for occ.
	OccView* myOccView;

};

#endif // MAINWINDOW_H 
