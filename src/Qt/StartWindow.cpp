#include <StartWindow.h>
#include <ui_StartWindow.h>
#include <OccViewer.h>

//QT5
#include <QToolBar>
#include <QTreeView>
#include <QMessageBox>
#include <QDockWidget>
#include <QtWidgets>

//OCC 7
#include <Graphic3d_GraphicDriver.hxx>
#include <OpenGl_GraphicDriver.hxx>

#include <StlMesh_Mesh.hxx>
#include <MeshVS_Mesh.hxx>
#include <XSDRAWSTLVRML_DataSource.hxx>
#include <MeshVS_MeshPrsBuilder.hxx>
#include <MeshVS_Drawer.hxx>
#include <RWStl.hxx>
#include <MeshVS_DrawerAttribute.hxx>
#include <Graphic3d_MaterialAspect.hxx>
#include <OSD_Path.hxx>

StartWindow::StartWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::StartWindow)
{
	ui->setupUi(this);
	setWindowIcon(QIcon(":/Qt/resources/FitAll.png"));
	myOccViewer = new OccViewer(this);
	setCentralWidget(myOccViewer);
	createActions();
	createMenus();
	createToolBars();
	setAttribute(Qt::WA_QuitOnClose,Standard_False);
}

StartWindow::~StartWindow()
{
}

void StartWindow::createActions(void)
{
	mExitAction = new QAction(tr("Exit"), this);
	mExitAction->setShortcut(tr("Ctrl+Q"));
	mExitAction->setIcon(QIcon(":/Qt/resources/close.png"));
	mExitAction->setStatusTip(tr("Exit the application"));
	connect(mExitAction, SIGNAL(triggered()), this, SLOT(close()));

	mReadSTLAction = new QAction(tr("ReadSTL"), this);
	mReadSTLAction->setShortcut(tr("Ctrl+R"));
	mReadSTLAction->setIcon(QIcon(":/Qt/resources/close.png"));
	mReadSTLAction->setStatusTip(tr("Read STL file"));
	connect(mReadSTLAction, SIGNAL(triggered()), this, SLOT(readSTL()));

	mAboutAction = new QAction(tr("About"), this);
	mAboutAction->setStatusTip(tr("About the application"));
	mAboutAction->setIcon(QIcon(":/Qt/resources/lamp.png"));
	connect(mAboutAction, SIGNAL(triggered()), this, SLOT(about()));
}

void StartWindow::createMenus(void)
{
	mFileMenu = menuBar()->addMenu(tr("&File"));
	mFileMenu->addAction(mExitAction);
	mFileMenu->addAction(mReadSTLAction);
	mHelpMenu = menuBar()->addMenu(tr("&Help"));
	mHelpMenu->addAction(mAboutAction);
}

void StartWindow::createToolBars(void)
{
	mFileToolBar = addToolBar(tr("&File"));
	mFileToolBar->addAction(mReadSTLAction);
	mHelpToolBar = addToolBar(tr("Help"));
	mHelpToolBar->addAction(mAboutAction);
}


void StartWindow::about()
{
	cout << sizeof(QApplication::topLevelWidgets()) << endl;

	myOccViewer->showGrid(Standard_True);
	QMessageBox::about(this, tr("About STACCATO"),
		tr("<h2>STACCATO: STefAn's Computational vibroaCoustics Analysis TOol</h2>"
		"<p>Copyright &copy; 2016 "
		"<p>STACCATO is using Qt and OpenCASCADE."));
}

void StartWindow::readSTL(void)
{
	
	QString fileNameSTL = QFileDialog::getOpenFileName(this,
		tr("Import STL File"), "", tr("STL Files (*.stl)"));

	if (!fileNameSTL.isEmpty() && !fileNameSTL.isNull()){
		OSD_Path aFile(fileNameSTL.toUtf8().constData());

		Handle(StlMesh_Mesh) aSTLMesh = RWStl::ReadFile(aFile);


		Handle(MeshVS_Mesh) aMesh = new MeshVS_Mesh();
		Handle(XSDRAWSTLVRML_DataSource) aDS = new XSDRAWSTLVRML_DataSource(aSTLMesh);

		aMesh->SetDataSource(aDS);
		aMesh->AddBuilder(new MeshVS_MeshPrsBuilder(aMesh), Standard_True);//False -> No selection

		aMesh->GetDrawer()->SetBoolean(MeshVS_DA_DisplayNodes, Standard_False); //MeshVS_DrawerAttribute
		aMesh->GetDrawer()->SetBoolean(MeshVS_DA_ShowEdges, Standard_False);
		aMesh->GetDrawer()->SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_NOM_BRASS);

		aMesh->SetColor(Quantity_NOC_AZURE);
		aMesh->SetDisplayMode(MeshVS_DMF_Shading); // Mode as defaut
		aMesh->SetHilightMode(MeshVS_DMF_WireFrame); // Wireframe as default hilight mode

		myOccViewer->getContext()->Display(aMesh);
	}

}

