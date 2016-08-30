#include "mainWindow.h"
#include "ui_mainWindow.h"
#include "occView.h"

//QT5
#include <QToolBar>
#include <QTreeView>
#include <QMessageBox>
#include <QDockWidget>

//OCC
#include <gp_Circ.hxx>
#include <gp_Elips.hxx>
#include <gp_Pln.hxx>
#include <gp_Lin2d.hxx>
#include <Geom_ConicalSurface.hxx>
#include <Geom_ToroidalSurface.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <GCE2d_MakeSegment.hxx>
#include <TopoDS.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TColgp_Array1OfPnt2d.hxx>
#include <BRepLib.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCone.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeTorus.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepFilletAPI_MakeChamfer.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepAlgoAPI_Common.hxx>

#include <AIS_Shape.hxx>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowIcon(QIcon(":/Qt/Qt/resources/FitAll.png"));
    myOccView = new OccView(this);
    setCentralWidget(myOccView);
    createActions();
    createMenus();
    createToolBars();
}

MainWindow::~MainWindow()
{
    delete ui;
} 

void MainWindow::createActions( void )
{
    mExitAction = new QAction(tr("Exit"), this);
    mExitAction->setShortcut(tr("Ctrl+Q"));
    mExitAction->setIcon(QIcon(":/Qt/resources/close.png"));
    mExitAction->setStatusTip(tr("Exit the application"));
    connect(mExitAction, SIGNAL(triggered()), this, SLOT(close()));

    mViewZoomAction = new QAction(tr("Zoom"), this);
    mViewZoomAction->setIcon(QIcon(":/Qt/resources/Zoom.png"));
    mViewZoomAction->setStatusTip(tr("Zoom the view"));
    connect(mViewZoomAction, SIGNAL(triggered()), myOccView, SLOT(zoom()));

    mViewPanAction = new QAction(tr("Pan"), this);
    mViewPanAction->setIcon(QIcon(":/Qt/resources/Pan.png"));
    mViewPanAction->setStatusTip(tr("Pan the view"));
    connect(mViewPanAction, SIGNAL(triggered()), myOccView, SLOT(pan()));

    mViewRotateAction = new QAction(tr("Rotate"), this);
    mViewRotateAction->setIcon(QIcon(":/Qt/resources/Rotate.png"));
    mViewRotateAction->setStatusTip(tr("Rotate the view"));
    connect(mViewRotateAction, SIGNAL(triggered()), myOccView, SLOT(rotate()));

    mViewResetAction = new QAction(tr("Reset"), this);
    mViewResetAction->setIcon(QIcon(":/Qt/resources/Home.png"));
    mViewResetAction->setStatusTip(tr("Reset the view"));
    connect(mViewResetAction, SIGNAL(triggered()), myOccView, SLOT(reset()));

    mViewFitallAction = new QAction(tr("Fit All"), this);
    mViewFitallAction->setIcon(QIcon(":/Qt/resources/FitAll.png"));
    mViewFitallAction->setStatusTip(tr("Fit all "));
    connect(mViewFitallAction, SIGNAL(triggered()), myOccView, SLOT(fitAll()));

    mMakeBoxAction = new QAction(tr("Box"), this);
    mMakeBoxAction->setIcon(QIcon(":/Qt/resources/box.png"));
    mMakeBoxAction->setStatusTip(tr("Make a box"));
    connect(mMakeBoxAction, SIGNAL(triggered()), this, SLOT(makeBox()));

    mMakeConeAction = new QAction(tr("Cone"), this);
    mMakeConeAction->setIcon(QIcon(":/Qt/resources/cone.png"));
    mMakeConeAction->setStatusTip(tr("Make a cone"));
    connect(mMakeConeAction, SIGNAL(triggered()), this, SLOT(makeCone()));

    mMakeSphereAction = new QAction(tr("Sphere"), this);
    mMakeSphereAction->setStatusTip(tr("Make a sphere"));
    mMakeSphereAction->setIcon(QIcon(":/Qt/resources/sphere.png"));
    connect(mMakeSphereAction, SIGNAL(triggered()), this, SLOT(makeSphere()));

    mMakeCylinderAction = new QAction(tr("Cylinder"), this);
    mMakeCylinderAction->setStatusTip(tr("Make a cylinder"));
    mMakeCylinderAction->setIcon(QIcon(":/Qt/resources/cylinder.png"));
    connect(mMakeCylinderAction, SIGNAL(triggered()), this, SLOT(makeCylinder()));

    mMakeTorusAction = new QAction(tr("Torus"), this);
    mMakeTorusAction->setStatusTip(tr("Make a torus"));
    mMakeTorusAction->setIcon(QIcon(":/Qt/resources/torus.png"));
    connect(mMakeTorusAction, SIGNAL(triggered()), this, SLOT(makeTorus()));

    mFilletAction = new QAction(tr("Fillet"), this);
    mFilletAction->setIcon(QIcon(":/Qt/resources/fillet.png"));
    mFilletAction->setStatusTip(tr("Test Fillet algorithm"));
    connect(mFilletAction, SIGNAL(triggered()), this, SLOT(makeFillet()));

    mChamferAction = new QAction(tr("Chamfer"), this);
    mChamferAction->setIcon(QIcon(":/Qt/resources/chamfer.png"));
    mChamferAction->setStatusTip(tr("Test chamfer algorithm"));
    connect(mChamferAction, SIGNAL(triggered()), this, SLOT(makeChamfer()));

    mExtrudeAction = new QAction(tr("Extrude"), this);
    mExtrudeAction->setIcon(QIcon(":/Qt/resources/extrude.png"));
    mExtrudeAction->setStatusTip(tr("Test extrude algorithm"));
    connect(mExtrudeAction, SIGNAL(triggered()), this, SLOT(makeExtrude()));

    mRevolveAction = new QAction(tr("Revolve"), this);
    mRevolveAction->setIcon(QIcon(":/Qt/resources/revolve.png"));
    mRevolveAction->setStatusTip(tr("Test revol algorithm"));
    connect(mRevolveAction, SIGNAL(triggered()), this, SLOT(makeRevol()));

    mLoftAction = new QAction(tr("Loft"), this);
    mLoftAction->setIcon(QIcon(":/Qt/resources/loft.png"));
    mLoftAction->setStatusTip(tr("Test loft algorithm"));
    connect(mLoftAction, SIGNAL(triggered()), this, SLOT(makeLoft()));

    mCutAction = new QAction(tr("Cut"), this);
    mCutAction->setIcon(QIcon(":/Qt/resources/cut.png"));
    mCutAction->setStatusTip(tr("Boolean operation cut"));
    connect(mCutAction, SIGNAL(triggered()), this, SLOT(testCut()));

    mFuseAction = new QAction(tr("Fuse"), this);
    mFuseAction->setIcon(QIcon(":/Qt/resources/fuse.png"));
    mFuseAction->setStatusTip(tr("Boolean operation fuse"));
    connect(mFuseAction, SIGNAL(triggered()), this, SLOT(testFuse()));

    mCommonAction = new QAction(tr("Common"), this);
    mCommonAction->setIcon(QIcon(":/Qt/resources/common.png"));
    mCommonAction->setStatusTip(tr("Boolean operation common"));
    connect(mCommonAction, SIGNAL(triggered()), this, SLOT(testCommon()));

    myHelixAction = new QAction(tr("Helix"), this);
    myHelixAction->setIcon(QIcon(":/Qt/resources/helix.png"));
    myHelixAction->setStatusTip(tr("Make helix shapes"));
    connect(myHelixAction, SIGNAL(triggered()), this, SLOT(testHelix()));

    mAboutAction = new QAction(tr("About"), this);
    mAboutAction->setStatusTip(tr("About the application"));
    mAboutAction->setIcon(QIcon(":/Qt/resources/lamp.png"));
    connect(mAboutAction, SIGNAL(triggered()), this, SLOT(about()));
}

void MainWindow::createMenus( void )
{
    mFileMenu = menuBar()->addMenu(tr("&File"));
    mFileMenu->addAction(mExitAction);

    mViewMenu = menuBar()->addMenu(tr("&View"));
    mViewMenu->addAction(mViewZoomAction);
    mViewMenu->addAction(mViewPanAction);
    mViewMenu->addAction(mViewRotateAction);
    mViewMenu->addSeparator();
    mViewMenu->addAction(mViewResetAction);
    mViewMenu->addAction(mViewFitallAction);

    mPrimitiveMenu = menuBar()->addMenu(tr("&Primitive"));
    mPrimitiveMenu->addAction(mMakeBoxAction);
    mPrimitiveMenu->addAction(mMakeConeAction);
    mPrimitiveMenu->addAction(mMakeSphereAction);
    mPrimitiveMenu->addAction(mMakeCylinderAction);
    mPrimitiveMenu->addAction(mMakeTorusAction);

    mModelingMenu = menuBar()->addMenu(tr("&Modeling"));
    mModelingMenu->addAction(mFilletAction);
    mModelingMenu->addAction(mChamferAction);
    mModelingMenu->addAction(mExtrudeAction);
    mModelingMenu->addAction(mRevolveAction);
    mModelingMenu->addAction(mLoftAction);
    mModelingMenu->addSeparator();
    mModelingMenu->addAction(mCutAction);
    mModelingMenu->addAction(mFuseAction);
    mModelingMenu->addAction(mCommonAction);
    mModelingMenu->addSeparator();
    mModelingMenu->addAction(myHelixAction);

    mHelpMenu = menuBar()->addMenu(tr("&Help"));
    mHelpMenu->addAction(mAboutAction);
}

void MainWindow::createToolBars( void )
{
    mNavigateToolBar = addToolBar(tr("&Navigate"));
    mNavigateToolBar->addAction(mViewZoomAction);
    mNavigateToolBar->addAction(mViewPanAction);
    mNavigateToolBar->addAction(mViewRotateAction);

    mViewToolBar = addToolBar(tr("&View"));
    mViewToolBar->addAction(mViewResetAction);
    mViewToolBar->addAction(mViewFitallAction);

    mPrimitiveToolBar = addToolBar(tr("&Primitive"));
    mPrimitiveToolBar->addAction(mMakeBoxAction);
    mPrimitiveToolBar->addAction(mMakeConeAction);
    mPrimitiveToolBar->addAction(mMakeSphereAction);
    mPrimitiveToolBar->addAction(mMakeCylinderAction);
    mPrimitiveToolBar->addAction(mMakeTorusAction);

    mModelingToolBar = addToolBar(tr("&Modeling"));
    mModelingToolBar->addAction(mFilletAction);
    mModelingToolBar->addAction(mChamferAction);
    mModelingToolBar->addAction(mExtrudeAction);
    mModelingToolBar->addAction(mRevolveAction);
    mModelingToolBar->addAction(mLoftAction);
    mModelingToolBar->addSeparator();
    mModelingToolBar->addAction(mCutAction);
    mModelingToolBar->addAction(mFuseAction);
    mModelingToolBar->addAction(mCommonAction);
    mModelingToolBar->addSeparator();
    mModelingToolBar->addAction(myHelixAction);

    mHelpToolBar = addToolBar(tr("Help"));
    mHelpToolBar->addAction(mAboutAction);
}

void MainWindow::about()
{
    QMessageBox::about(this, tr("About STACCATO"),
        tr("<h2>STACCATO: STefAn's Computational vibroaCoustics Analysis TOol</h2>"
        "<p>Copyright &copy; 2016 "
        "<p>STACCATO is using Qt and OpenCASCADE."));
}

void MainWindow::makeBox()
{
    TopoDS_Shape aTopoBox = BRepPrimAPI_MakeBox(3.0, 4.0, 5.0).Shape();
    Handle(AIS_Shape) anAisBox = new AIS_Shape(aTopoBox);

    anAisBox->SetColor(Quantity_NOC_AZURE);

    myOccView->getContext()->Display(anAisBox);
}

void MainWindow::makeCone()
{
    gp_Ax2 anAxis;
    anAxis.SetLocation(gp_Pnt(0.0, 10.0, 0.0));

    TopoDS_Shape aTopoReducer = BRepPrimAPI_MakeCone(anAxis, 3.0, 1.5, 5.0).Shape();
    Handle(AIS_Shape) anAisReducer = new AIS_Shape(aTopoReducer);

    anAisReducer->SetColor(Quantity_NOC_BISQUE);

    anAxis.SetLocation(gp_Pnt(8.0, 10.0, 0.0));
    TopoDS_Shape aTopoCone = BRepPrimAPI_MakeCone(anAxis, 3.0, 0.0, 5.0).Shape();
    Handle(AIS_Shape) anAisCone = new AIS_Shape(aTopoCone);

    anAisCone->SetColor(Quantity_NOC_CHOCOLATE);

    myOccView->getContext()->Display(anAisReducer);
    myOccView->getContext()->Display(anAisCone);
}

void MainWindow::makeSphere()
{
    gp_Ax2 anAxis;
    anAxis.SetLocation(gp_Pnt(0.0, 20.0, 0.0));

    TopoDS_Shape aTopoSphere = BRepPrimAPI_MakeSphere(anAxis, 3.0).Shape();
    Handle(AIS_Shape) anAisSphere = new AIS_Shape(aTopoSphere);

    anAisSphere->SetColor(Quantity_NOC_BLUE1);

    myOccView->getContext()->Display(anAisSphere);
}

void MainWindow::makeCylinder()
{
    gp_Ax2 anAxis;
    anAxis.SetLocation(gp_Pnt(0.0, 30.0, 0.0));

    TopoDS_Shape aTopoCylinder = BRepPrimAPI_MakeCylinder(anAxis, 3.0, 5.0).Shape();
    Handle(AIS_Shape) anAisCylinder = new AIS_Shape(aTopoCylinder);

    anAisCylinder->SetColor(Quantity_NOC_RED);

    anAxis.SetLocation(gp_Pnt(8.0, 30.0, 0.0));
    TopoDS_Shape aTopoPie = BRepPrimAPI_MakeCylinder(anAxis, 3.0, 5.0, M_PI_2 * 3.0).Shape();
    Handle(AIS_Shape) anAisPie = new AIS_Shape(aTopoPie);

    anAisPie->SetColor(Quantity_NOC_TAN);

    myOccView->getContext()->Display(anAisCylinder);
    myOccView->getContext()->Display(anAisPie);
}

void MainWindow::makeTorus()
{
    gp_Ax2 anAxis;
    anAxis.SetLocation(gp_Pnt(0.0, 40.0, 0.0));

    TopoDS_Shape aTopoTorus = BRepPrimAPI_MakeTorus(anAxis, 3.0, 1.0).Shape();
    Handle(AIS_Shape) anAisTorus = new AIS_Shape(aTopoTorus);

    anAisTorus->SetColor(Quantity_NOC_YELLOW);

    anAxis.SetLocation(gp_Pnt(8.0, 40.0, 0.0));
    TopoDS_Shape aTopoElbow = BRepPrimAPI_MakeTorus(anAxis, 3.0, 1.0, M_PI_2).Shape();
    Handle(AIS_Shape) anAisElbow = new AIS_Shape(aTopoElbow);

    anAisElbow->SetColor(Quantity_NOC_THISTLE);

    myOccView->getContext()->Display(anAisTorus);
    myOccView->getContext()->Display(anAisElbow);
}

void MainWindow::makeFillet()
{
    gp_Ax2 anAxis;
    anAxis.SetLocation(gp_Pnt(0.0, 50.0, 0.0));

    TopoDS_Shape aTopoBox = BRepPrimAPI_MakeBox(anAxis, 3.0, 4.0, 5.0).Shape();
    BRepFilletAPI_MakeFillet MF(aTopoBox);

    // Add all the edges to fillet.
    for (TopExp_Explorer ex(aTopoBox, TopAbs_EDGE); ex.More(); ex.Next())
    {
        MF.Add(1.0, TopoDS::Edge(ex.Current()));
    }

    Handle(AIS_Shape) anAisShape = new AIS_Shape(MF.Shape());
    anAisShape->SetColor(Quantity_NOC_VIOLET);

    myOccView->getContext()->Display(anAisShape);
}

void MainWindow::makeChamfer()
{
    gp_Ax2 anAxis;
    anAxis.SetLocation(gp_Pnt(8.0, 50.0, 0.0));

    TopoDS_Shape aTopoBox = BRepPrimAPI_MakeBox(anAxis, 3.0, 4.0, 5.0).Shape();
    BRepFilletAPI_MakeChamfer MC(aTopoBox);
    TopTools_IndexedDataMapOfShapeListOfShape aEdgeFaceMap;

    TopExp::MapShapesAndAncestors(aTopoBox, TopAbs_EDGE, TopAbs_FACE, aEdgeFaceMap);

    for (Standard_Integer i = 1; i <= aEdgeFaceMap.Extent(); ++i)
    {
        TopoDS_Edge anEdge = TopoDS::Edge(aEdgeFaceMap.FindKey(i));
        TopoDS_Face aFace = TopoDS::Face(aEdgeFaceMap.FindFromIndex(i).First());

        MC.Add(0.6, 0.6, anEdge, aFace);
    }

    Handle(AIS_Shape) anAisShape = new AIS_Shape(MC.Shape());
    anAisShape->SetColor(Quantity_NOC_TOMATO);

    myOccView->getContext()->Display(anAisShape);
}

void MainWindow::makeExtrude()
{
    // prism a vertex result is an edge.
    TopoDS_Vertex aVertex = BRepBuilderAPI_MakeVertex(gp_Pnt(0.0, 60.0, 0.0));
    TopoDS_Shape aPrismVertex = BRepPrimAPI_MakePrism(aVertex, gp_Vec(0.0, 0.0, 5.0));
    Handle(AIS_Shape) anAisPrismVertex = new AIS_Shape(aPrismVertex);

    // prism an edge result is a face.
    TopoDS_Edge anEdge = BRepBuilderAPI_MakeEdge(gp_Pnt(5.0, 60.0, 0.0), gp_Pnt(10.0, 60.0, 0.0));
    TopoDS_Shape aPrismEdge = BRepPrimAPI_MakePrism(anEdge, gp_Vec(0.0, 0.0, 5.0));
    Handle(AIS_Shape) anAisPrismEdge = new AIS_Shape(aPrismEdge);

    // prism a wire result is a shell.
    gp_Ax2 anAxis;
    anAxis.SetLocation(gp_Pnt(16.0, 60.0, 0.0));

    TopoDS_Edge aCircleEdge = BRepBuilderAPI_MakeEdge(gp_Circ(anAxis, 3.0));
    TopoDS_Wire aCircleWire = BRepBuilderAPI_MakeWire(aCircleEdge);
    TopoDS_Shape aPrismCircle = BRepPrimAPI_MakePrism(aCircleWire, gp_Vec(0.0, 0.0, 5.0));
    Handle(AIS_Shape) anAisPrismCircle = new AIS_Shape(aPrismCircle);

    // prism a face or a shell result is a solid.
    anAxis.SetLocation(gp_Pnt(24.0, 60.0, 0.0));
    TopoDS_Edge aEllipseEdge = BRepBuilderAPI_MakeEdge(gp_Elips(anAxis, 3.0, 2.0));
    TopoDS_Wire aEllipseWire = BRepBuilderAPI_MakeWire(aEllipseEdge);
    TopoDS_Face aEllipseFace = BRepBuilderAPI_MakeFace(gp_Pln(gp::XOY()), aEllipseWire);
    TopoDS_Shape aPrismEllipse = BRepPrimAPI_MakePrism(aEllipseFace, gp_Vec(0.0, 0.0, 5.0));
    Handle(AIS_Shape) anAisPrismEllipse = new AIS_Shape(aPrismEllipse);

    anAisPrismVertex->SetColor(Quantity_NOC_PAPAYAWHIP);
    anAisPrismEdge->SetColor(Quantity_NOC_PEACHPUFF);
    anAisPrismCircle->SetColor(Quantity_NOC_PERU);
    anAisPrismEllipse->SetColor(Quantity_NOC_PINK);

    myOccView->getContext()->Display(anAisPrismVertex);
    myOccView->getContext()->Display(anAisPrismEdge);
    myOccView->getContext()->Display(anAisPrismCircle);
    myOccView->getContext()->Display(anAisPrismEllipse);
}

void MainWindow::makeRevol()
{
    gp_Ax1 anAxis;

    // revol a vertex result is an edge.
    anAxis.SetLocation(gp_Pnt(0.0, 70.0, 0.0));
    TopoDS_Vertex aVertex = BRepBuilderAPI_MakeVertex(gp_Pnt(2.0, 70.0, 0.0));
    TopoDS_Shape aRevolVertex = BRepPrimAPI_MakeRevol(aVertex, anAxis);
    Handle(AIS_Shape) anAisRevolVertex = new AIS_Shape(aRevolVertex);

    // revol an edge result is a face.
    anAxis.SetLocation(gp_Pnt(8.0, 70.0, 0.0));
    TopoDS_Edge anEdge = BRepBuilderAPI_MakeEdge(gp_Pnt(6.0, 70.0, 0.0), gp_Pnt(6.0, 70.0, 5.0));
    TopoDS_Shape aRevolEdge = BRepPrimAPI_MakeRevol(anEdge, anAxis);
    Handle(AIS_Shape) anAisRevolEdge = new AIS_Shape(aRevolEdge);

    // revol a wire result is a shell.
    anAxis.SetLocation(gp_Pnt(20.0, 70.0, 0.0));
    anAxis.SetDirection(gp::DY());

    TopoDS_Edge aCircleEdge = BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(15.0, 70.0, 0.0), gp::DZ()), 1.5));
    TopoDS_Wire aCircleWire = BRepBuilderAPI_MakeWire(aCircleEdge);
    TopoDS_Shape aRevolCircle = BRepPrimAPI_MakeRevol(aCircleWire, anAxis, M_PI_2);
    Handle(AIS_Shape) anAisRevolCircle = new AIS_Shape(aRevolCircle);

    // revol a face result is a solid.
    anAxis.SetLocation(gp_Pnt(30.0, 70.0, 0.0));
    anAxis.SetDirection(gp::DY());

    TopoDS_Edge aEllipseEdge = BRepBuilderAPI_MakeEdge(gp_Elips(gp_Ax2(gp_Pnt(25.0, 70.0, 0.0), gp::DZ()), 3.0, 2.0));
    TopoDS_Wire aEllipseWire = BRepBuilderAPI_MakeWire(aEllipseEdge);
    TopoDS_Face aEllipseFace = BRepBuilderAPI_MakeFace(gp_Pln(gp::XOY()), aEllipseWire);
    TopoDS_Shape aRevolEllipse = BRepPrimAPI_MakeRevol(aEllipseFace, anAxis, M_PI_4);
    Handle(AIS_Shape) anAisRevolEllipse = new AIS_Shape(aRevolEllipse);

    anAisRevolVertex->SetColor(Quantity_NOC_LIMEGREEN);
    anAisRevolEdge->SetColor(Quantity_NOC_LINEN);
    anAisRevolCircle->SetColor(Quantity_NOC_MAGENTA1);
    anAisRevolEllipse->SetColor(Quantity_NOC_MAROON);

    myOccView->getContext()->Display(anAisRevolVertex);
    myOccView->getContext()->Display(anAisRevolEdge);
    myOccView->getContext()->Display(anAisRevolCircle);
    myOccView->getContext()->Display(anAisRevolEllipse);
}

void MainWindow::makeLoft()
{
    // bottom wire.
    TopoDS_Edge aCircleEdge = BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(0.0, 80.0, 0.0), gp::DZ()), 1.5));
    TopoDS_Wire aCircleWire = BRepBuilderAPI_MakeWire(aCircleEdge);

    // top wire.
    BRepBuilderAPI_MakePolygon aPolygon;
    aPolygon.Add(gp_Pnt(-3.0, 77.0, 6.0));
    aPolygon.Add(gp_Pnt(3.0, 77.0, 6.0));
    aPolygon.Add(gp_Pnt(3.0, 83.0, 6.0));
    aPolygon.Add(gp_Pnt(-3.0, 83.0, 6.0));
    aPolygon.Close();

    BRepOffsetAPI_ThruSections aShellGenerator;
    BRepOffsetAPI_ThruSections aSolidGenerator(true);

    aShellGenerator.AddWire(aCircleWire);
    aShellGenerator.AddWire(aPolygon.Wire());

    aSolidGenerator.AddWire(aCircleWire);
    aSolidGenerator.AddWire(aPolygon.Wire());

    // translate the solid.
    gp_Trsf aTrsf;
    aTrsf.SetTranslation(gp_Vec(18.0, 0.0, 0.0));
    BRepBuilderAPI_Transform aTransform(aSolidGenerator.Shape(), aTrsf);

    Handle(AIS_Shape) anAisShell = new AIS_Shape(aShellGenerator.Shape());
    Handle(AIS_Shape) anAisSolid = new AIS_Shape(aTransform.Shape());

    anAisShell->SetColor(Quantity_NOC_OLIVEDRAB);
    anAisSolid->SetColor(Quantity_NOC_PEACHPUFF);

    myOccView->getContext()->Display(anAisShell);
    myOccView->getContext()->Display(anAisSolid);
}

void MainWindow::testCut()
{
    gp_Ax2 anAxis;
    anAxis.SetLocation(gp_Pnt(0.0, 90.0, 0.0));

    TopoDS_Shape aTopoBox = BRepPrimAPI_MakeBox(anAxis, 3.0, 4.0, 5.0).Shape();
    TopoDS_Shape aTopoSphere = BRepPrimAPI_MakeSphere(anAxis, 2.5).Shape();
    TopoDS_Shape aCuttedShape1 = BRepAlgoAPI_Cut(aTopoBox, aTopoSphere);
    TopoDS_Shape aCuttedShape2 = BRepAlgoAPI_Cut(aTopoSphere, aTopoBox);

    gp_Trsf aTrsf;
    aTrsf.SetTranslation(gp_Vec(8.0, 0.0, 0.0));
    BRepBuilderAPI_Transform aTransform1(aCuttedShape1, aTrsf);

    aTrsf.SetTranslation(gp_Vec(16.0, 0.0, 0.0));
    BRepBuilderAPI_Transform aTransform2(aCuttedShape2, aTrsf);

    Handle(AIS_Shape) anAisBox = new AIS_Shape(aTopoBox);
    Handle(AIS_Shape) anAisSphere = new AIS_Shape(aTopoSphere);
    Handle(AIS_Shape) anAisCuttedShape1 = new AIS_Shape(aTransform1.Shape());
    Handle(AIS_Shape) anAisCuttedShape2 = new AIS_Shape(aTransform2.Shape());

    anAisBox->SetColor(Quantity_NOC_SPRINGGREEN);
    anAisSphere->SetColor(Quantity_NOC_STEELBLUE);
    anAisCuttedShape1->SetColor(Quantity_NOC_TAN);
    anAisCuttedShape2->SetColor(Quantity_NOC_SALMON);

    myOccView->getContext()->Display(anAisBox);
    myOccView->getContext()->Display(anAisSphere);
    myOccView->getContext()->Display(anAisCuttedShape1);
    myOccView->getContext()->Display(anAisCuttedShape2);
}

void MainWindow::testFuse()
{
    gp_Ax2 anAxis;
    anAxis.SetLocation(gp_Pnt(0.0, 100.0, 0.0));

    TopoDS_Shape aTopoBox = BRepPrimAPI_MakeBox(anAxis, 3.0, 4.0, 5.0).Shape();
    TopoDS_Shape aTopoSphere = BRepPrimAPI_MakeSphere(anAxis, 2.5).Shape();
    TopoDS_Shape aFusedShape = BRepAlgoAPI_Fuse(aTopoBox, aTopoSphere);

    gp_Trsf aTrsf;
    aTrsf.SetTranslation(gp_Vec(8.0, 0.0, 0.0));
    BRepBuilderAPI_Transform aTransform(aFusedShape, aTrsf);

    Handle(AIS_Shape) anAisBox = new AIS_Shape(aTopoBox);
    Handle(AIS_Shape) anAisSphere = new AIS_Shape(aTopoSphere);
    Handle(AIS_Shape) anAisFusedShape = new AIS_Shape(aTransform.Shape());

    anAisBox->SetColor(Quantity_NOC_SPRINGGREEN);
    anAisSphere->SetColor(Quantity_NOC_STEELBLUE);
    anAisFusedShape->SetColor(Quantity_NOC_ROSYBROWN);

    myOccView->getContext()->Display(anAisBox);
    myOccView->getContext()->Display(anAisSphere);
    myOccView->getContext()->Display(anAisFusedShape);
}

void MainWindow::testCommon()
{
    gp_Ax2 anAxis;
    anAxis.SetLocation(gp_Pnt(0.0, 110.0, 0.0));

    TopoDS_Shape aTopoBox = BRepPrimAPI_MakeBox(anAxis, 3.0, 4.0, 5.0).Shape();
    TopoDS_Shape aTopoSphere = BRepPrimAPI_MakeSphere(anAxis, 2.5).Shape();
    TopoDS_Shape aCommonShape = BRepAlgoAPI_Common(aTopoBox, aTopoSphere);

    gp_Trsf aTrsf;
    aTrsf.SetTranslation(gp_Vec(8.0, 0.0, 0.0));
    BRepBuilderAPI_Transform aTransform(aCommonShape, aTrsf);

    Handle(AIS_Shape) anAisBox = new AIS_Shape(aTopoBox);
    Handle(AIS_Shape) anAisSphere = new AIS_Shape(aTopoSphere);
    Handle(AIS_Shape) anAisCommonShape = new AIS_Shape(aTransform.Shape());

    anAisBox->SetColor(Quantity_NOC_SPRINGGREEN);
    anAisSphere->SetColor(Quantity_NOC_STEELBLUE);
    anAisCommonShape->SetColor(Quantity_NOC_ROYALBLUE);

    myOccView->getContext()->Display(anAisBox);
    myOccView->getContext()->Display(anAisSphere);
    myOccView->getContext()->Display(anAisCommonShape);
}

void MainWindow::testHelix()
{
    makeCylindricalHelix();

    makeConicalHelix();

    makeToroidalHelix();
}

void MainWindow::makeCylindricalHelix()
{
    Standard_Real aRadius = 3.0;
    Standard_Real aPitch = 1.0;

    // the pcurve is a 2d line in the parametric space.
    gp_Lin2d aLine2d(gp_Pnt2d(0.0, 0.0), gp_Dir2d(aRadius, aPitch));

    Handle(Geom2d_TrimmedCurve) aSegment = GCE2d_MakeSegment(aLine2d, 0.0, M_PI * 2.0).Value();

    Handle(Geom_CylindricalSurface) aCylinder = new Geom_CylindricalSurface(gp::XOY(), aRadius);

    TopoDS_Edge aHelixEdge = BRepBuilderAPI_MakeEdge(aSegment, aCylinder, 0.0, 6.0 * M_PI).Edge();

    gp_Trsf aTrsf;
    aTrsf.SetTranslation(gp_Vec(0.0, 120.0, 0.0));
    BRepBuilderAPI_Transform aTransform(aHelixEdge, aTrsf);

    Handle(AIS_Shape) anAisHelixCurve = new AIS_Shape(aTransform.Shape());

    myOccView->getContext()->Display(anAisHelixCurve);

    // sweep a circle profile along the helix curve.
    // there is no curve3d in the pcurve edge, so approx one.
    BRepLib::BuildCurve3d(aHelixEdge);

    gp_Ax2 anAxis;
    anAxis.SetDirection(gp_Dir(0.0, 4.0, 1.0));
    anAxis.SetLocation(gp_Pnt(aRadius, 0.0, 0.0));

    gp_Circ aProfileCircle(anAxis, 0.3);

    TopoDS_Edge aProfileEdge = BRepBuilderAPI_MakeEdge(aProfileCircle).Edge();
    TopoDS_Wire aProfileWire = BRepBuilderAPI_MakeWire(aProfileEdge).Wire();
    TopoDS_Face aProfileFace = BRepBuilderAPI_MakeFace(aProfileWire).Face();

    TopoDS_Wire aHelixWire = BRepBuilderAPI_MakeWire(aHelixEdge).Wire();

    BRepOffsetAPI_MakePipe aPipeMaker(aHelixWire, aProfileFace);

    if (aPipeMaker.IsDone())
    {
        aTrsf.SetTranslation(gp_Vec(8.0, 120.0, 0.0));
        BRepBuilderAPI_Transform aPipeTransform(aPipeMaker.Shape(), aTrsf);

        Handle(AIS_Shape) anAisPipe = new AIS_Shape(aPipeTransform.Shape());
        anAisPipe->SetColor(Quantity_NOC_CORAL);
        myOccView->getContext()->Display(anAisPipe);
    }
}

/**
 * make conical helix, it is the same as the cylindrical helix,
 * the only different is the surface.
 */
void MainWindow::makeConicalHelix()
{
    Standard_Real aRadius = 3.0;
    Standard_Real aPitch = 1.0;

    // the pcurve is a 2d line in the parametric space.
    gp_Lin2d aLine2d(gp_Pnt2d(0.0, 0.0), gp_Dir2d(aRadius, aPitch));

    Handle(Geom2d_TrimmedCurve) aSegment = GCE2d_MakeSegment(aLine2d, 0.0, M_PI * 2.0).Value();

    Handle(Geom_ConicalSurface) aCylinder = new Geom_ConicalSurface(gp::XOY(), M_PI / 6.0, aRadius);

    TopoDS_Edge aHelixEdge = BRepBuilderAPI_MakeEdge(aSegment, aCylinder, 0.0, 6.0 * M_PI).Edge();

    gp_Trsf aTrsf;
    aTrsf.SetTranslation(gp_Vec(18.0, 120.0, 0.0));
    BRepBuilderAPI_Transform aTransform(aHelixEdge, aTrsf);

    Handle(AIS_Shape) anAisHelixCurve = new AIS_Shape(aTransform.Shape());

    myOccView->getContext()->Display(anAisHelixCurve);

    // sweep a circle profile along the helix curve.
    // there is no curve3d in the pcurve edge, so approx one.
    BRepLib::BuildCurve3d(aHelixEdge);

    gp_Ax2 anAxis;
    anAxis.SetDirection(gp_Dir(0.0, 4.0, 1.0));
    anAxis.SetLocation(gp_Pnt(aRadius, 0.0, 0.0));

    gp_Circ aProfileCircle(anAxis, 0.3);

    TopoDS_Edge aProfileEdge = BRepBuilderAPI_MakeEdge(aProfileCircle).Edge();
    TopoDS_Wire aProfileWire = BRepBuilderAPI_MakeWire(aProfileEdge).Wire();
    TopoDS_Face aProfileFace = BRepBuilderAPI_MakeFace(aProfileWire).Face();

    TopoDS_Wire aHelixWire = BRepBuilderAPI_MakeWire(aHelixEdge).Wire();

    BRepOffsetAPI_MakePipe aPipeMaker(aHelixWire, aProfileFace);

    if (aPipeMaker.IsDone())
    {
        aTrsf.SetTranslation(gp_Vec(28.0, 120.0, 0.0));
        BRepBuilderAPI_Transform aPipeTransform(aPipeMaker.Shape(), aTrsf);

        Handle(AIS_Shape) anAisPipe = new AIS_Shape(aPipeTransform.Shape());
        anAisPipe->SetColor(Quantity_NOC_DARKGOLDENROD);
        myOccView->getContext()->Display(anAisPipe);
    }
}

void MainWindow::makeToroidalHelix()
{
    Standard_Real aRadius = 1.0;
    Standard_Real aSlope = 0.05;

    // the pcurve is a 2d line in the parametric space.
    gp_Lin2d aLine2d(gp_Pnt2d(0.0, 0.0), gp_Dir2d(aSlope, 1.0));

    Handle(Geom2d_TrimmedCurve) aSegment = GCE2d_MakeSegment(aLine2d, 0.0, M_PI * 2.0).Value();

    Handle(Geom_ToroidalSurface) aCylinder = new Geom_ToroidalSurface(gp::XOY(), aRadius * 5.0, aRadius);

    TopoDS_Edge aHelixEdge = BRepBuilderAPI_MakeEdge(aSegment, aCylinder, 0.0, 2.0 * M_PI / aSlope).Edge();

    gp_Trsf aTrsf;
    aTrsf.SetTranslation(gp_Vec(45.0, 120.0, 0.0));
    BRepBuilderAPI_Transform aTransform(aHelixEdge, aTrsf);

    Handle(AIS_Shape) anAisHelixCurve = new AIS_Shape(aTransform.Shape());

    myOccView->getContext()->Display(anAisHelixCurve);

    // sweep a circle profile along the helix curve.
    // there is no curve3d in the pcurve edge, so approx one.
    BRepLib::BuildCurve3d(aHelixEdge);

    gp_Ax2 anAxis;
    anAxis.SetDirection(gp_Dir(0.0, 0.0, 1.0));
    anAxis.SetLocation(gp_Pnt(aRadius * 6.0, 0.0, 0.0));

    gp_Circ aProfileCircle(anAxis, 0.3);

    TopoDS_Edge aProfileEdge = BRepBuilderAPI_MakeEdge(aProfileCircle).Edge();
    TopoDS_Wire aProfileWire = BRepBuilderAPI_MakeWire(aProfileEdge).Wire();
    TopoDS_Face aProfileFace = BRepBuilderAPI_MakeFace(aProfileWire).Face();

    TopoDS_Wire aHelixWire = BRepBuilderAPI_MakeWire(aHelixEdge).Wire();

    BRepOffsetAPI_MakePipe aPipeMaker(aHelixWire, aProfileFace);

    if (aPipeMaker.IsDone())
    {
        aTrsf.SetTranslation(gp_Vec(60.0, 120.0, 0.0));
        BRepBuilderAPI_Transform aPipeTransform(aPipeMaker.Shape(), aTrsf);

        Handle(AIS_Shape) anAisPipe = new AIS_Shape(aPipeTransform.Shape());
        anAisPipe->SetColor(Quantity_NOC_CORNSILK1);
        myOccView->getContext()->Display(anAisPipe);
    }
}
