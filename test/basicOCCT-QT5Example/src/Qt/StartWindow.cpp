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
#include <StartWindow.h>
#include <ui_StartWindow.h>
#include <OccViewer.h>
#include <STLVRML_DataSource.h>

//QT5
#include <QtWidgets>

//OCC 7
#include <StlMesh_Mesh.hxx> 
#include <MeshVS_Mesh.hxx>
#include <MeshVS_MeshPrsBuilder.hxx>
#include <MeshVS_Drawer.hxx>
#include <RWStl.hxx>
#include <MeshVS_DrawerAttribute.hxx>
#include <Graphic3d_MaterialAspect.hxx>
#include <OSD_Path.hxx>
#include <Geom_CartesianPoint.hxx>
#include <AIS_Line.hxx>
#include <AIS_Point.hxx>
#include <TopoDS_Vertex.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <Prs3d_PointAspect.hxx>
#include <GC_MakeSegment.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <TopoDS_Edge.hxx>
#include <AIS_InteractiveContext.hxx>
#include <BRep_Tool.hxx>
#include <TopoDS.hxx>
#include <Geom2d_CartesianPoint.hxx>
#include <ElCLib.hxx>
#include <MeshVS_SelectionModeFlags.hxx>
#include <TColStd_HPackedMapOfInteger.hxx>
#include <Select3D_SensitiveTriangle.hxx>
#include <MeshVS_MeshEntityOwner.hxx>
#include <Select3D_SensitiveTriangulation.hxx>
#include <Select3D_SensitiveFace.hxx>
#include <MeshVS_CommonSensitiveEntity.hxx>
#include <MeshVS_Buffer.hxx>


StartWindow::StartWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::StartWindow)
{
	ui->setupUi(this);
	myOccViewer = new OccViewer(this);
	setCentralWidget(myOccViewer);
	cout << "QApplication::topLevelWidgets().size: " << QApplication::topLevelWidgets().size() << endl;
}

StartWindow::~StartWindow()
{
	//delete myOccViewer;
}



void StartWindow::readSTL(void)
{

	QString fileNameSTL = QFileDialog::getOpenFileName(this,
		tr("Import STL File"), "", tr("STL Files (*.stl)"));

	if (!fileNameSTL.isEmpty() && !fileNameSTL.isNull()){
		OSD_Path aFile(fileNameSTL.toUtf8().constData());
		Handle(StlMesh_Mesh) aSTLMesh = RWStl::ReadFile(aFile);
		Handle(MeshVS_Mesh) aMesh = new MeshVS_Mesh();
		Handle(STLVRML_DataSource) aDS = new STLVRML_DataSource(aSTLMesh);
		aMesh->SetDataSource(aDS);
		aMesh->AddBuilder(new MeshVS_MeshPrsBuilder(aMesh), Standard_True);//False -> No selection
		aMesh->GetDrawer()->SetBoolean(MeshVS_DA_DisplayNodes, Standard_False); //MeshVS_DrawerAttribute
		aMesh->GetDrawer()->SetBoolean(MeshVS_DA_ShowEdges, Standard_False);
		aMesh->GetDrawer()->SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_NOM_BRASS);
		aMesh->SetColor(Quantity_NOC_AZURE);
		aMesh->SetDisplayMode(MeshVS_DMF_Shading); // Mode as defaut
		aMesh->SetHilightMode(MeshVS_DMF_WireFrame); // Wireframe as default hilight mode
		aMesh->GetDrawer()->SetColor(MeshVS_DA_EdgeColor, Quantity_NOC_YELLOW);

		// Hide all nodes by default
		Handle(TColStd_HPackedMapOfInteger) aNodes = new TColStd_HPackedMapOfInteger();
		Standard_Integer aLen = aSTLMesh->Vertices().Length();
		for (Standard_Integer anIndex = 1; anIndex <= aLen; anIndex++){
			aNodes->ChangeMap().Add(anIndex);
		}
		aMesh->SetHiddenNodes(aNodes);
		aMesh->SetSelectableNodes(aNodes);
		myOccViewer->getContext()->Display(aMesh);
		myOccViewer->getContext()->Deactivate(aMesh);
		myOccViewer->getContext()->Load(aMesh, -1, Standard_True);
		//myOccViewer->getContext()->Activate(aMesh, 1); // Node selection
		myOccViewer->getContext()->Activate(aMesh, 8); // Element selection
	}
}