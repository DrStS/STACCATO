// Created on: 2014-08-04
// Created by: Artem NOVIKOV
// Copyright (c) 2014 OPEN CASCADE SAS
//
// This file is part of Open CASCADE Technology software library.
//
// This library is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License version 2.1 as published
// by the Free Software Foundation, with special exception defined in the file
// OCCT_LGPL_EXCEPTION.txt. Consult the file LICENSE_LGPL_21.txt included in OCCT
// distribution for complete text of the license and disclaimer of any warranty.
//
// Alternatively, this file may be used under the terms of Open CASCADE
// commercial license or contractual agreement.


#include <HMeshToMeshVS_DataSource.h>
#include <HMesh.h>
#include <Message.h>
#include <STACCATO_Enum.h>

#include <Standard_Type.hxx>
#include <StlMesh_MeshTriangle.hxx>
#include <StlMesh_SequenceOfMeshTriangle.hxx>
#include <TColgp_SequenceOfXYZ.hxx>
#include <TColStd_DataMapOfIntegerInteger.hxx>
#include <TColStd_DataMapOfIntegerReal.hxx>
#include <Standard_Macro.hxx>

//IMPLEMENT_STANDARD_RTTIEXT(HMeshToMeshVS_DataSource,MeshVS_DataSource)

//================================================================
// Function : Constructor
// Purpose  :
//================================================================
HMeshToMeshVS_DataSource::HMeshToMeshVS_DataSource(HMesh& _HMesh)
{
  Standard_Integer numNodes = _HMesh.getNumNodes();
  warningOut << "Step 0: " << std::endl;

  for (Standard_Integer aNodeID = 1; aNodeID <= numNodes; aNodeID++)
  {
    myNodes.Add( aNodeID);
  }

  Standard_Integer numElements = _HMesh.getNumElements();
  warningOut << "Step 1" << std::endl;

  for (Standard_Integer anElemID = 1; anElemID <= numElements; anElemID++)
  {
    myElements.Add( anElemID);
  }

  myNodeCoords = new TColStd_HArray2OfReal(1, numNodes, 1, 3);
  for (Standard_Integer i = 0; i < numNodes; i++)
  {
	  myNodeCoords->SetValue(i+1, 1, _HMesh.getNodeCoords()[(i*3)+0]);
	  myNodeCoords->SetValue(i+1, 2, _HMesh.getNodeCoords()[(i*3)+1]);
	  myNodeCoords->SetValue(i+1, 3, _HMesh.getNodeCoords()[(i*3)+2]);
  }

  myElemNbNodes = new TColStd_HArray1OfInteger(1, numElements);
  for (Standard_Integer i = 0; i < numElements; i++)
  {
	  if (_HMesh.getElementTypes()[i] == STACCATO_PlainStrain4Node2D || _HMesh.getElementTypes()[i] == STACCATO_PlainStress4Node2D){
		  myElemNbNodes->SetValue(i+1, 4);
	  }
  }
 
  myElemNodes = new TColStd_HArray2OfInteger(1, numElements, 1, 4);
  myElemNormals = new TColStd_HArray2OfReal(1, numElements, 1, 3);

  int index = 0;
  for (Standard_Integer i = 0; i < numElements; i++)
  {
	  if (_HMesh.getElementTypes()[i] == STACCATO_PlainStrain4Node2D || _HMesh.getElementTypes()[i] == STACCATO_PlainStress4Node2D){

		  myElemNodes->SetValue(i + 1, 1, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 0]) + 1);
		  myElemNodes->SetValue(i + 1, 2, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 1]) + 1);
		  myElemNodes->SetValue(i + 1, 3, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 2]) + 1);
		  myElemNodes->SetValue(i + 1, 4, _HMesh.convertNodeLabelToNodeIndex(_HMesh.getElementTopology()[index + 3]) + 1);
		  index = index + 4;
		  /// Hack works only for Quad elements align to xy-plane
		  myElemNormals->SetValue(i + 1, 1, 0);
		  myElemNormals->SetValue(i + 1, 2, 0);
		  myElemNormals->SetValue(i + 1, 3, 1);
	  }
  }

}

//================================================================
// Function : GetGeom
// Purpose  :
//================================================================
Standard_Boolean HMeshToMeshVS_DataSource::GetGeom
(const Standard_Integer ID, const Standard_Boolean IsElement,
TColStd_Array1OfReal& Coords, Standard_Integer& NbNodes,
MeshVS_EntityType& Type) const
{

	errorOut << "ID: " << ID << std::endl;
	if (IsElement)
	{
		if (ID >= 1 && ID <= myElements.Extent())
		{
			Type = MeshVS_ET_Face;
			NbNodes = 4;

			for (Standard_Integer i = 1, k = 1; i <= NbNodes; i++)
			{
				Standard_Integer IdxNode = myElemNodes->Value(ID, i);
				for (Standard_Integer j = 1; j <= 3; j++, k++)
					Coords(k) = myNodeCoords->Value(IdxNode, j);
			}

			return Standard_True;
		}
		else
			return Standard_False;
	}
	else
		if (ID >= 1 && ID <= myNodes.Extent())
		{
			Type = MeshVS_ET_Node;
			NbNodes = 1;

			Coords(1) = myNodeCoords->Value(ID, 1);
			Coords(2) = myNodeCoords->Value(ID, 2);
			Coords(3) = myNodeCoords->Value(ID, 3);
			return Standard_True;
		}
		else
			return Standard_False;
}


//================================================================
// Function : GetGeomType
// Purpose  :
//================================================================
Standard_Boolean HMeshToMeshVS_DataSource::GetGeomType
( const Standard_Integer theID,
 const Standard_Boolean theIsElement,
 MeshVS_EntityType& theType ) const
{
	if (theIsElement)
	{
		theType = MeshVS_ET_Face;
		return Standard_True;
	}
	else
	{
		theType = MeshVS_ET_Node;
		return Standard_True;
	}
}

//================================================================
// Function : GetAddr
// Purpose  :
//================================================================
Standard_Address HMeshToMeshVS_DataSource::GetAddr
( const Standard_Integer, const Standard_Boolean ) const
{
  return NULL;
}

//================================================================
// Function : GetNodesByElement
// Purpose  :
//================================================================
Standard_Boolean HMeshToMeshVS_DataSource::GetNodesByElement
( const Standard_Integer theID,
 TColStd_Array1OfInteger& theNodeIDs,
 Standard_Integer& theNbNodes ) const
{

	if (theID >= 1 && theID <= myElements.Extent() && theNodeIDs.Length() >= 3)
	{
		Standard_Integer aLow = theNodeIDs.Lower();
		theNodeIDs(aLow + 0) = myElemNodes->Value(theID, 1);
		theNodeIDs(aLow + 1) = myElemNodes->Value(theID, 2);
		theNodeIDs(aLow + 2) = myElemNodes->Value(theID, 3);
		theNodeIDs(aLow + 3) = myElemNodes->Value(theID, 4);
		return Standard_True;
	}
	return Standard_False;
}

//================================================================
// Function : GetAllNodes
// Purpose  :
//================================================================
const TColStd_PackedMapOfInteger& HMeshToMeshVS_DataSource::GetAllNodes() const
{
  return myNodes;
}

//================================================================
// Function : GetAllElements
// Purpose  :
//================================================================
const TColStd_PackedMapOfInteger& HMeshToMeshVS_DataSource::GetAllElements() const
{
  return myElements;
}

//================================================================
// Function : GetNormal
// Purpose  :
//================================================================
Standard_Boolean HMeshToMeshVS_DataSource::GetNormal
(const Standard_Integer Id, const Standard_Integer Max,
Standard_Real& nx, Standard_Real& ny, Standard_Real& nz) const
{
	if (Id >= 1 && Id <= myElements.Extent() && Max >= 3)
	{
		nx = myElemNormals->Value(Id, 1);
		ny = myElemNormals->Value(Id, 2);
		nz = myElemNormals->Value(Id, 3);
		return Standard_True;
	}
	else
		return Standard_False;
}
