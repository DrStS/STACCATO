/*  Copyright &copy; 2017, Stefan Sicklinger, Munich
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
* \file HMesh.h
* This file holds the class HMesh which is holds a finite element h mesh
* \date 1/25/2017
**************************************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <map>
#include "STACCATO_Enum.h"

class OutputDatabase;
class Message;
/********//**
 * \brief Class HMesh has all data w.r.t. a finite element mesh
 ***********/
class HMesh{
public:
    /***********************************************************************************************
     * \brief Constructor, allocating the storage of the mesh
     * \param[in] _name name of the mesh
     * \author Stefan Sicklinger
     ***********/
    HMesh(std::string _name);
    /***********************************************************************************************
     * \brief Destructor
     * \author Stefan Sicklinger
     ***********/
    virtual ~HMesh();
	/***********************************************************************************************
	* \brief add a node to the mesh
	* \param[in] _label node label
	* \param[in] _xCoord 
	* \param[in] _yCoord
	* \param[in] _zCoord
	* \author Stefan Sicklinger
	***********/
	void addNode(int _label, double _xCoord, double _yCoord, double _zCoord);
	/***********************************************************************************************
	* \brief add a element to the mesh
	* \param[in] _label element label
	* \param[in] _type element type
	* \param[in] _yCoord
	* \param[in] _zCoord
	* \author Stefan Sicklinger
	***********/
	void addElement(int _label, STACCATO_Element_type _type, std::vector<int> _elementTopology);	
	/***********************************************************************************************
	* \brief get total number of nodes
	* \param[out] total number of nodes
	* \author Stefan Sicklinger
	***********/
	int getNumNodes(){ return nodeLabels.size() - referenceNodeLabel.size(); }
	/***********************************************************************************************
	* \brief get node labels
	* \param[out] reference to std vector int
	* \author Stefan Sicklinger
	***********/
	std::vector<int>& getNodeLabels() { return nodeLabels; }
	/***********************************************************************************************
	* \brief get element labels
	* \param[out] reference to std vector int
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<int>& getElementLabels() { return elementLabels; }
	/***********************************************************************************************
	* \brief get total number of elements
	* \param[out] total number of elements
	* \author Stefan Sicklinger
	***********/
	int getNumElements(){ return elementLabels.size(); }
	/***********************************************************************************************
	* \brief get node coords
	* \param[out] reference to std vector double
	* \author Stefan Sicklinger
	***********/
	std::vector<double>& getNodeCoords(){ return nodeCoords; }
	/***********************************************************************************************
	* \brief get node coords sorted
	* \param[out] reference to std vector double
	* \author Stefan Sicklinger
	***********/
	std::vector<double> & getNodeCoordsSortElement() { return nodeCoordsSortElementIndices; }
	/***********************************************************************************************
	* \brief get element types
	* \param[out] reference to std vector
	* \author Stefan Sicklinger
	***********/
	std::vector<STACCATO_Element_type>& getElementTypes(){ return elementTyps; }
	/***********************************************************************************************
	* \brief get element topology
	* \param[out] reference to std vector
	* \author Stefan Sicklinger
	***********/
	std::vector<int>& getElementTopology(){ return elementsTopology; }
	/***********************************************************************************************
	* \brief get number of nodes per element
	* \param[out] reference to std vector
	* \author Stefan Sicklinger
	***********/
	std::vector<int>& getNumNodesPerElement(){ return numNodesPerElem; }
	/***********************************************************************************************
	* \brief get number of DoFs per element
	* \param[out] reference to std vector
	* \author Stefan Sicklinger
	***********/
	std::vector<int>& getNumDoFsPerElement() { return numDoFsPerElem; }
	/***********************************************************************************************
	* \brief get number of DoFs per element
	* \param[out] reference to std vector
	* \author Stefan Sicklinger
	***********/
	std::vector<int>& getElementDoFList() { return elementDoFList; }
	/***********************************************************************************************
	* \brief get number of DoFs per element
	* \param[out] reference to std vector
	* \author Stefan Sicklinger
	***********/
//	std::vector<int>& getTypeDoFsPerElement() { return typeDoFsPerElement; }
	/***********************************************************************************************
	* \brief get relation of node index to element indexes: 1 to nn
	* \param[out] reference to std::vector<std::vector<int>>
	* \author Stefan Sicklinger
	***********/
	std::vector<std::vector<int>>& getNodeIndexToElementIndices(){ return nodeIndexToElementIndices; }
	/***********************************************************************************************
	* \brief get relation of element index to node indexes
	* \param[out] reference to s
	td::vector<std::vector<int>>
	* \author Stefan Sicklinger
	***********/
	std::vector<std::vector<int>>& getElementIndexToNodesIndices(){ return elementIndexToNodesIndices; }
	/***********************************************************************************************
	* \brief get relation of node index to DoF indexes
	* \param[out] reference to std::vector<std::vector<int>>
	* \author Stefan Sicklinger
	***********/
	std::vector<std::vector<int>>& getNodeIndexToDoFIndices(){ return nodeIndexToDoFIndices; }
	/***********************************************************************************************
	* \brief get number of DoFs of each node using node index
	* \param[out] number of DoF 
	* \author Stefan Sicklinger
	***********/
	int getNumDoFsPerNode(int _nodeIndex){ return numDoFsPerNode[_nodeIndex]; }
	/***********************************************************************************************
	* \brief convert node label to node index
	* \param[in] node label
	* \param[out] node index [0..nNodes]
	* \author Stefan Sicklinger
	***********/
	int convertNodeLabelToNodeIndex(int _nodeLabel){
		return nodeLabelToNodeIndexMap[_nodeLabel];
	}
	/***********************************************************************************************
	* \brief convert element label to element index
	* \param[in] element label
	* \param[out] element index [0..nElems]
	* \author Stefan Sicklinger
	***********/
	int convertElementLabelToElementIndex(int _elemLabel) {
		return elementLabelToElementIndexMap[_elemLabel];
	}
	/***********************************************************************************************
	* \brief Total number of DoF without internal DoFs and BCs
	* \param[out] reference to std vector double
	* \author Stefan Sicklinger
	***********/
	int getTotalNumOfDoFsRaw(){ return totalNumOfDoFsRaw; }
	/***********************************************************************************************
	* \brief Get domain dimension
	* \param[out] domain dimension
	* \author Stefan Sicklinger
	***********/
	int getDomainDimension() { return domainDimension; }
	/***********************************************************************************************
	* \brief build datastructure
	* \author Stefan Sicklinger
	***********/
	void buildDataStructure(void);
	/***********************************************************************************************
	* \brief build DOF graph and local element coord vectors
	* \author Stefan Sicklinger
	***********/
	void buildDoFGraph(void);
	/***********************************************************************************************
	* \brief Add a Node Set
	* \param[in] _name
	* \param[in] _nodeLabels
	* \author Harikrishnan Sreekumar
	***********/
	void addNodeSet(std::string, std::vector<int>);
	/***********************************************************************************************
	* \brief Add an Element Set
	* \param[in] _name
	* \param[in] _elemLabels
	* \author Harikrishnan Sreekumar
	***********/
	void addElemSet(std::string, std::vector<int>);
	/***********************************************************************************************
	* \brief Get Handle to the Killed Element-DoF List
	* \param[out] Handle to the Killed Element-DoF List
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<int>&  getElementDoFListRestricted() { return elementDoFListRestricted; }
	/***********************************************************************************************
	* \brief Get the list of DoFs with Dirichlet Condition
	* \param[out] dirichletDOF
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<int>  getRestrictedHomogeneousDoFList() { return restrictedHomogeneousDoFList; }
	/***********************************************************************************************
	* \brief Kills / Mark the DoFs with Dirichlet Condition as -1
	* \param[in] _nodeSetName
	* \param[in] _restrictedDOF
	* \author Harikrishnan Sreekumar
	***********/
	void killDirichletDOF(std::string _nodeSetName, std::vector<int> _restrictedDOF);
	/***********************************************************************************************
	* \brief Find and Return all Node Sets corresponding to the label
	* \param[in] _nodeSetName
	* \param[out] nodeSets
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<int> convertNodeSetNameToLabels(std::string _nodeSetName);
	/***********************************************************************************************
	* \brief Find and Return all Element Sets corresponding to the label
	* \param[in] _elemSetName
	* \param[out] nodeSets
	* \author Harikrishnan Sreekumar
	***********/
	std::vector<int> convertElementSetNameToLabels(std::string _elemSetName);
	bool hasParts;
	bool isSIM;
	std::vector<int> referenceNodeLabel;
private:
	/// mesh name
	std::string name;
    /// coordinates of all nodes
    std::vector<double> nodeCoords;
	/// coordinates of all nodes sorted for parallel mem access element index 0..ne
	std::vector<double> nodeCoordsSortElementIndices;
	/// DoF list sorted  element index 0..ne
	std::vector<int> elementDoFList;
	/// DoF list sorted  element index 0..ne with entries -1 for homogeneous Dirichlet DoF; -2 for non-homogeneous Dirichlet
	std::vector<int> elementDoFListRestricted;
	/// vector holding affected DOFs for Dirichlet
	std::vector<int> restrictedHomogeneousDoFList;
    /// labels of all nodes
	std::vector<int> nodeLabels;
    /// number of nodes of each element
	std::vector<int> numNodesPerElem; 
	/// number of DoFs of each element
	std::vector<int> numDoFsPerElem;
	/// type of DoFs of each element
	//std::vector<int> typeDoFsPerEle;
	/// number of DoFs of each node
	std::vector<int> numDoFsPerNode;
    /// nodes connectivity inside all elements using node labels
	std::vector<int> elementsTopology;
    /// labels of all elements
	std::vector<int> elementLabels;
	/// store element type by index
	std::vector<STACCATO_Element_type> elementTyps;
	/// map node label to node index
	std::map<int, int> nodeLabelToNodeIndexMap;
	/// map element label to element index
	std::map<int, int> elementLabelToElementIndexMap;
	/// relation of node index to element indexes: 1 to nn
	std::vector<std::vector<int>> nodeIndexToElementIndices;
	/// relation of element index to node indexes: 1 to ne
	std::vector<std::vector<int>> elementIndexToNodesIndices;
	/// total number of DoF without internal DoFs and BCs
	int totalNumOfDoFsRaw;
	/// domain dimension 1D, 2D or 3D
	int domainDimension;
	/// relation of node index to DoF indexes: 1 to nd
	std::vector<std::vector<int>> nodeIndexToDoFIndices;
	/// Avoid copy of a HMesh object copy constructor 
	HMesh(const HMesh&);
	/// Avoid copy of a HMesh object assignment operator
	HMesh& operator=(const HMesh&);
	/// Map holding NodeSets
	std::map<std::string, std::vector<int>> nodeSetsMap;
	/// Map holding ElementSets
	std::map<std::string, std::vector<int>> elemSetsMap;
public:
	OutputDatabase* myOutputDatabase;

};
