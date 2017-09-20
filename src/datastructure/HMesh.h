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

#ifndef HMESH_H_
#define HMESH_H_

#include <string>
#include <vector>
#include <map>
#include "STACCATO_Enum.h"

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
	* \brief plot the mesh
	* \author Stefan Sicklinger
	***********/
	void plot(void);		
	/***********************************************************************************************
	* \brief get total number of nodes
	* \param[out] total number of nodes
	* \author Stefan Sicklinger
	***********/
	int getNumNodes(){ return nodeLabels.size(); }
	/***********************************************************************************************
	* \brief get node labels
	* \param[out] reference to std vector int
	* \author Stefan Sicklinger
	***********/
	std::vector<int>& getNodeLabels() { return nodeLabels; }
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
	std::vector<int>& getTypeDoFsPerElement() { return typeDoFsPerElement; }
	/***********************************************************************************************
	* \brief get relation of node index to element indexes: 1 to nn
	* \param[out] reference to std::vector<std::vector<int>>
	* \author Stefan Sicklinger
	***********/
	std::vector<std::vector<int>>& getNodeIndexToElementIndices(){ return nodeIndexToElementIndices; }
	/***********************************************************************************************
	* \brief get relation of element index to node indexes
	* \param[out] reference to std::vector<std::vector<int>>
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
	* \brief Total number of DoF without internal DoFs and BCs
	* \param[out] reference to std vector double
	* \author Stefan Sicklinger
	***********/
	int getTotalNumOfDoFsRaw(){ return totalNumOfDoFsRaw; }
	/***********************************************************************************************
	* \brief build datastructure
	* \author Stefan Sicklinger
	***********/
	void buildDataStructure(void);
	/***********************************************************************************************
	* \brief Add a result to database
	* \param[in] _type
	* \param[in] _value
	* \author Stefan Sicklinger
	***********/
	void HMesh::addResultScalarFieldAtNodes(STACCATO_Result_type _type, double _value);
	/***********************************************************************************************
	* \brief Add a result to database
	* \param[in] _type
	* \param[out] reference to std vector
	* \author Stefan Sicklinger
	***********/
	std::vector<double>& HMesh::getResultScalarFieldAtNodes(STACCATO_Result_type _type);

private:
	/// mesh name
	std::string name;
    /// coordinates of all nodes
    std::vector<double> nodeCoords;
    /// labels of all nodes
	std::vector<int> nodeLabels;
    /// number of nodes of each element
	std::vector<int> numNodesPerElem;
	/// type of DoFs of each element
	std::vector<int> typeDoFsPerElement;
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
	/// relation of node index to DoF indexes: 1 to nd
	std::vector<std::vector<int>> nodeIndexToDoFIndices;
	/// result Vector node index to result value
	std::vector<double> resultUxRe;
	/// result Vector node index to result value
	std::vector<double> resultUyRe;
	/// result Vector node index to result value
	std::vector<double> resultUzRe;
    /// unit test class
    friend class TestFEMesh;
private:
	/// Avoid copy of a HMesh object copy constructor 
	HMesh(const HMesh&);
	/// Avoid copy of a HMesh object assignment operator
	HMesh& operator=(const HMesh&);
};
/***********************************************************************************************
 * \brief Print the mesh at once
 * \author Stefan Sicklinger
 ***********/

#endif /* HMESH_H_ */
