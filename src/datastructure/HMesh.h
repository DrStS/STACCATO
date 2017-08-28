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
	* \brief get total number of elements
	* \param[out] total number of elements
	* \author Stefan Sicklinger
	***********/
	int getNumElements(){ return elementLabels.size(); }
	/***********************************************************************************************
	* \brief get node coords
	* \param[out] std vector
	* \author Stefan Sicklinger
	***********/
	std::vector<double> getNodeCoords(){ return nodeCoords; }
	/***********************************************************************************************
	* \brief get element types
	* \param[out] std vector
	* \author Stefan Sicklinger
	***********/
	std::vector<STACCATO_Element_type> getElementTypes(){ return elementTyps; }
	/***********************************************************************************************
	* \brief get element topology
	* \param[out] std vector
	* \author Stefan Sicklinger
	***********/
	std::vector<int> getElementTopology(){ return elementsTopology; }
	/***********************************************************************************************
	* \brief convert node label to node index
	* \param[in] node label
	* \param[out] node index [0..nNodes]
	* \author Stefan Sicklinger
	***********/
	int convertNodeLabelToNodeIndex(int _nodeLabel){
		//std::vector<int>::iterator it;
		//it = std::find(nodeLabels.begin(), nodeLabels.end(), _nodeLabel);
		//return distance(nodeLabels.begin(), it);
		return nodeLabelToNodeIndexMap[_nodeLabel];
	}
	/***********************************************************************************************
	* \brief build datastructure
	* \author Stefan Sicklinger
	***********/
	void buildDataStructure(void){
		for (std::vector<int>::size_type i = 0; i != nodeLabels.size(); i++) {
			nodeLabelToNodeIndexMap[nodeLabels[i]] = i;
		}
	}


private:
	/// mesh name
	std::string name;
    /// coordinates of all nodes
    std::vector<double> nodeCoords;
    /// IDs of all nodes
	std::vector<int> nodeLabels;
    /// number of nodes of each element
	std::vector<int> numNodesPerElem;
    /// nodes connectivity inside all elements
	std::vector<int> elementsTopology;
    /// IDs of all elements (now it is not received from clients, therefore it is fixed as 1,2,3...)
	std::vector<int> elementLabels;
	/// Store element type
	std::vector<STACCATO_Element_type> elementTyps;
	/// Map node label to node index
	std::map<int, int> nodeLabelToNodeIndexMap;
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
