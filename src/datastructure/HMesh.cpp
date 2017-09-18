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
#include <assert.h>
#include <numeric>
#include "HMesh.h"
#include "Message.h"

HMesh::HMesh(std::string _name) : name(_name) {
}

HMesh::~HMesh() {
}

void HMesh::addNode(int _label, double _xCoord, double _yCoord, double _zCoord){
	nodeLabels.push_back(_label);
	nodeCoords.push_back(_xCoord);
	nodeCoords.push_back(_yCoord);
	nodeCoords.push_back(_zCoord);
}

void HMesh::addElement(int _label, STACCATO_Element_type _type, std::vector<int> _elementTopology){
	elementLabels.push_back(_label);
	elementTyps.push_back(_type);
	for (std::vector<int>::size_type i = 0; i != _elementTopology.size(); i++) {
		elementsTopology.push_back(_elementTopology[i]);
	}
}

void HMesh::addResultScalarFieldAtNodes(STACCATO_Result_type _type, double _value) {
	// Needs to be called for every node in the sequence nodeIndex = 0..nNodes
	if (_type == STACCATO_Ux_Re) {
		resultUxRe.push_back(_value);
	}
	else if (_type == STACCATO_Uy_Re) {
		resultUyRe.push_back(_value);
	}
	else if (_type == STACCATO_Uz_Re) {
		resultUzRe.push_back(_value);
	}
}

std::vector<double>&  HMesh::getResultScalarFieldAtNodes(STACCATO_Result_type _type) {
	if (_type == STACCATO_Ux_Re){
		return resultUxRe;
	}
	else if(_type == STACCATO_Uy_Re){
		return resultUyRe;
	}
	else if (_type == STACCATO_Uz_Re) {
		return resultUzRe;
	}
}

void HMesh::plot(){
	infoOut << "Element size vector: " << elementLabels.size() << std::endl;
}

void HMesh::buildDataStructure(void){
	//Node loop	
	for (std::vector<int>::size_type i = 0; i != nodeLabels.size(); i++) {
		nodeLabelToNodeIndexMap[nodeLabels[i]] = i;
	}

	//Element loop
	nodeIndexToElementIndices.resize(getNumNodes());
	numDoFsPerNode.resize(getNumNodes());
	elementIndexToNodesIndices.resize(getNumElements());
	numNodesPerElem.resize(getNumElements());

	int lastIndexInElementTopology = -1;

	for (std::vector<int>::size_type i = 0; i != elementLabels.size(); i++) {
		elementLabelToElementIndexMap[elementLabels[i]] = i;

		int numDoFsPerNodeCurrent;
		if (elementTyps[i] == STACCATO_PlainStrain4Node2D || elementTyps[i] == STACCATO_PlainStress4Node2D){
			numNodesPerElem[i]=4;
			//1. DoF -> u_x
			//2. DoF -> u_y
			numDoFsPerNodeCurrent = 2;
		}

		for (int j = 0; j < numNodesPerElem[i]; j++){
			lastIndexInElementTopology++;
			int nodeLabel = getElementTopology()[lastIndexInElementTopology];
			int nodeIndex = convertNodeLabelToNodeIndex(nodeLabel);
			nodeIndexToElementIndices[nodeIndex].push_back(i);
			elementIndexToNodesIndices[i].push_back(nodeIndex);	

			if (numDoFsPerNodeCurrent > numDoFsPerNode[nodeIndex]){
				numDoFsPerNode[nodeIndex] = numDoFsPerNodeCurrent;
			}
		}
	}

	//Node loop for DoFs
	int globalDoFIndex = 0;
	nodeIndexToDoFIndices.resize(getNumNodes());

	// Generate DoFPerEle -> DoFGlob 
	for (int i = 0; i < getNumNodes(); i++) {
		for (int j = 0; j < getNumDoFsPerNode(i); j++) {
			nodeIndexToDoFIndices[i].push_back(globalDoFIndex);
			globalDoFIndex++;
		}
	}
	//Total number of DoF without internal DoFs and BCs
	totalNumOfDoFsRaw = globalDoFIndex;

}