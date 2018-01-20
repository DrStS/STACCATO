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

void HMesh::addResultScalarFieldAtNodes(STACCATO_Result_type _type, std::vector<double> _valueVec) {
	if (_type == STACCATO_Ux_Re) {
		resultsUxRe.push_back(_valueVec);
	}
	else if (_type == STACCATO_Uy_Re) {
		resultsUyRe.push_back(_valueVec);
	}
	else if (_type == STACCATO_Uz_Re) {
		resultsUzRe.push_back(_valueVec);
	}
	else if (_type == STACCATO_Ux_Im) {
		resultsUxIm.push_back(_valueVec);
	}
	else if (_type == STACCATO_Uy_Im) {
		resultsUyIm.push_back(_valueVec);
	}
	else if (_type == STACCATO_Uz_Im) {
		resultsUzIm.push_back(_valueVec);
	}
	else if (_type == STACCATO_Magnitude_Re) {
		resultsMagRe.push_back(_valueVec);
	}
	else if (_type == STACCATO_Magnitude_Im) {
		resultsMagIm.push_back(_valueVec);
	}
	
}

std::vector<double>&  HMesh::getResultScalarFieldAtNodes(STACCATO_Result_type _type, int index) {
	if (_type == STACCATO_Ux_Re){
		return resultsUxRe[index];
	}
	else if(_type == STACCATO_Uy_Re){
		return resultsUyRe[index];
	}
	else if (_type == STACCATO_Uz_Re) {
		return resultsUzRe[index];
	}
	else if (_type == STACCATO_Ux_Im) {
		return resultsUxIm[index];
	}
	else if (_type == STACCATO_Uy_Im) {
		return resultsUyIm[index];
	}
	else if (_type == STACCATO_Uz_Im) {
		return resultsUzIm[index];
	}
	else if (_type == STACCATO_Magnitude_Re) {
		return resultsMagRe[index];
	}
	else if (_type == STACCATO_Magnitude_Im) {
		return resultsMagIm[index];
	}
}

void HMesh::addResultsTimeDescription(std::string _resultsTimeDescription) {
	resultsTimeDescription.push_back(_resultsTimeDescription);
}

std::vector<std::string>& HMesh::getResultsTimeDescription() {		// Getter Function to return all frequency steps
	return resultsTimeDescription;
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
			domainDimension = 2;
		}
		else if (elementTyps[i] == STACCATO_Tetrahedron10Node3D) {
			numNodesPerElem[i] = 10;
			//1. DoF -> u_x
			//2. DoF -> u_y
			//3. DoF -> u_z
			numDoFsPerNodeCurrent = 3;
			domainDimension = 3;
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

void HMesh::buildDoFGraph(void) {
	int domainDimension = 3;
	int numDoFsPerElement;
	//nodeCoordsSortElementIndices.resize(getNumNodes()*domainDimension);
	numDoFsPerElem.resize(getNumElements());
	//elementDoFList.resize(totalNumOfDoFsRaw);
	int lastIndex = 0;

	for (std::vector<int>::size_type iElement = 0; iElement < elementLabels.size(); iElement++) {
		int numNodesPerElement = numNodesPerElem[iElement];
		double * eleCoord = new double[numNodesPerElement*domainDimension];
		numDoFsPerElement = 0;
		//Loop over nodes of current element
		for (int j = 0; j < numNodesPerElement; j++)
		{
			int nodeIndex = elementIndexToNodesIndices[iElement][j];
			if (domainDimension == 3) {
				nodeCoordsSortElementIndices.push_back(nodeCoords[nodeIndex * 3 + 0]);
				nodeCoordsSortElementIndices.push_back(nodeCoords[nodeIndex * 3 + 1]);
				nodeCoordsSortElementIndices.push_back(nodeCoords[nodeIndex * 3 + 2]);
			}
			else if (domainDimension == 2) {
				// Extract x and y coord only; for 2D; z=0
				nodeCoordsSortElementIndices.push_back(nodeCoords[nodeIndex * 3 + 0]);
				nodeCoordsSortElementIndices.push_back(nodeCoords[nodeIndex * 3 + 1]);
			}

			// Generate DoF table
			for (int l = 0; l < numDoFsPerNode[nodeIndex]; l++) {
				elementDoFList.push_back(nodeIndexToDoFIndices[nodeIndex][l]);
				numDoFsPerElement++;
			}

		}
		numDoFsPerElem[iElement] = numDoFsPerElement;
	}
}

std::vector<double>& HMesh::getResultScalarFieldOfNode(STACCATO_Result_type _type, int _nodeLabel) {
	std::vector<double> results;
	for (int i = 0; i < this->getResultsTimeDescription().size(); i++){
		results.push_back(this->getResultScalarFieldAtNodes(_type, i)[_nodeLabel]);
		std::cout << results[i] << "\n";
	}
	return results;
}

int HMesh::getNodeIndexForLabel(int _nodeLabel) {
	for (int i = 0; i < this->getNodeLabels().size(); i++)
	{
		if (this->getNodeLabels()[i] == _nodeLabel)
			return i;
	}
}