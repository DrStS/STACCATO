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
	hasParts = false;
	isSIM = false;
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
		else if (elementTyps[i] == STACCATO_UmaElement) {
			numNodesPerElem[i] = 5;  // Take care of this
			//1. DoF -> u_x
			//2. DoF -> u_y
			//3. DoF -> u_z
			//4. DoF -> phi_x
			//5. DoF -> phi_y
			//6. DoF -> phi_z
			numDoFsPerNodeCurrent = 6;
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
				elementDoFListBC.push_back(nodeIndexToDoFIndices[nodeIndex][l]);	// Create a copy for Dirichlet DoF Killing
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

void HMesh::killDirichletDOF(std::string _nodeSetName, std::vector<int> _restrictedDOF) {
	std::vector<int> nodeSet = convertNodeSetNameToLabels(_nodeSetName);

	// Create the Map of boundary Dof List for all Nodes
	for (int iNode = 0; iNode < nodeSet.size(); iNode++) {
		int nodeIndex = convertNodeLabelToNodeIndex(nodeSet[iNode]);

		// Create a map of Dofs
		int numDoFsPerNode = getNumDoFsPerNode(nodeIndex);
		for (int iMap = 0; iMap < numDoFsPerNode; iMap++) {
			if (_restrictedDOF[iMap] == 1) {
				dirichletDOF.push_back(getNodeIndexToDoFIndices()[nodeIndex][iMap]);
			}
		}
	}

	// DOF Killing
	bool flag = false;
	for (int n = 0; n < nodeSet.size(); n++) {
		std::vector<int> indexAffected = getNodeIndexToDoFIndices()[convertNodeLabelToNodeIndex(nodeSet[n])];
		std::vector<int> affectedElements = getNodeIndexToElementIndices()[convertNodeLabelToNodeIndex(nodeSet[n])];
		// Create a list of Dofs with -1 indicating Dirichlet enforced DoF
		int totalDoFsPerElem = getNumDoFsPerNode(convertNodeLabelToNodeIndex(nodeSet[n]))*getNumNodesPerElement()[0];
		for (int iMap = 0; iMap < affectedElements.size(); iMap++) {
			for (int jMap = totalDoFsPerElem*affectedElements[iMap]; jMap < totalDoFsPerElem*affectedElements[iMap] + totalDoFsPerElem; jMap++)
			{
				for (int kMap = 0; kMap < indexAffected.size(); kMap++) {
					if (indexAffected[kMap] == elementDoFList[jMap]) {
						elementDoFListBC[jMap] = -1;
						flag = true;
					}
				}

			}
		}
	}
	if(flag)
		std::cout << ">> Dirichlet DoF killing performed on NODESET: " << _nodeSetName << std::endl;

}

void HMesh::addNodeSet(std::string _name, std::vector<int> _nodeLabels) {
	nodeSetsMap[_name] = _nodeLabels;
}

void HMesh::addElemSet(std::string _name, std::vector<int> _elemLabels) {
	elemSetsMap[_name] = _elemLabels;
}

std::vector<int> HMesh::convertNodeSetNameToLabels(std::string _nodeSetName) {
	if (nodeSetsMap.find(_nodeSetName) != nodeSetsMap.end()) {
		return nodeSetsMap.find(_nodeSetName)->second;
	}
	else
		return {};
}

std::vector<int> HMesh::convertElementSetNameToLabels(std::string _elemSetName) {
	if (elemSetsMap.find(_elemSetName) != elemSetsMap.end()) {
		return elemSetsMap.find(_elemSetName)->second;
	}
	else
		return{};
}