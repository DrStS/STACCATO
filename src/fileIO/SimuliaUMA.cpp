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
#include "AuxiliaryParameters.h"
#include "SimuliaUMA.h"
#include "Message.h"
#include "MemWatcher.h"
#include "HMesh.h"
//XML
#include "MetaDatabase.h"

//UMA
#ifdef USE_SIMULIA_UMA_API
#include <ads_CoreFESystemC.h>
#include <ads_CoreMeshC.h>
#include <uma_System.h>
#include <uma_ArrayInt.h>
#include <uma_SparseMatrix.h>
#endif
//#define DEBUG

#include <fstream>
#include <iostream>

SimuliaUMA::SimuliaUMA(std::string _fileName, HMesh& _hMesh, int _partId) : myHMesh(&_hMesh) {
	myFileName = _fileName;
	std::cout << ">> SIM Reader initialized for file " << myFileName << "[_X1][_X2][_X3].sim" << std::endl;
	MetaDatabase::getInstance()->simFile = myFileName;
	numNodes = 0;
	numDoFperNode = 0;
	openFile();
	myHMesh->hasParts = true;
	myHMesh->isSIM = true;

	hasInternalDOF_K = false;
	hasInternalDOF_M = false;
	hasInternalDOF_SD = false;

}

SimuliaUMA::~SimuliaUMA() {
}

void SimuliaUMA::openFile() {
	bool printDetails = false;
	bool printFile = false;
	bool printMap = true;
	std::vector<int> knowSomeNode = { 
	};
#ifdef USE_SIMULIA_UMA_API
	std::vector<std::string> mapTypeName;
	mapTypeName.push_back("DOFS");
	mapTypeName.push_back("NODES");
	mapTypeName.push_back("MODES");
	mapTypeName.push_back("ELEMENTS");
	mapTypeName.push_back("CASES");
	mapTypeName.push_back("Unknown");

	// File Checks
	bool enableFileCheck = false;
	if (enableFileCheck)
	{
		std::cout << ">> SIM File content check: " << std::endl;
		std::vector<std::string> checkMatrix;
		checkMatrix.push_back("GenericSystem_mass");
		checkMatrix.push_back("GenericSystem_stiffness");
		checkMatrix.push_back("GenericSystem_load");
		checkMatrix.push_back("GenericSystem_structuraldamping");
		checkMatrix.push_back("GenericSystem_StructuralDamping");
		checkMatrix.push_back("GenericSystem_structuralDamping");
		checkMatrix.push_back("GenericSystem_structural damping");
		checkMatrix.push_back("GenericSystem_Structural Damping");
		checkMatrix.push_back("GenericSystem_STRUCTURAL DAMPING");
		checkMatrix.push_back("GenericSystem_damping");
		checkMatrix.push_back("STRUCTURAL DAMPING");
		checkMatrix.push_back("structural damping");
		checkMatrix.push_back("Structural Damping");
		for (int i = 0; i < 5; i++)
		{
			std::string postfix;
			if (i == 0)
				postfix = "";
			else
				postfix = "_X" + std::to_string(i);
			std::string checkFile = myFileName + postfix + ".sim";
			std::cout << " > Checking file " << checkFile << std::endl;
			const char * findfile = checkFile.c_str();
			std::ifstream ifile(findfile);
			if (ifile) {
				uma_System systemCheck(checkFile.c_str());
				if (systemCheck.IsNull()) {
					std::cout << "  > Error: System not defined.\n";
				}
				else {
					std::cout << "  > Type :" << systemCheck.Type() << std::endl;
					for (int j = 0; j < checkMatrix.size(); j++) {
						if (!systemCheck.HasMatrix(checkMatrix[j].c_str()))
							std::cout << "  > " << checkMatrix[j] << ": NO " << std::endl;
						else
							std::cout << "  > " << checkMatrix[j] << ": YES " << std::endl;
					}
				}
			}
			else
				std::cout << "  > File does not exist." << std::endl;
		}
	}
	
	/* ***********************************
	* _X1 -> Stiffness Matrix
	* _X2 -> Mass Matrix
	* _X3 -> Structural Damping
	* _X4 -> Reserved for Viscous Damping
	*********************************** */
	stiffnessFileName = myFileName + "_X1.sim";
	massFileName = myFileName + "_X2.sim";
	structDampingFileName = myFileName + "_X3.sim";

	stiffnessUMA_key = "GenericSystem_stiffness";
	massUMA_key = "GenericSystem_mass";
	structuralDampingUMA_key = "GenericSystem_structuralDamping";

	char * simFileStiffness = const_cast<char*>(stiffnessFileName.c_str());
	char * simFileMass = const_cast<char*>(massFileName.c_str());
	char * simFileStructD = const_cast<char*>(structDampingFileName.c_str());

	std::cout << " > STIFFNESS SIM file      : " << simFileStiffness << std::endl;
	std::cout << " > MASS SIM file           : " << simFileMass << std::endl;
	std::cout << " > STRUCT DAMPING SIM file : " << simFileStructD << std::endl;


	// Checking for existence of the files. We avoid stiffness and mass files as they are a must. Abscence may throw an error
	bool structDampingExists = false;

	// Import all individual data
	importDatastructureSIM(simFileStiffness, stiffnessUMA_key, printDetails);
	importDatastructureSIM(simFileMass, massUMA_key, printDetails);
	importDatastructureSIM(simFileStructD, structuralDampingUMA_key, printDetails);
	//importDatastructureSIM(simFileStiffness, structuralDampingUMA_key, true);
	// Generate global map
	generateGlobalMap(printDetails);

	MetaDatabase::getInstance()->nodeToDofMapMeta = nodeToDofMap;
	MetaDatabase::getInstance()->nodeToGlobalMapMeta = nodeToGlobalMap;
	
	std::cout << ">> Importing SIM to HMesh ..." << std::endl;

	numNodes = nodeToGlobalMap.size();
	totalDOFs = 0;
	for (std::map<int, std::vector<int>>::iterator it = nodeToDofMap.begin(); it != nodeToDofMap.end(); ++it) {
		totalDOFs += it->second.size();
	}

	std::cout << " > Num Node: " << numNodes << " TotalDoFs: " << totalDOFs << std::endl;
	myHMesh->numUMADofs = totalDOFs;
	
	// Labeling
	std::vector<int> elementTopo;
	std::vector<double> coord;
	coord.push_back(0.00);
	coord.push_back(0.00);
	coord.push_back(0.00);

	// Node Labelling
	int lastHMeshNodeLabel = 0;
	if(myHMesh->hasParts)	{
		lastHMeshNodeLabel = myHMesh->getNodeLabels().back();
	}

	if (printDetails)
		std::cout << " > Added Nodes with Label: ";
	for (int i = 1; i <= totalDOFs; i++)	{
		elementTopo.push_back(lastHMeshNodeLabel +i);
		myHMesh->addNode(lastHMeshNodeLabel + i, coord[0], coord[1], coord[2]);
		if (printDetails)
			std::cout << lastHMeshNodeLabel + i << " | ";
	}
	if (printDetails)
		std::cout << std::endl;

	if (lastHMeshNodeLabel != 0)
		std::cout << ">> Simulia UMA: Adding Nodes and Element in append mode.\n";
	else
		std::cout << ">> Simulia UMA: Adding Nodes and Element as new model.\n";

	// Element Labelling
	int lastHMeshElementLabel = 0;
	if (myHMesh->hasParts) {
		lastHMeshElementLabel = myHMesh->getElementLabels().back();
	}
	myHMesh->addElement(lastHMeshElementLabel+1, STACCATO_UmaElement, elementTopo);

	if (printMap)
		printMapToFile();

	std::vector<int> inputdoflist;
	for (int iKN = 0; iKN < knowSomeNode.size(); iKN++)
	{
		auto search = nodeToGlobalMap.find(knowSomeNode[iKN]);
		auto search2 = nodeToDofMap.find(knowSomeNode[iKN]);
		if (search != nodeToGlobalMap.end())
		{
			std::cout << "==============================================" << std::endl;
			std::cout << "= Node: " << knowSomeNode[iKN] << " , #DOFS: " << search->second.size() << std::endl;
			std::cout << "= GI: Global Index, GL: Global Label" << std::endl;
			for (int i = 0; i < search->second.size(); i++) {
				std::cout << "== DOF: " << search2->second[i] << " GI: " << search->second[i] << " GL: " << search->second[i] + 1 << std::endl;
				inputdoflist.push_back(search->second[i] + 1);
			}
			std::cout << "==============================================" << std::endl;
		}
		else {
			std::cout << "==============================================" << std::endl;
			std::cout << "= Node: " << knowSomeNode[iKN] << " DOES NOT EXIST." << std::endl;
			std::cout << "==============================================" << std::endl;
		}
	}

	std::string _fileName = "Staccato_KMOR_InputDOF_List.dat";
	std::cout << ">> Writing " << _fileName << "..." << std::endl;
	std::ofstream myfile;
	myfile.open(_fileName);

	for (int i = 0; i < inputdoflist.size(); i++) {
		myfile << inputdoflist[i] << " ";
	}
	myfile << std::endl;
	myfile.close();
#endif
}

void SimuliaUMA::importDatastructureSIM(char* _fileName, char* _matrixkey, bool _printToScreen) {
#ifdef USE_SIMULIA_UMA_API
	std::cout << ">> Sensing SIM File: " << _fileName << " UMA Key: " << _matrixkey << std::endl;
	flag_SD = false;
	std::ifstream ifile(_fileName);
	if (!ifile) {
		if (std::string(_matrixkey) == structuralDampingUMA_key)
		{
			std::cout << ">> StructuralDamping file not found and hence skipped." << std::endl;
			flag_SD = true;
		}
		else {
			std::cout << ">> File not found." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	if (!flag_SD)
	{
		uma_System system(_fileName);
		if (system.IsNull()) {
			std::cout << ">> Error: System not defined.\n";
		}
		if (system.Type() != ads_GenericSystem) {
			std::cout << ">> Error: Not a Generic System.\n";
		}

		extractDatastructure(system, _matrixkey, _printToScreen);
	}
#endif
}

#ifdef USE_SIMULIA_UMA_API
void SimuliaUMA::extractDatastructure(const uma_System &system, char *matrixName, bool _printToScreen) {
	char * mapTypeName[] = { "DOFS", "NODES", "MODES", "ELEMENTS", "CASES", "Unknown" };
	// Check for matrix existence
	if (!system.HasMatrix(matrixName)) {
		std::cout << " >> Sparse matrix " << matrixName << " not found\n";
		exit(EXIT_FAILURE);
	}

	// Load the matrix
	uma_SparseMatrix smtx;
	system.SparseMatrix(smtx, matrixName);
	if (!smtx) {
		std::cout << " >> Sparse matrix " << matrixName << " cannot be accessed!\n";
		exit(EXIT_FAILURE);
	}

	printf(" Matrix %s - sparse type\n", matrixName);
	printf("  domain: rows %s, columns %s\n", mapTypeName[smtx.TypeRows()], mapTypeName[smtx.TypeColumns()]);
	printf("  size: rows %i, columns %i, entries %i", smtx.NumRows(), smtx.NumColumns(), smtx.NumEntries());
	if (smtx.IsSymmetric())
		printf("; symmetric");
	printf("\n");
	checkForInternalDofs(smtx, matrixName, _printToScreen);
}
#endif

#ifdef USE_SIMULIA_UMA_API
void SimuliaUMA::checkForInternalDofs(const uma_SparseMatrix &smtx, char *matrixName, bool _printToScreen) {

	uma_SparseIterator iter(smtx);
	int row, col; double val;

	if (_printToScreen) {
		int count = 0;
		int lastRow = 0;
		for (iter.First(); !iter.IsDone(); iter.Next(), count++) {
			iter.Entry(row, col, val);
			if (row != lastRow)
				printf("\n");
			if (row != col)
				printf("  %2i,%-2i:%g", row, col, val);
			else
				printf(" *%2i,%-2i:%g", row, col, val);

			lastRow = row;
		}
		printf("\n");

		// Map column DOFS to user nodes and dofs
		if (smtx.TypeColumns() != uma_Enum::DOFS)
			exit(EXIT_FAILURE);
		printf("\n  map columns [column: node-dof]:");
		uma_ArrayInt     nodes; smtx.MapColumns(nodes, uma_Enum::NODES); // test array
		uma_ArrayInt ldofs; smtx.MapColumns(ldofs, uma_Enum::DOFS);  // test vector
		for (int col = 0; col < nodes.Size(); col++) {
			if (col % 10 == 0)
				printf("\n");
			printf(" %3i:%3i-%1i", col, nodes[col], ldofs[col]);
		}
		printf("\n");
	}

	if (smtx.TypeColumns() != uma_Enum::DOFS)
		return;

	// Sensing internal dofs
	bool flagIDOF = false;
	uma_ArrayInt     nodes; smtx.MapColumns(nodes, uma_Enum::NODES); // test array
	uma_ArrayInt ldofs; smtx.MapColumns(ldofs, uma_Enum::DOFS);  // test vector
	for (int col = 0; col < nodes.Size(); col++) {
		if (nodes[col] >= 1000000000) {
			addInternalDof(matrixName, col, true);
			flagIDOF = true;
		}
	}

	if (flagIDOF)
		std::cout << " ! Presence of Internal DOF." << std::endl;
	else
		std::cout << " > No Internal DOFs." << std::endl;
}
#endif

void SimuliaUMA::addInternalDof(char *matrixName, int _index, bool _flag) {

	if (std::string(matrixName) == std::string(stiffnessUMA_key)) {
		hasInternalDOF_K = _flag;
		internalDOF_K.push_back(_index);
	}
	else if (std::string(matrixName) == std::string(massUMA_key)) {
		hasInternalDOF_M = _flag;
		internalDOF_M.push_back(_index);
	}
	else if (std::string(matrixName) == std::string(structuralDampingUMA_key)) {
		hasInternalDOF_SD = _flag;
		internalDOF_SD.push_back(_index);
	}
}

void SimuliaUMA::generateGlobalMap(bool _printToScreen) {
	std::cout << ">> Generating global map..." << std::endl;
#ifdef USE_SIMULIA_UMA_API
	// Stiffness
	uma_System system_K(stiffnessFileName.c_str());
	uma_SparseMatrix smtx_K;
	system_K.SparseMatrix(smtx_K, stiffnessUMA_key);
	uma_ArrayInt nodes_K; smtx_K.MapColumns(nodes_K, uma_Enum::NODES);
	uma_ArrayInt ldofs_K; smtx_K.MapColumns(ldofs_K, uma_Enum::DOFS);

	// Mass
	uma_System system_M(massFileName.c_str());
	uma_SparseMatrix smtx_M;
	system_M.SparseMatrix(smtx_M, massUMA_key);
	uma_ArrayInt nodes_M; smtx_M.MapColumns(nodes_M, uma_Enum::NODES);
	uma_ArrayInt ldofs_M; smtx_M.MapColumns(ldofs_M, uma_Enum::DOFS);

	// Create pool of all available nodes and dofs
	for (int col_K = 0; col_K < nodes_K.Size(); col_K++) {
		addEntryToNodeDofMap(nodes_K[col_K], ldofs_K[col_K]);
	}
	for (int col_M = 0; col_M < nodes_M.Size(); col_M++) {
		addEntryToNodeDofMap(nodes_M[col_M], ldofs_M[col_M]);
	}

	if (!flag_SD)
	{
		// Structural Damping
		uma_System system_SD(structDampingFileName.c_str());
		uma_SparseMatrix smtx_SD;
		system_SD.SparseMatrix(smtx_SD, structuralDampingUMA_key);
		uma_ArrayInt nodes_SD; smtx_SD.MapColumns(nodes_SD, uma_Enum::NODES);
		uma_ArrayInt ldofs_SD; smtx_SD.MapColumns(ldofs_SD, uma_Enum::DOFS);
		for (int col_SD = 0; col_SD < nodes_SD.Size(); col_SD++) {
			addEntryToNodeDofMap(nodes_SD[col_SD], ldofs_SD[col_SD]);
		}
	}

	// Display Map
	if (_printToScreen)
	{
		std::cout << ">> DOF Map: " << std::endl;
		for (std::map<int, std::vector<int>>::iterator it = nodeToDofMap.begin(); it != nodeToDofMap.end(); ++it) {
			std::cout << "Node: " << it->first << " DOFs: ";
			for (int j = 0; j < it->second.size(); j++)
			{
				std::cout << it->second[j] << " < ";
			}
			std::cout << std::endl;
		}
	}

	// Create Global Map
	int globalIndex = 0;
	for (std::map<int, std::vector<int>>::iterator it = nodeToDofMap.begin(); it != nodeToDofMap.end(); ++it) {
		for (int j = 0; j < it->second.size(); j++)
		{
			nodeToGlobalMap[it->first].push_back(globalIndex);
			globalIndex++;
		}
	}

	// Display Map
	if (_printToScreen)
	{
		std::cout << ">> Global Map: " << std::endl;
		for (std::map<int, std::vector<int>>::iterator it = nodeToGlobalMap.begin(); it != nodeToGlobalMap.end(); ++it) {
			std::cout << "Node: " << it->first << " DOFs: ";
			for (int j = 0; j < it->second.size(); j++)
			{
				std::cout << it->second[j] << " < ";
			}
			std::cout << std::endl;
		}
	}
#endif
}

void SimuliaUMA::addEntryToNodeDofMap(int _node, int _dof) {
	auto search = nodeToDofMap.find(_node);
	if (search != nodeToDofMap.end()) {
		bool found = false;
		for (int i = 0; i < search->second.size(); i++)
		{
			if (_dof == search->second[i]) {
				found = true;
			}
		}
		if (!found) {
			//std::cout << "Adding new dof to node " << _node << " with dof "<< _dof <<std::endl;
			search->second.push_back(_dof);
		}
	}
	else {
		/*if (_dof>3 && nodeToDofMap[_node].size() == 0)
		{
			std::cout << "Added missing" << std::endl;
			nodeToDofMap[_node].push_back(1);
			nodeToDofMap[_node].push_back(2);
			nodeToDofMap[_node].push_back(3);
		}*/
		nodeToDofMap[_node].push_back(_dof);
	}
}

void SimuliaUMA::printMapToFile() {
	std::string _fileName = "Staccato_KMOR_MAP_UMA.dat";
	std::cout << ">> Writing " << _fileName << "..." << std::endl;
	std::ofstream myfile;
	myfile.open(_fileName);
	myfile << std::scientific;
	myfile << "% UMA MAP | Generated with STACCCATO" << std::endl;
	myfile << "% NODE LABEL  |  LOCAL DOF  |  GLOBAL DOF" << std::endl;
	std::map<int, std::vector<int>>::iterator it2 = nodeToDofMap.begin();
	for (std::map<int, std::vector<int>>::iterator it = nodeToGlobalMap.begin(); it != nodeToGlobalMap.end(); ++it, ++it2) {
		for (int j = 0; j < it->second.size(); j++)
		{
			myfile << it->first << " " << it2->second[j] << " " << it->second[j] << std::endl;
		}
	}
	myfile << std::endl;
	myfile.close();
}