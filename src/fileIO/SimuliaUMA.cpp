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
#include "AuxiliaryFunctions.h"
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
	std::cout << ">> SIM Reader initialized for file " << myFileName << "[_X1][_X2][_X3][_X4].sim" << std::endl;
	numNodes = 0;
	numDoFperNode = 0;
	noStructuralDamping = false;
	noDamping = false;
	openFile();
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
		checkMatrix.push_back("GenericSystem_mass");					/* True for mass */
		checkMatrix.push_back("GenericSystem_stiffness");				/* True for stiffness */
		checkMatrix.push_back("GenericSystem_load");			
		checkMatrix.push_back("GenericSystem_structuralDamping");		/* True for structural damping */
		checkMatrix.push_back("GenericSystem_damping");
		checkMatrix.push_back("GenericSystem_viscousDamping");			/* True for viscous damping */
		checkMatrix.push_back("GenericSystem_modes");
		checkMatrix.push_back("ModalSystem_modal");
		checkMatrix.push_back("ModalSystem_modal");
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

				if (systemCheck.Type() != ads_GenericSystem)
					std::cout << ">> Error: Not a Generic System.\n";
				else
					std::cout << ">> Generic System.\n";

				if (systemCheck.Type() != ads_ModalSystem)
					std::cout << ">> Error: Not a Modal System.\n";
				else {
					std::cout << ">> Modal System.\n";
					std::cout << "  > Type :" << systemCheck.Type() << std::endl;
					for (int j = 0; j < checkMatrix.size(); j++) {
						if (!systemCheck.HasMatrix(checkMatrix[j].c_str()))
							std::cout << "  > " << checkMatrix[j] << ": NO " << std::endl;
						else {
							std::cout << "  > " << checkMatrix[j] << ": YES " << std::endl;

							// Check for MODES
							uma_SparseMatrix smtx;
							systemCheck.SparseMatrix(smtx, checkMatrix[j].c_str());


							if (smtx.TypeColumns() != uma_Enum::MODES)
								std::cout << " > MODES NOT FOUND" << std::endl;
							else
								std::cout << " > MODES FOUND" << std::endl;
						}
					}
				}

				if (systemCheck.Type() != ads_GenericSystem) 
					std::cout << ">> Error: Not a Generic System.\n";
				else {
					std::cout << "  > Type :" << systemCheck.Type() << std::endl;
					for (int j = 0; j < checkMatrix.size(); j++) {
						if (!systemCheck.HasMatrix(checkMatrix[j].c_str()))
							std::cout << "  > " << checkMatrix[j] << ": NO " << std::endl;
						else {
							std::cout << "  > " << checkMatrix[j] << ": YES " << std::endl;
						}
					}

				}
			}
			else
				std::cout << "  > File does not exist." << std::endl;
		}
		exit(EXIT_FAILURE);
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
	dampingFileName = myFileName + "_X4.sim";

	stiffnessUMA_key = "GenericSystem_stiffness";
	massUMA_key = "GenericSystem_mass";
	structuralDampingUMA_key = "GenericSystem_structuralDamping";
	dampingUMA_key = "GenericSystem_viscousDamping";

	char * simFileStiffness = const_cast<char*>(stiffnessFileName.c_str());
	char * simFileMass = const_cast<char*>(massFileName.c_str());
	char * simFileStructD = const_cast<char*>(structDampingFileName.c_str());
	char * simFileDamping = const_cast<char*>(dampingFileName.c_str());

	std::cout << " > STIFFNESS SIM file      : " << simFileStiffness << std::endl;
	std::cout << " > MASS SIM file           : " << simFileMass << std::endl;
	std::cout << " > STRUCT DAMPING SIM file : " << simFileStructD << std::endl;
	std::cout << " > VISCOUS DAMPING SIM file: " << simFileDamping << std::endl;

	// Import all individual data
	importDatastructureSIM(simFileStiffness, stiffnessUMA_key, printDetails);
	importDatastructureSIM(simFileMass, massUMA_key, printDetails);
	importDatastructureSIM(simFileStructD, structuralDampingUMA_key, printDetails);
	importDatastructureSIM(simFileDamping, dampingUMA_key, printDetails);

	// Generate global map
	generateGlobalMap(printDetails);

	numNodes = nodeToGlobalMap.size();
	totalDOFs = 0;
	for (std::map<int, std::vector<int>>::iterator it = nodeToDofMap.begin(); it != nodeToDofMap.end(); ++it) {
		totalDOFs += it->second.size();
	}

	std::cout << " > Num Node: " << numNodes << " TotalDoFs: " << totalDOFs << std::endl;
	
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
	std::ifstream ifile(_fileName);
	bool flag = true;
	if (!ifile) {
		if (std::string(_matrixkey) == structuralDampingUMA_key)
		{
			std::cout << ">> StructuralDamping file not found and hence skipped." << std::endl;
			noStructuralDamping = true;
			flag = false;
		}
		else if (std::string(_matrixkey) == dampingUMA_key) {
			std::cout << ">> ViscousDamping file not found and hence skipped." << std::endl;
			noDamping = true;
			flag = false;
		}
		else {
			std::cout << ">> File not found." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	if (flag)
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
	else
		printf("; non-symmetric");
	printf("\n");
}
#endif

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

	if (!noStructuralDamping)
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

	if (!noDamping) {
		// Viscous Damping
		// Structural Damping
		uma_System system_VD(dampingFileName.c_str());
		uma_SparseMatrix smtx_VD;
		system_VD.SparseMatrix(smtx_VD, dampingUMA_key);
		uma_ArrayInt nodes_VD; smtx_VD.MapColumns(nodes_VD, uma_Enum::NODES);
		uma_ArrayInt ldofs_VD; smtx_VD.MapColumns(ldofs_VD, uma_Enum::DOFS);
		for (int col_VD = 0; col_VD < nodes_VD.Size(); col_VD++) {
			addEntryToNodeDofMap(nodes_VD[col_VD], ldofs_VD[col_VD]);
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

void SimuliaUMA::ImportSIM(const char* _matrixkey, const char* _fileName, bool _printToScreen, bool _printToFile, std::vector<int>& _ia, std::vector<int> &_ja, std::vector<STACCATOComplexDouble> &_values) {
#ifdef USE_SIMULIA_UMA_API
	std::cout << ">>> Import SIM File: " << _fileName << " UMA Key: " << _matrixkey << std::endl;
	bool flag = false;
	std::ifstream ifile(_fileName);
	if (!ifile) {
		if (std::string(_matrixkey) == structuralDampingUMA_key)
		{
			std::cout << " > StructuralDamping file not found and hence skipped." << std::endl;
			flag = true;
		}
		else if (std::string(_matrixkey) == dampingUMA_key) {
			std::cout << ">> ViscousDamping file not found and hence skipped." << std::endl;
			flag = true;
		}
		else {

			std::cout << " > File not found." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	if (!flag)
	{
		uma_System system(_fileName);
		if (system.IsNull()) {
			std::cout << ">> Error: System not defined.\n";
		}
		if (system.Type() != ads_GenericSystem) {
			std::cout << ">> Error: Not a Generic System.\n";
		}

		char * mapTypeName[] = { "DOFS", "NODES", "MODES", "ELEMENTS", "CASES", "Unknown" };
		// Check for matrix existence
		if (!system.HasMatrix(_matrixkey)) {
			std::cout << " >> Sparse matrix " << _matrixkey << " not found\n";
			exit(EXIT_FAILURE);
		}

		// Load the matrix
		uma_SparseMatrix smtx;
		system.SparseMatrix(smtx, _matrixkey);
		if (!smtx) {
			std::cout << " >> Sparse matrix " << _matrixkey << " cannot be accessed!\n";
			exit(EXIT_FAILURE);
		}

		if (_printToScreen)
			PrintMatrix(system, _matrixkey, _printToScreen, false);

		if (!smtx.IsSymmetric()) {
			std::cout << " Error: System not Symmetric" << std::endl;
			//exit(EXIT_FAILURE);
		}

		if (std::string(_matrixkey) == std::string(stiffnessUMA_key)) {
			std::cout << " > Importing Ke ..." << std::endl;
			loadDataFromSIM(smtx, _printToFile, "Staccato_Sparse_Stiffness", _ia, _ja, _values);
		}
		else if (std::string(_matrixkey) == std::string(massUMA_key)) {
			std::cout << " > Importing Me ..." << std::endl;
			loadDataFromSIM(smtx, _printToFile, "Staccato_Sparse_Mass", _ia, _ja, _values);
		}
		else if (std::string(_matrixkey) == std::string(structuralDampingUMA_key)) {
			std::cout << " > Importing SDe ..." << std::endl;
			loadDataFromSIM(smtx, _printToFile, "Staccato_Sparse_StructuralDamping", _ia, _ja, _values);
		}
		else if (std::string(_matrixkey) == std::string(dampingUMA_key)) {
			std::cout << " > Importing De ..." << std::endl;
			loadDataFromSIM(smtx, _printToFile, "Staccato_Sparse_ViscousDamping", _ia, _ja, _values);
		}
	}
#endif
}

void SimuliaUMA::getSparseMatrixCSR(std::string _matName, std::vector<int>& _ia, std::vector<int> &_ja, std::vector<STACCATOComplexDouble> &_values, bool _fileexp) {
	bool matdisp = false;

	if (_matName == "stiffness") {
		char * simFileStiffness = const_cast<char*>(stiffnessFileName.c_str());
		ImportSIM(stiffnessUMA_key, simFileStiffness, matdisp, _fileexp, _ia, _ja, _values);
	}
	else if (_matName == "mass") {
		char * simFileMass = const_cast<char*>(massFileName.c_str());
		ImportSIM(massUMA_key, simFileMass, matdisp, _fileexp, _ia, _ja, _values);
	}
	else if (_matName == "structuraldamping") {
		char * simFileStructD = const_cast<char*>(structDampingFileName.c_str());
		if (!noStructuralDamping)
			ImportSIM(structuralDampingUMA_key, simFileStructD, matdisp, _fileexp, _ia, _ja, _values);
	}
	else if (_matName == "viscousdamping") {
		char * simFileDamping = const_cast<char*>(dampingFileName.c_str());
		if (!noDamping)
			ImportSIM(dampingUMA_key, simFileDamping, matdisp, _fileexp, _ia, _ja, _values);
	}
}

void SimuliaUMA::loadDataFromSIM(const uma_SparseMatrix &_smtx, bool _printToFile, std::string _filePrefix, std::vector<int>& _ia, std::vector<int> &_ja, std::vector<STACCATOComplexDouble> &_values) {
	prepMapCSR.clear();

	// Pre-allocation
	int nnz = _smtx.NumEntries();
	_ia.reserve(totalDOFs + 1);
	_ja.reserve(nnz);
	_values.reserve(nnz);

	uma_SparseIterator iter(_smtx);
	int row, col; double val;
	int count = 0;
	uma_ArrayInt nodes; _smtx.MapColumns(nodes, uma_Enum::NODES); // test array
	uma_ArrayInt ldofs; _smtx.MapColumns(ldofs, uma_Enum::DOFS);  // test vector

	int rem_row = -22;
	int nnz_row = 0;

	for (iter.First(); !iter.IsDone(); iter.Next(), count++) {
		iter.Entry(row, col, val);

		int fnode_row = nodes[row];
		int fdof_row = ldofs[row];
		int fnode_col = nodes[col];
		int fdof_col = ldofs[col];

		int globalrow = -1;
		int globalcol = -1;

		for (int i = 0; i < nodeToDofMap[fnode_row].size(); i++)
		{
			if (nodeToDofMap[fnode_row][i] == fdof_row)
				globalrow = nodeToGlobalMap[fnode_row][i];
		}
		for (int i = 0; i < nodeToDofMap[fnode_col].size(); i++)
		{
			if (nodeToDofMap[fnode_col][i] == fdof_col)
				globalcol = nodeToGlobalMap[fnode_col][i];
		}

		// Exception for Internal DOFs. Exchange Row and Column Index
		if (globalrow > globalcol && fnode_row >= 1000000000) {
			int temp = globalrow;
			globalrow = globalcol;
			globalcol = temp;
		}
		prepMapCSR[globalrow][globalcol + 1] = val;

		if (globalrow == globalcol) {
			if (val >= 1e36) {
				std::cout << "Error: DBC pivot found!";
				exit(EXIT_FAILURE);
			}
		}
	}

	for (size_t iTotalDof = 0; iTotalDof < totalDOFs; iTotalDof++)
	{
		_ia.push_back(_values.size() + 1);
		auto search = prepMapCSR.find(iTotalDof);
		if (search != prepMapCSR.end()) {
			for (std::map<int, double>::iterator it2 = prepMapCSR[iTotalDof].begin(); it2 != prepMapCSR[iTotalDof].end(); ++it2) {
				_ja.push_back(it2->first);
				_values.push_back({ it2->second,0 });
			}
		}
	}
	_ia.push_back(_values.size() + 1);

	if (_printToFile) {
		std::cout << ">> Printing to " << _filePrefix <<" Matrix CSR Format..." << std::endl;
		AuxiliaryFunctions::writeIntegerVectorDatFormat(_filePrefix + "_CSR_IA.dat", _ia);
		AuxiliaryFunctions::writeIntegerVectorDatFormat(_filePrefix + "_CSR_JA.dat", _ja);

		AuxiliaryFunctions::writeMKLComplexVectorDatFormat(_filePrefix + "_CSR_MAT.dat", _values);
	}
	prepMapCSR.clear();
}

std::map<int, std::vector<int>> SimuliaUMA::getNodeToDofMap() {
	return nodeToDofMap;
}

std::map<int, std::vector<int>> SimuliaUMA::getNodeToGlobalMap() {
	return nodeToGlobalMap;
}

#ifdef USE_SIMULIA_UMA_API
void SimuliaUMA::PrintMatrix(const uma_System &system, const char *matrixName, bool _printToScreen, bool _printToFile)
{
	char * mapTypeName[] = { "DOFS", "NODES", "MODES", "ELEMENTS", "CASES", "Unknown" };
	if (!system.HasMatrix(matrixName)) {
		return;
		printf("\nSparse matrix %s not found\n", matrixName);
	}

	uma_SparseMatrix smtx;
	system.SparseMatrix(smtx, matrixName);
	if (!smtx) {
		printf("\nSparse matrix %s cannot be not accessed\n", matrixName);
		return;
	}

	printf(" Matrix %s - sparse type\n", matrixName);
	printf("  domain: rows %s, columns %s\n", mapTypeName[smtx.TypeRows()], mapTypeName[smtx.TypeColumns()]);
	printf("  size: rows %i, columns %i, entries %i", smtx.NumRows(), smtx.NumColumns(), smtx.NumEntries());
	if (smtx.IsSymmetric())
		printf("; symmetric");
	else
		printf("; non-symmetric");
	printf("\n");
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
				printf("  %2i,%-2i:%g", row+1, col+1, val);
			else
				printf("  *%2i,%-2i:%g", row+1, col+1, val);

			lastRow = row;
		}
		printf("\n");

		// Map column DOFS to user nodes and dofs
		if (smtx.TypeColumns() != uma_Enum::DOFS)
			return;
		printf("\n  map columns [column: node-dof]:");
		uma_ArrayInt     nodes; smtx.MapColumns(nodes, uma_Enum::NODES); // test array
		std::vector<int> ldofs; smtx.MapColumns(ldofs, uma_Enum::DOFS);  // test vector
		for (int col = 0; col < nodes.Size(); col++) {
			if (col % 10 == 0)
				printf("\n");
			printf(" %3i:%3i-%1i", col+1, nodes[col], ldofs[col]);
		}
		printf("\n");
	}

	if (_printToFile) {
		std::ofstream myfile;
		myfile.open(std::string(matrixName) +".mtx");
		std::cout << ">>> Writing file to: " << std::string(matrixName) + ".mtx" << std::endl;
		myfile.precision(std::numeric_limits<double>::digits10 + 1);
		myfile << std::scientific;

		for (iter.First(); !iter.IsDone(); iter.Next()) {
			iter.Entry(row, col, val);
			myfile << row << " " << col << " " << val << std::endl;
		}
		myfile.close();
	}
}
#endif