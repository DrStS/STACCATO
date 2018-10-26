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
	std::cout << ">> SIM Reader initialized for file " << myFileName << std::endl;

	openFile();
}

SimuliaUMA::~SimuliaUMA() {
}

void SimuliaUMA::openFile() {
	myUMASystem = new uma_System(const_cast<char*>(myFileName.c_str()));
	if (myUMASystem->IsNull()) {
		std::cout << ">> Error: System not defined.\n";
		exit(EXIT_FAILURE);
	}
	if (myUMASystem->Type() != ads_GenericSystem) {
		std::cout << ">> Error: Not a Generic System.\n";
		exit(EXIT_FAILURE);
	}
}

int SimuliaUMA::collectDatastructureSIM(char* _key, std::map<int, std::vector<int>> &_dofMap) {
	std::cout << ">> Sensing SIM File: " << myFileName << " UMA Key: " << _key << std::endl;
	char * mapTypeName[] = { "DOFS", "NODES", "MODES", "ELEMENTS", "CASES", "Unknown" };
	// Check for matrix existence
	if (!myUMASystem->HasMatrix(_key)) {
		std::cout << " >> Sparse matrix " << _key << " not found\n";
		exit(EXIT_FAILURE);
	}

	// Load the matrix
	uma_SparseMatrix smtx;
	myUMASystem->SparseMatrix(smtx, _key);
	if (!smtx) {
		std::cout << " >> Sparse matrix " << _key << " cannot be accessed!\n";
		exit(EXIT_FAILURE);
	}

	printf(" Matrix %s - sparse type\n", _key);
	printf("  domain: rows %s, columns %s\n", mapTypeName[smtx.TypeRows()], mapTypeName[smtx.TypeColumns()]);
	printf("  size: rows %i, columns %i, entries %i", smtx.NumRows(), smtx.NumColumns(), smtx.NumEntries());
	if (smtx.IsSymmetric())
		printf("; symmetric");
	else {
		printf("; non-symmetric");
	}
	printf("\n");

	uma_ArrayInt nodes; smtx.MapColumns(nodes, uma_Enum::NODES);
	uma_ArrayInt ldofs; smtx.MapColumns(ldofs, uma_Enum::DOFS);
	
	// Create pool of all available nodes and dofs
	for (int col = 0; col < nodes.Size(); col++) {
		auto search = _dofMap.find(nodes[col]);
		if (search != _dofMap.end()) {
			bool found = false;
			for (int i = 0; i < search->second.size(); i++)
			{
				if (ldofs[col] == search->second[i]) {
					found = true;
				}
			}
			if (!found) {
				search->second.push_back(ldofs[col]);
			}
		}
		else {
			_dofMap[nodes[col]] .push_back(ldofs[col]);
		}
	}
	return 1;
}

bool SimuliaUMA::isUmaSymmetric(char* _key) {
	uma_SparseMatrix smtx;
	myUMASystem->SparseMatrix(smtx, _key);

	return smtx.IsSymmetric();
}

void SimuliaUMA::setCouplingMatFSI(std::map<int, std::map<int, double>> &_KASI) {
	myCouplingMatFSI.clear();
	myCouplingMatFSI = _KASI;
}

std::map<int, std::map<int, double>> SimuliaUMA::getCouplingMatFSI() {
	return myCouplingMatFSI;
}

void SimuliaUMA::loadSIMforUMA(std::string _key, std::vector<int>& _ia, std::vector<int> &_ja, std::vector<STACCATOComplexDouble> &_values, std::map<int, std::vector<int>> &_dofMap, std::map<int, std::vector<int>> &_globalMap, int _readMode, bool _flagUnymRead, bool _printToFile, int _numrows) {
	std::cout << " > Reading: " << _key << " in mode [unsy, rmod]: ["<< _flagUnymRead<< "," << _readMode << "]..."<<std::endl;
	int FF_start = -1; int FF_end = -1;
	int SS_start = -1; int SS_end = -1;
	if (_readMode > 0)
	{
		// get Fluid domain dofs and structre domain dofs
		std::map<int, std::vector<int>> fluidMap;
		std::map<int, std::vector<int>> structMap;
		for (std::map<int, std::vector<int>>::iterator it = _globalMap.begin(); it != _globalMap.end(); ++it) {
			if (it->first < 1000000000 && it->second.size() == 1)
				fluidMap[it->first] = it->second;
			else if (it->first < 1000000000 && (it->second.size() == 3 || it->second.size() == 6))
				structMap[it->first] = it->second;
 		}
		FF_start = fluidMap.begin()->second[0];		FF_end = fluidMap.rbegin()->second[0];
		SS_start = structMap.begin()->second[0];	SS_end = structMap.rbegin()->second[structMap.rbegin()->second.size() - 1];
		std::cout << ">> FSI: FF [" << FF_start << ":" << FF_end << "," << FF_end - FF_start << "], SS [" << SS_start << ":" << SS_end << "," << SS_end - SS_start << "]" << std::endl;
	}

	mySystemMatrixMapCSR.clear();

	// Pre-allocation
	uma_SparseMatrix smtx;
	myUMASystem->SparseMatrix(smtx, const_cast<char*>(_key.c_str()));

	int nnz = smtx.NumEntries();
	_ia.reserve(_numrows + 1);
	_ja.reserve(nnz);
	_values.reserve(nnz);

	uma_SparseIterator iter(smtx);
	int row, col; double val;
	int count = 0;
	uma_ArrayInt nodes; smtx.MapColumns(nodes, uma_Enum::NODES); // test array
	uma_ArrayInt ldofs; smtx.MapColumns(ldofs, uma_Enum::DOFS);  // test vector

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

		for (int i = 0; i < _dofMap[fnode_row].size(); i++)
		{
			if (_dofMap[fnode_row][i] == fdof_row)
				globalrow = _globalMap[fnode_row][i];
		}
		for (int i = 0; i < _dofMap[fnode_col].size(); i++)
		{
			if (_dofMap[fnode_col][i] == fdof_col)
				globalcol = _globalMap[fnode_col][i];
		}

		// Exception for Internal DOFs. Exchange Row and Column Index
		if(fnode_row >= 1000000000 || fnode_col>= 1000000000) {
				int temp = globalrow;
				globalrow = globalcol;
				globalcol = temp;
			}

		if (_readMode == 1)	// Extraction
		{
			if ((globalrow >= FF_start && globalrow <= FF_end) && (globalcol >= SS_start && globalcol <= SS_end)) {
				myCouplingMatFSI[globalrow][globalcol + 1] = -val;		// (-)K_ASI  negation is done while extraction
				val = 0;
			}
		}

		mySystemMatrixMapCSR[globalrow][globalcol + 1] = val;
		if (_flagUnymRead)		// Special lower triangular entries for symmetric matrices
			mySystemMatrixMapCSR[globalcol][globalrow + 1] = val;


		if (globalrow == globalcol) {
			if (val >= 1e36) {
				std::cout << "Error: DBC pivot found!"<<std::endl;
				//exit(EXIT_FAILURE);
			}
		}
	}

	if (_readMode==2)	{
		for (std::map<int, std::map<int, double>>::iterator it = myCouplingMatFSI.begin(); it != myCouplingMatFSI.end(); ++it) {
			mySystemMatrixMapCSR[it->first] = it->second;
		}
	}

	for (size_t iTotalDof = 0; iTotalDof < _numrows; iTotalDof++)
	{
		_ia.push_back(_values.size() + 1);
		auto search = mySystemMatrixMapCSR.find(iTotalDof);
		if (search != mySystemMatrixMapCSR.end()) {
			for (std::map<int, double>::iterator it2 = mySystemMatrixMapCSR[iTotalDof].begin(); it2 != mySystemMatrixMapCSR[iTotalDof].end(); ++it2) {
				_ja.push_back(it2->first);
				_values.push_back({ it2->second,0 });
			}
		}
	}
	_ia.push_back(_values.size() + 1);

	if (_printToFile) {
		std::string exportFilePrefix = "Staccato_Sparse_" + _key;
		std::cout << ">> Printing to " << exportFilePrefix << " Matrix CSR Format..." << std::endl;
		AuxiliaryFunctions::writeIntegerVectorDatFormat(exportFilePrefix + "_CSR_IA.dat", _ia);
		AuxiliaryFunctions::writeIntegerVectorDatFormat(exportFilePrefix + "_CSR_JA.dat", _ja);

		AuxiliaryFunctions::writeMKLComplexVectorDatFormat(exportFilePrefix + "_CSR_MAT.dat", _values);
	}
	mySystemMatrixMapCSR.clear();
	std::cout << " > Reading: " << _key << " in mode [unsy, rmod]: [" << _flagUnymRead << "," << _readMode << "]... Finished."<<std::endl;
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