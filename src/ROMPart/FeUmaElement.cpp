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
#include "FeUmaElement.h"
#include "Material.h"
#include "Message.h"
#include "MathLibrary.h"

#ifdef USE_SIMULIA_UMA_API
#include <ads_CoreFESystemC.h>
#include <ads_CoreMeshC.h>
#include <uma_System.h>
#include <uma_SparseMatrix.h>
#include <uma_ArrayInt.h>
#include <uma_IncoreMatrix.h>
#endif

#include <iostream>
#include <stdio.h>
#include <iomanip>

//XML
#include "MetaDatabase.h"

FeUmaElement::FeUmaElement(Material *_material) : FeElement(_material) {
	nodeToDofMap = MetaDatabase::getInstance()->nodeToDofMapMeta;
	nodeToGlobalMap = MetaDatabase::getInstance()->nodeToGlobalMapMeta;

	int numNodes = nodeToGlobalMap.size();
	totalDOFs = 0;
	for (std::map<int, std::vector<int>>::iterator it = nodeToDofMap.begin(); it != nodeToDofMap.end(); ++it) {
		totalDOFs += it->second.size();
	}
	mySparseSDReal = new MathLibrary::SparseMatrix<double>(totalDOFs, true, true);

	std::cout << " >> UMA Element Created with TotalDoFs: " << totalDOFs << std::endl;
}

FeUmaElement::~FeUmaElement() {
}

void FeUmaElement::computeElementMatrix(const double* _eleCoords) {
	bool matdisp = false;
	bool fileexp = false;

	std::string simFileK = MetaDatabase::getInstance()->simFile + "_X1.sim";
	const char * simFileStiffness = simFileK.c_str();
	std::string simFileM = MetaDatabase::getInstance()->simFile + "_X2.sim";
	const char * simFileMass = simFileM.c_str();
	std::string simFileStrD = MetaDatabase::getInstance()->simFile + "_X3.sim";
	const char * simFileSD = simFileStrD.c_str();
	
	bool structDampingExists = false;

	std::ifstream ifile(simFileSD);
	if (ifile) 
		structDampingExists = true;


	stiffnessUMA_key = "GenericSystem_stiffness";
	massUMA_key = "GenericSystem_mass";
	structuralDampingUMA_key = "GenericSystem_structuralDamping";
#ifdef USE_SIMULIA_UMA_API
//DebugSIM(stiffnessUMA_key, simFileStiffness, matdisp, fileexp);
//DebugSIM(massUMA_key, simFileMass, matdisp, fileexp);
ImportSIM(stiffnessUMA_key, simFileStiffness, matdisp, fileexp);
ImportSIM(massUMA_key, simFileMass, matdisp, fileexp);
ImportSIM(structuralDampingUMA_key, simFileSD, matdisp, fileexp);
#endif
}

#ifdef USE_SIMULIA_UMA_API
void FeUmaElement::PrintMatrix(const uma_System &system, const char *matrixName, bool _printToScreen, bool _printToFile)
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
				printf("  %2i,%-2i:%g", row, col, val);
			else
				printf("  *%2i,%-2i:%g", row, col, val);

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
			printf(" %3i:%3i-%1i", col, nodes[col], ldofs[col]);
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

void FeUmaElement::DebugSIM(const char* _matrixkey, const char* _fileName, bool _printToScreen, bool _printToFile) {
#ifdef USE_SIMULIA_UMA_API
	std::cout << ">>> Debug SIM File: " << _fileName << " UMA Key: " << _matrixkey << std::endl;
	uma_System system(_fileName);
	if (system.IsNull()) {
		std::cout << ">>> Error: System not defined.\n";
	}
	if (system.Type() != ads_GenericSystem) {
		std::cout << "Error: Struc. Damping Not a Generic System.\n";
	}

	PrintMatrix(system, _matrixkey, _printToScreen, _printToFile);
#endif
}

void FeUmaElement::ImportSIM(const char* _matrixkey, const char* _fileName, bool _printToScreen, bool _printToFile) {
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

		extractData(system, _matrixkey, _printToScreen, _printToFile);
	}
#endif
}

#ifdef USE_SIMULIA_UMA_API
void FeUmaElement::extractData(const uma_System &system, const char *matrixName, bool _printToScreen, bool _printToFile) {
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

	if (_printToScreen)
	{
		printf(" Matrix %s - sparse type\n", matrixName);
		printf("  domain: rows %s, columns %s\n", mapTypeName[smtx.TypeRows()], mapTypeName[smtx.TypeColumns()]);
		printf("  size: rows %i, columns %i, entries %i", smtx.NumRows(), smtx.NumColumns(), smtx.NumEntries());
		if (smtx.IsSymmetric())
			printf("; symmetric");
		else {
			std::cout << ";unsymmetric" << std::endl;
			exit(EXIT_FAILURE);
		}
		printf("\n");
	}
	if (!smtx.IsSymmetric()){
		std::cout << " Error: System not Symmetric" << std::endl;
		exit(EXIT_FAILURE);
	}

	uma_SparseIterator iter(smtx);
	int row, col; double val;
	int count = 0;
	uma_ArrayInt nodes; smtx.MapColumns(nodes, uma_Enum::NODES); // test array
	uma_ArrayInt ldofs; smtx.MapColumns(ldofs, uma_Enum::DOFS);  // test vector

	if (std::string(matrixName) == std::string(stiffnessUMA_key)) {
		std::cout << " > Importing Ke ..." << std::endl;
		mySparseKReal = new MathLibrary::SparseMatrix<double>(totalDOFs, true, true);

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
				if(nodeToDofMap[fnode_row][i] == fdof_row)
					globalrow = nodeToGlobalMap[fnode_row][i];
			}
			for (int i = 0; i < nodeToDofMap[fnode_col].size(); i++)
			{
				if (nodeToDofMap[fnode_col][i] == fdof_col)
					globalcol = nodeToGlobalMap[fnode_col][i];
			}

			//std::cout << "Entry @ " << globalrow << " , " << globalcol << std::endl;
			if (globalrow > globalcol) {
				int temp = globalrow;
				globalrow = globalcol;
				globalcol = temp;
			}

			(*mySparseKReal)(globalrow, globalcol) = val;
			myK_row.push_back(globalrow);
			myK_col.push_back(globalcol);			

			if (globalrow == globalcol) {
				if (val >= 1e36) {
					std::cout << "Error: DBC pivot found!";
					exit(EXIT_FAILURE);						
				}
			}
		}
		if (_printToFile) {
			std::cout << ">> Printing to file Staccato_Sparse_Stiffness.mtx..." << std::endl;
			(*mySparseKReal).writeSparseMatrixToFile("Staccato_Sparse_Stiffness", "mtx");
		}
	}
	else if (std::string(matrixName) == std::string(massUMA_key)) {
		std::cout << " > Importing Me ..." << std::endl;
		mySparseMReal = new MathLibrary::SparseMatrix<double>(totalDOFs, true, true);

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

			//std::cout << "Entry @ " << globalrow << " , " << globalcol << std::endl;
			if (fnode_row >= 1e9 || fnode_col >= 1e9) {
				if (val != 0) {
					std::cout << ">> ERROR: Has entry." << std::endl;
					exit(EXIT_FAILURE);
				}
			}

			if (globalrow <= globalcol)
				(*mySparseMReal)(globalrow, globalcol) = val;
			else
				(*mySparseMReal)(globalcol, globalrow) = val;

			myM_row.push_back(globalrow);
			myM_col.push_back(globalcol);
		}

		if (_printToFile) {
			std::cout << ">> Printing to file Staccato_Sparse_Mass.mtx..." << std::endl;
			(*mySparseMReal).writeSparseMatrixToFile("Staccato_Sparse_Mass", "mtx");
		}
	}
	else if (std::string(matrixName) == std::string(structuralDampingUMA_key)) {
		std::cout << " > Importing SDe ..." << std::endl;

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

			//std::cout << "Entry @ " << globalrow << " , " << globalcol << std::endl;
			if (globalrow > globalcol) {
				int temp = globalrow;
				globalrow = globalcol;
				globalcol = temp;
			}

			(*mySparseSDReal)(globalrow, globalcol) = val;
			mySD_row.push_back(globalrow);
			mySD_col.push_back(globalcol);

			if (globalrow == globalcol) {
				if (val >= 1e36) {
					std::cout << "Error: DBC pivot found!";
					exit(EXIT_FAILURE);
				}
			}
		}
		if (_printToFile) {
			std::cout << ">> Printing to file Staccato_Sparse_StructuralDamping.mtx..." << std::endl;
			(*mySparseSDReal).writeSparseMatrixToFile("Staccato_Sparse_StructuralDamping", "mtx");
		}
	}
}
#endif