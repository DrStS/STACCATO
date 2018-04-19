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
#include "memWatcher.h"
#include "HMesh.h"

//UMA
#ifdef SIMULIA_API_ON
#include <ads_CoreFESystemC.h>
#include <ads_CoreMeshC.h>
#include <uma_System.h>
#include <uma_SparseMatrix.h>
#include <uma_ArrayInt.h>
#endif

//XML
#include "MetaDatabase.h"

//#define DEBUG

SimuliaUMA::SimuliaUMA(std::string _fileName, HMesh& _hMesh, int _partId) : myHMesh(&_hMesh) {
	myFileName = _fileName;
	std::cout << ">> SIM Reader initialized for file " << myFileName << std::endl;
	numNodes = 0;
	numDoFperNode = 0;
	openFile();
	myHMesh->hasParts = true;
	myHMesh->isSIM = true;
}

SimuliaUMA::~SimuliaUMA() {
}

void SimuliaUMA::openFile() {
#ifdef SIMULIA_API_ON
	std::vector<std::string> mapTypeName;
	mapTypeName.push_back("DOFS");
	mapTypeName.push_back("NODES");
	mapTypeName.push_back("MODES");
	mapTypeName.push_back("ELEMENTS");
	mapTypeName.push_back("CASES");
	mapTypeName.push_back("Unknown");

	// Error --
	char * simFile = const_cast<char*>(myFileName.c_str());
	printf("SIM file: %s\n", simFile);
	// -- Error

	uma_System system(simFile);

	std::cout << "\n>> Importing SIM to HMesh ..." << std::endl;

	char* matrixName = "GenericSystem_stiffness";
	if (!system.HasMatrix(matrixName)) {
		return;
		printf("\nSparse matrix %s not found\n", matrixName);
	}
	uma_SparseMatrix smtx;
	system.SparseMatrix(smtx, matrixName);

	// Map column DOFS to user nodes and dofs
	if (smtx.TypeColumns() != uma_Enum::DOFS)
		return;

	// Map column DOFS to user nodes and dofs
	if (smtx.TypeColumns() != uma_Enum::DOFS)
		return;
	uma_ArrayInt     nodes; smtx.MapColumns(nodes, uma_Enum::NODES); // test array
	std::vector<int> ldofs; smtx.MapColumns(ldofs, uma_Enum::DOFS);  // test vector
	
	bool flag = true;
	for (int col = 0; col < nodes.Size(); col+= numDoFperNode) {
		std::vector<int> temp;
		int i = 0;
		i = col;
		temp.push_back(nodes[i]);
		while (nodes[col] == nodes[i] && i < nodes.Size()) {
			temp.push_back(ldofs[i]);
			i++;
			if (i >= nodes.Size())
				break;
		}
		simNodeMap.push_back(temp);
		if (flag) {
			numDoFperNode = simNodeMap[0].size() - 1;
		}
	}

	numNodes = simNodeMap.size();

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

	for (int i = 1; i <= numNodes; i++)	{
		elementTopo.push_back(lastHMeshNodeLabel +i);
		myHMesh->addNode(lastHMeshNodeLabel + i, coord[0], coord[1], coord[2]);
	}

	if (lastHMeshNodeLabel != 0)
		std::cout << "\n>> Simulia UMA: Adding Nodes and Element in append mode.\n";
	else
		std::cout << "\n>> Simulia UMA: Adding Nodes and Element as new model.\n";

	// Element Labelling
	int lastHMeshElementLabel = 0;
	if (myHMesh->hasParts) {
		lastHMeshElementLabel = myHMesh->getElementLabels().back();
	}
	myHMesh->addElement(lastHMeshElementLabel+1, STACCATO_UmaElement, elementTopo);

	// Printing
	std::cout << "\nMap for UMA-SIM:\n";
	for (int col = 0; col < nodes.Size(); col++) {
		if (col % 10 == 0)
			printf("\n");
		printf(" %3i:%3i-%1i", col, nodes[col], ldofs[col]);
	}
	printf("\n");
#endif
}



