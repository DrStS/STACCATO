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
#include "SimuliaODB.h"
#include "Message.h"
#include "memWatcher.h"
#include "HMesh.h"


/// SIMULIA includes
#include <odb_API.h>
#include <odb_Coupling.h>
#include <odb_MPC.h>
#include <odb_ShellSolidCoupling.h>
#include <odb_Enum.h>
//UMA
#include <ads_CoreFESystemC.h>
#include <uma_System.h>
#include <uma_IncoreMatrix.h>
#include <uma_Matrix.h>
#include <uma_Enum.h>
#include <uma_ArrayInt.h>

//#define DEBUG

SimuliaODB::SimuliaODB() {
	odb_initializeAPI();
}

SimuliaODB::~SimuliaODB() {
	odb_finalizeAPI();
}

void SimuliaODB::openODBFile(std::string _obdFilePath) {
	try {
#ifdef DEBUG_OUTPUT
		infoOut << "Open OBD file: " << _obdFilePath << std::endl;
#endif
		odb_Odb& odb = openOdb(odb_String(_obdFilePath.c_str()));
#ifdef DEBUG_OUTPUT
		infoOut << odb.name().CStr() << " '__________" << std::endl;
		infoOut << "analysisTitle: " << odb.analysisTitle().CStr() << std::endl;
		infoOut << "description: " << odb.description().CStr() << std::endl;
#endif
		odb_InstanceRepository& instanceRepo = odb.rootAssembly().instances();
		odb_InstanceRepositoryIT iter(instanceRepo);

		for (iter.first(); !iter.isDone(); iter.next())
		{
			odb_Instance& inst = instanceRepo[iter.currentKey()];
			const odb_SequenceNode& nodes = inst.nodes();
			int numOfNodes = nodes.size();
			const odb_SequenceElement& elements = inst.elements();
			int numOfElements = elements.size();

			// Works for one instance only
			myHMesh = new HMesh("default");

			//Nodes
#ifdef DEBUG_OUTPUT
			infoOut << "Total number of nodes: " << numOfNodes << std::endl;
#endif
			for (int i = 0; i < numOfNodes; i++)
			{
				const odb_Node aNode = nodes.node(i);
				const float * const coords = aNode.coordinates();
#ifdef DEBUG_OUTPUT
				char formattedOut[256];
				sprintf(formattedOut, " %9d [%10.6f %10.6f %10.6f]", aNode.label(),
					coords[0], coords[1], coords[2]);
				infoOut << formattedOut << std::endl;
#endif
				myHMesh->addNode(aNode.label(), coords[0], coords[1], coords[2]);
			}
			//Elements
#ifdef DEBUG_OUTPUT
			infoOut << "Total number of elements: " << numOfElements << std::endl;
#endif
			for (int i = 0; i < numOfElements; i++)
			{
				const odb_Element aElement = elements.element(i);



#ifdef DEBUG_OUTPUT
				infoOut << aElement.label() << " " << aElement.type().CStr() << " [";
#endif
				int elemConSize;
				const int* const conn = aElement.connectivity(elemConSize);
				std::vector<int> elementTopo;
				elementTopo.resize(elemConSize);
				for (int j = 0; j < elemConSize; j++){
#ifdef DEBUG_OUTPUT
					infoOut << " " << conn[j];
#endif
					elementTopo[j] = conn[j];
				}
#ifdef DEBUG_OUTPUT
				infoOut << " ] " << std::endl;
#endif
				if (aElement.geometry() == odb_Enum::QUAD4) {
					myHMesh->addElement(aElement.label(), STACCATO_PlainStress4Node2D, elementTopo);

				}
				else if (aElement.geometry() == odb_Enum::TETRA10) {
					myHMesh->addElement(aElement.label(), STACCATO_Tetrahedron10Node3D, elementTopo);
				}
				
				
			}
		}

		debugOut << "SimuliaODB::openODBFile: " << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
		myHMesh->buildDataStructure();
		debugOut << "SimuliaODB::openODBFile: " << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
		odb.close();//Change datastrc here HMesh should node be a member of odb
	}
	catch (odb_BaseException& exc) {
		errorOut << "odbBaseException caught" << std::endl;
		errorOut << "Abaqus error message: " << exc.UserReport().CStr() << std::endl;
		qFatal(exc.UserReport().CStr());
		//ToDo add error handling
	}
	catch (...) {
		errorOut << "Unknown Exception." << std::endl;
	}
}



