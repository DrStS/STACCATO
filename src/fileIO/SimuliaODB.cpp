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
#include "SimuliaODB.h"
#include "Message.h"
#include "HMesh.h"


/// SIMULIA includes
#include <odb_API.h>
#include <odb_Coupling.h>
#include <odb_MPC.h>
#include <odb_ShellSolidCoupling.h>

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
				elementTopo.resize(4);
				for (int j = 0; j < elemConSize; j++){
#ifdef DEBUG_OUTPUT
					infoOut << " " << conn[j];
#endif
					elementTopo[j] = conn[j];
				}
#ifdef DEBUG_OUTPUT
				infoOut << " ] " << std::endl;
#endif
				myHMesh->addElement(aElement.label(), STACCATO_PlainStrain4Node2D, elementTopo);
			}
		}
		myHMesh->buildDataStructure();
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



