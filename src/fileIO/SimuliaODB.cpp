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
#include <odb_SectionTypes.h>

//XML
#include "MetaDatabase.h"

//#define DEBUG

SimuliaODB::SimuliaODB(std::string _fileName, HMesh& myHMesh) : myHMesh(&myHMesh) {
	myFileName = _fileName;
	std::cout << ">> ODB Reader initialized for file " << myFileName << std::endl;
	odb_initializeAPI();
	openFile();
	myHMesh.hasParts = true;
}

SimuliaODB::~SimuliaODB() {
	odb_finalizeAPI();
}

void SimuliaODB::openFile() {
	try {
#ifdef DEBUG_OUTPUT
		infoOut << "Open OBD file: " << myFileName << std::endl;
#endif
		odb_Odb& odb = openOdb(odb_String(myFileName.c_str()));
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

			// Check for Imports
			for (STACCATO_XML::FILEIMPORT_const_iterator iFileImport(MetaDatabase::getInstance()->xmlHandle->FILEIMPORT().begin());
				iFileImport != MetaDatabase::getInstance()->xmlHandle->FILEIMPORT().end();
				++iFileImport)
			{
				if (std::string(iFileImport->Type()->c_str()) == "AbqODB") {
					for (int iImport = 0; iImport < iFileImport->IMPORT().size(); iImport++) {
						std::string importType = iFileImport->IMPORT()[iImport].Type()->c_str();

						if (importType == "Nodes") {
							if (std::string(iFileImport->IMPORT()[iImport].LIST()->c_str()) == "ALL") {
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
							}
							else
								std::cerr << "Unrecognized Node Import List.\n";
						}
						else if (importType == "Elements") {

							std::string translateSource = iFileImport->IMPORT()[iImport].TRANSLATETO().begin()->Source()->c_str();
							std::string translateTarget = iFileImport->IMPORT()[iImport].TRANSLATETO().begin()->Target()->c_str();

							if (std::string(iFileImport->IMPORT()[iImport].LIST()->c_str()) == "ALL") {

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
									for (int j = 0; j < elemConSize; j++) {
#ifdef DEBUG_OUTPUT
										infoOut << " " << conn[j];
#endif
										elementTopo[j] = conn[j];
									}
#ifdef DEBUG_OUTPUT
									infoOut << " ] " << std::endl;

#endif
									if (std::string(aElement.type().CStr()) == translateSource) {
										if (translateTarget == "STACCATO_Tetrahedron10Node3D")
											myHMesh->addElement(aElement.label(), STACCATO_Tetrahedron10Node3D, elementTopo);
										else if (translateTarget == "STACCATO_PlainStress4Node2D")
											myHMesh->addElement(aElement.label(), STACCATO_PlainStress4Node2D, elementTopo);
										else
											std::cerr << "STACCATO cannot recognize this element: " << translateTarget << std::endl;
									}
								}
							}
							else
								std::cerr << "Unrecognized Element Import List.\n";
						}
						else if (importType == "Sets") {
							// SETS
							// NODES
							for (int i = 0; i < iFileImport->IMPORT()[iImport].NODE().begin()->TRANSLATETO().size(); i++) {

								std::string translateSource = iFileImport->IMPORT()[iImport].NODE().begin()->TRANSLATETO()[i].Source()->c_str();
								std::string translateTarget = iFileImport->IMPORT()[iImport].NODE().begin()->TRANSLATETO()[i].Target()->c_str();

								odb_SetRepositoryIT setIter(odb.rootAssembly().nodeSets());
								for (setIter.first(); !setIter.isDone() && setIter.currentValue().type() == odb_Enum::NODE_SET; setIter.next()) {

									odb_Set set = setIter.currentValue();
									int setSize = set.size();

									if (std::string(set.name().CStr()) == translateSource) {
										odb_SequenceString names = set.instanceNames();
										int numInstances = names.size();

										int i;
										for (i = 0; i < numInstances; i++)
										{
											odb_String name = names.constGet(i);
											const odb_SequenceNode& nodesInMySet = set.nodes(name);
											int n_max = nodesInMySet.size();
											std::vector<int> nodeLabels;
											for (int n = 0; n < n_max; n++) {
												nodeLabels.push_back(nodesInMySet.node(n).label());
											}
											myHMesh->addNodeSet(translateTarget, nodeLabels);
										}
									}
								}
							}
							// ELEMENTS
							for (int i = 0; i < iFileImport->IMPORT()[iImport].ELEMENT().begin()->TRANSLATETO().size(); i++) {

								std::string translateSource = iFileImport->IMPORT()[iImport].ELEMENT().begin()->TRANSLATETO()[i].Source()->c_str();
								std::string translateTarget = iFileImport->IMPORT()[iImport].ELEMENT().begin()->TRANSLATETO()[i].Target()->c_str();

								odb_SetRepositoryIT setIter(odb.rootAssembly().nodeSets());

								for (setIter.first(); !setIter.isDone() && setIter.currentValue().type() == odb_Enum::ELEMENT_SET; setIter.next()) {

									odb_Set set = setIter.currentValue();
									int setSize = set.size();

									if (std::string(set.name().CStr()) == translateSource) {
										odb_SequenceString names = set.instanceNames();
										int numInstances = names.size();

										int i;
										for (i = 0; i < numInstances; i++)
										{
											odb_String name = names.constGet(i);

											const odb_SequenceElement& elemsInMySet = set.elements(name);
											int n_max = elemsInMySet.size();
											for (int n = 0; n < n_max; n++)
											{
												int elemConSize;
												const int* const conn = elemsInMySet.element(n).connectivity(elemConSize);
												std::vector<int> elemLabels;
												for (int j = 0; j < elemConSize; j++)
													elemLabels.push_back(elemsInMySet.element(n).label());
												myHMesh->addElemSet(translateTarget, elemLabels);
											}
										}
									}
								}
							}
						}
						else
							std::cerr << importType << " is not yet Supported or is Incorrect.\n";
					}

				}
			}

		}
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



