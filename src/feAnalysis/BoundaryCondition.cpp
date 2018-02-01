/*  Copyright &copy; 2018, Stefan Sicklinger, Munich
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
#include "BoundaryCondition.h"
#include "MetaDatabase.h"
#include "HMesh.h"
#include <iostream>
#include <complex>

BoundaryCondition::BoundaryCondition(HMesh& _hMesh) : myHMesh(& _hMesh) {
	
}

BoundaryCondition::~BoundaryCondition() {
}


void BoundaryCondition::addConcentratedForce(std::vector<double> &_rhsReal){


	unsigned int numNodes = myHMesh->getNumNodes();

	STACCATO_XML::LOADS_const_iterator iLoads(MetaDatabase::getInstance()->xmlHandle->LOADS().begin());
	for (int k = 0; k < iLoads->LOAD().size(); k++) {

		// Find NODESET
		int flag = 0;
		std::vector<int> nodeSet;
		for (int i = 0; i < myHMesh->getNodeSetsName().size(); i++) {
			if (myHMesh->getNodeSetsName().at(i) == std::string(iLoads->LOAD().at(k).NODESET().begin()->Name()->c_str())) {
				nodeSet = myHMesh->getNodeSets().at(i);
				std::cout << ">> " << std::string(iLoads->LOAD().at(k).Type()->c_str()) << " " << iLoads->LOAD().at(k).NODESET().begin()->Name()->c_str() << " is loaded.\n";
				flag = 1;
			}
		}
		if (flag == 0)
			std::cerr << ">> Error while Loading: NODESET " << std::string(iLoads->LOAD().at(k).NODESET().begin()->Name()->c_str()) << " not Found.\n";

		int flagLabel = 0;

		for (int j = 0; j < numNodes; j++)
		{
			int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
			for (int l = 0; l < numDoFsPerNode; l++) {
				for (int m = 0; m < nodeSet.size(); m++) {
					if (myHMesh->getNodeLabels()[j] == myHMesh->getNodeLabels()[nodeSet.at(m)]) {
						if (std::string(iLoads->LOAD().at(k).Type()->c_str()) == "ConcentratedForce") {
							flagLabel = 1;

							std::complex<double> temp_Fx(std::atof(iLoads->LOAD().at(k).REAL().begin()->X()->data()), std::atof(iLoads->LOAD().at(k).IMAGINARY().begin()->X()->data()));
							std::complex<double> temp_Fy(std::atof(iLoads->LOAD().at(k).REAL().begin()->Y()->data()), std::atof(iLoads->LOAD().at(k).IMAGINARY().begin()->Y()->data()));
							std::complex<double> temp_Fz(std::atof(iLoads->LOAD().at(k).REAL().begin()->Z()->data()), std::atof(iLoads->LOAD().at(k).IMAGINARY().begin()->Z()->data()));

							int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
								switch (l) {
								case 0:
									_rhsReal[dofIndex] += temp_Fx.real();
									break;
								case 1:
									_rhsReal[dofIndex] += temp_Fy.real();
									break;
								case 2:
									_rhsReal[dofIndex] += temp_Fz.real();
									break;
								default:
									break;
								}
							}
						}
					}
				}
			}
		if (flagLabel == 0)
			std::cerr << ">> Error while Loading: NODE of NODESET " << std::string(iLoads->LOAD().at(k).NODESET().begin()->Name()->c_str()) << " not found.\n";
	}
	std::cout << ">> Building RHS Finished." << std::endl;
}

void BoundaryCondition::addConcentratedForce(std::vector<MKL_Complex16> &_rhsComplex) {

	unsigned int numNodes = myHMesh->getNumNodes();

	STACCATO_XML::LOADS_const_iterator iLoads(MetaDatabase::getInstance()->xmlHandle->LOADS().begin());
	for (int k = 0; k < iLoads->LOAD().size(); k++) {

		// Find NODESET
		int flag = 0;
		std::vector<int> nodeSet;
		for (int i = 0; i < myHMesh->getNodeSetsName().size(); i++) {
			if (myHMesh->getNodeSetsName().at(i) == std::string(iLoads->LOAD().at(k).NODESET().begin()->Name()->c_str())) {
				nodeSet = myHMesh->getNodeSets().at(i);
				std::cout << ">> " << std::string(iLoads->LOAD().at(k).Type()->c_str()) << " " << iLoads->LOAD().at(k).NODESET().begin()->Name()->c_str() << " is loaded.\n";
				flag = 1;
			}
		}
		if (flag == 0)
			std::cerr << ">> Error while Loading: NODESET " << std::string(iLoads->LOAD().at(k).NODESET().begin()->Name()->c_str()) << " not Found.\n";

		int flagLabel = 0;

		for (int j = 0; j < numNodes; j++)
		{
			int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
			for (int l = 0; l < numDoFsPerNode; l++) {
				for (int m = 0; m < nodeSet.size(); m++) {
					if (myHMesh->getNodeLabels()[j] == myHMesh->getNodeLabels()[nodeSet.at(m)]) {
						if (std::string(iLoads->LOAD().at(k).Type()->c_str()) == "ConcentratedForce") {
							flagLabel = 1;

							std::complex<double> temp_Fx(std::atof(iLoads->LOAD().at(k).REAL().begin()->X()->data()), std::atof(iLoads->LOAD().at(k).IMAGINARY().begin()->X()->data()));
							std::complex<double> temp_Fy(std::atof(iLoads->LOAD().at(k).REAL().begin()->Y()->data()), std::atof(iLoads->LOAD().at(k).IMAGINARY().begin()->Y()->data()));
							std::complex<double> temp_Fz(std::atof(iLoads->LOAD().at(k).REAL().begin()->Z()->data()), std::atof(iLoads->LOAD().at(k).IMAGINARY().begin()->Z()->data()));

							int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
								switch (l) {
								case 0:
									_rhsComplex[dofIndex].real += temp_Fx.real();
									_rhsComplex[dofIndex].imag += temp_Fx.imag();
									break;
								case 1:
									_rhsComplex[dofIndex].real += temp_Fy.real();
									_rhsComplex[dofIndex].imag += temp_Fy.imag();
									break;
								case 2:
									_rhsComplex[dofIndex].real += temp_Fz.real();
									_rhsComplex[dofIndex].imag += temp_Fz.imag();
									break;
								default:
									break;
								}
						}
					}
				}
			}
		}
		if (flagLabel == 0)
			std::cerr << ">> Error while Loading: NODE of NODESET " << std::string(iLoads->LOAD().at(k).NODESET().begin()->Name()->c_str()) << " not found.\n";
	}
	std::cout << ">> Building RHS Finished." << std::endl;
}


