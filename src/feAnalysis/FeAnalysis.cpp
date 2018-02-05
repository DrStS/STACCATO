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

#define _USE_MATH_DEFINES
#include <math.h>
#include "Message.h"
#include "FeAnalysis.h"
#include "HMesh.h"
#include "BoundaryCondition.h"
#include "FeElement.h"
#include "FePlainStress4NodeElement.h"
#include "FeTetrahedron10NodeElement.h"
#include "FeUmaElement.h"
#include "Material.h"

#include "MathLibrary.h"
#include "Timer.h"
#include "MemWatcher.h"

#include "MetaDatabase.h"
#include <string.h>

#include <complex>
using namespace std::complex_literals;

FeAnalysis::FeAnalysis(HMesh& _hMesh) : myHMesh(&_hMesh) {
	
		// --- XML Testing ---------------------------------------------------------------------------------------------
		std::cout << "=============================================\n";
		std::cout << "=============== STACCATO IMPORT =============\n";
		std::cout << "=============================================\n\n";


		std::cout << ">> Name: " << MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin()->NAME() << std::endl;
		std::cout << ">> Type: " << MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin()->TYPE() << std::endl;

		for (STACCATO_XML::FREQUENCY_const_iterator i(MetaDatabase::getInstance()->xmlHandle->FREQUENCY().begin());
			i != MetaDatabase::getInstance()->xmlHandle->FREQUENCY().end();
			++i)
		{
			std::cout << ">> Frequency Distr.: " << i->Type() << std::endl;
			std::cout << " > Start: " << i->START_FREQ() << " Hz" << std::endl;
			std::cout << " > End  : " << i->END_FREQ() << " Hz" << std::endl;
			std::cout << " > Step : " << i->STEP_FREQ() << " Hz" << std::endl;
		}

		std::cout << ">> MATERIALS:" << std::endl;
		STACCATO_XML::MATERIALS_const_iterator temp(MetaDatabase::getInstance()->xmlHandle->MATERIALS().begin());
		for (int j = 0; j < temp->MATERIAL().size(); j++) {
			std::cout << " > NAME: " << temp->MATERIAL()[j].Name() << " Type: " << temp->MATERIAL()[j].Type() << "\n\t E   : " << temp->MATERIAL()[j].E() << "\n\t nu  : " << temp->MATERIAL()[j].nu() << "\n\t rho : " << temp->MATERIAL()[j].rho() << "\n\t eta : " << temp->MATERIAL()[j].eta() << std::endl;
		}
		std::cout << "\n=============================================\n\n";

		// Build DataStructure
		myHMesh->buildDataStructure();
		debugOut << "SimuliaODB || SimuliaUMA::openFile: " << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

		// Build DoFGraph
		anaysisTimer01.start();
		myHMesh->buildDoFGraph(); 
		anaysisTimer01.stop();

		// Normal Routine is skipped if there is a detection of SIM Import
		infoOut << "Duration for building DoF graph: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
		debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

		// Build XML NodeSets and ElementSets
		MetaDatabase::getInstance()->buildXML(*myHMesh);

		unsigned int numElements = myHMesh->getNumElements();
		unsigned int numNodes = myHMesh->getNumNodes();
		std::vector<FeElement*> allElements(numElements);

		std::cout << "Num Nodes: " << numNodes << "\nNum Elements: " << numElements << std::endl;

		// Section Material Assignement
		STACCATO_XML::SECTIONS_const_iterator iSection(MetaDatabase::getInstance()->xmlHandle->SECTIONS().begin());
		for (int j = 0; j < iSection->SECTION().size(); j++) {
			Material * elasticMaterial = new Material(std::string(iSection->SECTION()[j].MATERIAL()->c_str()));

			// Find the Corresponding Set
			std::vector<int> idList = myHMesh->convertElementSetNameToLabels(std::string(iSection->SECTION()[j].ELEMENTSET()->c_str()));

			if (!idList.empty()) {
				// Assign Elements in idList
				int lastIndex = 0;
				for (int iElement = 0; iElement < idList.size(); iElement++)
				{
					int elemIndex = myHMesh->convertElementLabelToElementIndex(idList[iElement]);
					if (myHMesh->getElementTypes()[elemIndex] == STACCATO_PlainStress4Node2D) {
						allElements[elemIndex] = new FePlainStress4NodeElement(elasticMaterial);
					}
					else	if (myHMesh->getElementTypes()[elemIndex] == STACCATO_Tetrahedron10Node3D) {
						allElements[elemIndex] = new FeTetrahedron10NodeElement(elasticMaterial);
					}
					else    if (myHMesh->getElementTypes()[elemIndex] == STACCATO_UmaElement) {
						allElements[elemIndex] = new FeUmaElement(elasticMaterial);
					}
					int numNodesPerElement = myHMesh->getNumNodesPerElement()[elemIndex];
					double*  eleCoords = &myHMesh->getNodeCoordsSortElement()[lastIndex];
					allElements[elemIndex]->computeElementMatrix(eleCoords);
					lastIndex += numNodesPerElement*myHMesh->getDomainDimension();
				}
			}
			else
				std::cerr << ">> Error while assigning Material to element: ELEMENTSET " << std::string(iSection->SECTION()[j].ELEMENTSET()->c_str()) << " not Found.\n";
		}

		anaysisTimer01.stop();
		infoOut << "Duration for element loop: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
		debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

		std::vector<double> freq;

		// Routine to accomodate Step Distribution
		double start_freq = std::atof(MetaDatabase::getInstance()->xmlHandle->FREQUENCY().begin()->START_FREQ()->c_str());
		freq.push_back(start_freq);		// Push back starting frequency in any case

		if (std::string(MetaDatabase::getInstance()->xmlHandle->FREQUENCY().begin()->Type()->data()) == "STEP") {		// Step Distribute
			double end_freq = std::atof(MetaDatabase::getInstance()->xmlHandle->FREQUENCY().begin()->END_FREQ()->c_str());
			double step_freq = std::atof(MetaDatabase::getInstance()->xmlHandle->FREQUENCY().begin()->STEP_FREQ()->c_str());
			double push_freq = start_freq + step_freq;

			while (push_freq <= end_freq) {
				freq.push_back(push_freq);
				push_freq += step_freq;
			}
		}

		int totalDoF = myHMesh->getTotalNumOfDoFsRaw();
		// Memory for output
		// Re
		std::vector<double> resultUxRe;
		std::vector<double> resultUyRe;
		std::vector<double> resultUzRe;
		std::vector<double> resultMagRe;
		// Im															
		std::vector<double> resultUxIm;
		std::vector<double> resultUyIm;
		std::vector<double> resultUzIm;
		std::vector<double> resultMagIm;

		resultUxRe.resize(numNodes);
		resultUyRe.resize(numNodes);
		resultUzRe.resize(numNodes);
		resultMagRe.resize(numNodes);
		resultUxIm.resize(numNodes);
		resultUyIm.resize(numNodes);
		resultUzIm.resize(numNodes);
		resultMagIm.resize(numNodes);

		// Allocate global matrix and vector memory
		// Real Only
		std::vector<double> bReal;
		std::vector<double> solReal;
		// Complex
		std::vector<MKL_Complex16> bComplex;
		std::vector<MKL_Complex16> solComplex;

		std::string analysisType = MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin()->TYPE()->data();

		if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
			bReal.resize(totalDoF);
			solReal.resize(totalDoF);
		}
		else if (analysisType == "STEADYSTATE_DYNAMIC") {
			bComplex.resize(totalDoF);
			solComplex.resize(totalDoF);
		}
		else {
			std::cerr << "Unsupported Analysis Type! \n-Hint: Check XML Input \n-Exiting STACCATO." << std::endl;
			exit(EXIT_FAILURE);
		}

		anaysisTimer02.start();
		// Prepare elementDof for Enforcing Dirichlet BC
		STACCATO_XML::BC_const_iterator iBC(MetaDatabase::getInstance()->xmlHandle->BC().begin());
		for (int j = 0; j < iBC->DBC().size(); j++) {

			// Prepare restricted DOF vector
			std::vector<int> restrictedDOF;
			restrictedDOF.push_back(1);				// 1: FIXED
			restrictedDOF.push_back(1);				// 0: FREE
			restrictedDOF.push_back(1);
			/* For 6 Dof*/
			restrictedDOF.push_back(1);				// 1: FIXED
			restrictedDOF.push_back(1);				// 0: FREE
			restrictedDOF.push_back(1);

			if (std::string(iBC->DBC()[j].REAL().begin()->X()->c_str()) == "") {
				restrictedDOF[0] = 0;
				std::cout << "x set Free!\n";
			}
			if (std::string(iBC->DBC()[j].REAL().begin()->Y()->c_str()) == "") {
				restrictedDOF[1] = 0;
				std::cout << "y set Free!\n";
			}
			if (std::string(iBC->DBC()[j].REAL().begin()->Z()->c_str()) == "") {
				restrictedDOF[2] = 0;
				std::cout << "z set Free!\n";
			}
			myHMesh->killDirichletDOF(std::string(iBC->DBC()[j].NODESET().begin()->Name()->c_str()), restrictedDOF);

		}
		anaysisTimer02.stop();
		infoOut << "Duration for killing DOF loop: " << anaysisTimer02.getDurationMilliSec() << " milliSec" << std::endl;

		anaysisTimer01.start();
		for (int iFreqCounter = 0; iFreqCounter < freq.size(); iFreqCounter++) {
			int lastIndex = 0;

			if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
				AReal = new MathLibrary::SparseMatrix<double>(totalDoF, true);
			}
			else if (analysisType == "STEADYSTATE_DYNAMIC") {
				AComplex = new MathLibrary::SparseMatrix<MKL_Complex16>(totalDoF, true);
			}

			std::cout << ">> Building Stiffness Matrix...";
			for (int iElement = 0; iElement < numElements; iElement++)
			{
				int numDoFsPerElement = myHMesh->getNumDoFsPerElement()[iElement];
				int*  eleDoFs = &myHMesh->getElementDoFListBC()[lastIndex];
				lastIndex += numDoFsPerElement;
				double omega = 2 * M_PI*freq[iFreqCounter];
				//Assembly routine symmetric stiffness
				for (int i = 0; i < numDoFsPerElement; i++) {
					if (eleDoFs[i] != -1) {
						for (int j = 0; j < numDoFsPerElement; j++) {
							if (eleDoFs[j] >= eleDoFs[i] && eleDoFs[j] != -1) {
								//K(1+eta*i)
								if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
									(*AReal)(eleDoFs[i], eleDoFs[j]) += allElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j];
								}
								else if (analysisType == "STEADYSTATE_DYNAMIC") {
									(*AComplex)(eleDoFs[i], eleDoFs[j]).real += allElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j];
									(*AComplex)(eleDoFs[i], eleDoFs[j]).imag += allElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j] * allElements[iElement]->getMaterial()->getDampingParameter();
								}
							}
						}
					}
				}
				//K - omega*omega*M
				//Assembly routine symmetric mass
				if (analysisType == "STEADYSTATE_DYNAMIC_REAL" || analysisType == "STEADYSTATE_DYNAMIC")
					for (int i = 0; i < numDoFsPerElement; i++) {
						if (eleDoFs[i] != -1) {
							for (int j = 0; j < numDoFsPerElement; j++) {
								if (eleDoFs[j] >= eleDoFs[i] && eleDoFs[j] != -1) {
									//K(1+eta*i) - omega*omega*M
									if (analysisType == "STEADYSTATE_DYNAMIC_REAL") {
										(*AReal)(eleDoFs[i], eleDoFs[j]) -= allElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j] * omega*omega;
									}
									else if (analysisType == "STEADYSTATE_DYNAMIC") {
										(*AComplex)(eleDoFs[i], eleDoFs[j]).real -= allElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j] * omega*omega;
									}
								}
							}
						}
					}
			}
			std::cout << " Finished." << std::endl;
			std::cout << ">> Building RHS ...\n";
			//Add cload rhs contribution 
			BoundaryCondition neumannBoundaryCondition(*myHMesh);
			if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
				neumannBoundaryCondition.addConcentratedForce(bReal);
			}
			else if (analysisType == "STEADYSTATE_DYNAMIC") {
				neumannBoundaryCondition.addConcentratedForce(bComplex);
			}

			anaysisTimer02.start();
			std::cout << ">> Rebuilding RHS Matrix for Dirichlets ...";
			// Implement DBC
			// Applying Dirichlet
			for (int m = 0; m < myHMesh->getDirichletDOF().size(); m++) {

				int dofIndex = myHMesh->getDirichletDOF()[m];
				if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
					(*AReal)(dofIndex, dofIndex) = 1;
					bReal[dofIndex] = 0;
				}
				else if (analysisType == "STEADYSTATE_DYNAMIC") {
					(*AComplex)(dofIndex, dofIndex).real = 1;
					(*AComplex)(dofIndex, dofIndex).imag = 1;
					bComplex[dofIndex].real = 0;
					bComplex[dofIndex].imag = 0;
				}

			}
			std::cout << " Finished." << std::endl;
			anaysisTimer02.stop();
			infoOut << "Duration for applying dirichlet conditions: " << anaysisTimer02.getDurationMilliSec() << " milliSec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

			anaysisTimer01.stop();
			infoOut << "Duration for assembly loop: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
			anaysisTimer01.start();
			anaysisTimer02.start();
			if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
				(*AReal).check();
			}
			else if (analysisType == "STEADYSTATE_DYNAMIC") {
				(*AComplex).check();
			}
			anaysisTimer01.stop();
			infoOut << "Duration for direct solver check: " << anaysisTimer01.getDurationSec() << " sec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
			anaysisTimer01.start();
			if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
				(*AReal).factorize();
			}
			else if (analysisType == "STEADYSTATE_DYNAMIC") {
				(*AComplex).factorize();
			}
			anaysisTimer01.stop();
			infoOut << "Duration for direct solver factorize: " << anaysisTimer01.getDurationSec() << " sec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
			anaysisTimer01.start();
			if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
				(*AReal).solveDirect(&solReal[0], &bReal[0]);
			}
			else if (analysisType == "STEADYSTATE_DYNAMIC") {
				(*AComplex).solveDirect(&solComplex[0], &bComplex[0]);
			}
			anaysisTimer01.stop();
			anaysisTimer02.stop();
			infoOut << "Duration for direct solver substitution : " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
			infoOut << "Total duration for direct solver: " << anaysisTimer02.getDurationSec() << " sec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

			if (myHMesh->isSIM) {
				std::cout << ">> Printing RHS.. \n";
				for (int i = 0; i < bReal.size(); i++)
				{
					std::cout << " << " << bReal[i];
				}
				std::cout << std::endl;

				std::cout << "\n>> Nodal Solution: \n";
				for (int i = 0; i < numNodes; i++) {
					std::cout << "- Node " << i << ": \n";
					for (int j = 0; j < 6; j++) {
						std::cout << "  - DoF " << j + 1 << ": " << solReal[i * 6 + j] << std::endl;
					}
					std::cout << std::endl;
				}
			}			

			anaysisTimer01.start();

			// Store results
			for (int j = 0; j < numNodes; j++)
			{
				int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
				for (int l = 0; l < numDoFsPerNode; l++) {
					int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
					if (l == 0) {
						if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
							resultUxRe[j] = solReal[dofIndex];
						}
						else if (analysisType == "STEADYSTATE_DYNAMIC") {
							resultUxRe[j] = solComplex[dofIndex].real;
							resultUxIm[j] = solComplex[dofIndex].imag;
						}
					}
					if (l == 1) {
						if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
							resultUyRe[j] = solReal[dofIndex];
						}
						else if (analysisType == "STEADYSTATE_DYNAMIC") {
							resultUyRe[j] = solComplex[dofIndex].real;
							resultUyIm[j] = solComplex[dofIndex].imag;
						}
					}
					if (l == 2) {
						if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
							resultUzRe[j] = solReal[dofIndex];
						}
						else if (analysisType == "STEADYSTATE_DYNAMIC") {
							resultUzRe[j] = solComplex[dofIndex].real;
							resultUzIm[j] = solComplex[dofIndex].imag;
						}
					}
				}

				resultMagRe[j] = sqrt(pow(resultUxRe[j], 2) + pow(resultUyRe[j], 2) + pow(resultUzRe[j], 2));
				if (analysisType == "STEADYSTATE_DYNAMIC")
					resultMagIm[j] = sqrt(pow(resultUxIm[j], 2) + pow(resultUyIm[j], 2) + pow(resultUzIm[j], 2));

				if (myHMesh->getDomainDimension() == 2) {
					resultUzRe[j] = 0.0;
					resultUzIm[j] = 0.0;
				}

			}

			// Store results to database
			myHMesh->addResultScalarFieldAtNodes(STACCATO_Ux_Re, resultUxRe);
			myHMesh->addResultScalarFieldAtNodes(STACCATO_Uy_Re, resultUyRe);
			myHMesh->addResultScalarFieldAtNodes(STACCATO_Uz_Re, resultUzRe);
			myHMesh->addResultScalarFieldAtNodes(STACCATO_Magnitude_Re, resultMagRe);

			myHMesh->addResultScalarFieldAtNodes(STACCATO_Ux_Im, resultUxIm);
			myHMesh->addResultScalarFieldAtNodes(STACCATO_Uy_Im, resultUyIm);
			myHMesh->addResultScalarFieldAtNodes(STACCATO_Uz_Im, resultUzIm);
			myHMesh->addResultScalarFieldAtNodes(STACCATO_Magnitude_Im, resultMagIm);

			myHMesh->addResultsTimeDescription(std::to_string(freq[iFreqCounter]));

			anaysisTimer01.start();

			if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
				(*AReal).cleanPardiso();
				delete AReal;
			}
			else if (analysisType == "STEADYSTATE_DYNAMIC") {
				(*AComplex).cleanPardiso();
				delete AComplex;
			}
			anaysisTimer01.stop();

			infoOut << "Duration for Cleaning System Matrix: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
		}
}

FeAnalysis::~FeAnalysis() {
}