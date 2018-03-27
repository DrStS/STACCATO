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
#include "VectorFieldResults.h"
#include <string.h>

#include <complex>

#include <OutputDatabase.h>

using namespace std::complex_literals;

FeAnalysis::FeAnalysis(HMesh& _hMesh) : myHMesh(&_hMesh) {

	// --- XML Testing ---------------------------------------------------------------------------------------------
	std::cout << "=============================================\n";
	std::cout << "=============== STACCATO IMPORT =============\n";
	std::cout << "=============================================\n\n";

	for (STACCATO_XML::ANALYSIS_const_iterator i(MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin());
		i != MetaDatabase::getInstance()->xmlHandle->ANALYSIS().end();
		++i)
	{
		std::cout << ">> Analysis : " << i - MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin() + 1 << std::endl;
		std::cout << " > Name: " << i->NAME() << std::endl;
		std::cout << " > Type: " << i->TYPE() << std::endl;

		std::cout << " > Frequency Distr.: " << i->FREQUENCY().begin()->Type() << std::endl;
		std::cout << " > Start: " << i->FREQUENCY().begin()->START_FREQ() << " Hz" << std::endl;
		std::cout << " > End  : " << i->FREQUENCY().begin()->END_FREQ() << " Hz" << std::endl;
		std::cout << " > Step : " << i->FREQUENCY().begin()->STEP_FREQ() << " Hz" << std::endl;
	}
	std::cout << "\n=============================================\n\n";
	// --------------------------------------------------------------------------------------------------------------

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

	anaysisTimer01.start();

	// Build XML NodeSets and ElementSets
	MetaDatabase::getInstance()->buildXML(*myHMesh);

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();
	std::vector<FeElement*> allElements(numElements);

	std::cout << ">> Num Nodes   : " << numNodes << "\n>> Num Elements: " << numElements << std::endl;

	// Section Material Assignement
	STACCATO_XML::PARTS_const_iterator iterParts(MetaDatabase::getInstance()->xmlHandle->PARTS().begin());
	for (int iPart = 0; iPart < iterParts->PART().size(); iPart++)
	{
		for (int j = 0; j < iterParts->PART()[iPart].SECTIONS().begin()->SECTION().size(); j++) {
			Material * elasticMaterial = new Material(std::string(iterParts->PART()[iPart].SECTIONS().begin()->SECTION()[j].MATERIAL()->c_str()), iPart);

			// Find the Corresponding Set
			std::vector<int> idList = myHMesh->convertElementSetNameToLabels(std::string(iterParts->PART()[iPart].SECTIONS().begin()->SECTION()[j].ELEMENTSET()->c_str()));

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
				std::cerr << ">> Error while assigning Material to element: ELEMENTSET " << std::string(iterParts->PART()[iPart].SECTIONS().begin()->SECTION()[j].ELEMENTSET()->c_str()) << " not Found.\n";
		}
	}

	std::cout << ">> Section Material Assignment is Complete." << std::endl;
	anaysisTimer01.stop();
	infoOut << "Duration for element loop: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

	std::vector<double> freq;

	for (STACCATO_XML::ANALYSIS_const_iterator iAnalysis(MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin());
		iAnalysis != MetaDatabase::getInstance()->xmlHandle->ANALYSIS().end();
		++iAnalysis)
	{
		OutputDatabase::Analysis analysisData;
		int frameTrack = 0;

		std::string analysisType = iAnalysis->TYPE()->data();
		std::cout << "==== Starting Anaylsis: " << iAnalysis->NAME()->data() << " ====" << std::endl;
		if (analysisType != "STATIC") {
			// Routine to accomodate Step Distribution
			double start_freq = std::atof(iAnalysis->FREQUENCY().begin()->START_FREQ()->c_str());
			freq.push_back(start_freq);		// Push back starting frequency in any case

			if (std::string(iAnalysis->FREQUENCY().begin()->Type()->data()) == "STEP") {		// Step Distribute
				double end_freq = std::atof(iAnalysis->FREQUENCY().begin()->END_FREQ()->c_str());
				double step_freq = std::atof(iAnalysis->FREQUENCY().begin()->STEP_FREQ()->c_str());
				double push_freq = start_freq + step_freq;

				while (push_freq <= end_freq) {
					freq.push_back(push_freq);
					push_freq += step_freq;
				}
			}
		}
		else	{
			freq.push_back(0);			// For Static Case Assign One Static Step
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

		STACCATO_Analysis_type currentAnalysisType;

		VectorFieldResults* displacementVector;
		displacementVector = new VectorFieldResults();
		displacementVector->setResultsType(STACCATO_Result_Displacement);

		if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
			currentAnalysisType = (analysisType == "STATIC") ? STACCATO_Analysis_Static : STACCATO_Analysis_DynamicReal;
		}
		else if (analysisType == "STEADYSTATE_DYNAMIC") {
			currentAnalysisType = STACCATO_Analysis_Dynamic;
		}
		else {
			std::cerr << "Unsupported Analysis Type! \n-Hint: Check XML Input \n-Exiting STACCATO." << std::endl;
			exit(EXIT_FAILURE);
		}

		anaysisTimer02.start();
		// Prepare elementDof for Enforcing Dirichlet BC
		for (int jBC = 0; jBC < iAnalysis->BCCASE().begin()->BC().size(); jBC++) {
			for (int  jPart = 0; jPart < iterParts->PART().size(); jPart++)
			{
				if (std::string(iterParts->PART()[jPart].Name()->c_str()) == std::string(iAnalysis->BCCASE().begin()->BC()[jBC].Instance()->c_str()))		// Part Instance Matching
				{
					for (int k = 0; k < iterParts->PART()[jPart].BC_DEF().begin()->DISPLACEMENT().size(); k++)
					{
						if (std::string(iAnalysis->BCCASE()[jBC].BC().begin()->Name()->data()) == std::string(iterParts->PART()[jPart].BC_DEF().begin()->DISPLACEMENT()[k].Name()->data())) {
							// Prepare restricted DOF vector
							std::vector<int> restrictedDOF;
							restrictedDOF.push_back(1);				// 1: FIXED
							restrictedDOF.push_back(1);				// 0: FREE
							restrictedDOF.push_back(1);
							/* For 6 Dof*/
							restrictedDOF.push_back(1);				// 1: FIXED
							restrictedDOF.push_back(1);				// 0: FREE
							restrictedDOF.push_back(1);

							if (std::string(iterParts->PART()[jPart].BC_DEF().begin()->DISPLACEMENT()[k].REAL().begin()->X()->c_str()) == "") {
								restrictedDOF[0] = 0;
								std::cout << "x set Free!\n";
							}
							if (std::string(iterParts->PART()[jPart].BC_DEF().begin()->DISPLACEMENT()[k].REAL().begin()->Y()->c_str()) == "") {
								restrictedDOF[1] = 0;
								std::cout << "y set Free!\n";
							}
							if (std::string(iterParts->PART()[jPart].BC_DEF().begin()->DISPLACEMENT()[k].REAL().begin()->Z()->c_str()) == "") {
								restrictedDOF[2] = 0;
								std::cout << "z set Free!\n";
							}
							myHMesh->killDirichletDOF(std::string(iterParts->PART()[jPart].BC_DEF().begin()->DISPLACEMENT()[k].NODESET().begin()->Name()->c_str()), restrictedDOF);
						}
					}
				}
			}
		}
		anaysisTimer02.stop();
		infoOut << "Duration for killing DOF loop: " << anaysisTimer02.getDurationMilliSec() << " milliSec" << std::endl;

		anaysisTimer01.start();
		for (int iFreqCounter = 0; iFreqCounter < freq.size(); iFreqCounter++) {
			int lastIndex = 0;
			int sizeofRHS = 0;

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
				int*  eleDoFs = &myHMesh->getElementDoFListRestricted()[lastIndex];
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
			std::cout << ">> Building RHS Matrix for Neumann...\n";
			//Add cload rhs contribution 
			anaysisTimer02.start();

			OutputDatabase::TimeStep timeStep;
			timeStep.startIndex = frameTrack;

			for (int iLoadCase = 0; iLoadCase < iAnalysis->LOADCASES().begin()->LOADCASE().size(); iLoadCase++) {

				BoundaryCondition<double> neumannBoundaryConditionReal(*myHMesh);
				BoundaryCondition<STACCATOComplexDouble> neumannBoundaryConditionComplex(*myHMesh);

				std::string loadCaseType = iAnalysis->LOADCASES().begin()->LOADCASE().at(iLoadCase).Type()->data();


				for (int m = 0; m < iAnalysis->LOADCASES().begin()->LOADCASE().at(iLoadCase).LOAD().size(); m++)
				{
					std::vector<std::string> loadNameToPartLocal;

					// Search for Load Description
					for (int jPart = 0; jPart < iterParts->PART().size(); jPart++)
					{
						if (std::string(iterParts->PART()[jPart].Name()->c_str()) == std::string(iAnalysis->LOADCASES().begin()->LOADCASE()[iLoadCase].LOAD()[m].Instance()->c_str()))		// Part Instance Matching
						{
							for (int jPartLoad = 0; jPartLoad < iterParts->PART()[jPart].LOADS().begin()->LOAD().size(); jPartLoad++)
							{
								if (std::string(iAnalysis->LOADCASES().begin()->LOADCASE().at(iLoadCase).LOAD().at(m).Name()->data()) == std::string(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].Name()->data()));
								{

									if (loadCaseType == "ConcentratedLoadCase" && std::string(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].Type()->data()) == "ConcentratedForce") {

										OutputDatabase::LoadCase loadCaseData;
										loadCaseData.name = std::string(iAnalysis->LOADCASES().begin()->LOADCASE()[iLoadCase].NamePrefix()->data());
										loadCaseData.startIndex = frameTrack;

										std::string loadCaseTypePart = iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].Type()->data();

										std::vector<int> nodeSet = myHMesh->convertNodeSetNameToLabels(std::string(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].NODESET().begin()->Name()->c_str()));

										if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
											// Get Load
											std::vector<double> loadVector(3);
											loadVector[0] = std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->X()->data());
											loadVector[1] = std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Y()->data());
											loadVector[2] = std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Z()->data());

											neumannBoundaryConditionReal.addConcentratedForceContribution(nodeSet, loadVector, bReal);
											loadCaseData.type = neumannBoundaryConditionReal.myCaseType;

											frameTrack ++;
											sizeofRHS += neumannBoundaryConditionReal.getNumberOfTotalCases();
										}
										else if (analysisType == "STEADYSTATE_DYNAMIC") {

											// Get Load
											std::vector<STACCATOComplexDouble> loadVector(3);
											loadVector[0] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->X()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->X()->data()) };
											loadVector[1] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Y()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->Y()->data()) };
											loadVector[2] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Z()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->Z()->data()) };

											neumannBoundaryConditionComplex.addConcentratedForceContribution(nodeSet, loadVector, bComplex);
											loadCaseData.type = neumannBoundaryConditionComplex.myCaseType;

											frameTrack ++;
											sizeofRHS += neumannBoundaryConditionComplex.getNumberOfTotalCases();
										}
										loadCaseData.unit = myHMesh->myOutputDatabase->getSyntaxForSubLoadCase(loadCaseData.type);
										timeStep.caseList.push_back(loadCaseData);

									}
									else if (loadCaseType == "RotateGenerate" && std::string(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].Type()->data()) == "DistributingCouplingForce") {

										// Routine to accomodate Step Distribution
										double start_theta = std::atof(iAnalysis->LOADCASES().begin()->LOADCASE().at(iLoadCase).START_THETA()->c_str());
										neumannBoundaryConditionReal.addBCCaseDescription(start_theta);			// Push back starting angle
										neumannBoundaryConditionComplex.addBCCaseDescription(start_theta);		// Push back starting angle
																												// Step Distribute
										double end_theta = std::atof(iAnalysis->LOADCASES().begin()->LOADCASE().at(iLoadCase).END_THETA()->c_str());
										double step_theta = std::atof(iAnalysis->LOADCASES().begin()->LOADCASE().at(iLoadCase).STEP_THETA()->c_str());
										double push_theta = start_theta + step_theta;

										while (push_theta <= end_theta) {
											neumannBoundaryConditionReal.addBCCaseDescription(push_theta);
											neumannBoundaryConditionComplex.addBCCaseDescription(push_theta);
											push_theta += step_theta;
										}

										std::vector<int> refNode = myHMesh->convertNodeSetNameToLabels(std::string(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REFERENCENODESET().begin()->Name()->c_str()));
										std::vector<int> couplingNodes = myHMesh->convertNodeSetNameToLabels(std::string(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].COUPLINGNODESET().begin()->Name()->c_str()));

										if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {

											// Get Load
											std::vector<double> loadVector(3);
											loadVector[0] = std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->X()->data());
											loadVector[1] = std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Y()->data());
											loadVector[2] = std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Z()->data());
											neumannBoundaryConditionReal.addRotatingForceContribution(refNode, couplingNodes, loadVector, bReal);

											for (int it = 0; it < neumannBoundaryConditionReal.getNumberOfTotalCases(); it++)
											{
												OutputDatabase::LoadCase loadCaseData;
												loadCaseData.type = neumannBoundaryConditionReal.myCaseType;
												loadCaseData.unit = myHMesh->myOutputDatabase->getSyntaxForSubLoadCase(loadCaseData.type);
												loadCaseData.name = std::string(iAnalysis->LOADCASES().begin()->LOADCASE()[iLoadCase].NamePrefix()->data()) + "_" + std::to_string(static_cast<int>(neumannBoundaryConditionReal.getBCCaseDescription()[it])) + loadCaseData.unit;
												loadCaseData.startIndex = frameTrack;
												timeStep.caseList.push_back(loadCaseData);

												frameTrack++;
											}
											sizeofRHS += neumannBoundaryConditionReal.getNumberOfTotalCases();
										}
										else if (analysisType == "STEADYSTATE_DYNAMIC") {

											// Get Load
											std::vector<STACCATOComplexDouble> loadVector(3);
											loadVector[0] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->X()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->X()->data()) };
											loadVector[1] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Y()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->Y()->data()) };
											loadVector[2] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Z()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->Z()->data()) };

											neumannBoundaryConditionComplex.addRotatingForceContribution(refNode, couplingNodes, loadVector, bComplex);

											for (int it = 0; it < neumannBoundaryConditionComplex.getNumberOfTotalCases(); it++)
											{
												OutputDatabase::LoadCase loadCaseData;
												loadCaseData.type = neumannBoundaryConditionComplex.myCaseType;
												loadCaseData.unit = myHMesh->myOutputDatabase->getSyntaxForSubLoadCase(loadCaseData.type);
												loadCaseData.name = std::string(iAnalysis->LOADCASES().begin()->LOADCASE()[iLoadCase].NamePrefix()->data()) + "_" + std::to_string(static_cast<int>(neumannBoundaryConditionComplex.getBCCaseDescription()[it])) + loadCaseData.unit;
												loadCaseData.startIndex = frameTrack;
												timeStep.caseList.push_back(loadCaseData);

												frameTrack++;
												sizeofRHS += neumannBoundaryConditionComplex.getNumberOfTotalCases();
											}
										}
									}
								}
							}
						}
					}
				}
			}

			std::cout << ">> Building RHS Matrix for Neumann... Finished.\n" << std::endl;
			anaysisTimer02.stop();
			infoOut << "Duration for applying Neumann conditions: " << anaysisTimer02.getDurationMilliSec() << " milliSec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

			anaysisTimer02.start();
			std::cout << ">> Rebuilding RHS Matrix for Dirichlets ...";
			// Implement DBC
			// Applying Dirichlet
			int size = sizeofRHS;
			for (int m = 0; m < myHMesh->getRestrictedHomogeneousDoFList().size(); m++) {

				int dofIndex = myHMesh->getRestrictedHomogeneousDoFList()[m];
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

			for (int l = 0; l < size; l++) {
				for (int m = 0; m < myHMesh->getRestrictedHomogeneousDoFList().size(); m++) {

					int dofIndex = l*myHMesh->getTotalNumOfDoFsRaw() + myHMesh->getRestrictedHomogeneousDoFList()[m];
					if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
						bReal[dofIndex] = 0;
					}
					else if (analysisType == "STEADYSTATE_DYNAMIC") {
						bComplex[dofIndex].real = 0;
						bComplex[dofIndex].imag = 0;
					}

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
				(*AReal).factorize(size);
			}
			else if (analysisType == "STEADYSTATE_DYNAMIC") {
				(*AComplex).factorize(size);
			}
			anaysisTimer01.stop();
			infoOut << "Duration for direct solver factorize: " << anaysisTimer01.getDurationSec() << " sec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
			anaysisTimer01.start();
			if (size == 1) {
				if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
					solReal.resize(bReal.size());
					(*AReal).solveDirect(&solReal[0], &bReal[0]);
				}
				else if (analysisType == "STEADYSTATE_DYNAMIC") {
					solComplex.resize(bComplex.size());
					(*AComplex).solveDirect(&solComplex[0], &bComplex[0]);
				}
			}
			else {
				std::cout << ">> Solving for all " << size << " RHSs.\n";
				if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
					solReal.resize(bReal.size());
					(*AReal).solveDirect(&solReal[0], &bReal[0], size);
				}
				else if (analysisType == "STEADYSTATE_DYNAMIC") {
					solComplex.resize(bComplex.size());
					(*AComplex).solveDirect(&solComplex[0], &bComplex[0], size);
				}
			}

			anaysisTimer01.stop();
			anaysisTimer02.stop();
			infoOut << "direct solver substitution : " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
			infoOut << "Total duration for direct solver: " << anaysisTimer02.getDurationSec() << " sec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

			anaysisTimer01.start();
			std::cout << ">> Storing for " << size << " RHSs.\n";
			for (int k = 0; k < size; k++) {
				// Store results
				for (int j = 0; j < numNodes; j++)
				{
					int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
					for (int l = 0; l < numDoFsPerNode; l++) {
						int dofIndex = k*totalDoF + myHMesh->getNodeIndexToDoFIndices()[j][l];
						if (l == 0) {
							if (analysisType == "STATIC" || analysisType == "STEADYSTATE_DYNAMIC_REAL") {
								resultUxRe[j] = solReal[l*totalDoF + dofIndex];
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
				displacementVector->addResultScalarFieldAtNodes(STACCATO_x_Re, resultUxRe);
				displacementVector->addResultScalarFieldAtNodes(STACCATO_y_Re, resultUyRe);
				displacementVector->addResultScalarFieldAtNodes(STACCATO_z_Re, resultUzRe);
				displacementVector->addResultScalarFieldAtNodes(STACCATO_Magnitude_Re, resultMagRe);

				displacementVector->addResultScalarFieldAtNodes(STACCATO_x_Im, resultUxIm);
				displacementVector->addResultScalarFieldAtNodes(STACCATO_y_Im, resultUyIm);
				displacementVector->addResultScalarFieldAtNodes(STACCATO_z_Im, resultUzIm);
				displacementVector->addResultScalarFieldAtNodes(STACCATO_Magnitude_Im, resultMagIm);
			}
			if (analysisType == "STATIC") {
				timeStep.timeDescription = "STATIC";
			}
			else if (analysisType == "STEADYSTATE_DYNAMIC_REAL" || analysisType == "STEADYSTATE_DYNAMIC") {
				timeStep.timeDescription = std::to_string(static_cast<int>(freq[iFreqCounter]));
			}

			anaysisTimer01.stop();

			infoOut << "Duration for Storing: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
			debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

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

			timeStep.unit = myHMesh->myOutputDatabase->getSyntaxForTime(currentAnalysisType);
			analysisData.timeSteps.push_back(timeStep);
		}
		// Add Result to OutputDataBase
		analysisData.name = myHMesh->myOutputDatabase->getSyntaxAnalysisDescription(std::string(iAnalysis->NAME()->data()), currentAnalysisType);
		analysisData.type = currentAnalysisType;
		analysisData.startIndex = myHMesh->myOutputDatabase->getStartIndexForNewAnalysis();
		myHMesh->myOutputDatabase->addNewAnalysisVectorField( analysisData, displacementVector);

		std::cout << "==== Anaylsis Completed: " << iAnalysis->NAME()->data() << " ====" << std::endl;
	}
	std::cout << ">> All Analyses Completed." << std::endl;
}


FeAnalysis::~FeAnalysis() {
}