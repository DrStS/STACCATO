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
#include "KrylovROMSubstructure.h"
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
#include <AuxiliaryFunctions.h>
#include <iomanip>

#ifdef USE_INTEL_MKL
#define MKL_DIRECT_CALL 1
#include <mkl.h>
#endif

KrylovROMSubstructure::KrylovROMSubstructure(HMesh& _hMesh) : myHMesh(&_hMesh) {
	std::cout << "=============== STACCATO ROM Analysis =============\n";

	/* -- Exporting ------------- */
	bool exportSparseMatrix = false;
	bool exportRHS = false;
	bool exportSolution = false;
	/* -------------------------- */

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

	// Assign all elements with respective assigned material section
	assignMaterialToElements();
	
	// Part Reduction
	STACCATO_XML::PARTS_const_iterator iterParts(MetaDatabase::getInstance()->xmlHandle->PARTS().begin());
	for (int iPart = 0; iPart < iterParts->PART().size(); iPart++)
	{
		if (std::string(iterParts->PART()[iPart].TYPE()->data()) == "FE_KMOR")
		{
			myHMesh->isKROM = true;
			std::cout << ">> KMOR procedure to be performed on FE part: " << std::string(iterParts->PART()[iPart].Name()->data()) << std::endl;

			// Getting ROM prerequisites
			/// Exapansion points
			if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->EXP_POINTS().begin()->Type()->c_str()) == "MANUAL")	{
				std::stringstream stream(std::string(iterParts->PART()[iPart].ROMDATA().begin()->EXP_POINTS().begin()->c_str()));
				while (stream) {
					double n;
					stream >> n;
					if (stream)
						myExpansionPoints.push_back(n);
				}
			}
			/// Krylov Order
			if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->KRYLOV_ORDER().begin()->Type()->c_str()) == "MANUAL") {
				myKrylovOrder = std::atoi(iterParts->PART()[iPart].ROMDATA().begin()->KRYLOV_ORDER().begin()->c_str());
			}
			/// Inputs
			if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->INPUTS().begin()->Type()->c_str()) == "NODES") {
				for (int iNodeSet = 0; iNodeSet < iterParts->PART()[iPart].ROMDATA().begin()->INPUTS().begin()->NODESET().size(); iNodeSet++) {
					std::vector<int> nodeSet = myHMesh->convertNodeSetNameToLabels(std::string(iterParts->PART()[iPart].ROMDATA().begin()->INPUTS().begin()->NODESET()[iNodeSet].Name()->c_str()));
					// Insert nodeSet entries to DOFS
					for (int jNodeSet = 0; jNodeSet < nodeSet.size(); jNodeSet++) {
						std::vector<int> dofIndices = myHMesh->getNodeIndexToDoFIndices()[myHMesh->convertNodeLabelToNodeIndex(nodeSet[jNodeSet])];
						myInputDOFS.insert(myInputDOFS.end(), dofIndices.begin(), dofIndices.end());
					}
					
				}
			}

			/*std::cout << ">> Recognized Inputs: ";
			for (int i = 0; i < inputDOFS.size(); i++)
			{
				std::cout << " << " << inputDOFS[i];
			}
			std::cout << std::endl;*/

			/// Outputs
			if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->OUTPUTS().begin()->Type()->c_str()) == "NODES") {
				/*for (int iNodeSet = 0; iNodeSet < iterParts->PART()[iPart].ROMDATA().begin()->OUTPUTS().begin()->NODESET().size(); iNodeSet++) {
					std::vector<int> nodeSet = myHMesh->convertNodeSetNameToLabels(std::string(iterParts->PART()[iPart].ROMDATA().begin()->OUTPUTS().begin()->NODESET()[iNodeSet].Name()->c_str()));
					// Insert nodeSet entries to DOFS
					for (int jNodeSet = 0; jNodeSet < nodeSet.size(); jNodeSet++) {
						std::vector<int> dofIndices = myHMesh->getNodeIndexToDoFIndices()[myHMesh->convertNodeLabelToNodeIndex(nodeSet[jNodeSet])];
						outputDOFS.insert(outputDOFS.end(), dofIndices.begin(), dofIndices.end());
					}

				}*/
			}
			else if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->OUTPUTS().begin()->Type()->c_str()) == "MIMO")	{
				myOutputDOFS = myInputDOFS;
			}

			// Size determination
			int FOM_DOF = myHMesh->getTotalNumOfDoFsRaw();
			int ROM_DOF = myExpansionPoints.size()*myKrylovOrder*myInputDOFS.size();	// Assuming MIMO

			std::cout << ">> -- ROM Data --" << std::endl;
			std::cout << " > Expansion Points: ";
			for (int i = 0; i < myExpansionPoints.size(); i++)
				std::cout << myExpansionPoints[i] << " . ";
			std::cout << std::endl;
			std::cout << " > Krylov order: " << myKrylovOrder << std::endl;
			std::cout << " > #Inputs: " << myInputDOFS.size() << " #Outputs: " << myOutputDOFS.size()<< std::endl;
			std::cout << " > FOM Dimensions: " << std::endl;
			std::cout << "  > System: " << FOM_DOF << "x" << FOM_DOF << std::endl;
			std::cout << "  >      B: " << FOM_DOF << "x" << myInputDOFS.size() << std::endl;
			std::cout << "  >      C: " << myOutputDOFS.size() << "x" << FOM_DOF << std::endl;
			std::cout << " > ROM Dimensions: " << std::endl;
			std::cout << "  > System: " << ROM_DOF << "x" << ROM_DOF << std::endl;
			std::cout << "  >    B_R: " << ROM_DOF << "x" << myInputDOFS.size() << std::endl;
			std::cout << "  >    C_R: " << myOutputDOFS.size() << "x" << ROM_DOF << std::endl;
			std::cout << " > Projection Matrices: " << std::endl;
			std::cout << "  >      V: " << FOM_DOF << "x" << ROM_DOF << std::endl; 
			std::cout << "  >      Z: " << FOM_DOF << "x" << ROM_DOF << std::endl;

			
			if (std::string(iterParts->PART()[iPart].TYPE()->data()) == "FE_KMOR") {	// Complex Data Instantiation
				KComplex = new MathLibrary::SparseMatrix<MKL_Complex16>(FOM_DOF, true, true);
				MComplex = new MathLibrary::SparseMatrix<MKL_Complex16>(FOM_DOF, true, true);

				myKComplexReduced.resize(ROM_DOF*ROM_DOF);
				myMComplexReduced.resize(ROM_DOF*ROM_DOF);

			}
			myB.resize(FOM_DOF*myInputDOFS.size());
			myC.resize(myOutputDOFS.size()*FOM_DOF);

			myBReduced.resize(ROM_DOF*myInputDOFS.size());
			myCReduced.resize(myOutputDOFS.size()*ROM_DOF);

			// Generate Input and Output Matrices
			for (int inpIter = 0; inpIter < myInputDOFS.size(); inpIter++) {
				myB[myInputDOFS[inpIter] + inpIter*FOM_DOF].real = 1;
			}
			for (int outIter = 0; outIter < myOutputDOFS.size(); outIter++) {
				myC[myOutputDOFS[outIter] * myOutputDOFS.size() + outIter].real = 1;
			}

			// Assemble global FOM system matrices
			if (std::string(iterParts->PART()[iPart].FILEIMPORT().begin()->Type()->c_str()) == "AbqODB") {
				std::cout << ">> Assembling FOM system matrices from ODB... " << std::endl;
				assembleGlobalMatrices(std::string(std::string(iterParts->PART()[iPart].TYPE()->data())));
			}
			else if (std::string(iterParts->PART()[iPart].FILEIMPORT().begin()->Type()->c_str()) == "AbqSIM") {
				std::cout << ">> Assembling FOM system matrices from SIM... " << std::endl;
				assembleUmaMatrices(std::string(std::string(iterParts->PART()[iPart].TYPE()->data())));
				std::cerr << "NOT IMPLEMENTED: Assign FOM system matrices with UMA element matrices.";
			}

			anaysisTimer01.start();
			if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->EXP_POINTS().begin()->Type()->c_str()) == "MANUAL") {
				std::cout << ">> Building projection basis WITHOUT automated MOR..." << std::endl;
				buildProjectionMatManual();
			}
			else if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->EXP_POINTS().begin()->Type()->c_str()) == "AUTO") {
				std::cerr << "NOT IMPLEMENTED: Building projection basis WITH automated MOR" << std::endl;
			}
			generateROM();
			anaysisTimer01.stop();
			std::cout << " > Duration for krylov reduced model generation of model " << std::string(iterParts->PART()[iPart].Name()->c_str()) << " : " << anaysisTimer01.getDurationMilliSec() << " ms" << std::endl;
		}
	}


	STACCATOComplexDouble ZeroComplex = { 0,0 };
	STACCATOComplexDouble OneComplex = { 1,0 };

	// Reading in the frequency range
	std::vector<double> freq;
	for (STACCATO_XML::ANALYSIS_const_iterator iAnalysis(MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin());
		iAnalysis != MetaDatabase::getInstance()->xmlHandle->ANALYSIS().end();
		++iAnalysis)
	{
		std::cout << std::endl << "==== Starting Anaylsis: " << iAnalysis->NAME()->data() << " ====" << std::endl;
		OutputDatabase::Analysis analysisData;
		int frameTrack = 0;

		STACCATO_Analysis_type currentAnalysisType;
		currentAnalysisType = STACCATO_Analysis_Dynamic;

		VectorFieldResults* displacementVector;
		displacementVector = new VectorFieldResults();
		displacementVector->setResultsType(STACCATO_Result_Displacement);
		displacementVector->setAnalysisType(currentAnalysisType);

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

		// Size determination
		int FOM_DOF = myHMesh->getTotalNumOfDoFsRaw();
		int ROM_DOF = myExpansionPoints.size()*myKrylovOrder*myInputDOFS.size();	// Assuming MIMO
		std::vector<STACCATOComplexDouble> results;

		// Determine right hand side
		std::cout << ">> Building RHS Matrix for Neumann...\n";
		int sizeofRHS = 0;
		OutputDatabase::TimeStep timeStep;
		timeStep.startIndex = frameTrack;
		std::vector<MKL_Complex16> bComplex;
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

									// Get Load
									std::vector<STACCATOComplexDouble> loadVector(3);
									loadVector[0] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->X()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->X()->data()) };
									loadVector[1] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Y()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->Y()->data()) };
									loadVector[2] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Z()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->Z()->data()) };

									neumannBoundaryConditionComplex.addConcentratedForceContribution(nodeSet, loadVector, bComplex);
									loadCaseData.type = neumannBoundaryConditionComplex.myCaseType;

									frameTrack++;
									sizeofRHS += neumannBoundaryConditionComplex.getNumberOfTotalCases();

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
									}
									sizeofRHS += neumannBoundaryConditionComplex.getNumberOfTotalCases();
								}
							}
						}
					}
				}
			}
		}
		std::vector<STACCATOComplexDouble> inputLoad;
		for (int iRHS = 0; iRHS < sizeofRHS; iRHS++)
		{
			std::vector<STACCATOComplexDouble> temp;
			temp.resize(myInputDOFS.size(), {0,0});
			for (int iInputDof = 0; iInputDof < myInputDOFS.size(); iInputDof++)
			{
				temp[iInputDof].real = bComplex[myInputDOFS[iInputDof]].real;
				temp[iInputDof].imag = bComplex[myInputDOFS[iInputDof]].imag;
			}
			inputLoad.insert(inputLoad.end(), temp.begin(), temp.end());
		}

		std::cout << ">> Building RHS Matrix for Neumann... Finished.\n" << std::endl;
		
		// Solving for each frequency
		anaysisTimer01.start();
#ifdef USE_INTEL_MKL		
		std::vector<lapack_int> pivot(ROM_DOF);	// Pivots for LU Decomposition
		for (int iFreqCounter = 0; iFreqCounter < freq.size(); iFreqCounter++) {
			std::vector<double> sampleResultRe(1, 0);
			std::vector<double> sampleResultIm(1, 0);

			//std::cout << ">> Computing frequency step at " << freq[iFreqCounter] << " Hz ..." << std::endl;
			double omega = 2 * M_PI*freq[iFreqCounter];
			STACCATOComplexDouble NegOmegaSquare = { -omega * omega,0 };

			// K_krylov_dyn = obj.K_R 
			std::vector<STACCATOComplexDouble> StiffnessAssembled;
			StiffnessAssembled.resize(ROM_DOF*ROM_DOF, ZeroComplex);
			MathLibrary::computeDenseVectorAdditionComplex(&myKComplexReduced[0], &StiffnessAssembled[0], &OneComplex, ROM_DOF*ROM_DOF);
			// K_krylov_dyn += -omega^2*obj.M_R 
			MathLibrary::computeDenseVectorAdditionComplex(&myMComplexReduced[0], &StiffnessAssembled[0], &NegOmegaSquare, ROM_DOF*ROM_DOF);


			// obj.B_R*obj.F
			std::vector<STACCATOComplexDouble> inputLoad_krylov(ROM_DOF*sizeofRHS, {0,0});
			MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, sizeofRHS, myInputDOFS.size(), &myBReduced[0], &inputLoad[0], &inputLoad_krylov[0], false, false, OneComplex, false, false, false);

			// z_krylov_freq = K_krylov_dyn\(obj.B_R*obj.F);
			// Factorize StiffnessAssembled
			LAPACKE_zgetrf(LAPACK_COL_MAJOR, ROM_DOF, ROM_DOF, &StiffnessAssembled[0], ROM_DOF, &pivot[0]);
			// Solve system
			LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', ROM_DOF, sizeofRHS, &StiffnessAssembled[0], ROM_DOF, &pivot[0], &inputLoad_krylov[0], ROM_DOF);

			// z_freq = obj.C_R*z_krylov_freq;
			std::vector<STACCATOComplexDouble> backprojected_sol(myOutputDOFS.size()*sizeofRHS, ZeroComplex);
			MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(myOutputDOFS.size(), sizeofRHS, ROM_DOF, &myCReduced[0], &inputLoad_krylov[0], &backprojected_sol[0], false, false, OneComplex, false, false, false);

			results.insert(results.end(), backprojected_sol.begin(), backprojected_sol.end());
			//std::cout << ">> Computing frequency step at " << freq[iFreqCounter] << " Hz ... Finished." << std::endl;

			/*sampleResultRe[0] = backprojected_sol[0].real;
			sampleResultIm[0] = backprojected_sol[0].imag;
			timeStep.timeDescription = std::to_string(static_cast<int>(freq[iFreqCounter]));
			displacementVector->addResultScalarFieldAtNodes(STACCATO_x_Re, sampleResultRe);
			displacementVector->addResultScalarFieldAtNodes(STACCATO_x_Im, sampleResultIm);

			timeStep.unit = myHMesh->myOutputDatabase->getSyntaxForTime(currentAnalysisType);
			analysisData.timeSteps.push_back(timeStep);*/

		}
		anaysisTimer01.stop();
		std::cout << " > Duration for backtransformation: " << anaysisTimer01.getDurationMilliSec() << " ms" << std::endl;

		// Add Result to OutputDataBase
		/*analysisData.name = myHMesh->myOutputDatabase->getSyntaxAnalysisDescription(std::string(iAnalysis->NAME()->data()), currentAnalysisType);
		analysisData.type = currentAnalysisType;
		analysisData.startIndex = myHMesh->myOutputDatabase->getStartIndexForNewAnalysis();
		myHMesh->myOutputDatabase->addNewAnalysisVectorField(analysisData, displacementVector);*/
#endif // USE_INTEL_MKL

		AuxiliaryFunctions::writeMKLComplexVectorDatFormat(std::string(iAnalysis->NAME()->data()) + "_KMOR_Results.dat", results);
		
		std::cout << "==== Anaylsis Completed: " << iAnalysis->NAME()->data() << " ====" << std::endl;
	}
	std::cout << ">> All Analyses Completed." << std::endl;
}

KrylovROMSubstructure::~KrylovROMSubstructure() {
}

void KrylovROMSubstructure::assignMaterialToElements() {
	anaysisTimer01.start();

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();
	std::cout << ">> Num Nodes   : " << numNodes << "\n>> Num Elements: " << numElements << std::endl;

	myAllElements.resize(numElements);
	allUMAElements.resize(numElements);
	// Section Material Assignement
	STACCATO_XML::PARTS_const_iterator iterParts(MetaDatabase::getInstance()->xmlHandle->PARTS().begin());

	for (int iPart = 0; iPart < iterParts->PART().size(); iPart++)
	{
		int matElCount = 0;
		// Map for mapping element labels to material
		std::map<int, std::string> elementLabelToMaterialMap;
		for (int j = 0; j < iterParts->PART()[iPart].SECTIONS().begin()->SECTION().size(); j++) {
			// Find the Corresponding Set
			std::vector<int> idList = myHMesh->convertElementSetNameToLabels(std::string(iterParts->PART()[iPart].SECTIONS().begin()->SECTION()[j].ELEMENTSET()->c_str()));
			matElCount += idList.size();

			if (!idList.empty()) {
				// Assign Elements in idList
				for (int iElement = 0; iElement < idList.size(); iElement++)
					elementLabelToMaterialMap[idList[iElement]] = std::string(iterParts->PART()[iPart].SECTIONS().begin()->SECTION()[j].MATERIAL()->c_str());
			}
			else
				std::cerr << ">> Error while mapping Material to element: ELEMENTSET " << std::string(iterParts->PART()[iPart].SECTIONS().begin()->SECTION()[j].ELEMENTSET()->c_str()) << " not Found.\n";

		}
		std::vector<int> allElemsLabel = myHMesh->getElementLabels();

		if ((!allElemsLabel.empty() && matElCount == numElements)) {
			// Assign Elements in idList
			int lastIndex = 0;
			for (int iElement = 0; iElement < allElemsLabel.size(); iElement++)
			{
				Material * elasticMaterial = new Material(elementLabelToMaterialMap[allElemsLabel[iElement]], iPart);

				int elemIndex = myHMesh->convertElementLabelToElementIndex(allElemsLabel[iElement]);
				if (myHMesh->getElementTypes()[elemIndex] == STACCATO_PlainStress4Node2D) {
					myAllElements[elemIndex] = new FePlainStress4NodeElement(elasticMaterial);
				}
				else if (myHMesh->getElementTypes()[elemIndex] == STACCATO_Tetrahedron10Node3D) {
					if (!myAllElements[elemIndex])
						myAllElements[elemIndex] = new FeTetrahedron10NodeElement(elasticMaterial);
					else
						std::cerr << ">> Warning: Attempt to assign already assigned element is skipped. ******************" << std::endl;
				}
				else if (myHMesh->getElementTypes()[elemIndex] == STACCATO_UmaElement) {
#ifdef USE_SIMULIA_UMA_API
					allUMAElements[elemIndex] = new FeUmaElement(elasticMaterial);
#endif
				}
				int numNodesPerElement = myHMesh->getNumNodesPerElement()[elemIndex];
				double*  eleCoords = &myHMesh->getNodeCoordsSortElement()[lastIndex];
				if (myHMesh->isSIM)
					allUMAElements[elemIndex]->computeElementMatrix(eleCoords);
				else
					myAllElements[elemIndex]->computeElementMatrix(eleCoords);
				lastIndex += numNodesPerElement * myHMesh->getDomainDimension();
			}
		}
		else
			std::cerr << ">> *Ignore if UMA* Error while assigning Material to element sets: Not all elements are assigned with a defined material." << std::endl;
	}

	std::cout << ">> Section Material Assignment is Complete." << std::endl;
	anaysisTimer01.stop();
	infoOut << "Duration for element loop: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
}

void KrylovROMSubstructure::assembleGlobalMatrices(std::string _type) {
	int FOM_DOF = myHMesh->getTotalNumOfDoFsRaw();
	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();
	int lastIndex = 0;

	// Assembling Element Stiffness Matrices
	for (int iElement = 0; iElement < numElements; iElement++)
	{
		int numDoFsPerElement = myHMesh->getNumDoFsPerElement()[iElement];
		int*  eleDoFs = &myHMesh->getElementDoFListRestricted()[lastIndex];
		lastIndex += numDoFsPerElement;
		
		if (!myHMesh->isSIM) {
			//Assembly routine symmetric stiffness
			for (int i = 0; i < numDoFsPerElement; i++) {
				if (eleDoFs[i] != -1) {
					for (int j = 0; j < numDoFsPerElement; j++) {
						if (eleDoFs[j] >= eleDoFs[i] && eleDoFs[j] != -1) {
							//K(1+eta*i)
							if (_type == "FE_KMOR_REAL") {
								//(*KReal)(eleDoFs[i], eleDoFs[j]) += myAllElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j];
							}
							else if (_type == "FE_KMOR") {
								(*KComplex)(eleDoFs[i], eleDoFs[j]).real += myAllElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j];
								(*KComplex)(eleDoFs[i], eleDoFs[j]).imag += myAllElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j] * myAllElements[iElement]->getMaterial()->getDampingParameter();
							}
							//K(1+eta*i) - omega*omega*M
							if (_type == "FE_KMOR_REAL") {
								//(*MReal)(eleDoFs[i], eleDoFs[j]) -= myAllElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j] * omega*omega;
							}
							else if (_type == "FE_KMOR") {
								(*MComplex)(eleDoFs[i], eleDoFs[j]).real += myAllElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j];
								//std::cout << ">" << myAllElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j] << " \n";
							}
						}
					}
				}
			}
		}
	}

	/// new CSR
	mySparseK = (*KComplex).convertToSparseDatatype();
	mySparseM = (*MComplex).convertToSparseDatatype();
	
	std::cout << " Finished." << std::endl;
}

void KrylovROMSubstructure::assembleUmaMatrices(std::string _type) {
	//Assembly routine symmetric stiffness
	int FOM_DOF = myHMesh->getTotalNumOfDoFsRaw();
	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();
	for (int iElement = 0; iElement < numElements; iElement++)
	{
		int num_SparseK_row = allUMAElements[iElement]->myK_row.size();
		int num_SparseK_col = allUMAElements[iElement]->myK_col.size();
		int num_SparseM_row = allUMAElements[iElement]->myM_row.size();
		int num_SparseM_col = allUMAElements[iElement]->myM_col.size();
		int num_SparseSD_row = allUMAElements[iElement]->mySD_row.size();
		int num_SparseSD_col = allUMAElements[iElement]->mySD_col.size();

		std::cout << ">> \nSparse Info: " << std::endl;
		std::cout << "    Stiffness : row " << num_SparseK_row << " col: " << num_SparseK_col << std::endl;
		std::cout << "         Mass : row " << num_SparseM_row << " col: " << num_SparseM_col << std::endl;
		std::cout << "Struc Damping : row " << num_SparseSD_row << " col: " << num_SparseSD_col << std::endl;

		std::vector<int> internalDOFs;
		for (std::map<int, std::vector<int>>::iterator it = allUMAElements[iElement]->nodeToGlobalMap.begin(); it != allUMAElements[iElement]->nodeToGlobalMap.end(); ++it) {
			if (it->first >= 1000000000)
				for (int j = 0; j < it->second.size(); j++)
					internalDOFs.push_back(it->second[j]);
		}

		for (int i = 0; i < num_SparseK_row; i++) {
			if (_type == "FE_KMOR") {
				(*KComplex)(allUMAElements[iElement]->myK_row[i], allUMAElements[iElement]->myK_col[i]).real = allUMAElements[iElement]->getSparseStiffnessMatrix()(allUMAElements[iElement]->myK_row[i], allUMAElements[iElement]->myK_col[i]);
			}
		}
		for (int i = 0; i < num_SparseSD_row; i++) {
			if (_type == "FE_KMOR") {
				(*KComplex)(allUMAElements[iElement]->mySD_row[i], allUMAElements[iElement]->mySD_col[i]).imag = allUMAElements[iElement]->getSparseStructuralDampingMatrix()(allUMAElements[iElement]->mySD_row[i], allUMAElements[iElement]->mySD_col[i]);
			}
		}
		//K - omega*omega*M
		//Assembly routine symmetric mass
		if (_type == "FE_KMOR")
			for (int i = 0; i < num_SparseM_row; i++) {
				(*MComplex)(allUMAElements[iElement]->myM_row[i], allUMAElements[iElement]->myM_col[i]).real = allUMAElements[iElement]->getSparseMassMatrix()(allUMAElements[iElement]->myM_row[i], allUMAElements[iElement]->myM_col[i]);
			}


		//std::cout << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
		//std::cout << "clearing...\n";
		delete allUMAElements[iElement]->mySparseKReal;
		delete allUMAElements[iElement]->mySparseMReal;
		//std::cout << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
		allUMAElements[iElement]->myK_row.clear();
		allUMAElements[iElement]->myK_col.clear();
		allUMAElements[iElement]->myM_row.clear();
		allUMAElements[iElement]->myM_col.clear();
		//std::cout << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
		std::cout << " Finished." << std::endl;

	}
	/// new CSR
	mySparseK = (*KComplex).convertToSparseDatatype();
	mySparseM = (*MComplex).convertToSparseDatatype();
}

void KrylovROMSubstructure::buildProjectionMatManual() {
	addKrylovModesForExpansionPoint(myExpansionPoints, myKrylovOrder);
}

void KrylovROMSubstructure::addKrylovModesForExpansionPoint(std::vector<double>& _expPoint, int _krylovOrder) {
	int FOM_DOF = myHMesh->getTotalNumOfDoFsRaw();
	std::cout << ">> Adding krylov modes for expansion points..."<< std::endl;

	STACCATOComplexDouble ZeroComplex = {0,0};
	STACCATOComplexDouble OneComplex = { 1,0 };
	STACCATOComplexDouble NegOneComplex = { -1,0 };
	for (int iEP = 0; iEP < _expPoint.size(); iEP++)
	{
		double omega = 2 * M_PI*_expPoint[iEP];
		int lastIndexV = myV.size();
		int lastIndexZ = myZ.size();
		//myV.resize(lastIndexV + FOM_DOF*myInputDOFS.size()*_krylovOrder);
		//myZ.resize(lastIndexZ + FOM_DOF*myOutputDOFS.size()*_krylovOrder);

		// Zero CSR
		MathLibrary::SparseMatrix<STACCATOComplexDouble>* zeroMatrix = new MathLibrary::SparseMatrix<STACCATOComplexDouble>(FOM_DOF, true, false);
		(*zeroMatrix)(0, 0).real = 0;
		sparse_matrix_t zeroMat = (*zeroMatrix).convertToSparseDatatype();

		sparse_matrix_t K_tilde;
		STACCATOComplexDouble negativeOmegaSquare;
		negativeOmegaSquare.real = -omega * omega;
		negativeOmegaSquare.imag = 0;

		// K_tilde = -(2*pi*obj.exp_points(k))^2*obj.M + 1i*2*pi*obj.exp_points(k)*obj.D + obj.K;
		MathLibrary::computeSparseMatrixAdditionComplex(&mySparseM, &mySparseK, &K_tilde, false, true, negativeOmegaSquare);
		// K_tilde = - K_tilde
		STACCATOComplexDouble negativeOne;
		negativeOne.real = -1;
		negativeOne.imag = 0;
		MathLibrary::computeSparseMatrixAdditionComplex(&K_tilde, &zeroMat, &K_tilde, false, true, negativeOne);
		//MathLibrary::print_csr_sparse_z(&K_tilde);
		// D_tilde = 2i*2*pi*obj.exp_points(k)*obj.M + obj.D;
		sparse_matrix_t D_tilde;
		STACCATOComplexDouble complexTwoOmega;
		complexTwoOmega.real = 0;
		complexTwoOmega.imag = 2 * omega;
		MathLibrary::computeSparseMatrixAdditionComplex(&mySparseM, &zeroMat, &D_tilde, false, true, complexTwoOmega);
		// QV = - K_tilde\obj.B;               % Initial Search Direction
		std::vector<STACCATOComplexDouble> QV;
		QV.resize(FOM_DOF*myInputDOFS.size());
		factorizeSparseMatrixComplex(&K_tilde, true, true, myInputDOFS.size());
		solveDirectSparseComplex(&K_tilde, true, true, myInputDOFS.size(), &QV[0], &myB[0]);

		if (iEP == 0)
		{
			// QV = QV / norm(QV, 'fro');
			STACCATOComplexDouble inv_normQV;
			inv_normQV.real = 1/(MathLibrary::computeDenseMatrixFrobeniusNormComplex(&QV[0], FOM_DOF, myInputDOFS.size()));
			inv_normQV.imag = 0;
			
			std::vector<STACCATOComplexDouble> y;
			STACCATOComplexDouble zero; zero.real = 0; zero.imag = 0;
			y.resize(FOM_DOF*myInputDOFS.size(), zero);

			MathLibrary::computeDenseVectorAdditionComplex(&QV[0], &y[0], &inv_normQV, QV.size());

			QV.swap(y);
		}
		else {
			for (int jQV = 0; jQV < myInputDOFS.size(); jQV++)
			{
				for (int j = 0; j < myV.size()/FOM_DOF; j++)
				{
					// h = QV(:, jQV)'*V(:,j);
					STACCATOComplexDouble h;
					MathLibrary::computeDenseDotProductComplex(&QV[jQV*FOM_DOF], &myV[j*FOM_DOF], &h, FOM_DOF, true);
					// QV(:,j)'*V(:,j)
					STACCATOComplexDouble hV = { 0,0 };
					MathLibrary::computeDenseDotProductComplex(&myV[j*FOM_DOF], &myV[j*FOM_DOF], &hV, FOM_DOF, true);
					STACCATOComplexDouble h_by_hV;
					h_by_hV.real = -(h.real * hV.real + h.imag*hV.imag) / (hV.real*hV.real + hV.imag*hV.imag);
					h_by_hV.imag = -(h.imag*hV.real - h.real * hV.imag) / (hV.real*hV.real + hV.imag*hV.imag);
					// WQ(:, jWQ, i) = WQ(:, jWQ, i) - h * V(:, j) / (V(:, j)'*V(:,j));
					MathLibrary::computeDenseVectorAdditionComplex(&myV[j*FOM_DOF], &QV[jQV*FOM_DOF], &h_by_hV, FOM_DOF);
				}
			}
		}

		// WQ(:,:,1) = QV;
		std::vector<STACCATOComplexDouble> WQ;
		WQ.swap(QV);
		// V = [V WQ(:, : , 1)];
		myV.insert(myV.end(), WQ.begin(), WQ.end());

		for (int i = 1; i < _krylovOrder; i++)
		{
			// D_tilde*WQ(:,:,i-1)
			std::vector<STACCATOComplexDouble> WQ_i_D;
			WQ_i_D.resize(FOM_DOF*myInputDOFS.size());
			MathLibrary::computeSparseMatrixDenseMatrixMultiplicationComplex(myInputDOFS.size(), FOM_DOF, FOM_DOF, &D_tilde, &WQ[(i-1)*FOM_DOF*myInputDOFS.size()], &WQ_i_D[0], false, false, ZeroComplex, true, false);
			// obj.M*WQ(:,:,i-1))
			std::vector<STACCATOComplexDouble> WQ_i_K;
			WQ_i_K.resize(FOM_DOF*myInputDOFS.size());
			MathLibrary::computeSparseMatrixDenseMatrixMultiplicationComplex(myInputDOFS.size(), FOM_DOF, FOM_DOF, &mySparseM, &WQ[(i - 1)*FOM_DOF*myInputDOFS.size()], &WQ_i_K[0], false, false, ZeroComplex, true, false);

			// (D_tilde*WQ(:,:,i-1)+ obj.M*WQ(:,:,i-1));
			MathLibrary::computeDenseVectorAdditionComplex(&WQ_i_D[0], &WQ_i_K[0], &OneComplex, FOM_DOF*myInputDOFS.size());

			// WQ(:,:,i) = -K_tilde\(D_tilde*WQ(:,:,i-1)+ obj.M*WQ(:,:,i-1));
			std::vector<STACCATOComplexDouble> WQ_i;
			WQ_i.resize(FOM_DOF*myInputDOFS.size(), ZeroComplex);
			solveDirectSparseComplex(&K_tilde, true, true, myInputDOFS.size(), &WQ_i[0], &WQ_i_K[0]);
			WQ.insert(WQ.end(), WQ_i.begin(), WQ_i.end());

			// Gram-Schmidt orthogonalization to all previous WQ
			for (int j = 0; j <= i-1; j++)
			{
				// h = WQ(:,:,i)'*WQ(:,:,j);
				STACCATOComplexDouble h = {0,0};
				std::vector<STACCATOComplexDouble> hMat;
				hMat.resize(FOM_DOF*myInputDOFS.size(), ZeroComplex);
				MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(myInputDOFS.size(), FOM_DOF, FOM_DOF, &WQ[(i)*FOM_DOF*myInputDOFS.size()], &WQ[j*FOM_DOF*myInputDOFS.size()], &hMat[0], true, false, OneComplex, false, false, false);
				// WQ(:,:,i) = WQ(:,:,i) - WQ(:,:,j)*h;
				MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(FOM_DOF, myInputDOFS.size(), myInputDOFS.size(), &WQ[(j)*FOM_DOF*myInputDOFS.size()], &hMat[0], &WQ[(i)*FOM_DOF*myInputDOFS.size()], false, true, negativeOne, true, false, false);
			}
			// Gram - Schmidt orthogonalization of WQ_i against V
			for (int jWQ = 0; jWQ < myInputDOFS.size(); jWQ++)
			{
				for (int j = 0; j < myV.size()/FOM_DOF ; j++)
				{
					// h = WQ(:,jWQ,i)'*V(:,j);
					STACCATOComplexDouble h = { 0,0 };
					MathLibrary::computeDenseDotProductComplex(&WQ[(i)*FOM_DOF*myInputDOFS.size() +jWQ*FOM_DOF], &myV[j*FOM_DOF], &h, FOM_DOF, true);
					// V(:,j)'*V(:,j)
					STACCATOComplexDouble hV = { 0,0 };
					MathLibrary::computeDenseDotProductComplex(&myV[j*FOM_DOF], &myV[j*FOM_DOF], &hV, FOM_DOF, true);
					STACCATOComplexDouble h_by_hV;
					h_by_hV.real = -(h.real * hV.real + h.imag*hV.imag) / (hV.real*hV.real + hV.imag*hV.imag);
					h_by_hV.imag = -(h.imag*hV.real- h.real * hV.imag) / (hV.real*hV.real + hV.imag*hV.imag);
					// WQ(:, jWQ, i) = WQ(:, jWQ, i) - h * V(:, j) / (V(:, j)'*V(:,j));
					MathLibrary::computeDenseVectorAdditionComplex(&myV[j*FOM_DOF], &WQ[(i)*FOM_DOF*myInputDOFS.size() + jWQ * FOM_DOF], &h_by_hV, FOM_DOF);
				}
			}
			// V = [V WQ(:,:,i)];
			myV.insert(myV.end(), std::next(WQ.begin()+i*FOM_DOF*myInputDOFS.size()-1), std::next(WQ.begin()+(i+1)*FOM_DOF*myInputDOFS.size()-1));
			// [V,~]=qr(V,0);
			MathLibrary::computeDenseMatrixQRDecompositionComplex(FOM_DOF, myV.size()/FOM_DOF, &myV[0], false);
		}
	}
	// MIMO
	myZ.assign(myV.begin(), myV.end());
	std::cout << ">> Adding krylov modes for expansion points... Finished." << std::endl;
}

void  KrylovROMSubstructure::factorizeSparseMatrixComplex(const sparse_matrix_t* _mat, const bool _symmetric, const bool _positiveDefinite, int _nRHS) {
#ifdef USE_INTEL_MKL
	sparse_index_base_t indextype;
	exportCSRTimer01.start();
	sparse_status_t status = mkl_sparse_z_export_csr(*_mat, &indextype, &m, &n, &pointerB, &pointerE, &columns, &values);

	rowIndex = pointerB;
	rowIndex[m] = pointerE[m - 1];
	exportCSRTimer01.stop();
	std::cout << "> Export completed in : " << exportCSRTimer01.getDurationMilliSec() << " (milliSec)" << std::endl;

	// Checks
	if (_symmetric)
		pardiso_mtype = 6;		// complex and symmetric 
	else
		pardiso_mtype = 13;		// complex and unsymmetric matrix


	pardisoinit(pardiso_pt, &pardiso_mtype, pardiso_iparm);
	// set pardiso default parameters
	for (int i = 0; i < 64; i++) {
		pardiso_iparm[i] = 0;
	}

	pardiso_iparm[0] = 1;    // No solver defaults
	pardiso_iparm[1] = 3;    // Fill-in reordering from METIS 
	pardiso_iparm[9] = 13;   // Perturb the pivot elements with 1E-13
							 //pardiso_iparm[23] = 1;   // 2-level factorization
							 //pardiso_iparm[36] = -99; // VBSR format
	pardiso_iparm[17] = -1;	 // Output: Number of nonzeros in the factor LU
	pardiso_iparm[18] = -1;	 // Output: Report Mflops
	pardiso_iparm[19] = 0;	 // Output: Number of CG iterations
							 // pardiso_iparm[27] = 1;   // PARDISO checks integer arrays ia and ja. In particular, PARDISO checks whether column indices are sorted in increasing order within each row.
	pardiso_maxfct = 1;	    // max number of factorizations
	pardiso_mnum = 1;		// which factorization to use
	pardiso_msglvl = 0;		// do NOT print statistical information
	pardiso_neq = m;		// number of rows of 
	pardiso_error = 1;		// Initialize error flag 
	pardiso_nrhs = _nRHS;	// number of right hand side
	pardiso_iparm[12] = 1;   //Improved accuracy using (non-) symmetric weighted matching.
	int len = 198;
	char buf[198];
	mkl_get_version_string(buf, len);
	printf("%s\n", buf);
	printf("\n");

	mkl_set_num_threads(STACCATO::AuxiliaryParameters::solverMKLThreads); // set number of threads to 1 for mkl call only
	std::cout << "Matrixtype for PARDISO: " << pardiso_mtype << std::endl;
	std::cout << "#Threads   for PARDISO: " << mkl_get_max_threads() << std::endl;

	linearSolverTimer01.start();
	linearSolverTimer02.start();
	pardiso_phase = 11;
	pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase,
		&pardiso_neq, values, rowIndex, columns, &pardiso_idum,
		&pardiso_nrhs, pardiso_iparm, &pardiso_msglvl, &pardiso_ddum, &pardiso_ddum,
		&pardiso_error);
	linearSolverTimer01.stop();
	if (pardiso_error != 0) {
		std::cout << "Error pardiso reordering failed with error code: " << pardiso_error
			<< std::endl;
		exit(EXIT_FAILURE);
	}

	linearSolverTimer01.start();
	pardiso_phase = 22;
	pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase,
		&pardiso_neq, values, rowIndex, columns, &pardiso_idum,
		&pardiso_nrhs, pardiso_iparm, &pardiso_msglvl, &pardiso_ddum, &pardiso_ddum,
		&pardiso_error);
	linearSolverTimer01.stop();
	if (pardiso_error != 0) {
		std::cout << "Info: Number of zero or negative pivot = " << pardiso_iparm[29] << std::endl;
		std::cout << "Info: Number of nonzeros in factors = " << pardiso_iparm[17] << std::endl;
		std::cout << "Error pardiso factorization failed with error code: " << pardiso_error
			<< std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "Factorization completed: " << linearSolverTimer01.getDurationMilliSec() << " (milliSec)" << std::endl;
	std::cout << "Info: Number of equation = " << pardiso_neq << std::endl;
	std::cout << "Info: Number of nonzeros in factors = " << pardiso_iparm[17] << std::endl;
	std::cout << "Info: Number of factorization FLOPS = " << pardiso_iparm[18] * 1000000.0 << std::endl;
	std::cout << "Info: Total peak memory on numerical factorization and solution (Mb) = " << (pardiso_iparm[14] + pardiso_iparm[15] + pardiso_iparm[16]) / 1000 << std::endl;
	std::cout << "Info: Number of positive eigenvalues = " << pardiso_iparm[21] << std::endl;
	std::cout << "Info: Number of negative eigenvalues = " << pardiso_iparm[22] << std::endl;
	std::cout << "Info: Number of zero or negative pivot = " << pardiso_iparm[29] << std::endl;
#endif
}

void KrylovROMSubstructure::solveDirectSparseComplex(const sparse_matrix_t* _mat, const bool _symmetric, const bool _positiveDefinite, int _nRHS, STACCATOComplexDouble* _x, STACCATOComplexDouble* _b) {
	//Computes x=A\b
#ifdef USE_INTEL_MKL
	pardiso_nrhs = _nRHS;	// number of right hand side
	linearSolverTimer01.start();
	pardiso_phase = 33; // forward and backward substitution
	mkl_set_num_threads(STACCATO::AuxiliaryParameters::solverMKLThreads); // set number of threads to 1 for mkl call only
	pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase,
		&pardiso_neq, values, rowIndex, columns, &pardiso_idum,
		&pardiso_nrhs, pardiso_iparm, &pardiso_msglvl, _b, _x, &pardiso_error);
	if (pardiso_error != 0)
	{
		std::cout << "Error pardiso forward and backward substitution failed with error code: " << pardiso_error
			<< std::endl;
		exit(EXIT_FAILURE);
	}
	linearSolverTimer01.stop();
	std::cout << "Number of iterative refinement steps performed: " << pardiso_iparm[6] << std::endl;
	std::cout << "Forward and backward substitution completed: " << linearSolverTimer01.getDurationMilliSec() << " (milliSec)" << std::endl;
	linearSolverTimer02.stop();
	std::cout << "=== Complete duration PARDISO: " << linearSolverTimer02.getDurationMilliSec() << " (milliSec) for system dimension " << m << "x" << n << std::endl;
#endif
}

void KrylovROMSubstructure::generateROM() {
#ifdef USE_INTEL_MKL
	std::cout << ">> Generating ROM..." << std::endl;
	int FOM_DOF = myHMesh->getTotalNumOfDoFsRaw();
	int ROM_DOF = myExpansionPoints.size()*myKrylovOrder*myInputDOFS.size();	// Assuming MIMO
	STACCATOComplexDouble ZeroComplex = { 0,0 };
	STACCATOComplexDouble OneComplex = { 1,0 };

	myKComplexReduced.resize(ROM_DOF*ROM_DOF);
	myMComplexReduced.resize(ROM_DOF*ROM_DOF);
	myBReduced.resize(ROM_DOF*myInputDOFS.size());
	myCReduced.resize(myOutputDOFS.size()*ROM_DOF);
	// obj.K_R = obj.Z'*obj.K*obj.V;
	// obj.K*obj.V
	std::vector<STACCATOComplexDouble> y;
	y.resize(FOM_DOF*ROM_DOF, ZeroComplex);
	MathLibrary::computeSparseMatrixDenseMatrixMultiplicationComplex(ROM_DOF, FOM_DOF, FOM_DOF, &mySparseK, &myV[0], &y[0], false, false, OneComplex, true, false);
	// obj.K_R = obj.Z'*y
	MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, ROM_DOF, FOM_DOF, &myZ[0], &y[0], &myKComplexReduced[0], true, false, OneComplex, false, false, false);

	// obj.M_R = obj.Z'*obj.M*obj.V;
	// obj.M*obj.V
	y.clear();
	y.resize(FOM_DOF*ROM_DOF, ZeroComplex);
	MathLibrary::computeSparseMatrixDenseMatrixMultiplicationComplex(ROM_DOF, FOM_DOF, FOM_DOF, &mySparseM, &myV[0], &y[0], false, false, OneComplex, true, false);
	// obj.M_R = obj.Z'*y
	MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, ROM_DOF, FOM_DOF, &myZ[0], &y[0], &myMComplexReduced[0], true, false, OneComplex, false, false, false);

	// obj.B_R = obj.Z'*obj.B;
	MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, myInputDOFS.size(), FOM_DOF, &myZ[0], &myB[0], &myBReduced[0], true, false, OneComplex, false, false, false);

	// obj.C_R = obj.C*obj.V;
	MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(myOutputDOFS.size(), ROM_DOF, FOM_DOF, &myC[0], &myV[0], &myCReduced[0], false, false, OneComplex, false, false, false);
	
	AuxiliaryFunctions::writeMKLComplexVectorDatFormat("myKR.dat", myKComplexReduced);
	AuxiliaryFunctions::writeMKLComplexVectorDatFormat("myMR.dat", myMComplexReduced);
	AuxiliaryFunctions::writeMKLComplexVectorDatFormat("myBR.dat", myBReduced);
	AuxiliaryFunctions::writeMKLComplexVectorDatFormat("myCR.dat", myCReduced);

	std::cout << ">> Generating ROM... Finished." << std::endl;

#endif // USE_INTEL_MKL

}