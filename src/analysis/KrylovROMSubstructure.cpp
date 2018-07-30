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
			myB = new MathLibrary::SparseMatrix<int>(FOM_DOF, myInputDOFS.size());
			myC = new MathLibrary::SparseMatrix<int>(myOutputDOFS.size(), FOM_DOF);

			myBReduced.resize(ROM_DOF*myInputDOFS.size());
			myCReduced.resize(myOutputDOFS.size()*ROM_DOF);

			// Generate Input and Output Matrices
			for (int inpIter = 0; inpIter < myInputDOFS.size(); inpIter++) {
				(*myB)(myInputDOFS[inpIter], inpIter) = 1;
			}
			for (int outIter = 0; outIter < myOutputDOFS.size(); outIter++) {
				(*myC)(outIter, myOutputDOFS[outIter]) = 1;
			}

			// Assemble global FOM system matrices
			if (std::string(iterParts->PART()[iPart].FILEIMPORT().begin()->Type()->c_str()) == "AbqODB") {
				std::cout << ">> Assembling FOM system matrices... " << std::endl;
				assembleGlobalMatrices(std::string(std::string(iterParts->PART()[iPart].TYPE()->data())));
			}
			else if (std::string(iterParts->PART()[iPart].FILEIMPORT().begin()->Type()->c_str()) == "AbqSIM") {
				std::cerr << "NOT IMPLEMENTED: Assign FOM system matrices with UMA element matrices.";
			}

			/*std::vector<double> tempLoadVec(FOM_DOF);

			std::vector<double> inputLoad;
			inputLoad.resize(inputDOFS.size());
			for (int i = 0; i < inputDOFS.size(); i++) {
				inputLoad[i] = tempLoadVec[inputDOFS[i]];
			}*/

			if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->EXP_POINTS().begin()->Type()->c_str()) == "MANUAL") {
				std::cout << ">> Building projection basis WITHOUT automated MOR..." << std::endl;
				buildProjectionMatManual();
			}
			else if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->EXP_POINTS().begin()->Type()->c_str()) == "AUTO") {
				std::cerr << "NOT IMPLEMENTED: Building projection basis WITH automated MOR" << std::endl;
			}

			//-----------------------------------------------------------------------------
			// Dot Product
			std::vector<double> vec1 = { 1,2,5,7,8 };
			std::vector<double> vec2 = { 9,10,11,12,13 };

			std::cout << "Test Dot Product vec1.vec2 = " << MathLibrary::computeDenseDotProduct(&vec1[0], &vec2[0], vec1.size()) << std::endl;

			// Dense Mat QR Decomposition
			std::vector<double> vecqr = { 1,2, 3,4, 5,6, 7,8, 9, 10};
			int nrow = 5;
			int ncol = 2;
			MathLibrary::computeDenseMatrixQRDecomposition(nrow, ncol, &vecqr[0]);

			std::cout << "QR Decomposition: " << std::endl;;
			for (int i = 0; i < nrow; i++)
			{
				for (int j = 0; j < ncol; j++)
				{
					std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1) << vecqr[i * ncol + j] << " , ";
				}
				std::cout << std::endl;
			}

			std::vector<double> vecortho;
			vecortho.resize(ncol*ncol);

			MathLibrary::computeDenseMatrixMatrixMultiplication(ncol, ncol, nrow, &vecqr[0], &vecqr[0], &vecortho[0], true, false, 1, false, false);
			std::cout << "Check Orthogonality: " << std::endl;;
			for (int i = 0; i < ncol; i++)
			{
				for (int j = 0; j < ncol; j++)
				{
					std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1) << vecortho[i * ncol + j] << " , ";
				}
				std::cout << std::endl;
			}

			// Sparse Mat + Sparse Mat Addition
			MathLibrary::SparseMatrix<MKL_Complex16>* Mat1 = new MathLibrary::SparseMatrix<MKL_Complex16>(3, true, true);
			MathLibrary::SparseMatrix<MKL_Complex16>* Mat2 = new MathLibrary::SparseMatrix<MKL_Complex16>(3, true, true);

			(*Mat1)(0, 0).real = 2;
			(*Mat1)(0, 0).imag = 10;
			(*Mat1)(1, 0).real = 11;
			(*Mat1)(1, 1).real = -2;
			(*Mat1)(1, 1).imag = -20;
			(*Mat1)(2, 2).real = 10;

			(*Mat2)(0, 0).real = 5;
			(*Mat2)(0, 0).imag = 7;
			(*Mat2)(0, 1).imag = 1;
			(*Mat2)(1, 1).real = 9;
			(*Mat2)(1, 1).imag = -1;


			//(*Mat1).writeSparseMatrixToFile("Test_Mat1", "dat");
			//(*Mat2).writeSparseMatrixToFile("Test_Mat2", "dat");

			//(*Mat1).createMKLSparseCSR();
			//(*Mat1).SparseSparseAddition(Mat2);

			MathLibrary::computeSparseMatrixAddition(Mat1, Mat2);
			//-----------------------------------------------------------------------------

			exit(0);
		}
	}


	/*// Reading in the frequency range
	std::vector<double> freq;
	for (STACCATO_XML::ANALYSIS_const_iterator iAnalysis(MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin());
		iAnalysis != MetaDatabase::getInstance()->xmlHandle->ANALYSIS().end();
		++iAnalysis)
	{
		std::cout << std::endl << "==== Starting Anaylsis: " << iAnalysis->NAME()->data() << " ====" << std::endl;

		std::string analysisType = iAnalysis->TYPE()->data();
		if (analysisType != "KRYLOV_ROM") {
			// Routine to access expansion points
			if (std::string(iAnalysis->FREQUENCY().begin()->Type()->data()) == "RANGE") {		// Frequency range of interest for Expansion points
				double start_freq = std::atof(iAnalysis->FREQUENCY().begin()->START_FREQ()->c_str());
				double end_freq = std::atof(iAnalysis->FREQUENCY().begin()->END_FREQ()->c_str());
				freq.push_back(start_freq);		// Push back starting frequency
				freq.push_back(end_freq);		// Push back ending frequency
			}
		}
		else {
			std::cerr << ">> Error while recognizing analysis type: Invalid analysis type.\n";
		}

		// Instantiating Stiffness and Mass Matrix
		int totalDoF = myHMesh->getTotalNumOfDoFsRaw();
		if (analysisType == "KRYLOV_ROM_REAL") {
			//KReal = new MathLibrary::SparseMatrix<double>(totalDoF, true, true);
			//MReal = new MathLibrary::SparseMatrix<double>(totalDoF, true, true);
		}
		else if (analysisType == "KRYLOV_ROM") {
			KComplex = new MathLibrary::SparseMatrix<MKL_Complex16>(totalDoF, true, true);
			MComplex = new MathLibrary::SparseMatrix<MKL_Complex16>(totalDoF, true, true);
		}
		
		// Assembling element stiffness and mass matrices
		assembleGlobalMatrices(analysisType);

		std::vector<int> inputDOFS;
	}*/


}

KrylovROMSubstructure::~KrylovROMSubstructure() {
}

void KrylovROMSubstructure::assignMaterialToElements() {
	anaysisTimer01.start();

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();
	std::cout << ">> Num Nodes   : " << numNodes << "\n>> Num Elements: " << numElements << std::endl;

	myAllElements.resize(numElements);
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

		if (!allElemsLabel.empty() && matElCount == numElements) {
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
					myAllElements[elemIndex] = new FeUmaElement(elasticMaterial);
#endif
				}
				int numNodesPerElement = myHMesh->getNumNodesPerElement()[elemIndex];
				double*  eleCoords = &myHMesh->getNodeCoordsSortElement()[lastIndex];
				myAllElements[elemIndex]->computeElementMatrix(eleCoords);
				lastIndex += numNodesPerElement * myHMesh->getDomainDimension();
			}
		}
		else
			std::cerr << ">> Error while assigning Material to element sets: Not all elements are assigned with a defined material." << std::endl;
	}

	std::cout << ">> Section Material Assignment is Complete." << std::endl;
	anaysisTimer01.stop();
	infoOut << "Duration for element loop: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
}

void KrylovROMSubstructure::assembleGlobalMatrices(std::string _type) {

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();
	int lastIndex = 0;

	// Assembling Element Stiffness Matrices
	std::cout << ">> Building System Matrices...";
	for (int iElement = 0; iElement < numElements; iElement++)
	{
		int numDoFsPerElement = myHMesh->getNumDoFsPerElement()[iElement];
		int*  eleDoFs = &myHMesh->getElementDoFListRestricted()[lastIndex];
		lastIndex += numDoFsPerElement;

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
					}
				}
			}
		}
		//Assembly routine symmetric mass
		for (int i = 0; i < numDoFsPerElement; i++) {
			if (eleDoFs[i] != -1) {
				for (int j = 0; j < numDoFsPerElement; j++) {
					if (eleDoFs[j] >= eleDoFs[i] && eleDoFs[j] != -1) {
						//K(1+eta*i) - omega*omega*M
						if (_type == "FE_KMOR_REAL") {
							//(*MReal)(eleDoFs[i], eleDoFs[j]) -= myAllElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j] * omega*omega;
						}
						else if (_type == "FE_KMOR") {
							(*MComplex)(eleDoFs[i], eleDoFs[j]).real -= myAllElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j];
						}
					}
				}
			}
		}
	}

	std::cout << " Finished." << std::endl;
}

void KrylovROMSubstructure::buildProjectionMatManual() {
	for (int iEP = 0; iEP < myExpansionPoints.size(); iEP++) {
		addKrylovModesForExpansionPoint(myExpansionPoints[iEP], myKrylovOrder);
	}
}

void KrylovROMSubstructure::addKrylovModesForExpansionPoint(double _expPoint, int _krylovOrder) {
	std::cout << ">> Adding krylov modes for expansion points...";
	int lastIndexV = myV.size();
	int lastIndexZ = myZ.size();
	myV.resize(lastIndexV + myInputDOFS.size()*_krylovOrder);
	myZ.resize(lastIndexZ + myOutputDOFS.size()*_krylovOrder);

	MathLibrary::SparseMatrix<MKL_Complex16> *K_tile = new MathLibrary::SparseMatrix<MKL_Complex16>(myHMesh->getTotalNumOfDoFsRaw(), true, true);
	
	double omega = 2 * M_PI*_expPoint;

	std::cout << " Finished." << std::endl;
}