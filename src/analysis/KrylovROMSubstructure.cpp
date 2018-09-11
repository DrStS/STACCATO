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

#include "SimuliaUMA.h"

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
	/* -- Properties of KMOR ---- */
	isMIMO = false;
	enablePropDamping = false;
	/* -------------------------- */
	
	/* -- Exporting ------------- */
	writeFOM = false;
	writeROM = true;
	exportRHS = true;
	exportSolution = true;
	writeTransferFunctions = false;
	writeProjectionmatrices = false;
	/* -------------------------- */

	// Part Reduction
	STACCATO_XML::PARTS_const_iterator iterParts(MetaDatabase::getInstance()->xmlHandle->PARTS().begin());
	for (int iPart = 0; iPart < iterParts->PART().size(); iPart++)
	{

		/* %%% Build FOM - Builds type sparse_matrix_t K, M, D %%% */
		exportCSRTimer01.start();
		if (std::string(iterParts->PART()[iPart].FILEIMPORT().begin()->Type()->c_str()) == "AbqODB")
		{
			myModelType = "FOM_ODB";
			buildAbqODB();
		}
		else if (std::string(iterParts->PART()[iPart].FILEIMPORT().begin()->Type()->c_str()) == "AbqSIM") {
			myModelType = "FOM_SIM";
			buildAbqSIM(iPart);
		}
		else {
			myModelType = "UNSUPPORTED";
			FOM_DOF = -1;
		}
		std::cout << "Physical memory consumption after FOM build: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
		exportCSRTimer01.stop();
		std::cout << " --> Duration loading FOM: " << exportCSRTimer01.getDurationMilliSec() << " milliSec" << std::endl;

		if (enablePropDamping)		// Note Dev: Hack; Implement essential XML
		{
			STACCATOComplexDouble alpha = { 100,0 };
			STACCATOComplexDouble beta = { 1e-5, 0 };
			// Zero CSR
			MathLibrary::SparseMatrix<STACCATOComplexDouble>* zeroMatrix = new MathLibrary::SparseMatrix<STACCATOComplexDouble>(FOM_DOF, true, false);
			(*zeroMatrix)(0, 0).real = 0;
			sparse_matrix_t zeroMat = (*zeroMatrix).convertToSparseDatatype();

			sparse_matrix_t betaK;
			MathLibrary::computeSparseMatrixAdditionComplex(&mySparseK, &zeroMat, &betaK, false, true, beta);

			MathLibrary::computeSparseMatrixAdditionComplex(&mySparseM, &betaK, &mySparseD, false, true, alpha);
		}

		/* %%% Execute Reduction %%% */
		if (std::string(iterParts->PART()[iPart].TYPE()->data()) == "FE_KMOR")
		{
			//myHMesh->isKROM = true;
			std::cout << ">> KMOR procedure to be performed on FE part: " << std::string(iterParts->PART()[iPart].Name()->data()) << std::endl;
			currentPart = std::string(iterParts->PART()[iPart].Name()->data());
			// Getting ROM prerequisites
			/// Exapansion points
			if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->EXP_POINTS().begin()->Type()->c_str()) == "MANUAL") {
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
					if (myModelType == "FOM_ODB")
					{
						std::vector<int> nodeSet = myHMesh->convertNodeSetNameToLabels(std::string(iterParts->PART()[iPart].ROMDATA().begin()->INPUTS().begin()->NODESET()[iNodeSet].Name()->c_str()));
						// Insert nodeSet entries to DOFS
						for (int jNodeSet = 0; jNodeSet < nodeSet.size(); jNodeSet++) {
							std::vector<int> dofIndices = myHMesh->getNodeIndexToDoFIndices()[myHMesh->convertNodeLabelToNodeIndex(nodeSet[jNodeSet])];
							myInputDOFS.insert(myInputDOFS.end(), dofIndices.begin(), dofIndices.end());
						}
					}
					else if (myModelType == "FOM_SIM") {
						auto search = nodeSetsMap.find(std::string(iterParts->PART()[iPart].ROMDATA().begin()->INPUTS().begin()->NODESET()[iNodeSet].Name()->c_str()));
						if (search != nodeSetsMap.end())
							myInputDOFS.insert(myInputDOFS.end(), search->second.begin(), search->second.end());
						else
							std::cout << "!! InputNodeSet not found!";
					}
				}
			}

			/// Outputs
			if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->OUTPUTS().begin()->Type()->c_str()) == "NODES") {
				std::cout << " !! Output DOFs found! Unsymmetric MIMO not yet supported." << std::endl;
				exit(EXIT_FAILURE);
			}
			else if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->OUTPUTS().begin()->Type()->c_str()) == "MIMO") {
				myOutputDOFS = myInputDOFS;
				isMIMO = true;
			}

			// Size prediction
			ROM_DOF = myExpansionPoints.size()*myKrylovOrder*myInputDOFS.size();	// Assuming MIMO
			std::cout << ">> -- ROM Data WITHOUT Deflation --" << std::endl;
			displayModelSize();

			generateInputOutputMatricesForFOM();

			/* %%% Reduce FOM to ROM %%% */
			anaysisTimer01.start();
			anaysisTimer02.start();
			if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->EXP_POINTS().begin()->Type()->c_str()) == "MANUAL") {
				std::cout << ">> Building projection basis WITHOUT automated MOR..." << std::endl;
				buildProjectionMatManual();
				std::cout << ">> Building projection basis WITHOUT automated MOR... Finished." << std::endl;
			}
			else if (std::string(iterParts->PART()[iPart].ROMDATA().begin()->EXP_POINTS().begin()->Type()->c_str()) == "AUTO") {
				std::cerr << "NOT IMPLEMENTED: Building projection basis WITH automated MOR" << std::endl;
			}
			anaysisTimer02.stop();
			std::cout << " --> Duration building projection matrices: " << anaysisTimer02.getDurationMilliSec() << " milliSec" << std::endl;

			// Size determination
			ROM_DOF = myV.size() / FOM_DOF;
			std::cout << ">> -- ROM Data With Deflation --" << std::endl;
			displayModelSize();

			anaysisTimer02.start();
			generateROM();
			anaysisTimer02.stop();
			anaysisTimer01.stop();
			std::cout << " --> Duration generating MOR: " << anaysisTimer02.getDurationMilliSec() << " milliSec" << std::endl;
			std::cout << " --> Duration for krylov reduced model generation of model " << std::string(iterParts->PART()[iPart].Name()->c_str()) << " : " << anaysisTimer01.getDurationMilliSec() << " ms" << std::endl;

			/* %%% Output ROM %%% */
			if (writeROM)
				exportROMToFiles();
		}
	}

	// The following is will have to be separeted from above reduction
	/* %% Performing Anaylsis (Back-Transformation) %%% */
	performAnalysis();
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
				int numNodesPerElement = myHMesh->getNumNodesPerElement()[elemIndex];
				double*  eleCoords = &myHMesh->getNodeCoordsSortElement()[lastIndex];

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

void KrylovROMSubstructure::getSystemMatricesODB() {
	MathLibrary::SparseMatrix<MKL_Complex16> *KComplex = new MathLibrary::SparseMatrix<MKL_Complex16>(FOM_DOF, true, true);
	MathLibrary::SparseMatrix<MKL_Complex16> *MComplex = new MathLibrary::SparseMatrix<MKL_Complex16>(FOM_DOF, true, true);

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();
	int lastIndex = 0;

	// Assembling Element Stiffness Matrices
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
						(*KComplex)(eleDoFs[i], eleDoFs[j]).real += myAllElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j];
						(*KComplex)(eleDoFs[i], eleDoFs[j]).imag += myAllElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j] * myAllElements[iElement]->getMaterial()->getDampingParameter();

						(*MComplex)(eleDoFs[i], eleDoFs[j]).real += myAllElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j];

					}
				}
			}
		}
	}

	if (writeFOM)	{
		(*KComplex).writeSparseMatrixToFile("Staccato_Sparse_Stiffness_ODB", "dat");
		(*MComplex).writeSparseMatrixToFile("Staccato_Sparse_Mass_ODB", "dat");
	}

	/// new CSR
	mySparseK = (*KComplex).convertToSparseDatatype();
	mySparseM = (*MComplex).convertToSparseDatatype();
}

void KrylovROMSubstructure::getSystemMatricesSIM() {

	std::vector<std::string> importSIMMatrices;
	importSIMMatrices.push_back("stiffness");
	importSIMMatrices.push_back("mass");
	importSIMMatrices.push_back("structuraldamping");

	for (size_t i = 0; i < importSIMMatrices.size(); i++)
	{
		std::cout << " > Imporing " << importSIMMatrices[i] << " SIM Matrix..." << std::endl;
		if (importSIMMatrices[i] == "stiffness") {
			acquireSparseMatrix(importSIMMatrices[i] , stiffnessCSR);
		}
		else if (importSIMMatrices[i] == "mass") {
			acquireSparseMatrix(importSIMMatrices[i], massCSR);
		}
		else if (importSIMMatrices[i] == "structuraldamping") {
			acquireSparseMatrix(importSIMMatrices[i], structdampingCSR);
		}
	}
}

void KrylovROMSubstructure::acquireSparseMatrix(std::string _key, KrylovROMSubstructure::csrStruct& _struct) {
	myUMAReader->getSparseMatrixCSR(_key, _struct.csr_ia, _struct.csr_ja, _struct.csr_values, writeFOM);

	if (_struct.csr_values.size() != 0) {
		std::cout << ">> Sparse Info " << _key << ": nnz = " << _struct.csr_values.size() << ". Size: " << _struct.csr_ia.size() - 1 << "x" << _struct.csr_ia.size() - 1 << std::endl;
		for (int i = 0; i < _struct.csr_ia.size() - 1; i++)
		{
			_struct.csrPointerB.push_back(_struct.csr_ia[i]);
			_struct.csrPointerE.push_back(_struct.csr_ia[i + 1]);
		}

		if (_key == "stiffness") {
			MathLibrary::createSparseCSRComplex(&mySparseK, _struct.csr_ia.size() - 1, _struct.csr_ia.size() - 1, &_struct.csrPointerB[0], &_struct.csrPointerE[0], &_struct.csr_ja[0], &_struct.csr_values[0]);
		}
		else if (_key == "mass") {
			MathLibrary::createSparseCSRComplex(&mySparseM, _struct.csr_ia.size() - 1, _struct.csr_ia.size() - 1, &_struct.csrPointerB[0], &_struct.csrPointerE[0], &_struct.csr_ja[0], &_struct.csr_values[0]);
		}
		else if (_key == "structuraldamping") {
			sparse_matrix_t sparseSD;
			MathLibrary::createSparseCSRComplex(&sparseSD, _struct.csr_ia.size() - 1, _struct.csr_ia.size() - 1, &_struct.csrPointerB[0], &_struct.csrPointerE[0], &_struct.csr_ja[0], &_struct.csr_values[0]);
			STACCATOComplexDouble ComplexOne = { 0,1 };
			MathLibrary::computeSparseMatrixAdditionComplex(&sparseSD, &mySparseK, &mySparseK, false, true, ComplexOne);
		}
		std::cout << " > Imporing " << _key << " SIM Matrix... Finished." << std::endl;
	}
	else
		std::cout << " > Imporing " << _key << " SIM Matrix... Skipped." << std::endl;
}

void KrylovROMSubstructure::buildProjectionMatManual() {
	addKrylovModesForExpansionPoint(myExpansionPoints, myKrylovOrder);

	if (writeProjectionmatrices) {

		ROM_DOF = myV.size()/ FOM_DOF;

		std::string filename = "C://software//repos//staccato//scratch//";
		AuxiliaryFunctions::writeMKLComplexDenseMatrixMtxFormat(filename + currentPart + "_myV.dat", myV, FOM_DOF, ROM_DOF, false); 
		if(!isMIMO)
			AuxiliaryFunctions::writeMKLComplexDenseMatrixMtxFormat(filename + currentPart + "_myV.dat", myZ, FOM_DOF, ROM_DOF, false);

	}
}

void KrylovROMSubstructure::addKrylovModesForExpansionPoint(std::vector<double>& _expPoint, int _krylovOrder) {
	std::cout << ">> Adding krylov modes for expansion points..."<< std::endl;
		
	STACCATOComplexDouble ZeroComplex = {0,0};
	STACCATOComplexDouble OneComplex = { 1,0 };
	STACCATOComplexDouble NegOneComplex = { -1,0 };

	std::cout << " >> Block Arnoldi Algorithm with BLOCK-wise Gram-Schmidt and WITH deflation strategy..." << std::endl;

	for (int iEP = 0; iEP < _expPoint.size(); iEP++)
	{
		double sigTol = 1e-09;
		std::cout << "  > Deflation Tolerance: " << sigTol << std::endl;
		std::cout << "  -------------------------------------------------------> Processing expansion point " << _expPoint[iEP] << " Hz..." << std::endl;
		double progress = (iEP + 1) * 100 / _expPoint.size();
		double omega = 2 * M_PI*_expPoint[iEP];

		// Zero CSR
		MathLibrary::SparseMatrix<STACCATOComplexDouble>* zeroMatrix = new MathLibrary::SparseMatrix<STACCATOComplexDouble>(FOM_DOF, true, false);
		(*zeroMatrix)(0, 0).real = 0;
		sparse_matrix_t zeroMat = (*zeroMatrix).convertToSparseDatatype();

		sparse_matrix_t K_tilde;
		STACCATOComplexDouble negativeOmegaSquare = { -omega * omega ,0 };
		STACCATOComplexDouble complexOmega = { 0, omega };

		// K_tilde = -(2*pi*obj.exp_points(k))^2*obj.M + 1i*2*pi*obj.exp_points(k)*obj.D + obj.K;
		MathLibrary::computeSparseMatrixAdditionComplex(&mySparseM, &mySparseK, &K_tilde, false, true, negativeOmegaSquare);
		if (enablePropDamping)
			MathLibrary::computeSparseMatrixAdditionComplex(&mySparseD, &K_tilde, &K_tilde, false, true, complexOmega);

		// K_tilde = - K_tilde
		STACCATOComplexDouble negativeOne;
		negativeOne.real = -1;
		negativeOne.imag = 0;
		MathLibrary::computeSparseMatrixAdditionComplex(&K_tilde, &zeroMat, &K_tilde, false, true, negativeOne);
		//MathLibrary::print_csr_sparse_z(&K_tilde);
		// QV = - K_tilde\obj.B;               % Initial Search Direction
		std::vector<STACCATOComplexDouble> QV;
		QV.resize(FOM_DOF*myInputDOFS.size());

		factorizeSparseMatrixComplex(&K_tilde, true, true, myInputDOFS.size());
		solveDirectSparseComplex(&K_tilde, true, true, myInputDOFS.size(), &QV[0], &myB[0]);

		// Orthogonalization of first set of vectors : iterative
		// procedure
		// Modification : if k >1 : also orthogonalize to all previous
		// vectors from V
		if (iEP > 0)
		{
			// q0 = QV;
			std::vector<STACCATOComplexDouble> q0;
			q0.insert(q0.end(), QV.begin(), QV.end());

			// eta = 1/sqrt(2);
			STACCATOComplexDouble eta = { 1 / std::sqrt(2),0 };

			STACCATOComplexDouble lim = { 0,0 };

			// qk_ = q0;
			std::vector<STACCATOComplexDouble> qk_;
			qk_.insert(qk_.end(), q0.begin(), q0.end());

			std::vector<STACCATOComplexDouble> qk;
			qk.resize(FOM_DOF*myInputDOFS.size());

			while (lim.real <= eta.real)
			{
				// sk = V'*qk_;
				std::vector<STACCATOComplexDouble> sk;
				sk.resize((myV.size() / FOM_DOF)*myInputDOFS.size());
				MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(myV.size() / FOM_DOF, myInputDOFS.size(), FOM_DOF, &myV[0], &qk_[0], &sk[0], true, false, OneComplex, false, false, false);

				// uk = -V*sk;
				std::vector<STACCATOComplexDouble> uk;
				uk.resize(FOM_DOF*myInputDOFS.size());
				MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(FOM_DOF, myInputDOFS.size(), myV.size() / FOM_DOF, &myV[0], &sk[0], &uk[0], false, true, NegOneComplex, false, false, false);

				// qk = qk_ + uk;
				MathLibrary::computeDenseVectorAdditionComplex(&qk_[0], &uk[0], &OneComplex, qk_.size());
				MathLibrary::copyDenseVectorComplex(&qk[0], &uk[0], qk.size());

				// lim = norm(qk, 'fro')/norm(qk_, 'fro');
				lim.real = MathLibrary::computeDenseMatrixFrobeniusNormComplex(&uk[0], FOM_DOF, myInputDOFS.size()) / MathLibrary::computeDenseMatrixFrobeniusNormComplex(&qk_[0], FOM_DOF, myInputDOFS.size());

				// qk_ = qk;
				MathLibrary::copyDenseVectorComplex(&qk_[0], &qk[0], qk_.size());
			}
			q0.clear();
			qk_.clear();
			// Q = qk;
			MathLibrary::copyDenseVectorComplex(&QV[0], &qk[0], QV.size());
			qk.clear();

		}
		std::cout << "  -------------------------------------------------------> Processing krylov order 1" << std::endl;
		// [Q,R,~]=qr(QV,0); 

		std::vector<STACCATOComplexDouble> tempTAU;
		MathLibrary::computeDenseMatrixPivotedQR_R_DecompositionComplex(FOM_DOF, myInputDOFS.size(), &QV[0], false, tempTAU);

		// Algo for rank-reveiling
		// ind=find(diag(abs(R))>sigTol);
		//std::cout << "  > Reveiling Rank: " << std::endl;
		int RR = reveilRankQR_R(&QV[0], FOM_DOF, myInputDOFS.size(), sigTol);
		//std::cout << "  > Rank reveiled: " << RR << std::endl;

		MathLibrary::computeDenseMatrixQR_Q_DecompositionComplex(FOM_DOF, myInputDOFS.size(), &QV[0], false, tempTAU);

		// Algo to eliminate dependent rows and columns
		QV.resize(FOM_DOF*RR);

		if (RR == 0) {
			std::cout << "  -------------------------------------------------------> RRQR: NO modes added: " << std::endl;
			continue;
		}
		else
			std::cout << "  -------------------------------------------------------> RRQR: Number of modes added: " << RR << std::endl;

		// V=[V Q(:,ind)];
		myV.insert(myV.end(), QV.begin(), QV.end());

		for (size_t iKr = 1; iKr < _krylovOrder; iKr++)
		{
			std::cout << "  -------------------------------------------------------> Processing krylov order " << iKr + 1 << std::endl;
			// obj.M*QV
			std::vector<STACCATOComplexDouble> WQ_i_K;
			WQ_i_K.resize(FOM_DOF*RR);
			MathLibrary::computeSparseMatrixDenseMatrixMultiplicationComplex(RR, FOM_DOF, FOM_DOF, &mySparseM, &QV[0], &WQ_i_K[0], false, false, ZeroComplex, true, false);

			// Q = -K_tilde\(obj.M*Q);
			solveDirectSparseComplex(&K_tilde, true, true, RR, &QV[0], &WQ_i_K[0]);
			WQ_i_K.clear();

			// w0 = Q; eta = 1/sqrt(2); lim = 0; wk_=w0;
			std::vector<STACCATOComplexDouble> w0;
			w0.insert(w0.end(), QV.begin(), QV.end());
			STACCATOComplexDouble eta = { 1 / std::sqrt(2),0 };

			STACCATOComplexDouble lim = { 0,0 };
			std::vector<STACCATOComplexDouble> wk_;
			wk_.insert(wk_.end(), w0.begin(), w0.end());

			std::vector<STACCATOComplexDouble> wk;
			wk.resize(FOM_DOF*RR);

			while (lim.real <= eta.real)
			{
				// sk = V'*wk_;
				std::vector<STACCATOComplexDouble> sk;
				sk.resize((myV.size() / FOM_DOF)*RR);
				MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(myV.size() / FOM_DOF, RR, FOM_DOF, &myV[0], &wk_[0], &sk[0], true, false, OneComplex, false, false, false);

				// uk = -V*sk;
				std::vector<STACCATOComplexDouble> uk;
				uk.resize(FOM_DOF*RR);
				MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(FOM_DOF, RR, myV.size() / FOM_DOF, &myV[0], &sk[0], &uk[0], false, true, NegOneComplex, false, false, false);

				// wk = wk_ + uk;
				MathLibrary::computeDenseVectorAdditionComplex(&wk_[0], &uk[0], &OneComplex, wk_.size());
				MathLibrary::copyDenseVectorComplex(&wk[0], &uk[0], wk.size());

				// lim = norm(wk, 'fro')/norm(wk_, 'fro');
				lim.real = MathLibrary::computeDenseMatrixFrobeniusNormComplex(&uk[0], FOM_DOF, RR) / MathLibrary::computeDenseMatrixFrobeniusNormComplex(&wk_[0], FOM_DOF, RR);

				// wk_ = wk;
				MathLibrary::copyDenseVectorComplex(&wk_[0], &wk[0], wk_.size());
			}
			// Q = wk;
			w0.clear();
			wk_.clear();
			MathLibrary::copyDenseVectorComplex(&QV[0], &wk[0], QV.size());
			wk.clear();

			// [Q,R,~]=qr(Q,0);
			std::vector<STACCATOComplexDouble> tau;
			MathLibrary::computeDenseMatrixPivotedQR_R_DecompositionComplex(FOM_DOF, QV.size() / FOM_DOF, &QV[0], false, tau);

			// ind=find(diag(abs(R))>sigTol);
			RR = reveilRankQR_R(&QV[0], FOM_DOF, RR, sigTol);
			MathLibrary::computeDenseMatrixQR_Q_DecompositionComplex(FOM_DOF, QV.size() / FOM_DOF, &QV[0], false, tau);


			if (RR == 0) {
				std::cout << "  -------------------------------------------------------> RRQR: NO modes added: " << std::endl;
				iKr = _krylovOrder;		// Break further iterations
			}
			else {
				std::cout << "  -------------------------------------------------------> RRQR: Number of modes added: " << RR << std::endl;
				QV.resize(FOM_DOF*RR);
				myV.insert(myV.end(), QV.begin(), QV.end());
			}
		}

		cleanPardiso();
		std::cout << (int)progress << "% completed; Performed " << iEP + 1 << " of " << _expPoint.size() << "." << std::endl;
	}
	// MIMO
	if (!isMIMO)
		myZ.assign(myV.begin(), myV.end());
	
	std::cout << ">> Adding krylov modes for expansion points... Finished." << std::endl;
}

void  KrylovROMSubstructure::factorizeSparseMatrixComplex(const sparse_matrix_t* _mat, const bool _symmetric, const bool _positiveDefinite, int _nRHS) {
#ifdef USE_INTEL_MKL
	sparse_index_base_t indextype;
	sparse_status_t status = mkl_sparse_z_export_csr(*_mat, &indextype, &m, &n, &pointerB, &pointerE, &columns, &values);

	rowIndex = pointerB;
	rowIndex[m] = pointerE[m - 1];

	sparse_checker_error_values check_err_val;
	sparse_struct pt;
	int error = 0;

	sparse_matrix_checker_init(&pt);
	pt.n = m;
	pt.csr_ia = rowIndex;
	pt.csr_ja = columns;
	pt.indexing = MKL_ONE_BASED;
	pt.matrix_structure = MKL_UPPER_TRIANGULAR;
	pt.print_style = MKL_C_STYLE;
	pt.message_level = MKL_PRINT;
	check_err_val = sparse_matrix_checker(&pt);

	printf("Matrix check details: (%d, %d, %d)\n", pt.check_result[0], pt.check_result[1], pt.check_result[2]);
	if (check_err_val == MKL_SPARSE_CHECKER_NONTRIANGULAR) {
		printf("Matrix check result: MKL_SPARSE_CHECKER_NONTRIANGULAR\n");
		error = 0;
	}
	else {
		if (check_err_val == MKL_SPARSE_CHECKER_SUCCESS) { printf("Matrix check result: MKL_SPARSE_CHECKER_SUCCESS\n"); }
		if (check_err_val == MKL_SPARSE_CHECKER_NON_MONOTONIC) { printf("Matrix check result: MKL_SPARSE_CHECKER_NON_MONOTONIC\n"); }
		if (check_err_val == MKL_SPARSE_CHECKER_OUT_OF_RANGE) { printf("Matrix check result: MKL_SPARSE_CHECKER_OUT_OF_RANGE\n"); }
		if (check_err_val == MKL_SPARSE_CHECKER_NONORDERED) { printf("Matrix check result: MKL_SPARSE_CHECKER_NONORDERED\n"); }
		error = 1;
	}

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
	std::cout << "Reordering completed: " << linearSolverTimer01.getDurationMilliSec() << " (milliSec)" << std::endl;
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

	STACCATOComplexDouble ZeroComplex = { 0,0 };
	STACCATOComplexDouble OneComplex = { 1,0 };
	myKComplexReduced.resize(ROM_DOF*ROM_DOF);
	myMComplexReduced.resize(ROM_DOF*ROM_DOF);
	myBReduced.resize(ROM_DOF*myInputDOFS.size());
	myCReduced.resize(myOutputDOFS.size()*ROM_DOF);
	// obj.K_R = obj.Z'*obj.K*obj.V;
	// y=obj.K*obj.V
	std::vector<STACCATOComplexDouble> y;
	y.resize(FOM_DOF*ROM_DOF, ZeroComplex);
	MathLibrary::computeSparseMatrixDenseMatrixMultiplicationComplex(ROM_DOF, FOM_DOF, FOM_DOF, &mySparseK, &myV[0], &y[0], false, false, OneComplex, true, false);
	// obj.K_R = obj.Z'*y
	if (isMIMO)
		MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, ROM_DOF, FOM_DOF, &myV[0], &y[0], &myKComplexReduced[0], true, false, OneComplex, false, false, false);
	else
		MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, ROM_DOF, FOM_DOF, &myZ[0], &y[0], &myKComplexReduced[0], true, false, OneComplex, false, false, false);
	// obj.M_R = obj.Z'*obj.M*obj.V;
	// obj.M*obj.V
	y.clear();
	y.resize(FOM_DOF*ROM_DOF, ZeroComplex);
	MathLibrary::computeSparseMatrixDenseMatrixMultiplicationComplex(ROM_DOF, FOM_DOF, FOM_DOF, &mySparseM, &myV[0], &y[0], false, false, OneComplex, true, false);
	// obj.M_R = obj.Z'*y
	if (isMIMO)
		MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, ROM_DOF, FOM_DOF, &myV[0], &y[0], &myMComplexReduced[0], true, false, OneComplex, false, false, false);
	else
		MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, ROM_DOF, FOM_DOF, &myZ[0], &y[0], &myMComplexReduced[0], true, false, OneComplex, false, false, false);

	if (enablePropDamping)
	{
		myDComplexReduced.resize(ROM_DOF*ROM_DOF);
		// obj.D_R = obj.Z'*obj.D*obj.V;
		y.clear();
		y.resize(FOM_DOF*ROM_DOF, ZeroComplex);
		MathLibrary::computeSparseMatrixDenseMatrixMultiplicationComplex(ROM_DOF, FOM_DOF, FOM_DOF, &mySparseD, &myV[0], &y[0], false, false, OneComplex, true, false);
		// obj.D_R = obj.Z'*y
		if (isMIMO)
			MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, ROM_DOF, FOM_DOF, &myV[0], &y[0], &myDComplexReduced[0], true, false, OneComplex, false, false, false);
		else
			MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, ROM_DOF, FOM_DOF, &myZ[0], &y[0], &myDComplexReduced[0], true, false, OneComplex, false, false, false);
	}

	// obj.B_R = obj.Z'*obj.B;
	if (isMIMO)
		MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, myInputDOFS.size(), FOM_DOF, &myV[0], &myB[0], &myBReduced[0], true, false, OneComplex, false, false, false);
	else
		MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, myInputDOFS.size(), FOM_DOF, &myZ[0], &myB[0], &myBReduced[0], true, false, OneComplex, false, false, false);

	// obj.C_R = obj.C*obj.V;
	MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(myOutputDOFS.size(), ROM_DOF, FOM_DOF, &myC[0], &myV[0], &myCReduced[0], false, false, OneComplex, false, false, false);
	myV.clear();
	std::cout << ">> Generating ROM... Finished." << std::endl;

#endif // USE_INTEL_MKL

}

void KrylovROMSubstructure::cleanPardiso() {
#ifdef USE_INTEL_MKL
	// clean pardiso
	pardiso_phase = -1; // deallocate memory
	pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase,
		&pardiso_neq, values, rowIndex, columns, &pardiso_idum,
		&pardiso_nrhs, pardiso_iparm, &pardiso_msglvl, &pardiso_ddum, &pardiso_ddum,
		&pardiso_error);
	if (pardiso_error != 0) {
		errorOut << "Error deallocation of pardiso failed with error code: " << pardiso_error
			<< std::endl;
		exit(EXIT_FAILURE);
	}
#endif
}

int KrylovROMSubstructure::reveilRankQR_R(const STACCATOComplexDouble* _mat, int _m, int _n, double _tol) {
	int RR = 0;
	//std::cout << "Diag entries: ";
	for (size_t iDiag = 0; iDiag < _n; iDiag++)
	{
		double abs = std::sqrt(_mat[iDiag*_m + iDiag].real*_mat[iDiag*_m + iDiag].real + _mat[iDiag*_m + iDiag].imag*_mat[iDiag*_m + iDiag].imag);
		//std::cout << _mat[iDiag*_m + iDiag].real << "+1i*" << _mat[iDiag*_m + iDiag].imag << " < ";
		if (abs > _tol) {
			RR++;
		}
		else
			iDiag = _n;		// End loop
	}
	return RR;
}

void KrylovROMSubstructure::buildAbqODB() {
	std::cout << ">> FOM Building from ODB .." << std::endl;
	myHMesh->buildDataStructure();
	debugOut << "SimuliaODB::openFile: " << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	anaysisTimer01.start();
	myHMesh->buildDoFGraph();
	anaysisTimer01.stop();

	infoOut << "Duration for building DoF graph: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	FOM_DOF = myHMesh->getTotalNumOfDoFsRaw();
	// Build XML NodeSets and ElementSets
	MetaDatabase::getInstance()->buildXML(*myHMesh);

	// Assign all elements with respective assigned material section
	assignMaterialToElements();

	std::cout << ">> Assembling FOM system matrices from ODB... " << std::endl;
	getSystemMatricesODB();
	std::cout << ">> Assembling FOM system matrices from ODB... Finished." << std::endl;
}

void KrylovROMSubstructure::buildAbqSIM(int _iPart) {

	STACCATO_XML::PARTS_const_iterator iterParts(MetaDatabase::getInstance()->xmlHandle->PARTS().begin());

	/* -- Prepare Reader -- */
#if defined(_WIN32) || defined(__WIN32__) 
	std::string filePath = "C:/software/repos/STACCATO/model/";
#endif
#if defined(__linux__) 
	std::string filePath = "/opt/software/repos/STACCATO/model/";
#endif
	filePath += std::string(iterParts->PART()[_iPart].FILEIMPORT().begin()->FILE()->data());
	myUMAReader = new SimuliaUMA(filePath, *myHMesh, _iPart);
	myHMesh = NULL;
	buildXMLforSIM(_iPart);

	std::cout << ">> Assembling FOM system matrices from SIM... " << std::endl;
	getSystemMatricesSIM();
	std::cout << ">> Assembling FOM system matrices from SIM... Finished." << std::endl;

	FOM_DOF = myUMAReader->totalDOFs;
}

void KrylovROMSubstructure::displayModelSize() {
	// Size determination
	std::cout << ">> -- Model Size --" << std::endl;
	std::cout << " > Expansion Points: ";
	for (int i = 0; i < myExpansionPoints.size(); i++)
		std::cout << myExpansionPoints[i] << " . ";
	std::cout << std::endl;
	std::cout << " > Krylov order: " << myKrylovOrder << std::endl;
	std::cout << " > #Inputs: " << myInputDOFS.size() << " #Outputs: " << myOutputDOFS.size() << std::endl;
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
}

void KrylovROMSubstructure::generateInputOutputMatricesForFOM() {
	myB.resize(FOM_DOF*myInputDOFS.size());
	myC.resize(myOutputDOFS.size()*FOM_DOF);

	// Generate Input and Output Matrices
	for (int inpIter = 0; inpIter < myInputDOFS.size(); inpIter++) {
		myB[myInputDOFS[inpIter] + inpIter * FOM_DOF].real = 1;
	}
	for (int outIter = 0; outIter < myOutputDOFS.size(); outIter++) {
		myC[myOutputDOFS[outIter] * myOutputDOFS.size() + outIter].real = 1;
	}
}

void KrylovROMSubstructure::exportROMToFiles() {
		std::string filename = "C://software//repos//staccato//scratch//";
		AuxiliaryFunctions::writeMKLComplexDenseMatrixMtxFormat(filename + currentPart + "_myKR.dat", myKComplexReduced, ROM_DOF, ROM_DOF, false);
		AuxiliaryFunctions::writeMKLComplexDenseMatrixMtxFormat(filename + currentPart + "_myMR.dat", myMComplexReduced, ROM_DOF, ROM_DOF, false);
		AuxiliaryFunctions::writeMKLComplexDenseMatrixMtxFormat(filename + currentPart + "_myBR.dat", myBReduced, ROM_DOF, myInputDOFS.size(), false);
		AuxiliaryFunctions::writeMKLComplexDenseMatrixMtxFormat(filename + currentPart + "_myCR.dat", myCReduced, myOutputDOFS.size(), ROM_DOF, false);
}

void KrylovROMSubstructure::performAnalysis() {
	STACCATO_XML::PARTS_const_iterator iterParts(MetaDatabase::getInstance()->xmlHandle->PARTS().begin());

	STACCATOComplexDouble ZeroComplex = { 0,0 };
	STACCATOComplexDouble OneComplex = { 1,0 };

	// Reading in the frequency range
	std::vector<double> freq;
	for (STACCATO_XML::ANALYSIS_const_iterator iAnalysis(MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin());
		iAnalysis != MetaDatabase::getInstance()->xmlHandle->ANALYSIS().end();
		++iAnalysis)
	{
		std::cout << std::endl << "==== Starting Anaylsis: " << iAnalysis->NAME()->data() << " ====" << std::endl;
		int frameTrack = 0;

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

		// Determine right hand side
		std::cout << ">> Building RHS Matrix for Neumann...\n";
		int sizeofRHS = 0;
		OutputDatabase::TimeStep timeStep;
		timeStep.startIndex = frameTrack;
		std::vector<MKL_Complex16> bComplex;
		for (int iLoadCase = 0; iLoadCase < iAnalysis->LOADCASES().begin()->LOADCASE().size(); iLoadCase++) {

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

									// Get Load
									std::vector<STACCATOComplexDouble> loadVector(3);
									loadVector[0] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->X()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->X()->data()) };
									loadVector[1] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Y()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->Y()->data()) };
									loadVector[2] = { std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].REAL().begin()->Z()->data()), std::atof(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].IMAGINARY().begin()->Z()->data()) };
									if (myModelType=="FOM_ODB")
									{
										BoundaryCondition<STACCATOComplexDouble> neumannBoundaryConditionComplex(*myHMesh);

										std::vector<int> nodeSet = myHMesh->convertNodeSetNameToLabels(std::string(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].NODESET().begin()->Name()->c_str()));
										neumannBoundaryConditionComplex.addConcentratedForceContribution(nodeSet, loadVector, bComplex);
										loadCaseData.type = neumannBoundaryConditionComplex.myCaseType;

										frameTrack++;
										sizeofRHS += neumannBoundaryConditionComplex.getNumberOfTotalCases();
									}
									else if (myModelType == "FOM_SIM") {
										bComplex.resize(FOM_DOF);
										std::vector<int> nodeSet;
										auto search = nodeSetsMap.find(std::string(iterParts->PART()[jPart].LOADS().begin()->LOAD()[jPartLoad].NODESET().begin()->Name()->c_str()));
										if (search != nodeSetsMap.end())
											nodeSet.insert(nodeSet.end(), search->second.begin(), search->second.end());
										else
											std::cout << "!! LoadNodeSet not found!";

										for (size_t iLoadAss = 0; iLoadAss < nodeSet.size(); iLoadAss++)
										{
											bComplex[nodeSet[iLoadAss]] = loadVector[0];
										}
										sizeofRHS += 1;
									}
								}
							}
						}
					}
				}
			}
		}
		if (exportRHS) {
			std::cout << ">> Writing RHS ...\n";
			AuxiliaryFunctions::writeMKLComplexVectorDatFormat(std::string(iAnalysis->NAME()->data()) + "_RHS.dat", bComplex);
		}

		std::vector<STACCATOComplexDouble> inputLoad;
		for (int iRHS = 0; iRHS < sizeofRHS; iRHS++)
		{
			std::vector<STACCATOComplexDouble> temp;
			temp.resize(myInputDOFS.size(), { 0,0 });
			for (int iInputDof = 0; iInputDof < myInputDOFS.size(); iInputDof++)
			{
				temp[iInputDof].real = bComplex[myInputDOFS[iInputDof]].real;
				temp[iInputDof].imag = bComplex[myInputDOFS[iInputDof]].imag;
			}
			inputLoad.insert(inputLoad.end(), temp.begin(), temp.end());
		}

		std::cout << ">> Building RHS Matrix for Neumann... Finished.\n" << std::endl;

		backTransformKMOR(std::string(iAnalysis->NAME()->data()), &freq, &inputLoad[0], sizeofRHS);

		std::cout << "==== Anaylsis Completed: " << iAnalysis->NAME()->data() << " ====" << std::endl;
	}
	std::cout << ">> All Analyses Completed." << std::endl;
}

void KrylovROMSubstructure::backTransformKMOR(std::string _analysisName, std::vector<double>* _freq, STACCATOComplexDouble* _inputLoad, int _numLoadCase ){
	STACCATOComplexDouble ZeroComplex = { 0,0 };
	STACCATOComplexDouble OneComplex = { 1,0 };

	std::vector<STACCATOComplexDouble> results;

	// Solving for each frequency
	anaysisTimer01.start();
#ifdef USE_INTEL_MKL		
	std::vector<lapack_int> pivot(ROM_DOF);	// Pivots for LU Decomposition
	for (int iFreqCounter = 0; iFreqCounter < _freq->size(); iFreqCounter++) {
		std::vector<double> sampleResultRe(1, 0);
		std::vector<double> sampleResultIm(1, 0);

		//std::cout << ">> Computing frequency step at " << freq[iFreqCounter] << " Hz ..." << std::endl;
		double omega = 2 * M_PI*_freq->at(iFreqCounter);
		STACCATOComplexDouble NegOmegaSquare = { -omega * omega,0 };

		// K_krylov_dyn = obj.K_R 
		std::vector<STACCATOComplexDouble> StiffnessAssembled;
		StiffnessAssembled.resize(ROM_DOF*ROM_DOF, ZeroComplex);
		MathLibrary::computeDenseVectorAdditionComplex(&myKComplexReduced[0], &StiffnessAssembled[0], &OneComplex, ROM_DOF*ROM_DOF);
		// K_krylov_dyn += -omega^2*obj.M_R 
		MathLibrary::computeDenseVectorAdditionComplex(&myMComplexReduced[0], &StiffnessAssembled[0], &NegOmegaSquare, ROM_DOF*ROM_DOF);
		if (enablePropDamping)
		{
			// K_krylov_dyn += 1i*2*pi*freqs_fine(i)*obj.D_R
			STACCATOComplexDouble complexOmega = { 0,omega };
			MathLibrary::computeDenseVectorAdditionComplex(&myDComplexReduced[0], &StiffnessAssembled[0], &complexOmega, ROM_DOF*ROM_DOF);
		}

		// obj.B_R*obj.F
		std::vector<STACCATOComplexDouble> inputLoad_krylov(ROM_DOF*_numLoadCase, { 0,0 });
		MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(ROM_DOF, _numLoadCase, myInputDOFS.size(), &myBReduced[0], _inputLoad, &inputLoad_krylov[0], false, false, OneComplex, false, false, false);


		// z_krylov_freq = K_krylov_dyn\(obj.B_R*obj.F);
		// Factorize StiffnessAssembled
		LAPACKE_zgetrf(LAPACK_COL_MAJOR, ROM_DOF, ROM_DOF, &StiffnessAssembled[0], ROM_DOF, &pivot[0]);
		// Solve system
		LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', ROM_DOF, _numLoadCase, &StiffnessAssembled[0], ROM_DOF, &pivot[0], &inputLoad_krylov[0], ROM_DOF);


		if (writeTransferFunctions) {
			// obj.H_R(:,:,i) = obj.C_R*(K_krylov_dyn\obj.B_R);
			std::vector<STACCATOComplexDouble> H_R(myInputDOFS.size()*myOutputDOFS.size(), { 0,0 });
			std::vector<STACCATOComplexDouble> temp;
			temp.insert(temp.end(), myBReduced.begin(), myBReduced.end());
			// temp = (K_krylov_dyn\obj.B_R)
			LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', ROM_DOF, myInputDOFS.size(), &StiffnessAssembled[0], ROM_DOF, &pivot[0], &temp[0], ROM_DOF);
			// H_R = obj.C_R*temp
			MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(myOutputDOFS.size(), myInputDOFS.size(), ROM_DOF, &myCReduced[0], &temp[0], &H_R[0], false, false, OneComplex, false, false, false);

			std::string filename = "C://software//repos//staccato//scratch//";
			AuxiliaryFunctions::writeMKLComplexDenseMatrixMtxFormat(filename + _analysisName + "_HR_freq" + std::to_string(_freq->at(iFreqCounter)) + ".mtx", H_R, myOutputDOFS.size(), myInputDOFS.size(), false);
		}

		// z_freq = obj.C_R*z_krylov_freq;
		std::vector<STACCATOComplexDouble> backprojected_sol(myOutputDOFS.size()*_numLoadCase, ZeroComplex);
		MathLibrary::computeDenseMatrixMatrixMultiplicationComplex(myOutputDOFS.size(), _numLoadCase, ROM_DOF, &myCReduced[0], &inputLoad_krylov[0], &backprojected_sol[0], false, false, OneComplex, false, false, false);

		results.insert(results.end(), backprojected_sol.begin(), backprojected_sol.end());
		//std::cout << ">> Computing frequency step at " << freq[iFreqCounter] << " Hz ... Finished." << std::endl;
	}
	anaysisTimer01.stop();
	std::cout << " --> Duration for backtransformation: " << anaysisTimer01.getDurationMilliSec() << " ms" << std::endl;

	if (exportSolution)
		AuxiliaryFunctions::writeMKLComplexVectorDatFormat(_analysisName + "_KMOR_Results.dat", results);
#endif // USE_INTEL_MKL
}

void KrylovROMSubstructure::buildXMLforSIM(int _iPart) {
	STACCATO_XML::PARTS_const_iterator iterParts(MetaDatabase::getInstance()->xmlHandle->PARTS().begin());
	// Node Sets
	for (int k = 0; k < iterParts->PART()[_iPart].SETS().begin()->NODESET().size(); k++) {
		// Recognize List for ALL or a List of IDs
		std::vector<int> idList;
		// Keyword: ALL
		if (std::string(iterParts->PART()[_iPart].SETS().begin()->NODESET()[k].LIST()->c_str()) == "ALL") {
			int alldof = myUMAReader->totalDOFs;
			for (size_t i = 0; i < alldof; i++)
				idList.push_back(i);
		}
		else {	// ID List
				// filter
			std::stringstream stream(std::string(iterParts->PART()[_iPart].SETS().begin()->NODESET()[k].LIST()->c_str()));
			while (stream) {
				int n;
				stream >> n;
				if (stream)
					idList.push_back(n-1);
			}
		}
		nodeSetsMap[std::string(iterParts->PART()[_iPart].SETS().begin()->NODESET()[k].Name()->c_str())] =  idList;
	}
}