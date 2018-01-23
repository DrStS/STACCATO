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
#include "FeMetaDatabase.h"
#include "FeElement.h"
#include "FePlainStress4NodeElement.h"
#include "FeTetrahedron10NodeElement.h"
#include "Material.h"

#include "MathLibrary.h"
#include "Timer.h"
#include "MemWatcher.h"

#include "MetaDatabase.h"
#include <string.h>

#include <complex>
using namespace std::complex_literals;

FeAnalysis::FeAnalysis(HMesh& _hMesh, FeMetaDatabase& _feMetaDatabase) : myHMesh(&_hMesh), myFeMetaDatabase(&_feMetaDatabase) {
	
	// --- XML Testing ---------------------------------------------------------------------------------------------

	char* inputFileName = "C:/software/repos/STACCATO/xsd/IP_STACCATO_XML.xml";
	
	MetaDatabase::init(inputFileName);
	
	std::cout << "==================================\n";
	std::cout << "========= STACCATO IMPORT ========\n";
	std::cout << "==================================\n\n";


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
		std::cout << " > NAME: " << temp->MATERIAL().at(j).Name() << " Type: " << temp->MATERIAL().at(j).Type() << " E, nu, rho " << temp->MATERIAL().at(j).E() << "," << temp->MATERIAL().at(j).nu() << "," << temp->MATERIAL().at(j).rho() << std::endl;
	}

	std::cout << ">> NODES: " << std::endl;
	STACCATO_XML::NODES_const_iterator i(MetaDatabase::getInstance()->xmlHandle->NODES().begin());
	for (int j = 0; j < i->NODE().size(); j++) {
		std::cout << " > ID: " << i->NODE().at(j).ID() << " X, Y, Z: " << i->NODE().at(j).X() << "," << i->NODE().at(j).Y() << "," << i->NODE().at(j).Z() << std::endl;
	}

	std::cout << ">> ELEMENTS: " << std::endl;
	STACCATO_XML::ELEMENTS_const_iterator temp_e(MetaDatabase::getInstance()->xmlHandle->ELEMENTS().begin());
	for (int j = 0; j < temp_e->ELEMENT().size(); j++) {
		std::cout << " > Type: " << temp_e->ELEMENT().at(j).Type() << " ID: " << temp_e->ELEMENT().at(j).ID() << " NODECONNECT: " << temp_e->ELEMENT().at(j).NODECONNECT() << std::endl;
	}

	std::cout << "\n==================================\n";

	MetaDatabase::getInstance()->exportXML();			// Routine: Export XML

	// ---- END OF XML Testing -------------------------------------------------------------------------

	unsigned int numElements = myHMesh->getNumElements();
	unsigned int numNodes = myHMesh->getNumNodes();

	Material * elasticMaterial = new Material();

	anaysisTimer01.start();
	myHMesh->buildDoFGraph();
	anaysisTimer01.stop();
	infoOut << "Duration for building DoF graph: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	anaysisTimer01.start();

	std::vector<FeElement*> allElements(numElements);
	int lastIndex = 0;
	for (int iElement = 0; iElement < numElements; iElement++)
	{
		if (myHMesh->getElementTypes()[iElement] == STACCATO_PlainStress4Node2D) {
			allElements[iElement] = new FePlainStress4NodeElement(elasticMaterial);
		}
		else	if (myHMesh->getElementTypes()[iElement] == STACCATO_Tetrahedron10Node3D) {
			allElements[iElement] = new FeTetrahedron10NodeElement(elasticMaterial);
		}
		int numNodesPerElement = myHMesh->getNumNodesPerElement()[iElement];
		double*  eleCoords = &myHMesh->getNodeCoordsSortElement()[lastIndex];
		allElements[iElement]->computeElementMatrix(eleCoords);
		lastIndex += numNodesPerElement*myHMesh->getDomainDimension();
	}
	anaysisTimer01.stop();
	infoOut << "Duration for element loop: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
	
	std::vector<double> freq;
	
	// Routine to Accomodate Step Distribution
	double start_freq = std::atof(MetaDatabase::getInstance()->xmlHandle->FREQUENCY().begin()->START_FREQ()->c_str());
	freq.push_back(start_freq);		// Push back starting frequency in any case

	if (std::string(MetaDatabase::getInstance()->xmlHandle->FREQUENCY().begin()->Type()->data()) == "STEP") {		// Step Distribute
		double end_freq  = std::atof(MetaDatabase::getInstance()->xmlHandle->FREQUENCY().begin()->END_FREQ()->c_str());
		double step_freq = std::atof(MetaDatabase::getInstance()->xmlHandle->FREQUENCY().begin()->STEP_FREQ()->c_str());
		double push_freq = start_freq + step_freq;
		
		while (push_freq <= end_freq)		{
			freq.push_back(push_freq);
			push_freq += step_freq;
		}
	}

	//freq.push_back(250);
	/*
	freq.push_back(100);
	freq.push_back(1000);
	freq.push_back(2000);
	freq.push_back(3000);
	freq.push_back(4000);
	freq.push_back(5000);
	freq.push_back(6000);
	freq.push_back(7000);
	freq.push_back(8000);
	*/
	anaysisTimer01.start();

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

	// Damping														
	std::cout << "\nDamping Factor of eta = " << elasticMaterial->getDampingParameter() << " is added to the system!\n\n";

	// Allocate global matrix and vector memory
	// Real Only
	std::vector<double> bReal;
	std::vector<double> solReal;
	// Complex
	std::vector<MKL_Complex16> bComplex;
	std::vector<MKL_Complex16> solComplex;

	std::string analysisType = MetaDatabase::getInstance()->xmlHandle->ANALYSIS().begin()->TYPE()->data();

	if (analysisType == "STATIC_REAL") {
		bReal.resize(totalDoF);
		solReal.resize(totalDoF);
	} else if(analysisType == "STATIC" || analysisType == "DYNAMIC")	{
		bComplex.resize(totalDoF);
		solComplex.resize(totalDoF);
	} else {
		std::cerr << "Unsupported Analysis Type! \n-Hint: Check XML Input \n-Exiting STACCATO." << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int iFreqCounter = 0; iFreqCounter < freq.size(); iFreqCounter++) {
		lastIndex = 0;

		MathLibrary::SparseMatrix<double> *AReal;
		MathLibrary::SparseMatrix<MKL_Complex16> *AComplex;
		if (analysisType == "STATIC_REAL") {
			AReal = new MathLibrary::SparseMatrix<double>(totalDoF, true);
		}
		else if (analysisType == "STATIC" || analysisType == "DYNAMIC") {
			AComplex = new MathLibrary::SparseMatrix<MKL_Complex16>(totalDoF, true);
		}

		std::cout << "test1" << std::endl;
		for (int iElement = 0; iElement < numElements; iElement++)
		{
			int numDoFsPerElement = myHMesh->getNumDoFsPerElement()[iElement];
			int*  eleDoFs = &myHMesh->getElementDoFList()[lastIndex];
			lastIndex += numDoFsPerElement;
			double omega = 2 * M_PI*freq[iFreqCounter];
			//Assembly routine symmetric stiffness
			for (int i = 0; i < numDoFsPerElement; i++) {
				for (int j = 0; j < numDoFsPerElement; j++) {
					if (eleDoFs[j] >= eleDoFs[i]) {
						//K(1+eta*i)
						if (analysisType == "STATIC_REAL") {
							(*AReal)(eleDoFs[i], eleDoFs[j]) += allElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j];
						}
						else if (analysisType == "STATIC" || analysisType == "DYNAMIC") {
							(*AComplex)(eleDoFs[i], eleDoFs[j]).real += allElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j];
							if (analysisType == "DYNAMIC")
								(*AComplex)(eleDoFs[i], eleDoFs[j]).imag += allElements[iElement]->getStiffnessMatrix()[i*numDoFsPerElement + j] * elasticMaterial->getDampingParameter();
						}
					}
				}
			}
			//K - omega*omega*M
			//Assembly routine symmetric mass
			if (analysisType == "DYNAMIC")
				for (int i = 0; i < numDoFsPerElement; i++) {
					for (int j = 0; j < numDoFsPerElement; j++) {
						if (eleDoFs[j] >= eleDoFs[i]) {
							//K(1+eta*i) - omega*omega*M
							(*AComplex)(eleDoFs[i], eleDoFs[j]).real -= allElements[iElement]->getMassMatrix()[i*numDoFsPerElement + j] * omega*omega;
						}
					}
				}
		}
		std::cout << "test2" << std::endl;
		//Add cload rhs contribution 
		for (int k = 0; k < MetaDatabase::getInstance()->xmlHandle->LOADS().begin()->LOAD().size(); k++) {
			
			for (int j = 0; j < numNodes; j++)
			{
				int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
				for (int l = 0; l < numDoFsPerNode; l++) {
					if (myHMesh->getNodeLabels()[j] == std::stoi(MetaDatabase::getInstance()->xmlHandle->LOADS().begin()->LOAD().at(k).OnNode()->data())) {  //3407
						
						std::complex<double> temp_Fx(std::atof(MetaDatabase::getInstance()->xmlHandle->LOADS().begin()->LOAD().at(k).Fx()->data()), std::atof(MetaDatabase::getInstance()->xmlHandle->LOADS().begin()->LOAD().at(k).iFx()->data()));
						std::complex<double> temp_Fy(std::atof(MetaDatabase::getInstance()->xmlHandle->LOADS().begin()->LOAD().at(k).Fy()->data()), std::atof(MetaDatabase::getInstance()->xmlHandle->LOADS().begin()->LOAD().at(k).iFy()->data()));
						std::complex<double> temp_Fz(std::atof(MetaDatabase::getInstance()->xmlHandle->LOADS().begin()->LOAD().at(k).Fz()->data()), std::atof(MetaDatabase::getInstance()->xmlHandle->LOADS().begin()->LOAD().at(k).iFz()->data()));

						int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
						if (analysisType == "STATIC_REAL") {
							switch (l) {
							case 0:
								bReal[dofIndex] += temp_Fx.real();
								break;
							case 1:
								bReal[dofIndex] += temp_Fy.real();
								break;
							case 2:
								bReal[dofIndex] += temp_Fz.real();
								break;
							default:
								break;
							}
						}
						else if (analysisType == "STATIC" || analysisType == "DYNAMIC") {
							switch (l) {
							case 0:
								bComplex[dofIndex].real += temp_Fx.real();
								bComplex[dofIndex].imag += temp_Fx.imag();
								break;
							case 1:
								bComplex[dofIndex].real += temp_Fy.real();
								bComplex[dofIndex].imag += temp_Fy.imag();
								break;
							case 2:
								bComplex[dofIndex].real += temp_Fz.real();
								bComplex[dofIndex].imag += temp_Fz.imag();
								break;
							default:
								break;
							}
						}

						
					}
				}
			}
		}
		
		//(*A).print();
		anaysisTimer01.stop();
		infoOut << "Duration for assembly loop: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
		debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
		anaysisTimer01.start();
		anaysisTimer02.start();
		if (analysisType == "STATIC_REAL") {
			(*AReal).check();
		}
		else if (analysisType == "STATIC" || analysisType == "DYNAMIC") {
			(*AComplex).check();
		}
		anaysisTimer01.stop();
		infoOut << "Duration for direct solver check: " << anaysisTimer01.getDurationSec() << " sec" << std::endl;
		debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
		anaysisTimer01.start();
		if (analysisType == "STATIC_REAL") {
			(*AReal).factorize();
		}
		else if (analysisType == "STATIC" || analysisType == "DYNAMIC") {
			(*AComplex).factorize();
		}
		anaysisTimer01.stop();
		infoOut << "Duration for direct solver factorize: " << anaysisTimer01.getDurationSec() << " sec" << std::endl;
		debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;
		anaysisTimer01.start();
		if (analysisType == "STATIC_REAL") {
			(*AReal).solveDirect(&solReal[0], &bReal[0]);
		}
		else if (analysisType == "STATIC" || analysisType == "DYNAMIC") {
			(*AComplex).solveDirect(&solComplex[0], &bComplex[0]);
		}
		anaysisTimer01.stop();
		anaysisTimer02.stop();
		infoOut << "Duration for direct solver substitution : " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
		infoOut << "Total duration for direct solver: " << anaysisTimer02.getDurationSec() << " sec" << std::endl;
		debugOut << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

		anaysisTimer01.start();

		// Store results
		for (int j = 0; j < numNodes; j++)
		{
			int numDoFsPerNode = myHMesh->getNumDoFsPerNode(j);
			for (int l = 0; l < numDoFsPerNode; l++) {
				int dofIndex = myHMesh->getNodeIndexToDoFIndices()[j][l];
				if (l == 0) {
					if (analysisType == "STATIC_REAL") {
						resultUxRe[j] = solReal[dofIndex];
					}
					else if (analysisType == "STATIC" || analysisType == "DYNAMIC") {
						resultUxRe[j] = solComplex[dofIndex].real;
						resultUxIm[j] = solComplex[dofIndex].imag;
					}
				}
				if (l == 1) {
					if (analysisType == "STATIC_REAL") {
						resultUyRe[j] = solReal[dofIndex];
					}
					else if (analysisType == "STATIC" || analysisType == "DYNAMIC") {
						resultUyRe[j] = solComplex[dofIndex].real;
						resultUyIm[j] = solComplex[dofIndex].imag;
					}
				}
				if (l == 2) {
					if (analysisType == "STATIC_REAL") {
						resultUzRe[j] = solReal[dofIndex];
					}
					else if (analysisType == "STATIC" || analysisType == "DYNAMIC") {
						resultUzRe[j] = solComplex[dofIndex].real;
						resultUzIm[j] = solComplex[dofIndex].imag;
					}
				}
			}

			resultMagRe[j] = sqrt(pow(resultUxRe[j], 2) + pow(resultUyRe[j], 2) + pow(resultUzRe[j], 2));
			if (analysisType == "STATIC" || analysisType == "DYNAMIC")
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
		
		if (analysisType == "STATIC_REAL") {
			(*AReal).cleanPardiso();
			delete AReal;
		}
		else if (analysisType == "STATIC" || analysisType == "DYNAMIC") {
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



