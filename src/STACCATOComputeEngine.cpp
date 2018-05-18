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
#include "STACCATOComputeEngine.h"
#include "HMesh.h"
#include "HMeshToVtkUnstructuredGrid.h"
#include "Reader.h"
#include "Timer.h"
#include "MemWatcher.h"
#include "SimuliaODB.h"
#include "SimuliaUMA.h"

#include "FeAnalysis.h"
#include "HMesh.h"
#include "MetaDatabase.h"

STACCATOComputeEngine::STACCATOComputeEngine(std::string _xmlFileName){
	// Intialize XML metadatabase singelton 
	MetaDatabase::init(_xmlFileName);
	// Works for one instance only
	myHMesh = new HMesh("default");
}


STACCATOComputeEngine::~STACCATOComputeEngine(){
}

void STACCATOComputeEngine::prepare(void) {


	int numParts = MetaDatabase::getInstance()->xmlHandle->PARTS().begin()->PART().size();
	std::cout << "There are " << numParts << " models.\n";

	anaysisTimer01.start();
	anaysisTimer03.start();

	STACCATO_XML::PARTS_const_iterator iterParts(MetaDatabase::getInstance()->xmlHandle->PARTS().begin());
	for (int iPart = 0; iPart < iterParts->PART().size(); iPart++)
	{
		if (std::string(iterParts->PART()[iPart].TYPE()->data()) == "FE")
		{
			for (int iFileImport = 0; iFileImport < iterParts->PART()[iPart].FILEIMPORT().size(); iFileImport++)			/// Assumption: Only One FileImport per Part
			{
                //Todo add global search path to xml file
#if defined(_WIN32) || defined(__WIN32__) 
std::string filePath = "C:/software/repos/STACCATO/model/";
#endif
#if defined(__linux__) 
std::string filePath = "/home/stefan/software/repos/STACCATO/model/";
#endif
				filePath += std::string(iterParts->PART()[iPart].FILEIMPORT()[iFileImport].FILE()->data());
				if (std::string(iterParts->PART()[iPart].FILEIMPORT()[iFileImport].Type()->data()) == "AbqODB") {
					Reader* fileReader = new SimuliaODB(filePath, *myHMesh, iPart);
				}
				else if (std::string(iterParts->PART()[iPart].FILEIMPORT()[iFileImport].Type()->data()) == "AbqSIM") {
					Reader* fileReader = new SimuliaUMA(filePath, *myHMesh, iPart);
				}
				else {
					std::cerr << ">> XML Error: Unidentified FileImport type " << iterParts->PART()[iPart].FILEIMPORT()[iFileImport].Type()->data() << std::endl;
				}
			}
		}
		else
			std::cerr << ">> XML Error: Unidentified Part Type: " << iterParts->PART()[iPart].TYPE()->data() << std::endl;
	}

	anaysisTimer01.stop();
	std::cout << "Duration for reading all file imports: " << anaysisTimer01.getDurationMilliSec() << " milliSec" << std::endl;
	std::cout << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;

}

void STACCATOComputeEngine::compute(void) {

	//Run FE Analysis
	FeAnalysis *mFeAnalysis = new FeAnalysis(*myHMesh);
	anaysisTimer03.stop();
	std::cout << "Duration for STACCATO Finite Element run: " << anaysisTimer03.getDurationSec() << " sec" << std::endl;
	std::cout << "Current physical memory consumption: " << memWatcher.getCurrentUsedPhysicalMemory() / 1000000 << " Mb" << std::endl;


	std::cout << ">> FeAnalysis Finished." << std::endl;

}

OutputDatabase* STACCATOComputeEngine::getOutputDatabase(void) {

	return (myHMesh->myOutputDatabase); 
}

void STACCATOComputeEngine::clean(void) {

}


