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
/***********************************************************************************************//**
* \file SimuliaUMA.h
* This file holds the class SimuliaUMA which adds the capability to read Abaqus SIM files
* \date 2/1/2018
**************************************************************************************************/
#pragma once

#include <string>
#include <assert.h>
#include <vector>
#include <map>

#include "ReadWriteFile.h"
#include "AuxiliaryParameters.h"

#ifdef USE_SIMULIA_UMA_API
class uma_System;
class uma_SparseMatrix;
#endif

class HMesh;
/********//**
* \brief This handles the output handling with Abaqus SIM
**************************************************************************************************/
class SimuliaUMA :public ReadWriteFile {
public:
	/***********************************************************************************************
	* \brief Constructor
	* \param[in] _filePath string which holds the path to the sim file
	* \author Harikrishnan Sreekumar
	***********/
	SimuliaUMA(std::string _fileName, HMesh& _hMesh, int _partId);
	/***********************************************************************************************
	* \brief Destructor
	*
	* \author Harikrishnan Sreekumar
	***********/
	virtual ~SimuliaUMA(void);
	/***********************************************************************************************
	* \brief Open die sim file
	* \param[in] _filePath string which holds the path to the sim file
	* \author Harikrishnan Sreekumar
	***********/
	void openFile();
	/***********************************************************************************************
	* \brief The routine collects the information to the common map which is passed as arguments
	* \param[in] Matrix key
	* \param[in/out] dof map
	* \author Harikrishnan Sreekumar
	***********/
	int collectDatastructureSIM(char* _key, std::map<int, std::vector<int>> &_dofMap);
	/***********************************************************************************************
	* \brief Open the matrix for the passed key and checks for symmetricity
	* \param[in] _key uma matrix key
	* \author Harikrishnan Sreekumar
	***********/
	bool isUmaSymmetric(char* _key);
	/***********************************************************************************************
	* \brief The routine reads in matrix entries and return the CSR format
	* \param[in] Matrix key
	* \param[in/out] CSR _ia
	* \param[in/out] CSR _ja
	* \param[in/out] CSR _values
	* \param[in] local dof map
	* \param[in] global dof map
	* \param[in] read mode. 0-> normal symmetric read | 1-> read with FSI extraction | 2-> read with FSI extraction
	* \param[in] _flagUnymRead Use symmetric read with entries in lower triangular entries also
	* \param[in] _printToFile Export the read matrix to file in dat CSR format
	* \param[in] _numrows Number of totaldof equal to all system matrices
	* \author Harikrishnan Sreekumar
	***********/
	void loadSIMforUMA(std::string _key, std::vector<int>& _ia, std::vector<int> &_ja, std::vector<STACCATOComplexDouble> &_values, std::map<int, std::vector<int>> &_dofMap, std::map<int, std::vector<int>> &_globalMap, std::vector<int> &_dbcpivot, int _readMode, bool _flagUnymRead, bool _printToFile, int _numrows);
	/***********************************************************************************************
	* \brief Function to intialize the coupling matrix incase of Fluid Structure interaction
	* \param[in] _KASI matrix in map form
	* \author Harikrishnan Sreekumar
	***********/
	void setCouplingMatFSI(std::map<int, std::map<int, double>> &_KASI);
	/***********************************************************************************************
	* \brief Function to retrieve the coupling matrix incase of Fluid Structure interaction
	* \param[out] _KASI matrix in map form
	* \author Harikrishnan Sreekumar
	***********/
	std::map<int, std::map<int, double>> getCouplingMatFSI();
	/***********************************************************************************************
	* \brief Prints the UMA imported matrix
	* \param[in] UMA System with the valid matrix
	* \param[in] Name of matrix
	* \param[in] Flag for displaying imported matrices
	* \param[in] Flag for exporitng imported matrices
	* \author Harikrishnan Sreekumar
	***********/
#ifdef USE_SIMULIA_UMA_API
	void PrintMatrix(const uma_System &system, const char *matrixName, bool _printToScreen, bool _printToFile);
#endif
private:
	/// UMA filename
	std::string myFileName;
	/// HMesh object 
	HMesh *myHMesh;	   
	/// The system matrix map
	std::map<int, std::map<int, double>> mySystemMatrixMapCSR;
	/// FSI Coupling matrix map
	std::map<int, std::map<int, double>> myCouplingMatFSI;
	/// the current system
	uma_System* myUMASystem;
};

