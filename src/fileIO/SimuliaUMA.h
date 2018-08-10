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
#include <Reader.h>
#include <vector>
#include <map>

#ifdef USE_SIMULIA_UMA_API
class uma_System;
class uma_SparseMatrix;
//#include <uma_SparseMatrix.h>
#endif

class HMesh;
/********//**
* \brief This handles the output handling with Abaqus SIM
**************************************************************************************************/
class SimuliaUMA :public Reader {
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
	* \brief Import routine to load datastructure from SIM
	* \param[in] SIM file name
	* \param[in] Matrix key
	* \param[in] Flag for displaying imported matrices and maps
	* \author Harikrishnan Sreekumar
	***********/
	void importDatastructureSIM(char* _file, char* _key, bool _printToScreen);

#ifdef USE_SIMULIA_UMA_API
	/***********************************************************************************************
	* \brief Extract datastructure from UMA
	* \param[in] UMA System with the valid matrix
	* \param[in] Name of matrix
	* \param[in] Flag for displaying imported matrices
	* \author Harikrishnan Sreekumar
	***********/
	void extractDatastructure(const uma_System &system, char *matrixName, bool _printToScreen);
#endif
	/***********************************************************************************************
	* \brief Routine to check for internal dofs (Can be disabled if not required)
	* \param[in] UMA System with the valid matrix
	* \param[in] Name of matrix
	* \param[in] Flag for displaying imported matrices
	* \param[in] Flag for exporitng imported matrices
	* \author Harikrishnan Sreekumar
	***********/
#ifdef USE_SIMULIA_UMA_API
	void checkForInternalDofs(const uma_SparseMatrix &smtx, char *matrixName, bool _printToScreen);
#endif
	/***********************************************************************************************
	* \brief Adds an internal dof to the datastructure
	* \param[in] Name of matrix
	* \param[in] GlobalIndex
	* \param[in] Flag to mark detection of internal dof
	* \author Harikrishnan Sreekumar
	***********/
	void addInternalDof(char *matrixName, int _index, bool _flag);
	/***********************************************************************************************
	* \brief Generates global map
	* \author Harikrishnan Sreekumar
	***********/
	void generateGlobalMap(bool _printToScreen);
	/***********************************************************************************************
	* \brief Generates a file with node to local dof and global dof map
	* \author Harikrishnan Sreekumar
	***********/
	void printMapToFile();
	/***********************************************************************************************
	* \brief Adds a detected node and its dof to the map (accumulates info from all SIMs and avoid duplicates)
	* \param[in] Node
	* \param[in] Dof
	* \author Harikrishnan Sreekumar
	***********/
	void addEntryToNodeDofMap(int _node, int _dof);
private:
	std::string myFileName;
	/// HMesh object 
	HMesh *myHMesh;
	/// Node Label and DoF vector
	std::vector<std::vector<int>> simNodeMap;
	/// Number of nodes
	int numNodes;
	/// Total number of dofs
	int totalDOFs;
	/// Number of DoFs per Node
	int numDoFperNode;

	// Flags
	bool hasInternalDOF_K;
	bool hasInternalDOF_M;
	bool hasInternalDOF_SD;
	bool flag_SD;

	// vectors with local dof of internaldofs
	std::vector<int> internalDOF_K;
	std::vector<int> internalDOF_M;
	std::vector<int> internalDOF_SD;

	// SIM File Names
	std::string stiffnessFileName;
	std::string massFileName ;
	std::string structDampingFileName;

	// Definition of matrix keys
	char* stiffnessUMA_key ;
	char* massUMA_key;
	char* structuralDampingUMA_key;

	// Maps
	std::map<int, std::vector<int>> nodeToDofMap;
	std::map<int, std::vector<int>> nodeToGlobalMap;

};
