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
* \file FeUmaElement.h
* This file holds the class of Abaqus SIM Element
* \date 1/31/2018
**************************************************************************************************/

#pragma once

#include <cstddef>
#include <assert.h>
#include <math.h>
#include <vector>
#include <FeElement.h>

#include <MathLibrary.h>

#ifdef USE_SIMULIA_UMA_API
class uma_System;
#endif

/********//**
* \brief Class FeUmaElement
**************************************************************************************************/
class FeUmaElement : public FeElement {
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Harikrishnan Sreekumar
	***********/
	FeUmaElement(Material *_material);
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	virtual ~FeUmaElement(void);
	/***********************************************************************************************
	* \brief Compute stiffness, mass and damping matrices
	* \param[in] _eleCoords Element cooord vector
	* \author Harikrishnan Sreekumar
	***********/
	void computeElementMatrix(const double* _eleCoords);
	/***********************************************************************************************
	* \brief Return pointer to double array
	* \author Harikrishnan Sreekumar
	***********/
	const std::vector<double> &  getStiffnessMatrix(void) const { return myKe; }
	/***********************************************************************************************
	* \brief Return pointer to double array
	* \author Harikrishnan Sreekumar
	***********/
	const std::vector<double> & getMassMatrix(void) const { return myMe; }
	/***********************************************************************************************
	* \brief Return pointer to double array
	* \author Stefan Sicklinger
	***********/
	const std::vector<double> & getStructuralDampingMatrix(void) const { return mySDe; }
	/***********************************************************************************************
	* \brief Return pointer to sparse class
	* \author Harikrishnan Sreekumar
	***********/
	const MathLibrary::SparseMatrix<double> & getSparseMassMatrix(void) const { return *mySparseMReal; }
	/***********************************************************************************************
	* \brief Return pointer to sparse class
	* \author Harikrishnan Sreekumar
	***********/
	const MathLibrary::SparseMatrix<double> & getSparseStiffnessMatrix(void) const { return *mySparseKReal; }
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
	/***********************************************************************************************
	* \brief Extract data from UMA
	* \param[in] UMA System with the valid matrix
	* \param[in] Name of matrix
	* \param[in] Flag for displaying imported matrices
	* \param[in] Flag for exporitng imported matrices
	* \author Harikrishnan Sreekumar
	***********/
#ifdef USE_SIMULIA_UMA_API
	void extractData(const uma_System &system, const char *matrixName, bool _printToScreen, bool _printToFile);
#endif
	/***********************************************************************************************
	* \brief Debug routine to visualize pure UMA without any map formulation
	* \param[in] Matrix key
	* \param[in] SIM file name
	* \param[in] Flag for displaying imported matrices
	* \param[in] Flag for exporitng imported matrices
	* \author Harikrishnan Sreekumar
	***********/
	void DebugSIM(const char* _matrixkey, const char* _fileName, bool _printToScreen, bool _printToFile);
	/***********************************************************************************************
	* \brief Import routine to load data from SIM
	* \param[in] Matrix key
	* \param[in] SIM file name
	* \param[in] Flag for displaying imported matrices
	* \param[in] Flag for exporitng imported matrices
	* \author Harikrishnan Sreekumar
	***********/
	void ImportSIM(const char* _matrixkey, const char* _fileName, bool _printToScreen, bool _printToFile);

	/// Stiffness Matrix
	MathLibrary::SparseMatrix<double> *mySparseMReal;
	MathLibrary::SparseMatrix<double> *mySparseKReal;

	/// [To be deleted when sparse addition is implemented]
	std::vector<int> myK_row;
	std::vector<int> myK_col;
	std::vector<int> myM_row;
	std::vector<int> myM_col;

	/// Maps
	std::map<int, std::vector<int>> nodeToDofMap;
	std::map<int, std::vector<int>> nodeToGlobalMap;

	/// Definition of keys
	char* stiffnessUMA_key;
	char* massUMA_key;
	char* structuralDampingUMA_key;

	/// Total system dimension
	int totalDOFs;
};