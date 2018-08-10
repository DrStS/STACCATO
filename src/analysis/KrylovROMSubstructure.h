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
/*************************************************************************************************
* \file KrylovROMSubstructure.h
* This file holds the class KrylovROMSubstructure which form the entire ROM Analysis
* \date 12/06/2018
**************************************************************************************************/
#pragma once

#include <string>
#include <assert.h>
#include <math.h>

#include <MathLibrary.h>

class HMesh;
class FeElement;
class FeUmaElement;
/**********
* \brief Class KrylovROMSubstructure holds and builds the whole ROM Analysis
* Input to this class is a FeMetaDatabase and a HMesh object
**************************************************************************************************/
class KrylovROMSubstructure {
public:
	/***********************************************************************************************
	* \brief Constructor
	* \param[in] _Hmesh reference to HMesh object
	* \param[in] _FeMetaDatabase reference to FeMetaDatabase object
	* \author Harikrishnan Sreekumar
	***********/
	KrylovROMSubstructure(HMesh& _HMesh);
	/***********************************************************************************************
	* \brief Destructor
	*
	* \author Harikrishnan Sreekumar
	***********/
	virtual ~KrylovROMSubstructure(void);
	/***********************************************************************************************
	* \brief Assigns all element to a material and calculate the element stiffness and mass matrix
	* \author Harikrishnan Sreekumar
	***********/
	void assignMaterialToElements(void);
	/***********************************************************************************************
	* \brief Assemble all element stiffness and mass matrices
	* \param[in] Type of analysis
	* \author Harikrishnan Sreekumar
	***********/
	void assembleGlobalMatrices(std::string _analysisType);
	/***********************************************************************************************
	* \brief Assemble UMA stiffness and mass matrices
	* \param[in] Type of analysis
	* \author Harikrishnan Sreekumar
	***********/
	void assembleUmaMatrices(std::string _analysisType);
	/***********************************************************************************************
	* \brief Build projection basis for second order Krylov subspaces for manual settings
	* \author Harikrishnan Sreekumar
	***********/
	void buildProjectionMatManual();
	/***********************************************************************************************
	* \brief Add krylov modes for second order Krylov subspaces for the expansion point
	* \param[in] expansion point
	* \param[in] krylov order
	* \author Harikrishnan Sreekumar
	***********/
	void addKrylovModesForExpansionPoint(std::vector<double>& _expPoint, int _krylovOrder);
	/***********************************************************************************************
	* \brief PARDISO factorization for sparse matrix
	* \param[in] _mat sparse matrix
	* \param[in] _symmetric symmetricity
	* \param[in] _positiveDefinite
	* \param[in] _nRHS number of RHS
	* \author Harikrishnan Sreekumar
	***********/
	void factorizeSparseMatrixComplex(const sparse_matrix_t* _mat, const bool _symmetric, const bool _positiveDefinite, int _nRHS);
	/***********************************************************************************************
	* \brief PARDISO solving for sparse matrix
	* \param[in] _mat sparse matrix
	* \param[in] _symmetric symmetricity
	* \param[in] _positiveDefinite
	* \param[in] _nRHS number of RHS
	* \param[out] _x solution vector
	* \param[in] _b right hand side
	* \author Harikrishnan Sreekumar
	***********/
	void solveDirectSparseComplex(const sparse_matrix_t* _mat, const bool _symmetric, const bool _positiveDefinite, int _nRHS, STACCATOComplexDouble* _x, STACCATOComplexDouble* _b);
	/***********************************************************************************************
	* \brief Generate reduced matrices from projection matrices
	* \author Harikrishnan Sreekumar
	***********/
	void generateROM();

	void cleanPardiso();

private:
	/// HMesh object 
	HMesh * myHMesh;
	/// All Elements
	std::vector<FeElement*> myAllElements;
	std::vector<FeUmaElement*> allUMAElements;

	// FOM Complex data
	/// Stiffness Matrix
	MathLibrary::SparseMatrix<MKL_Complex16> *KComplex;
	/// Mass Matrix
	MathLibrary::SparseMatrix<MKL_Complex16> *MComplex;
	
	/// Input Matrix
	std::vector<MKL_Complex16> myB;
	/// Output matrix
	std::vector<MKL_Complex16> myC;

	// ROM Complex data
	/// Dense reduced stiffness matrix
	std::vector<MKL_Complex16> myKComplexReduced;
	/// Dense reduced mass matrix
	std::vector<MKL_Complex16> myMComplexReduced;
	/// Dense reduced Input matrix
	std::vector<MKL_Complex16> myBReduced;
	/// Dense reduced Output matrix
	std::vector<MKL_Complex16> myCReduced;

	// KMOR Data
	/// Expansion points
	std::vector<double> myExpansionPoints;
	/// Krylov order
	int myKrylovOrder;
	/// Projection matrix spanning input subspace
	std::vector<MKL_Complex16> myV;
	/// Projection matrix spanning output subspace
	std::vector<MKL_Complex16> myZ;
	/// inputDOFS
	std::vector<int> myInputDOFS;
	/// outputDOFS
	std::vector<int> myOutputDOFS;

	/// Common Storage for PARDISO sparse matrix
	int* pointerE;
	int* pointerB;
	int* columns;
	int* rowIndex;

	// FOM Sparse
#ifdef USE_INTEL_MKL
	sparse_matrix_t mySparseK;
	sparse_matrix_t mySparseM;

	MKL_INT m;
	/// number of columns
	MKL_INT n;
	/// pardiso variable
	MKL_INT *pardiso_pt[64]; // this is related to internal memory management, see PARDISO manual
							 /// pardiso variable
	MKL_INT pardiso_iparm[64];
	/// pardiso variable
	MKL_INT pardiso_mtype;
	/// pardiso variable
	MKL_INT pardiso_maxfct;
	/// pardiso variable
	MKL_INT pardiso_mnum;
	/// pardiso variable
	MKL_INT pardiso_msglvl;
	/// pardiso variable
	MKL_INT pardiso_neq;
	/// pardiso variable
	MKL_INT pardiso_nrhs;
	/// pardiso variable
	MKL_INT pardiso_phase;
	/// pardiso variable
	double pardiso_ddum;
	/// pardiso variable
	MKL_INT pardiso_idum;
	/// pardiso variable
	MKL_INT pardiso_error;

	STACCATOComplexDouble* values;
	std::string currentPart;
	bool isMIMO;
#endif
};
