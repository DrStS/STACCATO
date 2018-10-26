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
class SimuliaUMA;
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
	void getSystemMatricesODB();
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
	/***********************************************************************************************
	* \brief This function clean Pardiso
	* \author Stefan Sicklinger
	***********/
	void cleanPardiso();
	/***********************************************************************************************
	* \brief Algorithm to reveil rank for the R matrix from QR decomposition
	* \param[in] _mat R matrix
	* \param[in] _m number of rows
	* \param[in] _n number of columns
	* \param[in] _tol threshold value
	* \author Harikrishnan Sreekumar
	***********/
	int reveilRankQR_R(const STACCATOComplexDouble* _mat, int _m, int _n, double _tol);
	/***********************************************************************************************
	* \brief This function carries out the ODB import routine for already prepared SimuliaODB reader
	* \author Harikrishnan Sreekumar
	***********/
	void buildAbqODB();
	/***********************************************************************************************
	* \brief This function carries out the SIM import routine by preparing the SimuliaUMA reader
	* \param[in] _iPart XML Part ID to instantiate SimuliaUMA reader
	* \author Harikrishnan Sreekumar
	***********/
	void buildAbqSIM(int _iPart);
	/***********************************************************************************************
	* \brief Function to display current FOM and ROM information
	* \author Harikrishnan Sreekumar
	***********/
	void displayModelSize();
	/***********************************************************************************************
	* \brief Function to generate input and output matrix for KMOR
	* \author Harikrishnan Sreekumar
	***********/
	void generateInputOutputMatricesForFOM();
	/***********************************************************************************************
	* \brief Function to export the reduced matrices
	* \author Harikrishnan Sreekumar
	***********/
	void exportROMToFiles();
	/***********************************************************************************************
	* \brief This function carries out the Substructuring analysis
	* \author Harikrishnan Sreekumar
	***********/
	void performAnalysis();
	/***********************************************************************************************
	* \brief This function carries out the back transformation from Krylov subspace to original space
	* \param[in] _analysisName Name of Current Analysis
	* \param[in] _freq Fine frequency for interpolation
	* \param[in] _inputLoad Input
	* \param[in] _numLoadCase Number of loadcases
	* \author Harikrishnan Sreekumar
	***********/
	void backTransformKMOR(std::string _analysisName, std::vector<double>* _freq, STACCATOComplexDouble* _inputLoad, int _numLoadCase);
	/***********************************************************************************************
	* \brief This function carries out the direct solve of FOM
	* \param[in] _analysisName Name of Current Analysis
	* \param[in] _freq Fine frequency for interpolation
	* \param[in] _inputLoad Input
	* \param[in] _numLoadCase Number of loadcases
	* \author Harikrishnan Sreekumar
	***********/
	void performSolveFOM(std::string _analysisName, std::vector<double>* _freq, STACCATOComplexDouble* _inputLoad, int _numLoadCase);
	/***********************************************************************************************
	* \brief Function to get node set information from XML for SIM import routine
	* \param[in] _iPart XML Part ID
	* \author Harikrishnan Sreekumar
	***********/
	void buildXMLforSIM(int iPart);
	/***********************************************************************************************
	* \brief Function to clear memory by removing the FOM data
	* \author Harikrishnan Sreekumar
	***********/
	void clearDataFOM();
	/***********************************************************************************************
	* \brief Generates global map
	* \author Harikrishnan Sreekumar
	***********/
	void generateCollectiveGlobalMap(std::map<int, std::vector<int>> &_dofMap, std::map<int, std::vector<int>> &_globalMap);
	/***********************************************************************************************
	* \brief Generates a file with node to local dof and global dof map
	* \author Harikrishnan Sreekumar
	***********/
	void printMapToFile();
private:
#ifdef USE_INTEL_MKL
	/// HMesh object 
	HMesh * myHMesh;
	/// All Elements
	std::vector<FeElement*> myAllElements;
	std::vector<FeUmaElement*> allUMAElements;

	// FOM Complex data	
	/// Input Matrix
	std::vector<STACCATOComplexDouble> myB;
	/// Output matrix
	std::vector<STACCATOComplexDouble> myC;
	// FOM Sparse
	sparse_matrix_t mySparseK;
	sparse_matrix_t mySparseM;
	sparse_matrix_t mySparseD;

	// ROM Complex data
	/// Dense reduced stiffness matrix
	std::vector<STACCATOComplexDouble> myKComplexReduced;
	/// Dense reduced mass matrix
	std::vector<STACCATOComplexDouble> myMComplexReduced;
	/// Dense reduced damping matrix
	std::vector<STACCATOComplexDouble> myDComplexReduced;
	/// Dense reduced Input matrix
	std::vector<STACCATOComplexDouble> myBReduced;
	/// Dense reduced Output matrix
	std::vector<STACCATOComplexDouble> myCReduced;

	// KMOR Data
	/// Expansion points
	std::vector<double> myExpansionPoints;
	/// Krylov order
	int myKrylovOrder;
	/// Projection matrix spanning input subspace
	std::vector<STACCATOComplexDouble> myV;
	/// Projection matrix spanning output subspace
	std::vector<STACCATOComplexDouble> myZ;
	/// inputDOFS
	std::vector<int> myInputDOFS;
	/// outputDOFS
	std::vector<int> myOutputDOFS;

	/// Common Storage for PARDISO sparse matrix
	int* pointerE;
	int* pointerB;
	int* columns;
	int* rowIndex;
	
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
	bool isSymMIMO;
	bool enablePropDamping;
	bool isSymmetricSystem;

	int FOM_DOF;
	int ROM_DOF;

	std::string myModelType;
	std::string myAnalysisType;

	// UMA Reader
	SimuliaUMA* myUMAReader;

	/// Map holding NodeSets
	std::map<std::string, std::vector<int>> nodeSetsMap;

	/// Struct to hold CSR details of sparse matrices
	struct csrStruct {
		std::vector<int> csr_ia;
		std::vector<int> csr_ja;
		std::vector<STACCATOComplexDouble> csr_values;
		std::vector<int> csrPointerB;
		std::vector<int> csrPointerE;
	}*systemCSR;

	// Export Flags
	bool writeFOM;
	bool writeROM;
	bool writeProjectionmatrices;
	bool exportRHS;
	bool exportSolution;
	bool writeTransferFunctions;

	// Maps
	std::map<int, std::vector<int>> nodeToDofCommonMap;
	std::map<int, std::vector<int>> nodeToGlobalCommonMap;

	// StaccatoAbaqusInputOutputInfoMap
	/// Sets with same index info
	/// List of node numbers for input
	std::vector<int> myAbaqusInputNodeList;
	/// List of corresponding dof numbers for input
	std::vector<int> myAbaqusInputDofList;
	/// List of node numbers for output
	std::vector<int> myAbaqusOutputNodeList;
	/// List of corresponding dof numbers for output
	std::vector<int> myAbaqusOutputDofList;

	int numDOF_u;	// Displacement nodes
	int numDOF_p;	// Pressure nodes
	int numDOF_ui;	// Internal displacement nodes
	int numDOF_pi;	// Internal pressure nodes
	int numUndetected;	// number of nodes undetected

	int totaldof;	// total number of nodes 
#endif
};
