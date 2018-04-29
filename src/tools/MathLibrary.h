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
* \file MathLibrary.h
* The header file of math functions in STACCATO.
* \date 8/28/2017
**************************************************************************************************/
#pragma once

#include <fstream>
#include <map>
#include <vector>
#include <assert.h>
#include "AuxiliaryParameters.h"
#include "Timer.h"

#include "AuxiliaryFunctions.h"

#ifdef USE_INTEL_MKL
#define MKL_DIRECT_CALL 1
#include <mkl.h>
#endif

#include <type_traits>

namespace MathLibrary {
	/***********************************************************************************************
	* \brief Compute the dot product of two dense vectors
	* \param[in] vec1 the 1st vector
	* \param[in] vec2 the 2nd vector
	* \return dot product
	* \author Stefan Sicklinger
	***********/
	double computeDenseDotProduct(const std::vector<double> &vec1, const std::vector<double> &vec2);
	/***********************************************************************************************
	* \brief Compute the dot product of two dense vectors
	* \param[in] vec1 the 1st vector
	* \param[in] vec2 the 2nd vector
	* \param[in] elements number of elements in vec1 (vec2)
	* \return dot product
	* \author Stefan Sicklinger
	***********/
	double computeDenseDotProduct(const double *vec1, const double *vec2, const int elements);
	/***********************************************************************************************
	* \brief Copy dense vector vec1 <- vec2
	* \param[in] vec1 the 1st vector
	* \param[in] vec2 the 2nd vector
	* \author Stefan Sicklinger
	***********/
	void copyDenseVector(double *vec1, const double *vec2, const int elements);
	/***********************************************************************************************
	* \brief Compute Euclidean norm of vector
	* \param[in] vec1 the 1st vector
	* \param[in] elements number of elements in vec1
	* \return Euclidean norm
	* \author Stefan Sicklinger
	***********/
	double computeDenseEuclideanNorm(const double *vec1, const int elements);
	/***********************************************************************************************
	* \brief Computes a vector-scalar product and adds the result to a vector. vec1 <- a*vec1 + vec2
	* \param[in] vec1 the 1st vector
	* \param[in] vec2 the 2nd vector
	* \param[in] a    scalar
	* \param[in] elements number of elements in vec1
	* \author Stefan Sicklinger
	***********/
	void computeDenseVectorAddition(double *vec1, double *vec2, const double a, const int elements);
	/***********************************************************************************************
	* \brief Computes vector scales by scalar vec1 <- vec1*a
	* \param[in] vec1 the 1st vector
	* \param[in] a    scalar
	* \param[in] elements number of elements in vec1
	* \author Stefan Sicklinger
	***********/
	void computeDenseVectorScalarMultiplication(double *vec1, const double a, const int elements);
	/***********************************************************************************************
	* \brief Compute the matrix product between two matrices (general)
	* \param[in] _m Specifies the number of rows of the matrix op(A) and of the matrix C.The value
	*  of m must be at least zero.
	* \param[in] _n Specifies the number of columns of the matrix op(B) and the number of columns
	*  of the matrix C. The value of n must be at least zero.
	* \param[in] _k Specifies the number of columns of the matrix op(A) and the number of rows
	*  of the matrix op(B).
	* \param[in] _A m rows by k columns
	* \param[in] _B k rows by n columns
	* \param[in/out] _C m rows by n columns
	* \param[in] _transposeA C=A^T*B
	* \param[in] false->C=A*B true -> C=alpha*A*B
	* \param[in] _alpha scalar
	* \param[in] false-> C=A*B true -> C+=A*B
	* \author Stefan Sicklinger
	***********/
	void computeDenseMatrixMatrixMultiplication(int _m, int _n, int _k, const double *_A, const double *_B, double *_C, const bool _transposeA, const bool _multByScalar, const double _alpha, const bool _addPrevious, const bool _useIntelSmall);
	/***********************************************************************************************
	* \brief Compute dense matrix-vector product
	* \param[in] _m Specifies the number of rows of the matrix A and vector length b
	* \param[in] _n Specifies the number of columns of the matrix A
	* \param[in] _A m rows by _n columns
	* \param[in] _b vector of length _n
	* \param[in/out] _c vector of length _m
	* \author Stefan Sicklinger
	***********/
	void computeDenseMatrixVectorMultiplication(int _m, int _n, const double *_A, const double *_b, double *_c);
	/***********************************************************************************************
	* \brief Computes the Cross Product of two vectors
	* \param[in] Vector 1
	* \param[in] Vector 2
	* \param[out] Cross Product
	* \author Stefan Sicklinger
	***********/
	std::vector<double> computeVectorCrossProduct(std::vector<double> &_v1, std::vector<double> &_v2);
	/***********************************************************************************************
	* \brief Solves 3X3 Linear System of Equations
	* \author Stefan Sicklinger
	***********/
	std::vector<double> solve3x3LinearSystem(std::vector<double>& _A, std::vector<double>& _b, double _EPS);
	/***********************************************************************************************
	* \brief Computes Determinant of Matrix
	* \author Stefan Sicklinger
	***********/
	double det3x3(std::vector<double>& _A);
	/********//**
			  * \brief This is a template class does compressed sparse row matrix computations: CSR Format (3-Array Variation)
			  *
			  * \author Stefan Sicklinger
			  **************************************************************************************************/
	template<class T>
	class SparseMatrix {
	public:
		typedef std::vector<std::map<size_t, T> > mat_t;
		typedef size_t row_iter;
		typedef std::map<size_t, T> col_t;
		typedef typename col_t::iterator col_iter;
		/***********************************************************************************************
		* \brief Constructor for symmetric matrices
		* \param[in] _m is the number of rows & columns
		* \param[in] _isSymmetric only the upper triangular form is stored
		* \param[in] _isPositiveDefinite
		* \author Stefan Sicklinger
		***********/
		SparseMatrix(const size_t _m, const bool _isSymmetric, const bool _isPositiveDefinite) {
			m = _m;
			n = _m;
			isSquare = true;
			isSymmetric = _isSymmetric;
			isPositiveDefinite = _isPositiveDefinite;
			alreadyCalled = false;
			if (!((typeid(T) == typeid(double)) || (typeid(T) == typeid(float)))) {
				assert(0);
			}
			mat = new mat_t(m);
			rowIndex = new std::vector<int>(m + 1);

		}
		/***********************************************************************************************
		* \brief Constructor for unsymmetric matrices
		* \param[in] _m is the number of rows
		* \param[in] _n is the number of columns
		* \author Stefan Sicklinger
		***********/
		SparseMatrix(const size_t _m, const size_t _n) {
			m = _m;
			n = _n;
			isSquare = false;
			isSymmetric = false;
			alreadyCalled = false;
			mat = new mat_t(m);
			rowIndex = new std::vector<int>(m + 1);
		}
		/***********************************************************************************************
		* \brief Destructor
		* \author Stefan Sicklinger
		***********/
		virtual ~SparseMatrix() {
			delete mat;
		}
		;
		/***********************************************************************************************
		* \brief Operator overloaded () for assignment of value e.g. A(i,j)=10
		* \param[in] i is the number of rows
		* \param[in] j is the number of columns
		* \author Stefan Sicklinger
		***********/
		inline T& operator()(size_t i, size_t j) {
			if (i >= m || j >= n)
				assert(0);
			if (i > j && isSymmetric == true)
				assert(0);
			//not allowed
			return (*mat)[i][j];
		}
		/***********************************************************************************************
		* \brief Operator overloaded () for assignment of value e.g. A(i,j)=10
		* \param[in] i is the number of rows
		* \param[in] j is the number of columns
		* \author Stefan Sicklinger
		***********/
		inline T operator()(size_t i, size_t j) const {
			if (i >= m || j >= n)
				assert(0);
			if (i > j && isSymmetric == true)
				assert(0);
			//not allowed
			return (*mat)[i][j];
		}
		/***********************************************************************************************
		* \brief Operator overloaded * for Matrix vector multiplication
		* \return std vector
		* \author Stefan Sicklinger
		***********/
		std::vector<T> operator*(const std::vector<T>& x) { //Computes y=A*x
			if (this->m != x.size())
				assert(0);
			//not allowed

			std::vector<T> y(this->m, 0);
			T sum;
			T sumSym;

			row_iter ii;
			col_iter jj;

			for (ii = 0; ii < m; ii++) {
				sum = 0;
				for (jj = (*mat)[ii].begin(); jj != (*mat)[ii].end(); jj++) {
					sum += (*jj).second * x[(*jj).first];

					if ((ii != (*jj).first) && isSymmetric) { //not on the main diagonal
															  //   std::cout << (*ii).first << " ssss "<< (*jj).second <<" tttt "<< x[(*ii).first] << "uuuu" << (*jj).first << std::endl;
						y[(*jj).first] += (*jj).second * x[ii];
					}

				}
				y[ii] += sum;
			}

			return y;
		}
		/***********************************************************************************************
		* \brief This function is a fast alternative to the operator overloading alternative
		* \param[in] x vector to be multiplied
		* \param[out] y result vector
		* \param[in] elements are the number of entries in the vector
		* \author Stefan Sicklinger
		***********/
		void mulitplyVec(const T* x, T* y, const size_t elements) { //Computes y=A*x
			if (this->m != elements)
				assert(0);
			//not allowed
			T sum;
			size_t iter;
			for (iter = 0; iter < elements; iter++) {
				y[iter] = 0;
			}

			row_iter ii;
			col_iter jj;

			for (ii = 0; ii < m; ii++) {
				sum = 0;
				for (jj = (*mat)[ii].begin(); jj != (*mat)[ii].end(); jj++) {
					sum += (*jj).second * x[(*jj).first];
					if ((ii != (*jj).first) && isSymmetric) { //not on the main diagonal
															  //   std::cout << (*ii).first << " ssss "<< (*jj).second <<" tttt "<< x[(*ii).first] << "uuuu" << (*jj).first << std::endl;
						y[(*jj).first] += (*jj).second * x[ii];
					}
				}
				y[ii] += sum;
			}
		}

		/***********************************************************************************************
		* \brief This function analysis and factorize the matrix
		* \author Stefan Sicklinger
		***********/
		void factorize(int nRHS) {
#ifdef USE_INTEL_MKL
			this->determineCSR();
			if (isSymmetric) {
				if (std::is_same<T, MKL_Complex16>::value) pardiso_mtype = 6;		// complex and symmetric 
				else if (std::is_same<T, double>::value) {
					pardiso_mtype = -2;			// real and symmetric indefinite
					if (isPositiveDefinite) {
						pardiso_mtype = 2;  // real symmetric positive definite
					}
					
				}
			}
			else {
				if (std::is_same<T, MKL_Complex16>::value) pardiso_mtype = 13;		// complex and unsymmetric matrix
				else if (std::is_same<T, double>::value) pardiso_mtype = 11;		// real and unsymmetric matrix
			}


			pardisoinit(pardiso_pt, &pardiso_mtype, pardiso_iparm);
			// set pardiso default parameters
			for (int i = 0; i < 64; i++) {
				//pardiso_iparm[i] = 0;
			}

			pardiso_iparm[0] = 1;    // No solver defaults
			pardiso_iparm[1] = 3;    // Fill-in reordering from METIS 
			pardiso_iparm[9] = 13;   // Perturb the pivot elements with 1E-13
			pardiso_iparm[23] = 1;   // 2-level factorization
			pardiso_iparm[36] = -99; // VBSR format
			pardiso_iparm[17] = -1;	 // Output: Number of nonzeros in the factor LU
			pardiso_iparm[18] = -1;	 // Output: Report Mflops
			pardiso_iparm[19] = 0;	 // Output: Number of CG iterations
		 // pardiso_iparm[27] = 1;   // PARDISO checks integer arrays ia and ja. In particular, PARDISO checks whether column indices are sorted in increasing order within each row.
			pardiso_maxfct = 1;	    // max number of factorizations
			pardiso_mnum = 1;		// which factorization to use
			pardiso_msglvl = 0;		// do NOT print statistical information
			pardiso_neq = m;		// number of rows of 
			pardiso_error = 1;		// Initialize error flag 
			pardiso_nrhs = nRHS;	// number of right hand side

			// Print MKL Version
			int len = 198;
			char buf[198];
			mkl_get_version_string(buf, len);
			printf("%s\n", buf);
			printf("\n");

			mkl_set_num_threads(STACCATO::AuxiliaryParameters::solverMKLThreads); // set number of threads to 1 for mkl call only
			std::cout << "Matrixtype for PARDISO: " << pardiso_mtype << std::endl;
			linearSolverTimer01.start();
			linearSolverTimer02.start();
			pardiso_phase = 11;
			pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase,
				&pardiso_neq, &values[0], &((*rowIndex)[0]), &columns[0], &pardiso_idum,
				&pardiso_nrhs, pardiso_iparm, &pardiso_msglvl, &pardiso_ddum, &pardiso_ddum,
				&pardiso_error);
			linearSolverTimer01.stop();
			std::cout << "Reordering completed: "<< linearSolverTimer01.getDurationMilliSec() <<" (milliSec)" << std::endl;

			linearSolverTimer01.start();
			pardiso_phase = 22;
			pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase,
				&pardiso_neq, &values[0], &((*rowIndex)[0]), &columns[0], &pardiso_idum,
				&pardiso_nrhs, pardiso_iparm, &pardiso_msglvl, &pardiso_ddum, &pardiso_ddum,
				&pardiso_error);
			linearSolverTimer01.stop();
			if (pardiso_error != 0) {
				std::cout << "Info: Number of zero or negative pivot = " << pardiso_iparm[29] << std::endl;
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
		/***********************************************************************************************
		* \brief This function checks the matrix
		* \author Stefan Sicklinger
		***********/
		void check() {
#ifdef USE_INTEL_MKL
			this->determineCSR();
			sparse_checker_error_values check_err_val;
			sparse_struct pt;
			int error = 0;

			sparse_matrix_checker_init(&pt);
			pt.n = m;
			pt.csr_ia = &(*rowIndex)[0];
			pt.csr_ja = &columns[0];
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
#endif
		}
		/***********************************************************************************************
		* \brief This function performs the prepare of a solution
		* \param[out] pointer to solution vector _x
		* \param[in]  pointer to rhs vector _b
		* \author Stefan Sicklinger
		***********/
		void solveDirect(T* _x, T* _b) { //Computes x=A\b
#ifdef USE_INTEL_MKL
			// pardiso forward and backward substitution
		 // pardiso_iparm[5] = 0; // write solution to b if true otherwise to x (default)
			linearSolverTimer01.start();
			pardiso_phase = 33; // forward and backward substitution
			mkl_set_num_threads(STACCATO::AuxiliaryParameters::solverMKLThreads); // set number of threads to 1 for mkl call only
			pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase,
				&pardiso_neq, &values[0], &((*rowIndex)[0]), &columns[0], &pardiso_idum,
				&pardiso_nrhs, pardiso_iparm, &pardiso_msglvl, _b, _x, &pardiso_error);
			if (pardiso_error != 0)
			{
				errorOut << "Error pardiso forward and backward substitution failed with error code: " << pardiso_error
					<< std::endl;
				exit(EXIT_FAILURE);
			}
			linearSolverTimer01.stop();
			std::cout << "Forward and backward substitution completed: " << linearSolverTimer01.getDurationMilliSec() << " (milliSec)" << std::endl;
			linearSolverTimer02.stop();
			std::cout << "=== Complete duration PARDISO: " << linearSolverTimer02.getDurationMilliSec() << " (milliSec)" << std::endl;
#endif
		}

		/***********************************************************************************************
		* \brief This function clean Pardiso
		* \author Stefan Sicklinger
		***********/
		void cleanPardiso() {
#ifdef USE_INTEL_MKL
			// clean pardiso
			pardiso_phase = -1; // deallocate memory
			pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase,
				&pardiso_neq, &values[0], &((*rowIndex)[0]), &columns[0], &pardiso_idum,
				&pardiso_nrhs, pardiso_iparm, &pardiso_msglvl, &pardiso_ddum, &pardiso_ddum,
				&pardiso_error);
			if (pardiso_error != 0) {
				errorOut << "Error deallocation of pardiso failed with error code: " << pardiso_error
					<< std::endl;
				exit(EXIT_FAILURE);
			}
#endif
		}

		/***********************************************************************************************
		* \brief This function prepares iterative solver
		* \param[in]  pointer to rhs vector _b
		* \param[out] pointer to solution vector _x
		* \author Stefan Sicklinger
		***********/
		void initIterativeSolver(double* _x, double* _b) {
#ifdef USE_INTEL_MKL
			//////////////////////////////////////////////////////////////
			this->determineCSR();
			//const int m --> number of dofs
			/*---------------------------------------------------------------------------
			* Allocate storage for the ?par parameters and the solution/rhs/residual vectors
			*---------------------------------------------------------------------------*/
			//number of the non-restarted FGMRES iterations
			int numberofNonRestartedIterations = 2500;// std::min(m, 2500);
			int numberofMaxIterations = numberofNonRestartedIterations;

			//Allocate storage
			uint64_t memoryLength = ((2 * numberofNonRestartedIterations + 1)*m + numberofNonRestartedIterations * (numberofNonRestartedIterations + 9) / 2 + 1);
			printf("Try to allocate %f Gb \n", (double)memoryLength * sizeof(double) / 1000000000);
			fgmres_memory = new double[memoryLength];
			double *trueResidual; // r=A*x-b
			double *rhs; // copy of _b
			trueResidual = new double[m];
			rhs = new double[m];
			//Allocate CSR precon-matrix
			double  *pIlut;
			MKL_INT *jPIlut;
			MKL_INT *iPIlut;
			MKL_INT maxfilIlut = m / 2 - 1; // Maximum fill-in, which is half of the preconditioner bandwidth. The number of non-zero elements in the rows of the preconditioner cannot exceed (2*maxfil+1).
			double tolIlut = 1.E-6;

			MKL_INT pIlutLength = (2 * maxfilIlut + 1)*m - maxfilIlut*(maxfilIlut + 1) + 1;
			printf("Try to allocate %f Gb for precond \n", (double)pIlutLength * sizeof(double) / 1000000000);
			//ilut// pIlut = new double[pIlutLength];
			pIlut = new double[values.size()];
			jPIlut = new MKL_INT[pIlutLength];
			iPIlut = new MKL_INT[m + 1];
			MKL_INT errorFlagIlut;
			double  *tmpVecIlut;
			tmpVecIlut = new double[m];

			/*---------------------------------------------------------------------------
			* Initialize the solver
			*---------------------------------------------------------------------------*/
			dfgmres_init(&m, _x, _b, &fgmres_RCI_request, fgmres_ipar, fgmres_dpar, fgmres_memory);
			printf("--------------------------------------------------\n");
			printf(" The SIMPLE example of usage of RCI FGMRES solver\n");
			printf("to solve a non-symmetric indefinite non-degenerate\n");
			printf("       algebraic system of linear equations\n");
			printf("--------------------------------------------------\n\n");
			if (fgmres_RCI_request != 0) {
				iterativeSolverHandleError();
			}
			/*---------------------------------------------------------------------------
			* Set the desired parameters
			*---------------------------------------------------------------------------*/
			//specifies the maximum number of iterations.The default value is min(150, n).
			fgmres_ipar[4] = numberofNonRestartedIterations;
			//specifies the number of the non-restarted FGMRES iterations. 
			//To run the restarted version of the FGMRES method, assign the number of iterations to ipar[14] before the restart. 
			//The default value is min(150, n), which means that by default the non-restarted version of FGMRES method is used.
			fgmres_ipar[14] = numberofMaxIterations;
			fgmres_ipar[8] = 0; //do not do the residual stopping test
			fgmres_ipar[9] = 1; //do request for the user defined stopping test
			fgmres_ipar[11] = 1; //do the check of the norm of the next generated vector automatically
			fgmres_ipar[7] = 1; // deactivate max iter check --> fgmres_ipar[4] ignored
			fgmres_ipar[10] = 1; //--> activate precon
			fgmres_dpar[0] = 1.0E-6; ////set the relative tolerance to 1.0D-3 instead of default value 1.0D-6
			/*---------------------------------------------------------------------------
			* Save the right-hand side in vector b for future use
			*---------------------------------------------------------------------------*/
			int inc = 1;
			dcopy(&m, _b, &inc, rhs, &inc);
			/*---------------------------------------------------------------------------
			* Calculate ILUT preconditioner.
			*                      !ATTENTION!
			* DCSRILUT routine uses some IPAR, DPAR set by DFGMRES_INIT routine.
			* Important for DCSRILUT default entries set by DFGMRES_INIT are
			* ipar[1] = 6 - output of error messages to the screen,
			* ipar[5] = 1 - allow output of errors,
			* ipar[6] = 1 - output warn messages if any and continue
			* ipar[30]= 0 - abort DCSRILUT calculations if routine meets zero diagonal element.
			*
			* If ILUT is going to be used out of Intel(R) MKL FGMRES context, than the values
			* of ipar[1], ipar[5], ipar[6], ipar[30], and dpar[30] should be user
			* provided before the DCSRILUT routine call.
			*
			* In this example, specific for DCSRILUT entries are set in turn:
			* ipar[30]= 1 - change small diagonal value to that given by dpar[31],
			* dpar[30]= 1.E-5  instead of the default value set by DFGMRES_INIT.
			*                  It is the target value of the diagonal value if it is
			*                  small as compared to the given tolerance multiplied
			*                  by the matrix row norm and the routine should
			*                  change it rather than abort DCSRILUT calculations.
			*---------------------------------------------------------------------------*/
			fgmres_ipar[30] = 1;
			fgmres_dpar[30] = 1.E-6;

			//ilut//dcsrilut(&m, &values[0], &((*rowIndex)[0]), &columns[0], pIlut, iPIlut, jPIlut, &tolIlut, &maxfilIlut, fgmres_ipar, fgmres_dpar, &errorFlagIlut);
			dcsrilu0(&m, &values[0], &((*rowIndex)[0]), &columns[0], pIlut, fgmres_ipar, fgmres_dpar, &errorFlagIlut);

			if (errorFlagIlut != 0)
			{
				printf("Preconditioner dcsrilut has returned the ERROR code %d\n", errorFlagIlut);
				iterativeSolverHandleError();
			}
			/*---------------------------------------------------------------------------
			* Check the correctness and consistency of the newly set parameters
			*---------------------------------------------------------------------------*/
			dfgmres_check(&m, _x, rhs, &fgmres_RCI_request, fgmres_ipar, fgmres_dpar, fgmres_memory);
			if (fgmres_RCI_request != 0) {
				iterativeSolverHandleError();
			}
			iterativeSolverPrintInfo();
			while (1) {
				/*---------------------------------------------------------------------------
				* Compute the solution by RCI (P)FGMRES solver without preconditioning
				* Reverse Communication starts here
				*---------------------------------------------------------------------------*/
				dfgmres(&m, _x, rhs, &fgmres_RCI_request, fgmres_ipar, fgmres_dpar, fgmres_memory);
				/*---------------------------------------------------------------------------
				* If RCI_request=0, then the solution was found with the required precision
				*---------------------------------------------------------------------------*/
				if (fgmres_RCI_request == 0) {
					// Finalize
					iterativeSolverFinalize(_x, rhs);
					/*-------------------------------------------------------------------------*/
					/* Release internal Intel(R) MKL memory that might be used for computations         */
					/* NOTE: It is important to call the routine below to avoid memory leaks   */
					/* unless you disable Intel(R) MKL Memory Manager                                   */
					/*-------------------------------------------------------------------------*/
					MKL_Free_Buffers();
					break;
				}
				/*---------------------------------------------------------------------------
				* If RCI_request=1, then compute the vector A*tmp[ipar[21]-1]
				* and put the result in vector tmp[ipar[22]-1]
				*---------------------------------------------------------------------------
				* NOTE that ipar[21] and ipar[22] contain FORTRAN style addresses,
				* therefore, in C code it is required to subtract 1 from them to get C style
				* addresses
				*---------------------------------------------------------------------------*/
				if (fgmres_RCI_request == 1)
				{
					if (isSymmetric) {
						char upperDiag = 'U';
						mkl_dcsrsymv(&upperDiag, &m, &values[0], &((*rowIndex)[0]), &columns[0], &fgmres_memory[fgmres_ipar[21] - 1], &fgmres_memory[fgmres_ipar[22] - 1]);
					}
					else {
						char transposeA = 'N';
						mkl_dcsrgemv(&transposeA, &m, &values[0], &((*rowIndex)[0]), &columns[0], &fgmres_memory[fgmres_ipar[21] - 1], &fgmres_memory[fgmres_ipar[22] - 1]);
					}
				}
				/*---------------------------------------------------------------------------
				* If RCI_request=2, then do the user-defined stopping test
				* The residual stopping test for the computed solution is performed here
				* NOTE: from this point vector rhs[N] is no longer containing the right-hand
				* side of the problem! It contains the current FGMRES approximation to the
				* solution. If you need to keep the right-hand side, save it in some other
				* vector before the call to dfgmres routine. Here we saved it in vector b[N]
				*---------------------------------------------------------------------------*/
				if (fgmres_RCI_request == 2)
				{
					/* Request to the dfgmres_get routine to put the solution into rhs[N] via ipar[12]
					WARNING: beware that the call to dfgmres_get routine with ipar[12]=0 at this
					stage may destroy the convergence of the FGMRES method, therefore, only advanced
					users should exploit this option with care */
					fgmres_ipar[12] = 1;
					/* Get the current computed solution in the vector rhs[N] */
					int currentIterCount;
					dfgmres_get(&m, _x, rhs, &fgmres_RCI_request, fgmres_ipar, fgmres_dpar, fgmres_memory, &currentIterCount);
					/* Compute the current true residual via Intel(R) MKL (Sparse) BLAS routines */
					if (isSymmetric) {
						char upperDiag = 'U';
						mkl_dcsrsymv(&upperDiag, &m, &values[0], &((*rowIndex)[0]), &columns[0], rhs, trueResidual);
					}
					else {
						char transposeA = 'N';
						mkl_dcsrgemv(&transposeA, &m, &values[0], &((*rowIndex)[0]), &columns[0], rhs, trueResidual);
					}
					double minusOne = -1.0E0;
					int inc = 1;
					daxpy(&m, &minusOne, _b, &inc, trueResidual, &inc);
					double residualNorm = dnrm2(&m, trueResidual, &inc);
					printf("currentIterCount: %d \t residualNorm: %f \n", currentIterCount, residualNorm);
					if (currentIterCount == 800) {
						// Finalize
						iterativeSolverFinalize(_x, rhs);
						/*-------------------------------------------------------------------------*/
						/* Release internal Intel(R) MKL memory that might be used for computations         */
						/* NOTE: It is important to call the routine below to avoid memory leaks   */
						/* unless you disable Intel(R) MKL Memory Manager                                   */
						/*-------------------------------------------------------------------------*/
						MKL_Free_Buffers();
						break;
					}
				}
				/*---------------------------------------------------------------------------
				* If RCI_request=3, then apply the preconditioner on the vector
				* tmp[ipar[21]-1] and put the result in vector tmp[ipar[22]-1]
				*---------------------------------------------------------------------------
				* NOTE that ipar[21] and ipar[22] contain FORTRAN style addresses,
				* therefore, in C code it is required to subtract 1 from them to get C style
				* addresses
				* Here is the recommended usage of the result produced by ILUT routine
				* via standard Intel(R) MKL Sparse Blas solver routine mkl_dcsrtrsv.
				*---------------------------------------------------------------------------*/
				if (fgmres_RCI_request == 3)
				{
					char cvar1 = 'L';
					char cvar = 'N';
					char cvar2 = 'U';
					//ilut//mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &m, pIlut, iPIlut, jPIlut, &fgmres_memory[fgmres_ipar[21] - 1], tmpVecIlut);
					mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &m, pIlut, &((*rowIndex)[0]), &columns[0], &fgmres_memory[fgmres_ipar[21] - 1], tmpVecIlut);

					cvar1 = 'U';
					cvar = 'N';
					cvar2 = 'N';
					//ilut//mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &m, pIlut, iPIlut, jPIlut, tmpVecIlut, &fgmres_memory[fgmres_ipar[22] - 1]);
					mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &m, pIlut, &((*rowIndex)[0]), &columns[0], tmpVecIlut, &fgmres_memory[fgmres_ipar[22] - 1]);
					printf("Preconditioner applied!\n");
				}
				/*---------------------------------------------------------------------------
				* If RCI_request=anything else, then dfgmres subroutine failed
				* to compute the solution vector: computed_solution[N]
				*---------------------------------------------------------------------------*/
				if (!(fgmres_RCI_request == 0 || fgmres_RCI_request == 1 || fgmres_RCI_request == 2 || fgmres_RCI_request == 3))
				{
					iterativeSolverHandleError();
					break;
				}
			}// while
			 //////////////////////////////////////////////////////////////
#endif
		}
		/***********************************************************************************************
		* \brief This function prepares Conjugate Gradient iterative solver
		* \param[in]  pointer to rhs vector _b
		* \param[out] pointer to solution vector _x
		* \author Harikrishnan Sreekumar
		***********/
		void initCGIterativeSolver(double* _x, double* _b) {
#ifdef USE_INTEL_MKL
			this->determineCSR();

			/*---------------------------------------------------------------------------*/
			/* Allocate storage for the solver ?par and temporary storage tmp            */
			/*---------------------------------------------------------------------------*/
			// maximum number of iterations
			int numberofMaxIterations = 2500;

			// Allocate Memory
			uint64_t memoryLength = (m * 4);
			dcg_memory = new double[memoryLength];
			double *rhs; // copy of _b
			rhs = new double[m];

			/*---------------------------------------------------------------------------*/
			/* Some additional variables to use with the RCI (P)CG solver                */
			/*---------------------------------------------------------------------------*/
			char matdes[3];
			matdes[0] = 'd';
			matdes[1] = 'l';
			matdes[2] = 'n';

			/*---------------------------------------------------------------------------*/
			/* Initialize the solver                                                     */
			/*---------------------------------------------------------------------------*/
			dcg_init(&m, _x, _b, &dcg_RCI_request, dcg_ipar, dcg_dpar, dcg_memory);
			if (dcg_RCI_request != 0)
				iterativeSolverHandleError();
			/*---------------------------------------------------------------------------
			* Save the right-hand side in vector b for future use
			*---------------------------------------------------------------------------*/
			int inc = 1;
			dcopy(&m, _b, &inc, _b, &inc);
			/*---------------------------------------------------------------------------*/
			/* Set the desired parameters:                                               */
			/* INTEGER parameters:                                                        */
			/* set the maximal number of iterations to 100                               */
			/* LOGICAL parameters:                                                       */
			/* run the Preconditioned version of RCI (P)CG with preconditioner C_inverse */
			/* DOUBLE parameters                                                         */
			/* -                                                                         */
			/*---------------------------------------------------------------------------*/
			dcg_ipar[4] = numberofMaxIterations;
			dcg_ipar[10] = 1;
			/*---------------------------------------------------------------------------*/
			/* Check the correctness and consistency of the newly set parameters         */
			/*---------------------------------------------------------------------------*/
			dcg_check(&m, _x, _b, &dcg_RCI_request, dcg_ipar, dcg_dpar, dcg_memory);
			if (dcg_RCI_request != 0)
				iterativeSolverHandleError();
			/*---------------------------------------------------------------------------*/
			/* Compute the solution by RCI (P)CG solver                                  */
			/* Reverse Communications starts here                                        */
			/*---------------------------------------------------------------------------*/
			while (1) {
				dcg(&m, _x, _b, &dcg_RCI_request, dcg_ipar, dcg_dpar, dcg_memory);
				/*---------------------------------------------------------------------------*/
				/* If rci_request=0, then the solution was found according to the requested  */
				/* stopping tests. In this case, this means that it was found after 100      */
				/* iterations.                                                               */
				/*---------------------------------------------------------------------------*/
				if (dcg_RCI_request == 0) {
					iterativeSolverCGFinalize(_x, _b);
					break;
				}
				/*---------------------------------------------------------------------------*/
				/* If rci_request=1, then compute the vector A*tmp[0]                      */
				/* and put the result in vector tmp[n]                                     */
				/*---------------------------------------------------------------------------*/
				else if (dcg_RCI_request == 1)
				{
					char tr = 'u';
					mkl_dcsrsymv(&tr, &m, &values[0], &((*rowIndex)[0]), &columns[0], dcg_memory, &dcg_memory[m]);
				}
				/*---------------------------------------------------------------------------*/
				/* If rci_request=2, then do the user-defined stopping test: compute the     */
				/* Euclidean norm of the actual residual using Intel(R) MKL routines and check if     */
				/* it is less than 1.E-8                                                     */
				/*---------------------------------------------------------------------------*/
				else if (dcg_RCI_request == 2)
				{
					char tr = 'u';
					double eone = -1.E0;
					MKL_INT ione = 1;
					double* temp;
					temp = new double[m];
					mkl_dcsrsymv(&tr, &m, &values[0], &((*rowIndex)[0]), &columns[0], _x, temp);
					daxpy(&m, &eone, _b, &ione, temp, &ione);
					double euclidean_norm = dnrm2(&m, temp, &ione);
					/*---------------------------------------------------------------------------*/
					/* The solution has been found according to the user-defined stopping test   */
					/*---------------------------------------------------------------------------*/
					if (euclidean_norm <= 1.e-8) {
						iterativeSolverCGFinalize(_x, _b);
						break;
					}
				}
				/*---------------------------------------------------------------------------*/
				/* If rci_request=3, then compute apply the preconditioner matrix C_inverse  */
				/* on vector tmp[2*n] and put the result in vector tmp[3*n]                  */
				/*---------------------------------------------------------------------------*/
				else if (dcg_RCI_request == 3)
				{
					double one = 1.E0;
					mkl_dcsrsv(&matdes[2], &m, &one, matdes, &values[0], &columns[0] , &((*rowIndex)[0]), &((*rowIndex)[1]), &dcg_memory[2 * m], &dcg_memory[3 * m]);

				}
				else
					iterativeSolverHandleError();
			}
#endif
		}
		/***********************************************************************************************
		* \brief This function print CSR Row and Column Vector
		* \author Harikrishnan Sreekumar
		***********/
		void writeCSRtoFile(std::string _prefix, std::string _format) {
			if (_format == "dat" || _format == "DAT" || _format == ".dat" || _format == ".DAT") {
				determineCSR();
				AuxiliaryFunctions::writeIntegerVectorDatFormat(_prefix + "_CSR_IA.dat", *rowIndex);
				AuxiliaryFunctions::writeIntegerVectorDatFormat(_prefix + "_CSR_JA.dat", columns);

				writeMat(std::is_floating_point<T>{}, _prefix + "_CSR_MAT.dat", values);
			}
			else if (_format == "mtx" || _format == "MTX" || _format == ".mtx" || _format == ".MTX") {
				writeMtxMat(std::is_floating_point<T>{}, _prefix + "_CSR_MAT.mtx");
			}
			else
				std::cout << "Unsupported File Format."<< std::endl;

		}

	private:
		/// pointer to the vector of maps
		mat_t* mat;
		/// true if a square matrix should be stored
		bool isSquare;
		/// true if a symmetric matrix should be stored
		bool isSymmetric;
		/// Matrix is SPD
		bool isPositiveDefinite;
		/// number of rows
		MKL_INT m;
		/// number of columns
		MKL_INT n;
		/// A real array that contains the non-zero elements of a sparse matrix. The non-zero elements are mapped into the values array using the row-major upper triangular storage mapping.
		std::vector<T> values;
		/// Element i of the integer array columns is the number of the column that contains the i-th element in the values array.
		std::vector<int> columns;
		/// Element j of the integer array rowIndex gives the index of the element in the values array that is first non-zero element in a row j.
		std::vector<int>* rowIndex;
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
		/// GMRES INTEGER array, this parameter specifies the integer set of data for the RCI FGMRES computations :
		MKL_INT fgmres_ipar[128];
		/// FGEMRES DOUBLE PRECISION array, this parameter specifies the double precision set of data for the RCI CG computations, specifically:
		double fgmres_dpar[128];
		/// FGMRES memory DOUBLE PRECISION array of size((2 * ipar(15) + 1)*n + ipar(15)*(ipar(15) + 9) / 2 + 1)) used to supply the double precision temporary space for the RCI FGMRES computations, specifically:
		double *fgmres_memory;
		///INTEGER.Gives information about the result of the routine.
		MKL_INT fgmres_RCI_request;
		/// determineCSR already called
		bool alreadyCalled;
		/// This parameter specifies the integer set of data for the RCI CG computations
		MKL_INT dcg_ipar[128];
		/// this parameter is used to specify the double precision set of data for the RCI CG computations
		double dcg_dpar[128];
		/// array of size (n*4)for SRHS, and (n*(3+nrhs))for MRHS. This parameter is used to supply the double precision temporary space for the RCI CG computations
		double *dcg_memory;
		///INTEGER.Gives information about the result of work of the RCI CG routines.
		MKL_INT dcg_RCI_request;
		/// Contains the value of the current iteration number
		MKL_INT dcg_itercount;

		/***********************************************************************************************
		* \brief This fills the three vectors of the CSR format (one-based)
		* \author Stefan Sicklinger
		***********/
		void determineCSR() {
			if (!alreadyCalled) {
				std::cout << "!alreadyCalled" << std::endl;
				row_iter ii;
				col_iter jj;
				size_t ele_row = 0; //elements in current row
				std::cout << std::scientific;

				for (ii = 0; ii < m; ii++) {
					(*rowIndex)[ii] = (ele_row + 1);
					for (jj = (*mat)[ii].begin(); jj != (*mat)[ii].end(); jj++) {
						columns.push_back(((*jj).first) + 1);
						values.push_back((*jj).second);
						ele_row++;
					}

				}
				(*rowIndex)[m] = (ele_row + 1);
			}
			// This will free the memory
			mat->clear();
			mat_t dummyMat;
			mat->swap(dummyMat);

			alreadyCalled = true;
		}
		/***********************************************************************************************
		* \brief This function handles iterative solver errors
		* \author Stefan Sicklinger
		***********/
		void iterativeSolverHandleError() {
			/*-------------------------------------------------------------------------*/
			/* Release internal Intel(R) MKL memory that might be used for computations         */
			/* NOTE: It is important to call the routine below to avoid memory leaks   */
			/* unless you disable Intel(R) MKL Memory Manager                                   */
			/*-------------------------------------------------------------------------*/
			printf("\nThe iterative solver FAILED as the solver has returned the ERROR code %d", fgmres_RCI_request);
			MKL_Free_Buffers();
		}

		/***********************************************************************************************
		* \brief This function prints iterative solver RCI info
		* \author Stefan Sicklinger
		***********/
		void iterativeSolverPrintInfo() {
			printf("Some info about the current run of RCI FGMRES method:\n\n");
			if (fgmres_ipar[7])
			{
				printf("As ipar[7]=%d, the automatic test for the maximal number of ", fgmres_ipar[7]);
				printf("iterations will be\nperformed\n");
			}
			else
			{
				printf("As ipar[7]=%d, the automatic test for the maximal number of ", fgmres_ipar[7]);
				printf("iterations will be\nskipped\n");
			}
			printf("+++\n");
			if (fgmres_ipar[8])
			{
				printf("As ipar[8]=%d, the automatic residual test will be performed\n", fgmres_ipar[8]);
			}
			else
			{
				printf("As ipar[8]=%d, the automatic residual test will be skipped\n", fgmres_ipar[8]);
			}
			printf("+++\n");
			if (fgmres_ipar[9])
			{
				printf("As ipar[9]=%d, the user-defined stopping test will be ", fgmres_ipar[9]);
				printf("requested via\nRCI_request=2\n");
			}
			else
			{
				printf("As ipar[9]=%d, the user-defined stopping test will not be ", fgmres_ipar[9]);
				printf("requested, thus,\nRCI_request will not take the value 2\n");
			}
			printf("+++\n");
			if (fgmres_ipar[10])
			{
				printf("As ipar[10]=%d, the Preconditioned FGMRES iterations will be ", fgmres_ipar[10]);
				printf("performed, thus,\nthe preconditioner action will be requested via RCI_request=3\n");
			}
			else
			{
				printf("As ipar[10]=%d, the Preconditioned FGMRES iterations will not ", fgmres_ipar[10]);
				printf("be performed,\nthus, RCI_request will not take the value 3\n");
			}
			printf("+++\n");
			if (fgmres_ipar[11])
			{
				printf("As ipar[11]=%d, the automatic test for the norm of the next ", fgmres_ipar[11]);
				printf("generated vector is\nnot equal to zero up to rounding and ");
				printf("computational errors will be performed,\nthus, RCI_request will not take the value 4\n");
			}
			else
			{
				printf("As ipar[11]=%d, the automatic test for the norm of the next ", fgmres_ipar[11]);
				printf("generated vector is\nnot equal to zero up to rounding and ");
				printf("computational errors will be skipped,\nthus, the user-defined test ");
				printf("will be requested via RCI_request=4\n");
			}
			printf("+++\n\n");
		}
		/***********************************************************************************************
		* \brief This finalizes the iterative solver
		* \param[in]  pointer to rhs vector _rhs
		* \param[out] pointer to solution vector _x
		* \author Stefan Sicklinger
		***********/
		void iterativeSolverFinalize(double* _x, double* _rhs) {
			// Finalize
			fgmres_ipar[12] = 0;
			int currentIterCount;
			dfgmres_get(&m, _x, _rhs, &fgmres_RCI_request, fgmres_ipar, fgmres_dpar, fgmres_memory, &currentIterCount);
			/*---------------------------------------------------------------------------
			* Print solution vector: computed_solution[N] and the number of iterations: itercount
			*---------------------------------------------------------------------------*/
			printf(" The system has been solved \n");
			printf("\n The following solution has been obtained: \n");
			for (int i = 0; i < 3; i++)
			{
				printf("computed_solution[%d]=", i);
				printf("%e\n", _x[i]);
			}
			printf("\n Number of iterations: %d\n", currentIterCount);
		}
		/***********************************************************************************************
		* \brief This finalizes the CG iterative solver
		* \param[in]  pointer to rhs vector _rhs
		* \param[out] pointer to solution vector _x
		* \author Harikrishnan Sreekumar
		***********/
		void iterativeSolverCGFinalize(double* _x, double* _rhs) {
			dcg_get(&m, _x, _rhs, &dcg_RCI_request, dcg_ipar, dcg_dpar, dcg_memory, &dcg_itercount);
			printf("================ ITERATIVE CONJUGATE GRADIENT SOLVER =================\n");
			printf("This example has successfully PASSED through all steps of computation!\n");
			printf("Number of iterations: %d\n", dcg_itercount);
			printf("======================================================================\n\n");
			/*-------------------------------------------------------------------------*/
			/* Release internal Intel(R) MKL memory that might be used for computations         */
			/* NOTE: It is important to call the routine below to avoid memory leaks   */
			/* unless you disable Intel(R) MKL Memory Manager                                   */
			/*-------------------------------------------------------------------------*/
			MKL_Free_Buffers();
		}
		/***********************************************************************************************
		* \brief Tag Dispatching. Performs Double Writing Code for the Matrix
		* \param[in] type_trail - true_type for Double and false_type for Complex
		* \param[in] File Name
		* \param[in] Double Vector
		* \author Harikrishnan Sreekumar
		***********/
		void writeMat(std::true_type, std::string _fileName, std::vector<T> &_vector) {
			AuxiliaryFunctions::writeDoubleVectorDatFormat(_fileName, values);
		}
		/***********************************************************************************************
		* \brief Tag Dispatching. Performs MKLComplex Writing Code for the Matrix
		* \param[in] type_trail - true_type for Double and false_type for Complex
		* \param[in] File Name
		* \param[in] MKLComplex Vector
		* \author Harikrishnan Sreekumar
		***********/
		void writeMat(std::false_type, std::string _fileName, std::vector<T> &_vector) {
			AuxiliaryFunctions::writeMKLComplexVectorDatFormat(_fileName, values);
		}
		/***********************************************************************************************
		* \brief Tag Dispatching. Performs Double Writing Code for the Matrix to Mtx File
		* \param[in] type_trail - true_type for Double and false_type for Complex
		* \param[in] File Name
		* \author Harikrishnan Sreekumar
		***********/
		void writeMtxMat(std::true_type, std::string _fileName) {
			std::cout << ">> Writing " << _fileName << "#" << m << "x" << n << "..." << std::endl;
			size_t ii_counter;
			size_t jj_counter;

			std::ofstream myfile;
			myfile.open(_fileName);
			myfile.precision(std::numeric_limits<double>::digits10 + 1);
			myfile << std::scientific;
			for (ii_counter = 0; ii_counter < m; ii_counter++) {
				for (jj_counter = 0; jj_counter < n; jj_counter++) {
					if ((*mat)[ii_counter].find(jj_counter) != (*mat)[ii_counter].end()) {
						myfile << ii_counter << "\t" << jj_counter << "\t" << (*mat)[ii_counter].find(jj_counter)->second << std::endl;
					}
				}
				//myfile << std::endl;
			}
			myfile << std::endl;
			myfile.close();
		}
		/***********************************************************************************************
		* \brief Tag Dispatching. Performs MKLComplex Writing Code for the Matrix to Mtx File
		* \param[in] type_trail - true_type for Double and false_type for Complex
		* \param[in] File Name
		* \author Harikrishnan Sreekumar
		***********/
		void writeMtxMat(std::false_type, std::string _fileName) {
			std::cout << ">> Writing " << _fileName << "#" << m << "x" << n << "..." << std::endl;
			size_t ii_counter;
			size_t jj_counter;

			std::ofstream myfile;
			myfile.open(_fileName);
			myfile.precision(std::numeric_limits<double>::digits10 + 1);
			myfile << std::scientific;
			for (ii_counter = 0; ii_counter < m; ii_counter++) {
				for (jj_counter = 0; jj_counter < n; jj_counter++) {
					if ((*mat)[ii_counter].find(jj_counter) != (*mat)[ii_counter].end()) {
						myfile << ii_counter << "\t" << jj_counter << "\t" << (*mat)[ii_counter].find(jj_counter)->second.real << "+" << (*mat)[ii_counter].find(jj_counter)->second.real << "i" << std::endl;
					}
				}
				//myfile << std::endl;
			}
			myfile << std::endl;
			myfile.close();
		}
	};
	/***********************************************************************************************
	* \brief Gauss points for 2D bilinear elements
	* \author Stefan Sicklinger
	***********/
	static const double tmpSqrt13 = sqrt(1.0 / 3.0);
	const double quadGaussPoints2D4Point[8] = { tmpSqrt13, tmpSqrt13, -tmpSqrt13, tmpSqrt13, -tmpSqrt13, -tmpSqrt13, tmpSqrt13, -tmpSqrt13 };
	/***********************************************************************************************
	* \brief Gauss points for 3D quadratic tet element
	* \author Stefan Sicklinger
	***********/
	static const double tmpA = (5 + 3 * sqrt(5.0)) / 20.0;
	static const double tmpB = (5 - sqrt(5.0)) / 20.0;
	const double tetGaussPoints3D4Points[16] = {
		tmpA, tmpB, tmpB, tmpB,
		tmpB, tmpA, tmpB, tmpB,
		tmpB, tmpB, tmpA, tmpB,
		tmpB, tmpB, tmpB, tmpA
	};
	const double tetGaussWeights3D4Points = 1.0 / 4.0;

	static const double tmpG1A = (7. - sqrt(15.0)) / 34.;
	static const double tmpG1B = 1. - 3. * tmpG1A;
	static const double tmpG2A = 7. / 17. - tmpG1A;
	static const double tmpG2B = 1. - 3. * tmpG2A;
	static const double tmpG3A = (10. - 2. * sqrt(15.0)) / 40.;
	static const double tmpG3B = 1. / 2. - tmpG3A;
	static const double tmpG4A = 1. / 4.;
	static const double tmpW1 = (2665. + 14. * sqrt(15.0)) / 37800.;
	static const double tmpW2 = (2665. - 14. * sqrt(15.0)) / 37800.;
	static const double tmpW3 = 10. / 189.;
	static const double tmpW4 = 16. / 135.;

	const double tetGaussPoints3D15Points[60] = {
		tmpG1B, tmpG1A, tmpG1A, tmpG1A,
		tmpG1A, tmpG1B, tmpG1A, tmpG1A,
		tmpG1A, tmpG1A, tmpG1B, tmpG1A,
		tmpG1A, tmpG1A, tmpG1A, tmpG1B,

		tmpG2B, tmpG2A, tmpG2A, tmpG2A,
		tmpG2A, tmpG2B, tmpG2A, tmpG2A,
		tmpG2A, tmpG2A, tmpG2B, tmpG2A,
		tmpG2A, tmpG2A, tmpG2A, tmpG2B,

		tmpG3B, tmpG3B, tmpG3A, tmpG3A,
		tmpG3B, tmpG3A, tmpG3B, tmpG3A,
		tmpG3B, tmpG3A, tmpG3A, tmpG3B,
		tmpG3A, tmpG3B, tmpG3B, tmpG3A,
		tmpG3A, tmpG3B, tmpG3A, tmpG3B,
		tmpG3A, tmpG3A, tmpG3B, tmpG3B,

		tmpG4A, tmpG4A, tmpG4A, tmpG4A
	};
	const double tetGaussWeights3D15Points[15] = { tmpW1 ,tmpW1 ,tmpW1 ,tmpW1, tmpW2 ,tmpW2 ,tmpW2 ,tmpW2, tmpW3 ,tmpW3 ,tmpW3 ,tmpW3, tmpW3, tmpW3, tmpW4 };
} /* namespace Math */
