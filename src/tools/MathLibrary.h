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
#ifndef MATHLIBRARY_H_
#define MATHLIBRARY_H_

#include <fstream>
#include <map>
#include <vector>
#include <assert.h>

#include "AuxiliaryParameters.h"
#ifdef USE_INTEL_MKL
#define MKL_DIRECT_CALL 1
#include <mkl.h>
#endif


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
	void computeDenseVectorAddition(double *vec1, const double *vec2, const double a, const int elements);
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
		 * \author Stefan Sicklinger
		 ***********/
		SparseMatrix(const size_t _m, const bool _isSymmetric) {
			m = _m;
			n = _m;
			isSquare = true;
			isSymmetric = _isSymmetric;
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
		void factorize() {
#ifdef USE_INTEL_MKL
			this->determineCSR();
			if (isSymmetric) {
				if(std::is_same<T, MKL_Complex16>::value) pardiso_mtype = 6;		// complex and symmetric 
				else if (std::is_same<T, double>::value) pardiso_mtype  = -2;		// real and symmetric indefinite
			}
			else {
				if (std::is_same<T, MKL_Complex16>::value) pardiso_mtype = 13;		// complex and unsymmetric matrix
				else if (std::is_same<T, double>::value) pardiso_mtype   = 11;		// real and unsymmetric matrix
			}

			// set pardiso default parameters
			pardisoinit(pardiso_pt, &pardiso_mtype, pardiso_iparm);

			pardiso_iparm[1] = 2; //The parallel (OpenMP) version of the nested dissection algorithm
			pardiso_iparm[18] = -1; //Report Mflops 
			pardiso_maxfct = 1; // max number of factorizations
			pardiso_mnum = 1; // which factorization to use
			pardiso_msglvl = 1; // do NOT print statistical information
			pardiso_neq = m; // number of rows of 
			pardiso_error = 0; //Initialize error flag 
			//pardiso_iparm[27] = 1; // PARDISO checks integer arrays ia and ja. In particular, PARDISO checks whether column indices are sorted in increasing order within each row.
			pardiso_nrhs = 1; // number of right hand side
			pardiso_phase = 12; // analysis and factorization
			//pardiso_iparm[36] = -90;
			mkl_set_num_threads(STACCATO::AuxiliaryParameters::solverMKLThreads);

			pardiso(pardiso_pt, &pardiso_maxfct, &pardiso_mnum, &pardiso_mtype, &pardiso_phase,
				&pardiso_neq, &values[0], &((*rowIndex)[0]), &columns[0], &pardiso_idum,
				&pardiso_nrhs, pardiso_iparm, &pardiso_msglvl, &pardiso_ddum, &pardiso_ddum,
				&pardiso_error);
			if (pardiso_error != 0) {
				errorOut << "Error pardiso factorization failed with error code: " << pardiso_error
					<< std::endl;
				exit(EXIT_FAILURE);
			}
			infoOut << "Reordering and factorization completed" << std::endl;
			infoOut << "Info: Number of equation = " << pardiso_neq << std::endl;
			infoOut << "Info: Number of nonzeros in factors = " << pardiso_iparm[17] << std::endl;
			infoOut << "Info: Number of factorization FLOPS = " << pardiso_iparm[18]*1000000.0 << std::endl;
			infoOut << "Info: Total peak memory on numerical factorization and solution (Mb) = " << (pardiso_iparm[14]+ pardiso_iparm[15]+pardiso_iparm[16])/1000 << std::endl;
			infoOut << "Info: Number of positive eigenvalues = " << pardiso_iparm[21] << std::endl;
			infoOut << "Info: Number of negative eigenvalues = " << pardiso_iparm[22] << std::endl;
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
		 * \return std vector
		 * \author Stefan Sicklinger
		 ***********/
		void solve(T* _x, T* _b) { //Computes x=A\b
#ifdef USE_INTEL_MKL
			// pardiso forward and backward substitution
			pardiso_phase = 33; // forward and backward substitution
			//pardiso_iparm[5] = 0; // write solution to b if true otherwise to x (default)
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
			infoOut << "Forward and backward substitution completed" << std::endl;
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
		 * \brief This prints the matrix in CSR style i j value
		 * \author Stefan Sicklinger
		 ***********/
		void printCSR() {
			row_iter ii;
			col_iter jj;
			size_t ele_row; //elements in current row
			std::cout << std::scientific;

			for (ii = 0; ii < m; ii++) {
				for (jj = (*mat)[ii].begin(); jj != (*mat)[ii].end(); jj++) {
					std::cout << ii << ' ';
					std::cout << (*jj).first << ' ';
					std::cout << (*jj).second << std::endl;
				}
			}
			std::cout << std::endl;
		}
		/***********************************************************************************************
		 * \brief This prints the matrix in full style
		 * \author Stefan Sicklinger
		 ***********/
		void print() {
			size_t ii_counter;
			size_t jj_counter;

			std::ofstream myfile;
			myfile.open("dynStiff.dat");
			myfile.precision(std::numeric_limits<double>::digits10 + 1);
			myfile << std::scientific;
			for (ii_counter = 0; ii_counter < m; ii_counter++) {
				for (jj_counter = 0; jj_counter < n; jj_counter++) {
					if ((*mat)[ii_counter].find(jj_counter) != (*mat)[ii_counter].end()) {
						myfile << ii_counter << "\t" << jj_counter << "\t" << (*mat)[ii_counter].find(jj_counter)->second << std::endl;
					}
					else {
						//myfile << '\t' << 0.0;
					}
				}
				//myfile << std::endl;
			}
			myfile << std::endl;
			myfile.close();
		}
	private:
		/// pointer to the vector of maps
		mat_t* mat;
		/// true if a square matrix should be stored
		bool isSquare;
		/// true if a symmetric matrix should be stored
		bool isSymmetric;
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
		/// determineCSR already called
		bool alreadyCalled;
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
	static const double tmpA = (5 + 3 * sqrt(5.0))/20.0;
	static const double tmpB = (5 - sqrt(5.0)) / 20.0;
	const double tetGaussPoints3D4Points[16] = { 
	tmpA, tmpB, tmpB, tmpB,
	tmpB, tmpA, tmpB, tmpB,
	tmpB, tmpB, tmpA, tmpB,
	tmpB, tmpB, tmpB, tmpA
	};
	const double tetGaussWeights3D4Points = 1.0/4.0;

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
	const double tetGaussWeights3D15Points[15] = {tmpW1 ,tmpW1 ,tmpW1 ,tmpW1, tmpW2 ,tmpW2 ,tmpW2 ,tmpW2, tmpW3 ,tmpW3 ,tmpW3 ,tmpW3, tmpW3, tmpW3, tmpW4};
} /* namespace Math */
#endif /* MATHLIBRARY_H_ */
