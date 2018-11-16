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
#ifdef USE_INTEL_MKL
#define USE_INTEL_MKL_BLAS
#endif

#include "MathLibrary.h"
using namespace std;

namespace MathLibrary {

	double computeDenseDotProduct(const double *vec1, const double *vec2, const int elements) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		return cblas_ddot(elements, vec1, 1, vec2, 1);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}

	void computeDenseDotProductComplex(const STACCATOComplexDouble *vec1, const STACCATOComplexDouble *vec2, STACCATOComplexDouble* dotProduct, const int elements, bool _conjugateVec1) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		if (_conjugateVec1)
			cblas_zdotc_sub(elements, vec1, 1, vec2, 1, dotProduct);
		else
			cblas_zdotu_sub(elements, vec1, 1, vec2, 1, dotProduct);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}

	double computeDenseDotProduct(const std::vector<double> &vec1, const std::vector<double> &vec2) {
#ifdef USE_INTEL_MKL
		return 0;// cblas_ddot(vec1.size(), &vec1[0], 1, &vec2[0], 1);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}


	void copyDenseVector(double *vec1, const double *vec2, const int elements){
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		cblas_dcopy(elements, vec2, 1, vec1, 1);
#endif
	}

	void copyDenseVectorComplex(STACCATOComplexDouble *vec1, const STACCATOComplexDouble *vec2, const int elements) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		cblas_zcopy(elements, vec2, 1, vec1, 1);
#endif
	}

	double computeDenseEuclideanNorm(const double *vec1, const int elements){
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		return cblas_dnrm2 (elements, vec1, 1);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}

	double computeDenseEuclideanNormComplex(const STACCATOComplexDouble *vec1, const int elements) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		return cblas_dznrm2(elements, vec1, 1);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}

	void computeDenseVectorAddition(double *vec1, double *vec2, const double a, const int elements){
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		cblas_daxpy(elements, a, vec1, 1, vec2, 1);
#endif
	}

	void computeDenseVectorAdditionComplex(const STACCATOComplexDouble *vec1, STACCATOComplexDouble *vec2, const STACCATOComplexDouble* a, const int elements) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		cblas_zaxpy(elements, a, vec1, 1, vec2, 1);
#endif
	}

	void computeDenseVectorScalarMultiplication(double *vec1, const double a, const int elements){
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		cblas_dscal (elements, a, vec1, 1);
#endif
	}

	void computeDenseMatrixMatrixMultiplication(int _m, int _n, int _k, const double *_A, const double *_B, double *_C, const bool _transposeA, const bool _multByScalar, const double _alpha, const bool _addPrevious, const bool _useIntelSmall){

#ifdef USE_INTEL_MKL_BLAS
			CBLAS_TRANSPOSE transposeA;
			int ka, nb;
			if (!_transposeA) {
				transposeA = CblasNoTrans;
				ka = _k;
				//	nb = _n;
			}
			else {
				transposeA = CblasTrans;
				ka = _m;
				//	nb = _n;
			}
			double alpha;
			if (!_multByScalar) {
				alpha = 1.0;
			}
			else {
				alpha = _alpha;
			}
			double beta;
			if (!_addPrevious) {
				beta = 0.0;
			}
			else {
				beta = 1.0;
			}
			mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
			cblas_dgemm(CblasRowMajor, transposeA, CblasNoTrans, _m, _n, _k, alpha, _A, ka, _B, _n, beta, _C, _n);
#endif
#ifndef USE_INTEL_MKL_BLAS
			assert(_A != NULL);
			assert(_B != NULL);
			for (int i = 0; i < _m; i++) {
				for (int j = 0; j < _n; j++) {
					double sum = 0.0;
					for (int l = 0; l < _k; l++) {
						if (!_transposeA) {
							sum += _A[i * _k + l] * _B[l * _n + j];
						}
						if (_transposeA) {
							sum += _A[l * _m + i] * _B[l * _n + j];
						}
					}
					if (_multByScalar) {
						sum = _alpha*sum;
					}
					if (_addPrevious) {
						_C[i*_n + j] += sum;
					}
					if (!_addPrevious) {
						_C[i*_n + j] = sum;
					}
				}
			}
#endif
	}

	void computeDenseMatrixMatrixMultiplicationComplex(int _m, int _n, int _k, const STACCATOComplexDouble *_A, const STACCATOComplexDouble *_B, STACCATOComplexDouble *_C, const bool _transposeA, const bool _multByScalar, const STACCATOComplexDouble _alpha, const bool _addPrevious, const bool _useIntelSmall, const bool _rowMajor) {

#ifdef USE_INTEL_MKL_BLAS
		CBLAS_LAYOUT layout = CblasRowMajor;
		CBLAS_TRANSPOSE transposeA;
		int lda, ldb, ldc;
		STACCATOComplexDouble alpha;
		if (!_multByScalar) {
			alpha.real = 1.0;
			alpha.imag = 0.0;
		}
		else {
			alpha = _alpha;
		}
		STACCATOComplexDouble beta;
		if (!_addPrevious) {
			beta.real = 0.0;
			beta.imag = 0.0;
		}
		else {
			beta.real = 1.0;
			beta.imag = 0.0;
		}

		if (_rowMajor && _transposeA) {
			layout = CblasRowMajor;
			transposeA = CblasConjTrans;
			lda = _m;
		}
		else if (_rowMajor && !_transposeA) {
			layout = CblasRowMajor;
			transposeA = CblasNoTrans;
			lda = _k;
		}
		else if (!_rowMajor && _transposeA) {
			layout = CblasColMajor;
			transposeA = CblasConjTrans;
			lda = _k;
		}
		else if (!_rowMajor && !_transposeA) {
			layout = CblasColMajor;
			transposeA = CblasNoTrans;
			lda = _m;
		}

		if (_rowMajor) {
			ldb = _n;
			ldc = _n;
		}
		else {
			ldb = _k;
			ldc = _m;
		}

		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		cblas_zgemm(layout, transposeA, CblasNoTrans, _m, _n, _k, &alpha, _A, lda, _B, ldb, &beta, _C, ldc);
#endif
	}

	
	void computeDenseMatrixVectorMultiplication(int _m, int _n, const double *_A, const double *_b, double *_c){
		assert(_A != NULL);
		assert(_b != NULL);
		for (int i = 0; i < _m; i++){
			double sum = 0.0;
			for (int l = 0; l < _n; l++){
				sum += _A[i * _n + l] * _b[l];
			}
			_c[i] = sum;
		}
	}

	void computeDenseMatrixVectorMultiplicationComplex(int _m, int _n, int _k, const STACCATOComplexDouble *_A, const STACCATOComplexDouble *_x, STACCATOComplexDouble *_y, const bool _transposeA, const bool _multByScalar, const STACCATOComplexDouble _alpha, const bool _addPrevious, const bool _useIntelSmall, const bool _rowMajor) {
#ifdef USE_INTEL_MKL_BLAS
		CBLAS_LAYOUT layout = CblasRowMajor;
		CBLAS_TRANSPOSE transposeA;
		int lda, ldb, ldc;
		STACCATOComplexDouble alpha;
		if (!_multByScalar) {
			alpha.real = 1.0;
			alpha.imag = 0.0;
		}
		else {
			alpha = _alpha;
		}
		STACCATOComplexDouble beta;
		if (!_addPrevious) {
			beta.real = 0.0;
			beta.imag = 0.0;
		}
		else {
			beta.real = 1.0;
			beta.imag = 0.0;
		}

		if (_rowMajor && _transposeA) {
			layout = CblasRowMajor;
			transposeA = CblasConjTrans;
			lda = _m;
		}
		else if (_rowMajor && !_transposeA) {
			layout = CblasRowMajor;
			transposeA = CblasNoTrans;
			lda = _k;
		}
		else if (!_rowMajor && _transposeA) {
			layout = CblasColMajor;
			transposeA = CblasConjTrans;
			lda = _k;
		}
		else if (!_rowMajor && !_transposeA) {
			layout = CblasColMajor;
			transposeA = CblasNoTrans;
			lda = _m;
		}

		/*if (_rowMajor) {
			ldb = _n;
			ldc = _n;
		}
		else {
			ldb = _k;
			ldc = _m;
		}*/

		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		cblas_zgemv(layout, transposeA, _m, _n, &alpha, _A, lda, _x, 1, &beta, _y, 1);

#endif // USE_INTEL_MKL_BLAS

	}


	std::vector<double> computeVectorCrossProduct(std::vector<double> &_v1, std::vector<double> &_v2) {
		std::vector<double> crossProduct(3);
		crossProduct[0] = _v1[1] * _v2[2] - _v2[1] * _v1[2];
		crossProduct[1] = -(_v1[0] * _v2[2] - _v2[0] * _v1[2]);
		crossProduct[2] = _v1[0] * _v2[1] - _v2[0] * _v1[1];
		return crossProduct;
	}

	std::vector<double> solve3x3LinearSystem(std::vector<double>& _A, std::vector<double>& _b, double _EPS) {
		std::vector<double> A(9, 0);
		std::vector<double> b(3, 0);

		double detA = det3x3(_A);
		if (fabs(detA) < _EPS)
			return{};
		for (int i = 0; i < 3; i++)
			b[i] = _b[i];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 9; j++)
				A[j] = _A[j];
			for (int j = 0; j < 3; j++)
				A[j * 3 + i] = b[j];
			_b[i] = det3x3(A) / detA;
		}
		return _b;
	}

	double det3x3(std::vector<double>& _A) {
		return _A[0] * _A[4] * _A[8] + _A[1] * _A[5] * _A[6] + _A[2] * _A[3] * _A[7]
			- _A[0] * _A[5] * _A[7] - _A[1] * _A[3] * _A[8] - _A[2] * _A[4] * _A[6];
	}

	void computeDenseMatrixQRDecomposition(int _m, int _n, double *_A) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		std::vector<double> tau;
		tau.resize(_m < _n ? _m : _n);
		// QR Factorization
		LAPACKE_dgeqrfp(CblasRowMajor, _m, _n, _A, _n, &tau[0]);
		// Generation of Orthogonal Q
		LAPACKE_dorgqr(CblasRowMajor, _m, _n, tau.size(), _A, _n, &tau[0]);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}

	void computeDenseMatrixQRDecompositionComplex(int _m, int _n, STACCATOComplexDouble *_A, bool _rowMajor) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		int lda;
		 int layout;
		if (_rowMajor) {
			lda = _n;
			layout = LAPACK_ROW_MAJOR;
		}
		else {
			lda = _m;
			layout = LAPACK_COL_MAJOR;
		}
		std::vector<STACCATOComplexDouble> tau;
		tau.resize(_m < _n ? _m : _n);
		// QR Factorization
		LAPACKE_zgeqrfp(layout, _m, _n, _A, lda, &tau[0]);
		// Generation of Orthogonal Q
		LAPACKE_zungqr(layout, _m, _n, tau.size(), _A, lda, &tau[0]);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}
	
	void computeDenseMatrixQR_Q_DecompositionComplex(int _m, int _n, STACCATOComplexDouble *_A, bool _rowMajor, std::vector<STACCATOComplexDouble>& _tau) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		int lda;
		int layout;
		if (_rowMajor) {
			lda = _n;
			layout = LAPACK_ROW_MAJOR;
		}
		else {
			lda = _m;
			layout = LAPACK_COL_MAJOR;
		}
		// Generation of Orthogonal Q
		LAPACKE_zungqr(layout, _m, _n, _tau.size(), _A, lda, &_tau[0]);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}

	void computeDenseMatrixPivotedQR_R_DecompositionComplex(int _m, int _n, STACCATOComplexDouble *_A, bool _rowMajor, std::vector<STACCATOComplexDouble>& _tau) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		int lda;
		int layout;
		std::vector<int> jpvt(_n);
		if (_rowMajor) {
			lda = _n;
			layout = LAPACK_ROW_MAJOR;
		}
		else {
			lda = _m;
			layout = LAPACK_COL_MAJOR;
		}
		//std::vector<STACCATOComplexDouble> tau;
		_tau.resize(_m < _n ? _m : _n);
		// QR Factorization
		LAPACKE_zgeqp3(layout, _m, _n, _A, lda, &jpvt[0], &_tau[0]);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}
	
	void computeSparseMatrixAdditionComplex(const sparse_matrix_t* _matA, const sparse_matrix_t* _matB, sparse_matrix_t* _matC, const bool _conjugatetransposeA, const bool _multByScalar, STACCATOComplexDouble _alpha) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		sparse_operation_t operationA;
		if (!_conjugatetransposeA)
			operationA = SPARSE_OPERATION_NON_TRANSPOSE;
		else
			operationA = SPARSE_OPERATION_CONJUGATE_TRANSPOSE;

		if (!_multByScalar){
			_alpha.real = 1;
			_alpha.imag = 0;
		}

		sparse_status_t status = mkl_sparse_z_add(operationA, *_matA, _alpha, *_matB, _matC);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}
	void computeSparseMatrixMultiplicationComplex(const sparse_matrix_t* _matA, const sparse_matrix_t* _matB, sparse_matrix_t* _matC, bool _conjugatetransposeA, bool _conjugatetransposeB, bool _symmetricA, bool _symmetricB) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		matrix_descr descrA;
		sparse_operation_t operationA;
		if (!_conjugatetransposeA)
			operationA = SPARSE_OPERATION_NON_TRANSPOSE;
		else
			operationA = SPARSE_OPERATION_CONJUGATE_TRANSPOSE;

		if (_symmetricA)
		{
			descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
			descrA.mode = SPARSE_FILL_MODE_UPPER;
			descrA.diag = SPARSE_DIAG_NON_UNIT;
		}
		else
			descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

		matrix_descr descrB;
		sparse_operation_t operationB;
		if (!_conjugatetransposeB)
			operationB = SPARSE_OPERATION_NON_TRANSPOSE;
		else
			operationB = SPARSE_OPERATION_CONJUGATE_TRANSPOSE;

		if (_symmetricA)
		{
			descrB.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
			descrB.mode = SPARSE_FILL_MODE_UPPER;
			descrB.diag = SPARSE_DIAG_NON_UNIT;
		}
		else
			descrB.type = SPARSE_MATRIX_TYPE_GENERAL;


		// Single stage operation
		sparse_status_t status = mkl_sparse_sp2m(operationA, descrA, *_matA, operationB, descrB, *_matB, SPARSE_STAGE_FULL_MULT, _matC);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}

	void computeSparseMatrixDenseMatrixMultiplicationComplex(int _col, int _ldx, int _ldy, const sparse_matrix_t* _matA, const STACCATOComplexDouble *_matX, STACCATOComplexDouble *_matY, const bool _conjugatetransposeA, const bool _multByScalar, STACCATOComplexDouble _alpha, const bool _symmetricA, const bool _addPrevious) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		matrix_descr descrA;
		sparse_operation_t operationA;
		int kY;
		if (!_conjugatetransposeA) 
			operationA = SPARSE_OPERATION_NON_TRANSPOSE;
		else 
			operationA = SPARSE_OPERATION_CONJUGATE_TRANSPOSE; 

		if (_symmetricA)
		{
			descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
			descrA.mode = SPARSE_FILL_MODE_UPPER;
			descrA.diag = SPARSE_DIAG_NON_UNIT;
		}
		else
			descrA.type = SPARSE_MATRIX_TYPE_GENERAL;


		if (!_multByScalar) {
			_alpha.real = 1;
			_alpha.imag = 0;
		}

		STACCATOComplexDouble beta;
		if (!_addPrevious) {
			beta.real = 0.0;
			beta.imag = 0.0;
		}
		else {
			beta.real = 1.0;
			beta.imag = 1.0;
		}

		sparse_status_t status = mkl_sparse_z_mm(operationA, _alpha, *_matA, descrA, SPARSE_LAYOUT_COLUMN_MAJOR, _matX, _col, _ldx, beta, _matY, _ldy);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}

	void printToScreen_csr_sparse_z(sparse_matrix_t* _mat)
	{
#ifdef USE_INTEL_MKL
		// Read Matrix Data and Print it
		int row, col;
		sparse_index_base_t indextype;
		int * bi, *ei;
		int * j;
		MKL_Complex16* rv;
		sparse_status_t status = mkl_sparse_z_export_csr(*_mat, &indextype, &row, &col, &bi, &ei, &j, &rv);
		if (status == SPARSE_STATUS_SUCCESS)
		{
			printf("SparseMatrix(%d x %d) [base:%d]\n", row, col, indextype);
			std::cout << "nnz: " << ei[row-1] - 1 << std::endl;
			for (int r = 0; r < row; ++r)
			{
				for (int idx = bi[r]; idx < ei[r]; ++idx)
				{
					printf("<%d, %d> \t %.9E +1i*%.9E\n", r + 1, j[idx - 1], rv[idx - 1].real, rv[idx - 1].imag);
				}
			}
		}
#endif
	}

	void printToFile_csr_sparse_z(sparse_matrix_t* _mat)
	{
#ifdef USE_INTEL_MKL
		// Read Matrix Data and Print it
		int row, col;
		sparse_index_base_t indextype;
		int * bi, *ei;
		int * j;
		MKL_Complex16* rv;
		sparse_status_t status = mkl_sparse_z_export_csr(*_mat, &indextype, &row, &col, &bi, &ei, &j, &rv);
		std::vector<STACCATOComplexDouble> values;
		std::vector<int> jaCSRR;
		if (status == SPARSE_STATUS_SUCCESS)
		{
			printf("SparseMatrix(%d x %d) [base:%d]\n", row, col, indextype);
			std::cout << "nnz: " << ei[row - 1] - 1 << std::endl;
			for (int r = 0; r < row; ++r)
			{
				for (int idx = bi[r]; idx < ei[r]; ++idx)
				{
					values.push_back(rv[idx - 1]);
					jaCSRR.push_back(j[idx - 1]);
				}
			}
		}
		std::vector<int> testpB;
		std::vector<int> testpE;
		for (int i = 0; i < row; i++)
		{
			testpB.push_back(bi[i]);
			testpE.push_back(ei[i + 1]);
		}
		testpB.push_back(ei[row - 1]);


		std::string exportFilePrefix = "Staccato_Sparse_Export";
		std::cout << ">> Printing to " << exportFilePrefix << " Matrix CSR Format..." << std::endl;
		AuxiliaryFunctions::writeIntegerVectorDatFormat(exportFilePrefix + "_CSR_IA.dat", testpB);
		AuxiliaryFunctions::writeIntegerVectorDatFormat(exportFilePrefix + "_CSR_JA.dat", jaCSRR);

		AuxiliaryFunctions::writeMKLComplexVectorDatFormat(exportFilePrefix + "_CSR_MAT.dat", values);
#endif
	}

	void createSparseCSRComplex(sparse_matrix_t* _mat, int _m, int _n, int* _pointerB, int* _pointerE, int* _columns, STACCATOComplexDouble* _entries) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		sparse_status_t status = mkl_sparse_z_create_csr(_mat, SPARSE_INDEX_BASE_ONE, _m, _n, _pointerB, _pointerE, _columns, _entries);
#endif
	}

	double computeDenseMatrixFrobeniusNormComplex(const STACCATOComplexDouble *vec1, const int _m, const int _n) {
#ifdef USE_INTEL_MKL
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		return LAPACKE_zlange(CblasRowMajor, 'E', _m, _n, vec1, _n);
#endif // USE_INTEL_MKL

	}
} /* namespace Math */

	