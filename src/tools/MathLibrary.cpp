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
		return cblas_ddot(elements, vec1, 1, vec2, 1);
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
		cblas_dcopy(elements, vec2, 1, vec1, 1);
#endif
	}

	double computeDenseEuclideanNorm(const double *vec1, const int elements){
#ifdef USE_INTEL_MKL
		return cblas_dnrm2 (elements, vec1, 1);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}

	void computeDenseVectorAddition(double *vec1, double *vec2, const double a, const int elements){
#ifdef USE_INTEL_MKL
		cblas_daxpy(elements, a, vec1, 1, vec2, 1);
#endif
	}

	void computeDenseVectorScalarMultiplication(double *vec1, const double a, const int elements){
#ifdef USE_INTEL_MKL
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

	void computeDenseMatrixMatrixMultiplicationComplex(int _m, int _n, int _k, const STACCATOComplexDouble *_A, const STACCATOComplexDouble *_B, STACCATOComplexDouble *_C, const bool _transposeA, const bool _multByScalar, const STACCATOComplexDouble _alpha, const bool _addPrevious, const bool _useIntelSmall) {

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
		STACCATOComplexDouble alpha;
		if (!_multByScalar) {
			alpha.real = 1.0;
			alpha.imag = 1.0;
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
			beta.imag = 1.0;
		}
		mkl_set_num_threads(STACCATO::AuxiliaryParameters::denseVectorMatrixThreads);
		cblas_zgemm(CblasRowMajor, transposeA, CblasNoTrans, _m, _n, _k, &alpha, _A, ka, _B, _n, &beta, _C, _n);

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
		std::vector<double> tau;
		tau.resize(_m < _n ? _m : _n);
		// QR Factorization
		LAPACKE_dgeqrf(CblasRowMajor, _m, _n, _A, _n, &tau[0]);
		// Generation of Orthogonal Q
		LAPACKE_dorgqr(CblasRowMajor, _m, _n, tau.size(), _A, _n, &tau[0]);
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}
	void computeSparseMatrixAddition(MathLibrary::SparseMatrix<MKL_Complex16>* _mat1, MathLibrary::SparseMatrix<MKL_Complex16>* _mat2) {
#ifdef USE_INTEL_MKL
		const sparse_matrix_t* sparsemat1 = _mat1->createMKLSparseCSR();
		const sparse_matrix_t* sparsemat2 = _mat2->createMKLSparseCSR();

		MKL_Complex16 alpha;
		alpha.real = 1;
		alpha.imag = 0;
		sparse_matrix_t* sparseSum;
		mkl_sparse_z_add(SPARSE_OPERATION_NON_TRANSPOSE, *sparsemat1, alpha, *sparsemat2, sparseSum);
		/*
		// Read Matrix Data and Print it
			int row, col;
			sparse_index_base_t indextype;
			int * bi, *ei;
			int * j;
			MKL_Complex16* rv;
			sparse_status_t status = mkl_sparse_z_export_csr(csrA, &indextype, &row, &col, &bi, &ei, &j, &rv);
			if (status == SPARSE_STATUS_SUCCESS)
			{
				printf("SparseMatrix(%d x %d) [base:%d]\n", row, col, indextype);
				for (int r = 0; r<row; ++r)
				{
					for (int idx = bi[r]; idx<ei[r]; ++idx)
					{
						printf("<%d, %d> \t %f +1i*%f\n", r, j[idx], rv[idx].real, rv[idx].imag);
					}
				}
			}*/
#endif
#ifndef USE_INTEL_MKL
		return 0;
#endif
	}
} /* namespace Math */