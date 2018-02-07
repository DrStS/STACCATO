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
		return 0;// cblas_ddot(elements, vec1, 1, vec2, 1);
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
} /* namespace Math */