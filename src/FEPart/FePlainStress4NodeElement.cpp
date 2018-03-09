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
#include "FePlainStress4NodeElement.h"
#include "Material.h"
#include "Message.h"
#include "MathLibrary.h"


FePlainStress4NodeElement::FePlainStress4NodeElement(Material *_material) : FeElement(_material) {
	myKe.resize(64);
	myMe.resize(64);
}

FePlainStress4NodeElement::~FePlainStress4NodeElement() {
}

void FePlainStress4NodeElement::computeElementMatrix(const double* _eleCoords){
	//Allocate local memory on the stack
	double N[4];
	double dNx[4];
	double dNy[4];
	double Jdet;
	double B[24] = {0};
	double B_T_times_Emat[24] = {0};

	double ni = myMaterial->getPoissonsRatio();
	double E = myMaterial->getYoungsModulus();
	double tmp = E / (1 - ni*ni);
	double Emat[9] = { tmp, tmp*ni, 0, tmp*ni, tmp, 0, 0, 0, tmp*0.5*(1 - ni) };

	//Gauss integration loop
	for (int k = 0; k < 4; k++) {

		evalQuad4IsoPShapeFunDer(_eleCoords, MathLibrary::quadGaussPoints2D4Point[2 * k], MathLibrary::quadGaussPoints2D4Point[(2 * k) + 1], N, dNx, dNy, Jdet);
		//Compute element stiffness matrix
		//B matrix 3 x 8
		for (int i = 0; i < 4; i++) {
			B[2 * i]              = dNx[i];
			B[((2 * i)+1)+8]      = dNy[i];
			B[((2 * i)) + 16]     = dNy[i];
			B[((2 * i) + 1) + 16] = dNx[i];
		}
		//Weights are 1 and thickness is assumed to be 1
		double t = 1.0;
		//Compute Ke=+det(J)*t*Wi*Wj*transpose(B)*Emat*B;
		MathLibrary::computeDenseMatrixMatrixMultiplication(8, 3, 3, B, Emat, B_T_times_Emat, true, false, 1.0, false,false);
		MathLibrary::computeDenseMatrixMatrixMultiplication(8, 8, 3, B_T_times_Emat, B, &myKe[0], false, true, Jdet*t, true,false);
		//Compute mass matrix
		double rho = myMaterial->getDensity();
		memset(B, 0, 24*sizeof(double));
		//N matrix 2 x 8
		for (int i = 0; i < 4; i++) {
			B[2 * i] = N[i];
			B[((2 * i) + 1) + 8] = N[i];
		}
		//Weights are 1 and thickness is assumed to be 1
		//Compute Me=+det(J)*t*Wi*Wj*rho*transpose(N)*N;
		MathLibrary::computeDenseMatrixMatrixMultiplication(8, 8, 2, B, B, &myMe[0], true, true, Jdet*rho*t, true,false);
		//Compute viscous damping matrix ...
	}

	//Lumping
	double b[8]={ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	double c[8];
	MathLibrary::computeDenseMatrixVectorMultiplication(8, 8, &myMe[0], b, c);
	std::fill(myMe.begin(), myMe.end(), 0.0);
	for (int i = 0; i < 8; i++) {
		myMe[i * 8 + i] = c[i];
	}


}

void FePlainStress4NodeElement::evalQuad4IsoPShapeFunDer(const double* _eleCoords, const double _xi, const double _eta, double *_N, double *_dNx, double *_dNy, double &_Jdet)
{
	_N[0] = (1 - _xi)*(1 - _eta) / 4;
	_N[1] = (1 + _xi)*(1 - _eta) / 4;
	_N[2] = (1 + _xi)*(1 + _eta) / 4;
	_N[3] = (1 - _xi)*(1 + _eta) / 4;
	double x21 = _eleCoords[2] - _eleCoords[0];
	double x31 = _eleCoords[4] - _eleCoords[0];
	double x41 = _eleCoords[6] - _eleCoords[0];
	double x32 = _eleCoords[4] - _eleCoords[2];
	double x42 = _eleCoords[6] - _eleCoords[2];
	double x43 = _eleCoords[6] - _eleCoords[4];
	double y21 = _eleCoords[3] - _eleCoords[1];
	double y31 = _eleCoords[5] - _eleCoords[1];
	double y41 = _eleCoords[7] - _eleCoords[1];
	double y32 = _eleCoords[5] - _eleCoords[3];
	double y42 = _eleCoords[7] - _eleCoords[3];
	double y43 = _eleCoords[7] - _eleCoords[5];

	double Jdet8 = -y31*x42 + x31*y42 + (y21*x43 - x21*y43)*_xi + (x32*y41 - x41*y32)*_eta;
	_Jdet = Jdet8 / 8;
	_dNx[0] = (-y42 + y43*_xi + y32*_eta) / Jdet8;
	_dNx[1] = (y31 - y43*_xi - y41*_eta) / Jdet8;
	_dNx[2] = (y42 - y21*_xi + y41*_eta) / Jdet8;
	_dNx[3] = (-y31 + y21*_xi - y32*_eta) / Jdet8;
	_dNy[0] = (x42 - x43*_xi - x32*_eta) / Jdet8;
	_dNy[1] = (-x31 + x43*_xi + x41*_eta) / Jdet8;
	_dNy[2] = (-x42 + x21*_xi - x41*_eta) / Jdet8;
	_dNy[3] = (x31 - x21*_xi + x32*_eta) / Jdet8;
}

