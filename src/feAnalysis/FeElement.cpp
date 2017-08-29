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
#include "FeElement.h"
#include "Message.h"
#include "MathLibrary.h"


FeElement::FeElement() {
}

FeElement::~FeElement() {
}

void FeElement::computeElementStiffness(const double* _eleCoords, const double *_Emat, double *_Ke){
	static const double tmpSqrt13 = sqrt(1.0 / 3.0);
	const double quadGaussPoints2DBiLinear[8] = { tmpSqrt13, tmpSqrt13, -tmpSqrt13, tmpSqrt13, -tmpSqrt13,- tmpSqrt13, tmpSqrt13, -tmpSqrt13 };

	// allocate local memory on the stack
	double N[4];
	double dNx[4];
	double dNy[4];
	double Jdet;
	double B[24] = { 0 };

	for (int k = 0; k < 8; k++) {
		evalQuad4IsoPShapeFunDer(_eleCoords, quadGaussPoints2DBiLinear[2 * k], quadGaussPoints2DBiLinear[(2 * k) + 1], N, dNx, dNy, Jdet);

		for (int i = 0; i < 4; i++) {
			B[2 * i]     = dNx[i];
			B[((2 * i)+1)+8] = dNy[i];
			B[((2 * i)) + 16] = dNy[i];
			B[((2 * i) + 1) + 16] = dNx[i];
		}
	}

}

void FeElement::evalQuad4IsoPShapeFunDer(const double* _eleCoords, const double _xi, const double _eta, double *_N, double *_dNx, double *_dNy, double &_Jdet)
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

