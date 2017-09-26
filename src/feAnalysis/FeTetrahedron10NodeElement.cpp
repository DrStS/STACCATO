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
#include "FeTetrahedron10NodeElement.h"
#include "Material.h"
#include "Message.h"
#include "MathLibrary.h"


FeTetrahedron10NodeElement::FeTetrahedron10NodeElement(Material *_material) : FeElement(_material) {
	myKe.resize(900);
	myMe.resize(900);
}

FeTetrahedron10NodeElement::~FeTetrahedron10NodeElement() {
}

void FeTetrahedron10NodeElement::computeElementMatrix(const double* _eleCoords){
	//Allocate local memory on the stack
	double   N[10];
	double dNx[10];
	double dNy[10];
	double dNz[10];
	double Jdet;
	double B[180] = {0};
	double B_T_times_Emat[180] = {0};

	double ni = 1./3.;// myMaterial->getPoissonsRatio();
	double E = 480.0;// myMaterial->getYoungsModulus();
	double tmp = E / ((1 + ni)*(1 - 2*ni));
	double Emat[36] = { 
	tmp*(1-ni), tmp*ni,   tmp*ni, 0. , 0. , 0.,	
	tmp*ni, tmp*(1 - ni), tmp*ni, 0. , 0. , 0.,
	tmp*ni, tmp*ni,   tmp*(1-ni), 0. , 0. , 0.,
	0.,         0.,           0., tmp*(0.5 - ni) , 0. , 0.,
	0.,         0.,           0., 0. , tmp*(0.5 - ni) , 0.,
	0.,         0.,           0., 0. , 0. ,tmp*(0.5 - ni) };

	//Gauss integration loop
	for (int k = 0; k < 4; k++) {

		evalTet10IsoPShapeFunDer(_eleCoords, MathLibrary::tetGaussPoints3D4Points[(4 * k) + 0], MathLibrary::tetGaussPoints3D4Points[(4 * k) + 1], MathLibrary::tetGaussPoints3D4Points[(4 * k) + 2], MathLibrary::tetGaussPoints3D4Points[(4 * k) + 3], N, dNx, dNy, dNz, Jdet);
		//Compute element stiffness matrix
		//B matrix 6 x 30
		for (int i = 0; i < 10; i++) {
			B[((3 * i) + 0)+0] = dNx[i] ;
			B[((3 * i) + 1)+30] = dNy[i] ;
			B[((3 * i) + 2)+60] = dNz[i] ;

			B[((3 * i) + 0) + 90] = dNy[i] ;
			B[((3 * i) + 1) + 90] = dNx[i] ;

			B[((3 * i) + 1) + 120] = dNz[i] ;
			B[((3 * i) + 2) + 120] = dNy[i] ;

			B[((3 * i) + 0) + 150] = dNz[i] ;
			B[((3 * i) + 2) + 150] = dNx[i] ;
		}
		//Compute Ke=+det(J)*Wi*transpose(B)*Emat*B;

		MathLibrary::computeDenseMatrixMatrixMultiplication(30, 6, 6, B, Emat, B_T_times_Emat, true, false, 1.0, false,false);
		MathLibrary::computeDenseMatrixMatrixMultiplication(30, 30, 6, B_T_times_Emat, B, &myKe[0], false, true, MathLibrary::tetGaussWeights3D4Points/(6.0*Jdet), true,false);
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
		//MathLibrary::computeDenseMatrixMatrixMultiplication(8, 8, 2, B, B, &myMe[0], true, true, Jdet*rho*t, true,false);
		//Compute viscous damping matrix ...
	}

	//Lumping
	double b[8]={ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	double c[8];
	//MathLibrary::computeDenseMatrixVectorMultiplication(8, 8, &myMe[0], b, c);
	//std::fill(myMe.begin(), myMe.end(), 0.0);
	//for (int i = 0; i < 8; i++) {
	//	myMe[i * 8 + i] = c[i];
	//}


}

void FeTetrahedron10NodeElement::evalTet10IsoPShapeFunDer(const double* _eleCoords, const double _xi1, const double _xi2,  const double _xi3, const double _xi4, double *_N, double *_dNx, double *_dNy, double *_dNz, double &_Jdet)
{
	/*    _eleCoords[0]  -> x1
		_eleCoords[1]  -> y1
		_eleCoords[2]  -> z1
		_eleCoords[3]  -> x2
		_eleCoords[4]  -> y2
		_eleCoords[5]  -> z2
		_eleCoords[6]  -> x3
		_eleCoords[7]  -> y3
		_eleCoords[8]  -> z3
		_eleCoords[9]  -> x4
		_eleCoords[10] -> y4
		_eleCoords[11] -> z4
		_eleCoords[12] -> x5
		_eleCoords[13] -> y5
		_eleCoords[14] -> z5
		_eleCoords[15] -> x6
		_eleCoords[16] -> y6
		_eleCoords[17] -> z6
		_eleCoords[18] -> x7
		_eleCoords[19] -> y7
		_eleCoords[20] -> z7
		_eleCoords[21] -> x8
		_eleCoords[22] -> y8
		_eleCoords[23] -> z8
		_eleCoords[24] -> x9
		_eleCoords[25] -> y9
		_eleCoords[26] -> z9
		_eleCoords[27] -> x10 
		_eleCoords[28] -> y10 
		_eleCoords[29] -> z10*/


	double Jx1 = 4 * (_eleCoords[0]*(_xi1 - 1 / 4) + _eleCoords[12] *_xi2 + _eleCoords[18] *_xi3 + _eleCoords[21] *_xi4);	
	double Jy1 = 4 * (_eleCoords[1]*(_xi1 - 1 / 4) + _eleCoords[13] *_xi2 + _eleCoords[19] *_xi3 + _eleCoords[22] *_xi4);
	double Jz1 = 4 * (_eleCoords[2] *(_xi1 - 1 / 4) + _eleCoords[14] * _xi2 + _eleCoords[20] *_xi3 + _eleCoords[23] *_xi4);
	double Jx2 = 4 * (_eleCoords[12] *_xi1 + _eleCoords[3] *(_xi2 - 1 / 4) + _eleCoords[15] *_xi3 + _eleCoords[24] *_xi4);
	double Jy2 = 4 * (_eleCoords[13] *_xi1 + _eleCoords[4] *(_xi2 - 1 / 4) + _eleCoords[16] *_xi3 + _eleCoords[25] *_xi4);
	double Jz2 = 4 * (_eleCoords[14] *_xi1 + _eleCoords[5] *(_xi2 - 1 / 4) + _eleCoords[17] *_xi3 + _eleCoords[26] *_xi4);
	double Jx3 = 4 * (_eleCoords[18] *_xi1 + _eleCoords[15] *_xi2 + _eleCoords[6] *(_xi3 - 1 / 4) + _eleCoords[27] *_xi4);
	double Jy3 = 4 * (_eleCoords[19] *_xi1 + _eleCoords[16] *_xi2 + _eleCoords[7] *(_xi3 - 1 / 4) + _eleCoords[28] *_xi4);
	double Jz3 = 4 * (_eleCoords[20] *_xi1 + _eleCoords[17] *_xi2 + _eleCoords[8] *(_xi3 - 1 / 4) + _eleCoords[29] *_xi4);
	double Jx4 = 4 * (_eleCoords[21] *_xi1 + _eleCoords[24] *_xi2 + _eleCoords[27] *_xi3 + _eleCoords[9] *(_xi4 - 1 / 4));
	double Jy4 = 4 * (_eleCoords[22] *_xi1 + _eleCoords[25] *_xi2 + _eleCoords[28] *_xi3 + _eleCoords[10] *(_xi4 - 1 / 4));
	double Jz4 = 4 * (_eleCoords[23] *_xi1 + _eleCoords[26] *_xi2 + _eleCoords[29] *_xi3 + _eleCoords[11] *(_xi4 - 1 / 4));
	
	
	double Jx12 = Jx1 - Jx2; 
	double Jx13 = Jx1 - Jx3; 
	double Jx14 = Jx1 - Jx4; 
	double Jx23 = Jx2 - Jx3;
	double Jx24 = Jx2 - Jx4; 
	double Jx34 = Jx3 - Jx4; 
	double Jy12 = Jy1 - Jy2; 
	double Jy13 = Jy1 - Jy3;
	double Jy14 = Jy1 - Jy4; 
	double Jy23 = Jy2 - Jy3; 
	double Jy24 = Jy2 - Jy4; 
	double Jy34 = Jy3 - Jy4;
	double Jz12 = Jz1 - Jz2; 
	double Jz13 = Jz1 - Jz3; 
	double Jz14 = Jz1 - Jz4;
	double Jz23 = Jz2 - Jz3; 
	double Jz24 = Jz2 - Jz4; 
	double Jz34 = Jz3 - Jz4;
	double Jx21 = -Jx12; 
	double Jx31 = -Jx13; 
	double Jx41 = -Jx14; 
	double Jx32 = -Jx23; 
	double Jx42 = -Jx24;
	double Jx43 = -Jx34; 
	double Jy21 = -Jy12;
	double Jy31 = -Jy13;
	double Jy41 = -Jy14; 
	double Jy32 = -Jy23;
	double Jy42 = -Jy24; 
	double Jy43 = -Jy34; 
	double Jz21 = -Jz12; 
	double Jz31 = -Jz13; 
	double Jz41 = -Jz14;
	double Jz32 = -Jz23; 
	double Jz42 = -Jz24; 
	double Jz43 = -Jz34;
	_Jdet = Jx21*(Jy23*Jz34 - Jy34*Jz23) + Jx32*(Jy34*Jz12 - Jy12*Jz34) + Jx43*(Jy12*Jz23 - Jy23*Jz12);
	double a1 = Jy42*Jz32 - Jy32*Jz42;
	double a2 = Jy31*Jz43 - Jy34*Jz13;
	double a3 = Jy24*Jz14 - Jy14*Jz24;
	double a4 = Jy13*Jz21 - Jy12*Jz31;
	double b1 = Jx32*Jz42 - Jx42*Jz32; 
	double b2 = Jx43*Jz31 - Jx13*Jz34;
	double b3 = Jx14*Jz24 - Jx24*Jz14; 
	double b4 = Jx21*Jz13 - Jx31*Jz12;
	double c1 = Jx42*Jy32 - Jx32*Jy42; 
	double c2 = Jx31*Jy43 - Jx34*Jy13;
	double c3 = Jx24*Jy14 - Jx14*Jy24; 
	double c4 = Jx13*Jy21 - Jx12*Jy31;

	_N[0] = _xi1*(2*_xi1-1);
	_N[1] = _xi2*(2*_xi2-1);
	_N[2] = _xi3*(2 * _xi3 - 1);
	_N[3] = _xi4*(2 * _xi4 - 1);
	_N[4] = 4*_xi1*_xi2;
	_N[5] = 4 * _xi2*_xi3;
	_N[6] = 4 * _xi3*_xi1;
	_N[7] = 4 * _xi1*_xi4;
	_N[8] = 4 * _xi2*_xi4;
	_N[9] = 4 * _xi3*_xi4;
	_dNx[0] = (4 * _xi1 - 1)*a1;
	_dNx[1] = (4 * _xi2 - 1)*a2;
    _dNx[2] = (4 * _xi3 - 1)*a3;
    _dNx[3] = (4 * _xi4 - 1)*a4;
	_dNx[4] = 4 * (_xi1*a2 + _xi2*a1);
	_dNx[5] = 4 * (_xi2*a3 + _xi3*a2);
	_dNx[6] = 4 * (_xi3*a1 + _xi1*a3);
	_dNx[7] = 4 * (_xi1*a4 + _xi4*a1);
	_dNx[8] = 4 * (_xi2*a4 + _xi4*a2);
	_dNx[9] = 4 * (_xi3*a4 + _xi4*a3);
	_dNy[0] = (4 * _xi1 - 1)*b1;
	_dNy[1] = (4 * _xi2 - 1)*b2;
	_dNy[2] = (4 * _xi3 - 1)*b3;
	_dNy[3] = (4 * _xi4 - 1)*b4;
	_dNy[4] = 4 * (_xi1*b2 + _xi2*b1);
	_dNy[5] = 4 * (_xi2*b3 + _xi3*b2);
	_dNy[6] = 4 * (_xi3*b1 + _xi1*b3);
	_dNy[7] = 4 * (_xi1*b4 + _xi4*b1);
	_dNy[8] = 4 * (_xi2*b4 + _xi4*b2);
	_dNy[9] = 4 * (_xi3*b4 + _xi4*b3);
	_dNz[0] = (4 * _xi1 - 1)*c1;
	_dNz[1] = (4 * _xi2 - 1)*c2;
	_dNz[2] = (4 * _xi3 - 1)*c3;
	_dNz[3] = (4 * _xi4 - 1)*c4;
	_dNz[4] = 4 * (_xi1*c2 + _xi2*c1);
	_dNz[5] = 4 * (_xi2*c3 + _xi3*c2);
	_dNz[6] = 4 * (_xi3*c1 + _xi1*c3);
	_dNz[7] = 4 * (_xi1*c4 + _xi4*c1);
	_dNz[8] = 4 * (_xi2*c4 + _xi4*c2);
	_dNz[9] = 4 * (_xi3*c4 + _xi4*c3);
}

