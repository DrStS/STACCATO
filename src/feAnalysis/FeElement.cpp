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

void FeElement::evalQuad4IsoPShapeFunDer(const double* eleCoords, const double xi, const double eta, double *N, double *dNx, double *dNy, double &Jdet)
{
	N[0] = (1 - xi)*(1 - eta) / 4;
	N[1] = (1 + xi)*(1 - eta) / 4;
	N[2] = (1 + xi)*(1 + eta) / 4;
	N[3] = (1 - xi)*(1 + eta) / 4;
	double x21 = eleCoords[1] - eleCoords[0];
	double x31 = eleCoords[2] - eleCoords[0];
	double x41 = eleCoords[3] - eleCoords[0];
	double x32 = eleCoords[2] - eleCoords[1];
	double x42 = eleCoords[3] - eleCoords[1];
	double x43 = eleCoords[3] - eleCoords[2];
	double y21 = eleCoords[5] - eleCoords[4];
	double y31 = eleCoords[6] - eleCoords[4];
	double y41 = eleCoords[7] - eleCoords[4];
	double y32 = eleCoords[6] - eleCoords[5];
	double y42 = eleCoords[7] - eleCoords[5];
	double y43 = eleCoords[7] - eleCoords[6];
	double Jdet8 = -y31*x42 + x31*y42 + (y21*x43 - x21*y43)*xi + (x32*y41 - x41*y32)*eta;
	Jdet = Jdet8 / 8;
	dNx[0] = (-y42 + y43*xi + y32*eta) / Jdet8;
	dNx[1] = (y31 - y43*xi - y41*eta) / Jdet8;
	dNx[2] = (y42 - y21*xi + y41*eta) / Jdet8;
	dNx[3] = (-y31 + y21*xi - y32*eta) / Jdet8;
	dNy[0] = (x42 - x43*xi - x32*eta) / Jdet8;
	dNy[1] = (-x31 + x43*xi + x41*eta) / Jdet8;
	dNy[2] = (-x42 + x21*xi - x41*eta) / Jdet8;
	dNy[3] = (x31 - x21*xi + x32*eta) / Jdet8;
}

