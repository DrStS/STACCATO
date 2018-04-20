/*  Copyright &copy; 2018, Dr. Stefan Sicklinger, Munich \n
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

#include <AuxiliaryFunctions.h>
#include <fstream>
#include <iostream>

AuxiliaryFunctions::AuxiliaryFunctions()
{
}

AuxiliaryFunctions::~AuxiliaryFunctions()
{
}

void AuxiliaryFunctions::writeDoubleVector(std::string _fileName, std::vector<double> &_vector) {
	std::cout << ">> Writing " << _fileName << "#" << _vector.size() <<"..." << std::endl;
	size_t ii_couter;
	std::ofstream myfile;
	myfile.open(_fileName);
	myfile.precision(std::numeric_limits<double>::digits10 + 1);
	myfile << std::scientific;
	for (ii_couter = 0; ii_couter < _vector.size(); ii_couter++)
	{
		myfile << _vector[ii_couter] << std::endl;
	}
	myfile << std::endl;
	myfile.close();

}

void AuxiliaryFunctions::writeIntegerVector(std::string _fileName, std::vector<int> &_vector) {
	std::cout << ">> Writing " << _fileName << "#" << _vector.size() << "..." << std::endl;
	size_t ii_couter;
	std::ofstream myfile;
	myfile.open(_fileName);
	myfile << std::scientific;
	for (ii_couter = 0; ii_couter < _vector.size(); ii_couter++)
	{
		myfile << _vector[ii_couter] << std::endl;
	}
	myfile << std::endl;
	myfile.close();
}

void AuxiliaryFunctions::writeMKLComplexVector(std::string _fileName, std::vector<MKL_Complex16> &_vector) {
	std::cout << ">> Writing " << _fileName << "#C:" << _vector.size() << "..." << std::endl;
	size_t ii_couter;
	std::ofstream myfile;
	myfile.open(_fileName);
	myfile.precision(std::numeric_limits<double>::digits10 + 1);
	myfile << std::scientific;
	for (ii_couter = 0; ii_couter < _vector.size(); ii_couter++)
	{
		myfile << _vector[ii_couter].real << "\t"<< _vector[ii_couter].imag << std::endl;
	}
	myfile << std::endl;
	myfile.close();
}