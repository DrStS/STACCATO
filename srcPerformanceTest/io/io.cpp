// Libraries
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>
#include <limits>
#include <iomanip>
#include <vector>

// MKL
#include <mkl.h>

// Header files
#include "io.hpp"

// Reads dense Mtx file
void io::readMtxDense(std::vector<MKL_Complex16> &A, std::string _filepath, std::string _filename, bool _isComplex){
	// Local variables
	size_t rowSize, colSize, entrySize;
	double _real, _imag;
	// Open file
	std::ifstream input;
	input.open(_filepath + _filename);
	//input.precision(std::numeric_limits<float>::digits8);
	// Ignore first line
	while (input.peek() == '%') input.ignore(2048, '\n');
	// Get matrix dimension
	input >> rowSize >> colSize;
	entrySize = rowSize * colSize;

	if (!input){
		std::cout << "File not found: " << _filepath + _filename << std::endl;
	}
	else {
		std::cout << ">> Reading matrix from "<< _filepath + _filename << " ... " << std::endl;
		std::cout << ">> Matrix size: " << rowSize << " x " << colSize << std::endl;
		A.resize(entrySize+1);	// Causes segmentation fault without +1
		clock_t io_time;
		io_time = clock();
		// Complex matrix
		if (_isComplex){
			std::cout << ">> Matrix type: COMPLEX" << std::endl;
			int i = 0;
			while (!input.eof()) {
				input >> _real >> _imag;
				MKL_Complex16 temp;
				temp.real = _real;
				temp.imag = _imag;
				A[i] = temp;
				i++;
			}
			io_time = clock() - io_time;
			std::cout << ">> Matrix " << _filename << " read" << std::endl;
			std::cout << ">>>> Time taken = " << ((float)io_time)/CLOCKS_PER_SEC << " (sec)" << "\n" << std::endl;
		}
		// Real matrix
		else if (!_isComplex){
			std::cout << ">> Matrix type: REAL" << std::endl;
			int i = 0;
			while (!input.eof()) {
				input >> _real >> _imag;
				MKL_Complex16 temp;
				temp.real = _real;
				temp.imag = 0;
				A[i] = temp;
				i++;
			}
			io_time = clock() - io_time;
			std::cout << ">> Matrix " << _filename << " read" << std::endl;
			std::cout << ">>>> Time taken = " << ((float)io_time)/CLOCKS_PER_SEC << " (sec)" << "\n" << std::endl;
		}
	}
	input.close();
}

// Writes solution vector
void io::writeSolVecComplex(std::vector<MKL_Complex16> &sol, std::string _filepath, std::string _filename){
	std::ofstream output;
	output.open(_filepath + _filename);
	clock_t io_time;
	io_time = clock();
	// Write header
	if (!output.is_open()){
		std::cout << ">> ERROR: Unable to open output file for solution vector" << std::endl;
	}
	else{
		output << std::setw(25) << std::left << "Real" << "    ";
		output << std::setw(25) << std::left << "Imag" << "\r\n";
		// Write data
		for (size_t i = 0; i < sol.size(); i++){
			output << std::setprecision(16) << std::setw(25) << std::left << sol[i].real << "    ";
			output << std::setprecision(16) << std::setw(25) << std::left << sol[i].imag << "\r\n";
		}
	}
	// Close file
	output.close();
	io_time = clock() - io_time;
	// Output messages
	std::cout << ">> Solution vector written in " << _filepath + _filename << std::endl;
	std::cout << ">>>> Vector size = " << sol.size() << std::endl;
	std::cout << ">>>> Time taken = " << ((float)io_time)/CLOCKS_PER_SEC << " (sec)" << "\n" << std::endl;
}
