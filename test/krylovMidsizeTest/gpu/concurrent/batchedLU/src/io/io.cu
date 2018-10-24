// Libraries
#include <fstream>
#include <iostream>
#include <string>
#include <limits>
#include <iomanip>

// THRUST
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// CUCOMPLEX
#include <cuComplex.h>

// Header files
#include "io.cuh"
#include "../helper/Timer.cuh"

// Namespace
using namespace staccato;

// Reads dense Mtx file
void io::readMtxDense(thrust::host_vector<cuDoubleComplex> &A, std::string _filepath, std::string _filename, bool _isComplex){
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
        std::cout << "File not found." << std::endl;
        exit(1);
    }
    else {
        //std::cout << ">> Reading matrix from "<< _filepath + _filename << " ... " << std::endl;
        //std::cout << ">> Matrix size: " << rowSize << " x " << colSize << std::endl;
        A.resize(entrySize+1);	// Causes segmentation fault without +1
        timerIO.start();
        // Complex matrix
        if (_isComplex){
            //std::cout << ">> Matrix type: COMPLEX" << std::endl;
            int i = 0;
            while (!input.eof()) {
                input >> _real >> _imag;
                cuDoubleComplex temp;
                temp.x = _real;
                temp.y = _imag;
                A[i] = temp;
                i++;
            }
            timerIO.stop();
            //std::cout << ">> Matrix " << _filename << " read" << std::endl;
            //std::cout << ">>>> Time taken = " << timerIO.getDurationMicroSec()*1e-6 << "\n" << std::endl;
        }
        // Real matrix
        else if (!_isComplex){
            //std::cout << ">> Matrix type: REAL" << std::endl;
            int i = 0;
            while (!input.eof()) {
                input >> _real >> _imag;
                cuDoubleComplex temp;
                temp.x = _real;
                temp.y = _imag;
                A[i] = temp;
                i++;
            }
            timerIO.stop();
            //std::cout << ">> Matrix " << _filename << " read" << std::endl;
            //std::cout << ">>>> Time taken = " << timerIO.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;
        }
    }
    input.close();
}

// Writes solution vector
void io::writeSolVecComplex(thrust::host_vector<cuDoubleComplex> &sol, std::string _filepath, std::string _filename){
    std::ofstream output;
    output.open(_filepath + _filename);
    timerIO.start();
    // Write header
    if (!output.is_open()){
        std::cout << ">> ERROR: Unable to open output file for solution vector" << std::endl;
    }
    else{
        output << std::setw(25) << std::left << "Real" << "    ";
        output << std::setw(25) << std::left << "Imag" << "\r\n";
        // Write data
        for (size_t i = 0; i < sol.size(); i++){
            output << std::setprecision(16) << std::setw(25) << std::left << sol[i].x << "    ";
            output << std::setprecision(16) << std::setw(25) << std::left << sol[i].y << "\r\n";
        }
    }
    // Close file
    timerIO.stop();
    // Output messages
    std::cout << ">> Solution vector written in " << _filepath + _filename << std::endl;
    std::cout << ">>>> Vector size = " << sol.size() << std::endl;
    std::cout << ">>>> Time taken = " << timerIO.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;
}
