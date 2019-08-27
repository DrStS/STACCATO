/*  Copyright &copy; 2019, Stefan Sicklinger, Munich
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
/*************************************************************************************************
* \file io.cu
* Written by Ji-Ho Yang
* This file contains functions for I/O
* \date 7/12/2019
**************************************************************************************************/

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
void io::readMtxDense(
                      thrust::host_vector<cuDoubleComplex> &A, std::string _filepath, std::string _filename,
                      int &row, int &col, int &nnz, bool _isComplex
                     ){
    // Local variables
    double _real, _imag;
    // Open file
    std::ifstream input;
    input.open(_filepath + _filename);
    //input.precision(std::numeric_limits<float>::digits8);
    // Ignore first line
    while (input.peek() == '%') input.ignore(2048, '\n');

    // Get matrix dimension
    input >> row >> col;
    nnz = row * col;

    if (!input){
        std::cout << "File not found" << std::endl;
        std::cout << "Please check filenames and frequency range" << std::endl;
        std::cout << "... Aborting ..." << std::endl;
        exit(1);
    }
    else {
        //std::cout << ">> Reading matrix from "<< _filepath + _filename << " ... " << std::endl;
        //std::cout << ">> Matrix size: " << row << " x " << col << std::endl;
        A.resize(nnz+1);	// Causes segmentation fault without +1
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
                input >> _real;
                cuDoubleComplex temp;
                temp.x = _real;
                temp.y = 0;
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

// Reads sparse Mtx file
void io::readMtxSparse(
                       thrust::host_vector<cuDoubleComplex> &A, std::string _filepath, std::string _filename,
                       int &row, int &col, int &nnz, bool _isComplex
                      ){
    // Local variables
    double _real, _imag;
    int _row, _col;
    // Open file
    std::ifstream input;
    input.open(_filepath + _filename);
    // Ignore first line
    while (input.peek() == '%') input.ignore(2048, '\n');
    // Get matrix dimension
    input >> row >> col >> nnz;

    if (!input){
        std::cout << "File not found." << std::endl;
        exit(1);
    }
    else {
        A.resize(row*col+1);	// Causes segmentation fault without +1 THIS NEEDS TO BE FIXED FOR SPARSE OPERATIONS
        timerIO.start();
        // Complex matrix
        if (_isComplex){
            int idx = 0;
            while (!input.eof()) {
                input >> _row >> _col >> _real >> _imag;
                cuDoubleComplex temp;
                idx = (_row-1) + (_col-1)*row;
                temp.x = _real;
                temp.y = 0;
                A[idx] = temp;
            }
            timerIO.stop();
        }
        // Real matrix
        else if (!_isComplex){
            int idx = 0;
            while (!input.eof()) {
               input >> _row >> _col >> _real;
               cuDoubleComplex temp;
               idx = (_row-1) + (_col-1)*row;
               temp.x = _real;
               temp.y = 0;
               A[idx] = temp;
            }
            timerIO.stop();
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

// Writes matrix in mtx format (currently only supports full complex matrices)
void io::writeMtxDenseComplex(thrust::host_vector<cuDoubleComplex> &mat, int row, int col, std::string _filepath, std::string _filename){
    std::ofstream output;
    output.open(_filepath + _filename);
    timerIO.start();
    // Write header
    if (!output.is_open()){
        std::cout << ">> ERROR: Unable to open output file for the matrix" << std::endl;
    }
    else{
        output << std::left << "%%MatrixMarket matrix array complex general\r\n";
        output << std::left << row << " " << col << "\r\n";
        // Write data
        for (size_t i = 0; i < mat.size(); ++i){
            output << std::setprecision(16) << std::left << mat[i].x << " ";
            output << std::setprecision(16) << std::left << mat[i].y << "\r\n";
        }
    }
    // Close file
    timerIO.stop();
    // Output messages
    std::cout << ">> Matrix written in " << _filepath + _filename << std::endl;
    std::cout << ">>>> Matrix row = " << row << std::endl;
    std::cout << ">>>> Matrix column = " << col << std::endl;
    std::cout << ">>>> Matrix size = " << mat.size() << std::endl;
    std::cout << ">>>> Time taken = " << timerIO.getDurationMicroSec()*1e-6 << " (sec)" << "\n" << std::endl;
}
