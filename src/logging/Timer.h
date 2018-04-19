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
/***********************************************************************************************//**
 * \file Timer.h
 * This file holds the class timer
 * \date 9/6/2017
 **************************************************************************************************/
#pragma once

#include <iostream>
#include <chrono>

/********//**
 * \brief This manages the timings for delta output
 **************************************************************************************************/
class Timer{
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	Timer(void){
	}
    /***********************************************************************************************
     * \brief Destructor
     * \author Stefan Sicklinger
     ***********/
	virtual ~Timer() {
	}
	/***********************************************************************************************
	* \brief Start the timer
	* \author Stefan Sicklinger
	***********/
	void start(void) {
		startTime = std::chrono::high_resolution_clock::now();
	}
	/***********************************************************************************************
	* \brief Stop the timer
	* \author Stefan Sicklinger
	***********/
	void stop(void) {
		stopTime = std::chrono::high_resolution_clock::now();
	}
	/***********************************************************************************************
	* \brief get duration in milli sec
	* \author Stefan Sicklinger
	***********/
	auto getDurationMilliSec(void) {
		return std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime).count();
	}
	/***********************************************************************************************
	* \brief get duration in mico sec
	* \author Stefan Sicklinger
	***********/
	auto getDurationMicroSec(void) {
		return std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime).count();
	}
	/***********************************************************************************************
	* \brief get duration in sec
	* \author Stefan Sicklinger
	***********/
	auto getDurationSec(void) {
		return std::chrono::duration_cast<std::chrono::seconds>(stopTime - startTime).count();
	}

private:
	std::chrono::high_resolution_clock::time_point startTime;
	std::chrono::high_resolution_clock::time_point stopTime;

};


extern Timer anaysisTimer01;
extern Timer anaysisTimer02;
extern Timer anaysisTimer03;
extern Timer linearSolverTimer01;

