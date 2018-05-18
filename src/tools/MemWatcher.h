

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
 * \file MemWatcher.h
 * This file holds the class MemWatcher
 * \date 9/6/2017
 **************************************************************************************************/
#pragma once

#include <cstddef>

#if defined(_WIN32) || defined(__WIN32__) 
#include <windows.h>
#include <psapi.h>
#endif

#if defined(__linux__) 

#endif

/********//**
 * \brief This helps to analyse the memory footprint
 **************************************************************************************************/
class MemWatcher {
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	MemWatcher(void){
	}
    /***********************************************************************************************
     * \brief Destructor
     * \author Stefan Sicklinger
     ***********/
	virtual ~MemWatcher() {
	}
	/***********************************************************************************************
	* \brief Physical memory currently used by current process in bytes
	* \author Stefan Sicklinger
	***********/
	size_t getCurrentUsedPhysicalMemory (void) {
        #if defined(_WIN32) || defined(__WIN32__) 
		PROCESS_MEMORY_COUNTERS_EX pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&pmc), sizeof(pmc));
		SIZE_T physMemUsedByMe = pmc.WorkingSetSize;
		return physMemUsedByMe;
        #endif
        #if defined(__linux__) 
        return 0;
        #endif
	}
	/***********************************************************************************************
	* \brief Physical memory currently used by current process in bytes
	* \author Stefan Sicklinger
	***********/
	size_t getCurrentUsedVirtualMemory(void) {
        #if defined(_WIN32) || defined(__WIN32__) 
		PROCESS_MEMORY_COUNTERS_EX pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&pmc), sizeof(pmc));
		SIZE_T virtualMemUsedByMe = pmc.PrivateUsage;
		return virtualMemUsedByMe;
        #endif
        #if defined(__linux__) 
        return 0;
        #endif
	}


private:



};

extern MemWatcher memWatcher;
