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
 * \file SimuliaODB.h
 * This file holds the class SimuliaODb which adds the capability to read Abaqus odb files
 * \date 1/18/2017
 **************************************************************************************************/
#pragma once


#include <string>
#include <assert.h>

#include "ReadWriteFile.h"

class HMesh;
/********//**
 * \brief This handles the output handling with Abaqus ODB
 **************************************************************************************************/
class SimuliaODB :public ReadWriteFile {
public:
    /***********************************************************************************************
     * \brief Constructor
     * \param[in] _filePath string which holds the path to the obd file
     * \author Stefan Sicklinger
     ***********/
	SimuliaODB(std::string _fileName, HMesh& _hMesh, int _partID);
    /***********************************************************************************************
     * \brief Destructor
     *
     * \author Stefan Sicklinger
     ***********/
	virtual ~SimuliaODB(void);
	/***********************************************************************************************
	* \brief Open die odb file
	* \param[in] _filePath string which holds the path to the obd file
	* \author Stefan Sicklinger
	***********/
	void openFile();

private:
	std::string myFileName;
	/// HMesh object 
	HMesh *myHMesh;
	/// Part Id wrt XML
	int myPartId;
};
