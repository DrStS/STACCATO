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
 * \file FileROM.h
 * This file holds the class FileROM which reads and writes ROM files generated by STACCATO
 * \date 9/11/2018
 **************************************************************************************************/
#pragma once
#include "ReadWriteFile.h"
#include "AuxiliaryParameters.h"
#include <string>
#include <vector>

namespace H5 {
	class H5File;
	class Group;
}
class ReadWriteFile;
/********//**
 * \brief This handles the file IO of reduced roder models
 **************************************************************************************************/
class FileROM :public ReadWriteFile {
public:
    /***********************************************************************************************
     * \brief Constructor
     * \param[in] _fileName string which holds the file name
     * \param[in] _filePath string which holds the file path
     * \author Stefan Sicklinger
     ***********/
	FileROM(std::string _fileName, std::string _filePath);
    /***********************************************************************************************
     * \brief Destructor
     *
     * \author Stefan Sicklinger
     ***********/
	virtual ~FileROM(void);
	/***********************************************************************************************
	 * \brief Create binary container
	 * \author Stefan Sicklinger
	 * \param[in] _forceWrite is true will wipe out any per existing data in the container 
	 ***********/
	void createContainer(bool _forceWrite);
	/***********************************************************************************************
    * \brief Open binary container
    * \author Stefan Sicklinger
    * \param[in] _writePermission true will enable write access to container
    ***********/
	void openContainer(bool _writePermission);
	/***********************************************************************************************
	* \brief Add complex dense matrix
	* \author Stefan Sicklinger
	* \param[in] _matrixName
	* \param[in] _values
	***********/
	void addComplexDenseMatrix(std::string _matrixName, std::vector<STACCATOComplexDouble>& _values);
	/***********************************************************************************************
     * \brief Close binary container
     * \author Stefan Sicklinger
     ***********/
	void closeContainer(void);

private:
	/// my file name;
	std::string myFileName;
	/// my file path;
	std::string myFilePath;
	/// my file handle
	H5::H5File* myHDF5FileHandle;
	/// my group handle
	H5::Group* myHDF5groupOperators;
};
