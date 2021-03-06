/*  Copyright &copy; 2018, Stefan Sicklinger, Munich
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
 * \file FileResults.h
 * This file holds the class FileResults which reads and writes result files generated by STACCATO
 * \date 11/18/2018
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
 * \brief This handles the file IO of results of STACCATO mainly transfer functions
 **************************************************************************************************/
class FileResults :public ReadWriteFile {
public:
    /***********************************************************************************************
     * \brief Constructor
     * \param[in] _fileName string which holds the file name
     * \param[in] _filePath string which holds the file path
     * \author Stefan Sicklinger
     ***********/
	FileResults(std::string _fileName, std::string _filePath);
    /***********************************************************************************************
     * \brief Destructor
     *
     * \author Stefan Sicklinger
     ***********/
	virtual ~FileResults(void);
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
	void addComplexDenseMatrixFRF(std::string _matrixName, std::vector<STACCATOComplexDouble>& _values);
	/***********************************************************************************************
    * \brief Add complex dense matrix
    * \author Stefan Sicklinger
    * \param[in] _matrixName
    * \param[in] _values
	* \param[in] _numColumns
	* \param[in] _numRows
    ***********/
	void addComplexDenseMatrixFRF(std::string _matrixName, std::vector<STACCATOComplexDouble>& _values, unsigned int _numColumns, unsigned int _numRows);
	/***********************************************************************************************
    * \brief Add input output map for dense ROM matrices 
    * \author Stefan Sicklinger
    * \param[in] _inputNodeLabel
    * \param[in] _inputDoFLabel
    * \param[in] _outputNodeLabel
    * \param[in] _outputDoFLabel
    ***********/
	void addInputOutputMapFRF(const std::vector<unsigned int>& _inputNodeLabel, const std::vector<unsigned int>& _inputDoFLabel, const std::vector<unsigned int>& _outputNodeLabel, const std::vector<unsigned int>& _outputDoFLabel);
	/***********************************************************************************************
	* \brief Add input output map for dense ROM matrices
	* \author Stefan Sicklinger
	* \param[in] _frequencyVector
	***********/
	void addFrequencyVectorFRF(const std::vector<double>& _frequencyVector);
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
private:
	/***********************************************************************************************
	* \brief Add input output map for dense ROM matrices
	* \author Stefan Sicklinger
	* \param[in] _containerName
	* \param[in] _nodeLabel
	* \param[in] _DoFLabel
	***********/
	void addNodeToDoFLabelMap(std::string _containerName, const std::vector<unsigned int>& _nodeLabel, const std::vector<unsigned int>& _DoFLabel);
};