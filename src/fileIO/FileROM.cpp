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
#include "FileROM.h"
#include "AuxiliaryParameters.h"
//HDF5
#ifdef USE_HDF5
#include "H5Cpp.h"
#include "Timer.h"
#endif

FileROM::FileROM(std::string _fileName, std::string _filePath) : myFileName(_fileName), myFilePath(_filePath) {
#ifdef USE_HDF5
	H5::Exception::dontPrint();
	myHDF5FileHandle = NULL;

/*

		STACCATOComplexDouble *kdynRead = new STACCATOComplexDouble[LENGTH];

		dataset->read(kdynRead, mtype);

		std::cout << std::endl << "Field real : " << std::endl;
		for (i = 0; i < LENGTH; i++)
			std::cout << kdyn[i].real - kdynRead[i].real << " ";
		std::cout << std::endl;
		std::cout << std::endl << "Field imag : " << std::endl;
		for (i = 0; i < LENGTH; i++)
			std::cout << kdyn[i].imag - kdynRead[i].imag << " ";
		std::cout << std::endl;

		delete dataset;
		delete file;
*/
		myHDF5FileHandle = NULL;
#endif
}


FileROM::~FileROM() {
	delete myHDF5FileHandle;
	delete myHDF5groupOperators;
}

void FileROM::createContainer(bool _forceWrite) {
#ifdef USE_HDF5

	try
	{
		if (_forceWrite) {
			myHDF5FileHandle = new H5::H5File(myFilePath + myFileName, H5F_ACC_TRUNC);
			myHDF5groupOperators = new H5::Group(myHDF5FileHandle->createGroup("/OperatorsDenseROM"));
		}
		else {
			myHDF5FileHandle = new H5::H5File(myFilePath + myFileName, H5F_ACC_EXCL);
			myHDF5groupOperators = new H5::Group(myHDF5FileHandle->createGroup("/OperatorsDenseROM"));
		}
	}
	catch (H5::FileIException error)
	{
		std::cout << "Error: Cannot create file" << std::endl;
		std::cout << "File already exists!" << std::endl;
	}
#endif // USE_HDF5
}

void FileROM::openContainer(bool _writePermission) {
#ifdef USE_HDF5
	try
	{
		if (_writePermission) {
			myHDF5FileHandle = new H5::H5File(myFilePath + myFileName, H5F_ACC_RDWR);
}
		else {
			myHDF5FileHandle = new H5::H5File(myFilePath + myFileName, H5F_ACC_RDONLY);
		}
	}
	catch (H5::FileIException error)
	{
		std::cout << "Error: Cannot open file" << std::endl;
		std::cout << "File already exists!" << std::endl;
	}
#endif // USE_HDF5
	
}


void FileROM::addComplexDenseMatrix(std::string _matrixName, std::vector<STACCATOComplexDouble>& _values, unsigned int _numColumns, unsigned int _numRows) {
#ifdef USE_HDF5
	try
	{
		unsigned int size = _values.size();
		hsize_t dim[] = { size }; 
		H5::DataSpace space(1, dim);
		H5::CompType mtype(sizeof(STACCATOComplexDouble));
		mtype.insertMember("real", HOFFSET(STACCATOComplexDouble, real), H5::PredType::NATIVE_DOUBLE);
		mtype.insertMember("imag", HOFFSET(STACCATOComplexDouble, imag), H5::PredType::NATIVE_DOUBLE);
		H5::DataSet* dataset;
		dataset = new H5::DataSet(myHDF5FileHandle->createDataSet("OperatorsDenseROM/"+_matrixName, mtype, space));
		dataset->write(_values.data(), mtype);
		/// Add matrix dimension information to container
		H5::DataSpace attrDataspaceScalar(H5S_SCALAR);
		H5::Attribute attribute = dataset->createAttribute("Matrix # columns", H5::PredType::STD_I32BE, attrDataspaceScalar);
		int attrDataScalar[1] = { _numColumns };
		attribute.write(H5::PredType::NATIVE_INT, attrDataScalar);
		attribute = dataset->createAttribute("Matrix # rows", H5::PredType::STD_I32BE, attrDataspaceScalar);
		attrDataScalar[0] = _numRows;
		attribute.write(H5::PredType::NATIVE_INT, attrDataScalar);
		delete dataset;
	} 
	catch (H5::DataSetIException error)
	{
		std::cout << "Error: DataSet operations" << std::endl;
	}
	catch (H5::DataSpaceIException error)
	{
		std::cout << "Error: DataSpace operations" << std::endl;
	}
	catch (H5::DataTypeIException error)
	{
		std::cout << "Error: DataType operations" << std::endl;
	}
#endif // USE_HDF5
}

void FileROM::addInputOutputMapROM(const std::vector<unsigned int>& _inputNodeLabel, const std::vector<unsigned int>& _inputDoFLabel, const std::vector<unsigned int>& _outputNodeLabel, const std::vector<unsigned int>& _outputDoFLabel) {
	addNodeToDoFLabelMap("OperatorsDenseROM/inputMap",   _inputNodeLabel, _inputDoFLabel);
	addNodeToDoFLabelMap("OperatorsDenseROM/outputMap", _outputNodeLabel, _outputDoFLabel);
}

void FileROM::addComplexDenseMatrix(std::string _matrixName, std::vector<STACCATOComplexDouble>& _values) {
	_values.size();
	addComplexDenseMatrix(_matrixName, _values, sqrt(_values.size()), sqrt(_values.size()));
}

void FileROM::closeContainer(void) {
#ifdef USE_HDF5
	myHDF5groupOperators->close();
	myHDF5FileHandle->close();
#endif // USE_HDF5
}


void FileROM::addNodeToDoFLabelMap(std::string _containerName, const std::vector<unsigned int>& _nodeLabel, const std::vector<unsigned int>& _DoFLabel) {
#ifdef USE_HDF5
	try
	{
		struct nodeLabelDoFLabel
		{
			unsigned int nodeLabel;
			unsigned int DoFLabel;
		};
		if (_nodeLabel.size() == _DoFLabel.size())
		{
			std::vector < nodeLabelDoFLabel> nodeLabelDoFLabelTmp;
			nodeLabelDoFLabelTmp.resize(_nodeLabel.size());
			for (int i = 0; i < _nodeLabel.size(); i++) {
				nodeLabelDoFLabelTmp[i].nodeLabel = _nodeLabel[i];
				nodeLabelDoFLabelTmp[i].DoFLabel = _DoFLabel[i];
			}
			unsigned int size = _nodeLabel.size();
			hsize_t dim[] = { size };
			H5::DataSpace space(1, dim);
			H5::CompType mtype(sizeof(nodeLabelDoFLabel));
			mtype.insertMember("nodeLabel", HOFFSET(nodeLabelDoFLabel, nodeLabel), H5::PredType::NATIVE_UINT);
			mtype.insertMember("DoFLabel", HOFFSET(nodeLabelDoFLabel, DoFLabel), H5::PredType::NATIVE_UINT);
			H5::DataSet* dataset;
			dataset = new H5::DataSet(myHDF5FileHandle->createDataSet(_containerName, mtype, space));
			dataset->write(nodeLabelDoFLabelTmp.data(), mtype);
			delete dataset;
		}
	}
	catch (H5::DataSetIException error)
	{
		std::cout << "Error: DataSet operations" << std::endl;
	}
	catch (H5::DataSpaceIException error)
	{
		std::cout << "Error: DataSpace operations" << std::endl;
	}
	catch (H5::DataTypeIException error)
	{
		std::cout << "Error: DataType operations" << std::endl;
	}
#endif // USE_HDF5
}