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
#include "FileFOM.h"
#include "AuxiliaryParameters.h"
#include "ReadWriteFile.h"
//HDF5
#ifdef USE_HDF5
#include "H5Cpp.h"
#include "Timer.h"
#endif

FileFOM::FileFOM(std::string _fileName, std::string _filePath) : myFileName(_fileName), myFilePath(_filePath), ReadWriteFile() {
#ifdef USE_HDF5
	//H5::Exception::dontPrint();
	myHDF5FileHandle = nullptr;
	myHDF5FileHandle = nullptr;
#endif
}


FileFOM::~FileFOM() {
	delete myHDF5FileHandle;
	delete myHDF5groupOperators;
}

void FileFOM::createContainer(bool _forceWrite) {
#ifdef USE_HDF5

	try
	{
		if (_forceWrite) {
			myHDF5FileHandle = new H5::H5File(myFilePath + myFileName, H5F_ACC_TRUNC);
		}
		else {
			myHDF5FileHandle = new H5::H5File(myFilePath + myFileName, H5F_ACC_EXCL);
		}
		myHDF5groupOperators = new H5::Group(myHDF5FileHandle->createGroup("/OperatorsSparseFOM"));
		myHDF5groupOperators->createGroup("/OperatorsSparseFOM/Kre");
		myHDF5groupOperators->createGroup("/OperatorsSparseFOM/Kim");
		myHDF5groupOperators->createGroup("/OperatorsSparseFOM/D");
		myHDF5groupOperators->createGroup("/OperatorsSparseFOM/M");
	}
	catch (H5::FileIException error)
	{
		std::cout << "Error: Cannot create file" << std::endl;
		std::cout << "File already exists!" << std::endl;
	}
#endif // USE_HDF5
}

void FileFOM::openContainer(bool _writePermission) {
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

void FileFOM::addRealSparseMatrix(std::string _matrixName, const std::vector<int>& _iA, const std::vector<int>& _jA, const std::vector<double>& _values) {
	try
	{
		H5::DataSet* dataset1;
		unsigned int size1 = _iA.size();
		hsize_t dim1[] = { size1 };
		H5::DataSpace space1(1, dim1);
		dataset1 = new H5::DataSet(myHDF5FileHandle->createDataSet("OperatorsSparseFOM/" + _matrixName + "/iIndices", H5::PredType::NATIVE_INT, space1));
		dataset1->write(_iA.data(), H5::PredType::NATIVE_INT);
		delete dataset1;
		H5::DataSet* dataset2;
		unsigned int size2 = _jA.size();
		hsize_t dim2[] = { size2 };
		H5::DataSpace space2(1, dim2);
		dataset2 = new H5::DataSet(myHDF5FileHandle->createDataSet("OperatorsSparseFOM/" + _matrixName + "/jIndices", H5::PredType::NATIVE_INT, space2));
		dataset2->write(_jA.data(), H5::PredType::NATIVE_INT);
		delete dataset2;
		H5::DataSet* dataset3;
		unsigned int size3 = _values.size();
		hsize_t dim3[] = { size3 };
		H5::DataSpace space3(1, dim3);
		dataset3 = new H5::DataSet(myHDF5FileHandle->createDataSet("OperatorsSparseFOM/" + _matrixName + "/values", H5::PredType::NATIVE_DOUBLE, space3));
		dataset3->write(_values.data(), H5::PredType::NATIVE_DOUBLE);
		delete dataset3;/**/
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
}

void FileFOM::addNodeAndDoFLabel(const std::vector<unsigned int>& _nodeLabel, const std::vector<unsigned int>& _DoFLabel) {
	addNodeToDoFLabelMap("OperatorsSparseFOM/nodeToDoFLabelMap", _nodeLabel, _DoFLabel);
}

void FileFOM::closeContainer(void) {
#ifdef USE_HDF5
	myHDF5groupOperators->close();
	myHDF5FileHandle->close();
#endif // USE_HDF5
}


void FileFOM::addNodeToDoFLabelMap(std::string _containerName, const std::vector<unsigned int>& _nodeLabel, const std::vector<unsigned int>& _DoFLabel) {
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
}