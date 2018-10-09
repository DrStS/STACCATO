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

const int   LENGTH = 80000000;

FileROM::FileROM(std::string _fileName, std::string _filePath) : myFileName(_fileName), myFilePath(_filePath) {
#ifdef USE_HDF5
	try
	{
		/*
		 * Initialize the data
		 */
		int  i;
		std::cout << "TEST" << std::endl;
		STACCATOComplexDouble *kdyn = new STACCATOComplexDouble[LENGTH];

		std::cout << "TEST2" << std::endl;
		for (i = 0; i < LENGTH; i++)
		{
			kdyn[i].real = (double)std::rand() / RAND_MAX;
			kdyn[i].imag = (double)std::rand() / RAND_MAX;
		}
		/*
		 * Turn off the auto-printing when failure occurs so that we can
		 * handle the errors appropriately
		 */
anaysisTimer01.start();
		H5::Exception::dontPrint();
		/*
		 * Create the data space.
		 */
		hsize_t dim[] = { LENGTH };   /* Dataspace dimensions */
		H5::DataSpace space(1, dim);
		/*
		 * Create the file.
		 */
		H5::H5File* file = new H5::H5File("Kdyn.h5", H5F_ACC_TRUNC);
		H5::Group* group = new H5::Group(file->createGroup("/Data"));
		/*
		 * Create the memory datatype.
		 */
		H5::CompType mtype(sizeof(STACCATOComplexDouble));
		mtype.insertMember("real", HOFFSET(STACCATOComplexDouble, real), H5::PredType::NATIVE_DOUBLE);
		mtype.insertMember("imag", HOFFSET(STACCATOComplexDouble, imag), H5::PredType::NATIVE_DOUBLE);
		/*
		 * Create the dataset.
		 */
		H5::DataSet* dataset;
		dataset = new H5::DataSet(file->createDataSet("Data/K_dyn", mtype, space));
		/*
		 * Write data to the dataset;
		 */

		hsize_t dims[1] = { 2 };
		int attr_data[2] = { 100, 200 };
		H5::DataSpace attr_dataspace = H5::DataSpace(1, dims);
		const H5std_string	ATTR_NAME("Units");
		H5::Attribute attribute = dataset->createAttribute(ATTR_NAME, H5::PredType::STD_I32BE,
			attr_dataspace);
		attribute.write(H5::PredType::NATIVE_INT, attr_data);
		attribute.write(H5::PredType::NATIVE_INT, attr_data);


		dataset->write(kdyn, mtype);
		/*
		 * Release resources
		 */
anaysisTimer01.stop();
        std::cout << "Time for complex: " << anaysisTimer01.getDurationMilliSec() << std::endl;
		

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
	}  // end of try block
    // catch failure caused by the H5File operations
	catch (H5::FileIException error)
	{

		
	}
	// catch failure caused by the DataSet operations
	catch (H5::DataSetIException error)
	{

	
	}
	// catch failure caused by the DataSpace operations
	catch (H5::DataSpaceIException error)
	{


	}
	// catch failure caused by the DataSpace operations
	catch (H5::DataTypeIException error)
	{


	}
#endif
}


FileROM::~FileROM() {
	
}

void FileROM::createContainer(bool _forceWrite) {

	try
	{
		if (_forceWrite) {
			myHDF5FileHandle = new H5::H5File(myFilePath + myFileName, H5F_ACC_TRUNC);
		}
		else {
			myHDF5FileHandle = new H5::H5File(myFilePath + myFileName, H5F_ACC_EXCL);
		}
	}
		catch (H5::FileIException error)
	{
			std::cout << "Error: Cannot create file" << std::endl;
			std::cout << "File already exists!" << std::endl;
	}
}



