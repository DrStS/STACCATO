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
 * \file FeMetaDatabase.h
 * This file holds the class FeMetaDatabase which holds all meta data for the FE analysis
 * i.e. lightweight descriptive data, e.g. material, section, anaylsis type, solver properties
 * \date 8/28/2017
 **************************************************************************************************/

#ifndef FEMETADATABASE_H_
#define FEMETADATABASE_H_

#include <string>
#include <assert.h>

/********//**
 * \brief This handles the output handling with Abaqus ODB
 **************************************************************************************************/
class FeMetaDatabase{
public:
    /***********************************************************************************************
     * \brief Constructor
     * \param[in] _obdFilePath string which holds the path to the obd file
     * \author Stefan Sicklinger
     ***********/
	FeMetaDatabase(void);
    /***********************************************************************************************
     * \brief Destructor
     *
     * \author Stefan Sicklinger
     ***********/
	virtual ~FeMetaDatabase(void);
private:

};


#endif /* FEMETADATABASE_H_ */
