/*  Copyright &copy; 2018, Dr. Stefan Sicklinger, Munich \n
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
 * \file MetaDatabase.h
 * This file holds the class of MetaDatabase.
 * \date 20/1/2018
 **************************************************************************************************/
#ifndef METADATABASE_H_
#define METADATABASE_H_

#include <vector>
#include <string>
#include <map>

#include "IP_STACCATO_XML.hxx"
/********//**
* \brief Class MetaDatabase handles all the Metadata inside STACCATO
***********/
class MetaDatabase {
public:
	static void init(char*);
	static MetaDatabase* getInstance();
	virtual ~MetaDatabase();
private:
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	MetaDatabase();
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	MetaDatabase(char *);
	static MetaDatabase* metaDatabase;
public:
	std::auto_ptr<STACCATO_XML> xmlHandle;
	std::auto_ptr<STACCATO_XML> getXMLHandle();
	void buildXML();
	void exportXML();
};

#endif /* METADATABASE_H_ */