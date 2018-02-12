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
 * This file holds the class of MetaDatabase which holds all meta data for the FE analysis
 * i.e. lightweight descriptive data, e.g. material, section, anaylsis type, solver properties
 * \date 20/1/2018
 **************************************************************************************************/
#ifndef METADATABASE_H_
#define METADATABASE_H_

#include <vector>
#include <string>
#include <map>
#include <HMesh.h>

#include "IP_STACCATO_XML.hxx"

// XERCES
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/XMLFormatter.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>

/**********
* \brief Class MetaDatabase handles all the Metadata inside STACCATO
***********/
class MetaDatabase {
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
	MetaDatabase(std::string);
	static MetaDatabase* metaDatabase;
public:
	std::auto_ptr<STACCATO_XML> xmlHandle;
	static void init(std::string);
	static MetaDatabase* getInstance();
	virtual ~MetaDatabase();
	void printXML();
	void buildXML(HMesh& _hMesh);
	void exportXML();
	void outputXML(xercesc::DOMDocument* _pmyDOMDocument, std::string _filePath);

};

#endif /* METADATABASE_H_ */