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
#include <iostream>
#include <string.h>
#include <assert.h>
#include <sstream>

#include "MetaDatabase.h"

using namespace std;

MetaDatabase *MetaDatabase::metaDatabase = NULL;

void MetaDatabase::init(char* inputFileName) {
	assert(metaDatabase == NULL);
	metaDatabase = new MetaDatabase(inputFileName);
}

MetaDatabase* MetaDatabase::getInstance() {
	assert(metaDatabase != NULL);
	return metaDatabase;
}

auto_ptr<STACCATO_XML> MetaDatabase::getXMLHandle() {
	return xmlHandle;
}

MetaDatabase::MetaDatabase() {

}

MetaDatabase::MetaDatabase(char *inputFileName) {
	try
	{
		xmlHandle = auto_ptr<STACCATO_XML>(STACCATO_XML_(inputFileName));
	}
	catch (const xml_schema::exception& e) {
		cerr << e << endl;
		exit(EXIT_FAILURE);
	}
}

MetaDatabase::~MetaDatabase() {
	metaDatabase = NULL;
}

void MetaDatabase::buildXML() {

	std::cout << "==================================\n";
	std::cout << "========= STACCATO IMPORT ========\n";
	std::cout << "==================================\n\n";

	std::cout << ">> Name: " << xmlHandle->ANALYSIS().begin()->NAME() << std::endl;
	std::cout << ">> Type: " << xmlHandle->ANALYSIS().begin()->NAME() << std::endl;

	for (STACCATO_XML::FREQUENCY_const_iterator i(xmlHandle->FREQUENCY().begin());
		i != xmlHandle->FREQUENCY().end();
		++i)
	{
		std::cout << ">> Frequency Distr.: " << i->Type() << endl;
		std::cout << " > Start: " << i->START_FREQ() << " Hz" << endl;
		std::cout << " > End  : " << i->END_FREQ() << " Hz" << endl;
		std::cout << " > Step : " << i->STEP_FREQ() << " Hz" << endl;
	}
	std::cout << ">> NODES: " << std::endl;
	for (STACCATO_XML::NODES_const_iterator i(xmlHandle->NODES().begin());
		i != xmlHandle->NODES().end(); i++)
	{
		for (int j = 0; j < i->NODE().size(); j++) {
			std::cout << " > ID: " << i->NODE().at(j).ID() << " X, Y, Z: " << i->NODE().at(j).X() << "," << i->NODE().at(j).Y() << "," << i->NODE().at(j).Z() << std::endl;
		}
	}

	std::cout << "\n==================================\n";
}

void MetaDatabase::exportXML() {
	// DOM export
}
