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
using namespace xercesc;

MetaDatabase *MetaDatabase::metaDatabase = NULL;

void MetaDatabase::init(std::string inputFileName) {
	assert(metaDatabase == NULL);
	metaDatabase = new MetaDatabase(inputFileName);
	std::cout << ">> xml Import: " << inputFileName << " successful.\n\n";
}

MetaDatabase* MetaDatabase::getInstance() {
	assert(metaDatabase != NULL);
	return metaDatabase;
}

MetaDatabase::MetaDatabase() {

}

MetaDatabase::MetaDatabase(std::string inputFileName) {
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

void MetaDatabase::printXML() {

	/*std::cout << "==================================\n";
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
			std::cout << " > ID: " << i->NODE()[j].ID() << " X, Y, Z: " << i->NODE()[j].X() << "," << i->NODE()[j].Y() << "," << i->NODE()[j].Z() << std::endl;
		}
	}

	std::cout << "\n==================================\n";*/
}

/// we need to move that
void MetaDatabase::buildXML(HMesh& _hMesh) {
	STACCATO_XML::PARTS_const_iterator iterParts(MetaDatabase::getInstance()->xmlHandle->PARTS().begin());
	for (int iPart = 0; iPart < iterParts->PART().size(); iPart++)
	{
		if (std::string(iterParts->PART()[iPart].TYPE()->data()) == "FE" || std::string(iterParts->PART()[iPart].TYPE()->data()) == "FE_KMOR")
		{
			// Add STACCATO_XML-User Entered Sets
			// Element Sets
			for (int k = 0; k < iterParts->PART()[iPart].SETS().begin()->ELEMENTSET().size(); k++) {
				// Recognize List for ALL or a List of IDs
				std::vector<int> idList;
				// Keyword: ALL
				if (std::string(iterParts->PART()[iPart].SETS().begin()->ELEMENTSET()[k].LIST()->c_str()) == "ALL") {
					idList = _hMesh.getElementLabels();
				}
				else {	// ID List
						// filter
					std::stringstream stream(std::string(iterParts->PART()[iPart].SETS().begin()->ELEMENTSET()[k].LIST()->c_str()));
					while (stream) {
						int n;
						stream >> n;
						if (stream)
							idList.push_back(n);
					}
				}
				_hMesh.addElemSet(std::string(iterParts->PART()[iPart].SETS().begin()->ELEMENTSET()[k].Name()->c_str()), idList);
			}
			// Node Sets
			for (int k = 0; k < iterParts->PART()[iPart].SETS().begin()->NODESET().size(); k++) {
				// Recognize List for ALL or a List of IDs
				std::vector<int> idList;
				// Keyword: ALL
				if (std::string(iterParts->PART()[iPart].SETS().begin()->NODESET()[k].LIST()->c_str()) == "ALL") {
					idList = _hMesh.getNodeLabels();
				}
				else {	// ID List
						// filter
					std::stringstream stream(std::string(iterParts->PART()[iPart].SETS().begin()->NODESET()[k].LIST()->c_str()));
					while (stream) {
						int n;
						stream >> n;
						if (stream)
							idList.push_back(n);
					}
				}
				_hMesh.addNodeSet(std::string(iterParts->PART()[iPart].SETS().begin()->NODESET()[k].Name()->c_str()), idList);
			}

			// Reference Node
			for (int k = 0; k < iterParts->PART()[iPart].LOADS().begin()->LOAD().size(); k++) {
				if (std::string(iterParts->PART()[iPart].LOADS().begin()->LOAD()[k].Type()->c_str()) == "DistributingCouplingForce") {
					std::vector<int> nextLabel = { _hMesh.getNodeLabels().back() + 1 };
					_hMesh.addNode(nextLabel[0], std::atof(iterParts->PART()[iPart].LOADS().begin()->LOAD()[k].REFERENCENODE().begin()->X()->c_str()), std::atof(iterParts->PART()[iPart].LOADS().begin()->LOAD()[k].REFERENCENODE().begin()->Y()->c_str()), std::atof(iterParts->PART()[iPart].LOADS().begin()->LOAD()[k].REFERENCENODE().begin()->Z()->c_str()));
					_hMesh.referenceNodeLabel.push_back(nextLabel[0]);
					_hMesh.addNodeSet(std::string(iterParts->PART()[iPart].LOADS().begin()->LOAD()[k].REFERENCENODESET().begin()->Name()->c_str()), nextLabel);

					std::cout << ">> Reference Node/NodeSet Found: Assigned with node label " << nextLabel[0] << ".\n";
				}
			}
		}
	}

}

void MetaDatabase::exportXML() {
	// DOM export
	XMLPlatformUtils::Initialize();

	DOMImplementation* xmlImpl = NULL;
	xmlImpl = DOMImplementationRegistry::getDOMImplementation(XMLString::transcode("core"));

	if (xmlImpl != NULL) {
		try {
			DOMDocument* xmlExportDoc = xmlImpl->createDocument(0, XMLString::transcode("STACCATO_XML"), 0);

			DOMElement* pRoot = xmlExportDoc->getDocumentElement();
			// Analysis --
			DOMElement* pElement = xmlExportDoc->createElement(XMLString::transcode("ANALYSIS"));

			DOMElement* pChildElement = xmlExportDoc->createElement(XMLString::transcode("NAME"));
			pChildElement->setTextContent(XMLString::transcode("FELGE"));
			pElement->appendChild(pChildElement);
			
			pChildElement = xmlExportDoc->createElement(XMLString::transcode("TYPE"));
			pChildElement->setTextContent(XMLString::transcode("DYNAMIC"));
			pElement->appendChild(pChildElement);

			pRoot->appendChild(pElement);
			// --

			// Frequency --
			pElement = xmlExportDoc->createElement(XMLString::transcode("FREQUENCY"));
			pElement->setAttribute(XMLString::transcode("Type"), XMLString::transcode("Step"));

			pChildElement = xmlExportDoc->createElement(XMLString::transcode("START_FREQ"));
			pChildElement->setTextContent(XMLString::transcode("1000"));
			pElement->appendChild(pChildElement);
			pChildElement = xmlExportDoc->createElement(XMLString::transcode("END_FREQ"));
			pChildElement->setTextContent(XMLString::transcode("2500"));
			pElement->appendChild(pChildElement);
			pChildElement = xmlExportDoc->createElement(XMLString::transcode("STEP_FREQ"));
			pChildElement->setTextContent(XMLString::transcode("500"));
			pElement->appendChild(pChildElement);

			pRoot->appendChild(pElement);
			// --

			// Materials --
			pElement = xmlExportDoc->createElement(XMLString::transcode("MATERIALS"));

			pChildElement = xmlExportDoc->createElement(XMLString::transcode("MATERIAL"));
			pChildElement->setAttribute(XMLString::transcode("Name"), XMLString::transcode("Aluminium"));
			pChildElement->setAttribute(XMLString::transcode("Type"), XMLString::transcode("Isotropic"));
			
			DOMElement* pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("E"));
			pChildChildElement->setTextContent(XMLString::transcode("210000"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("nu"));
			pChildChildElement->setTextContent(XMLString::transcode("0.3"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("rho"));
			pChildChildElement->setTextContent(XMLString::transcode("7.85e-9"));
			pChildElement->appendChild(pChildChildElement);

			pElement->appendChild(pChildElement);
			pRoot->appendChild(pElement);
			// --

			// Nodes --
			pElement = xmlExportDoc->createElement(XMLString::transcode("NODES"));

			pChildElement = xmlExportDoc->createElement(XMLString::transcode("FILE"));
			pChildElement->setTextContent(XMLString::transcode("ModelName.odb"));
			pElement->appendChild(pChildElement);

			pRoot->appendChild(pElement);
			// --

			// Elements --
			pElement = xmlExportDoc->createElement(XMLString::transcode("ELEMENTS"));

			pChildElement = xmlExportDoc->createElement(XMLString::transcode("FILE"));
			pChildElement->setTextContent(XMLString::transcode("ModelName.odb"));
			pElement->appendChild(pChildElement);

			pRoot->appendChild(pElement);
			// --

			// LOADS --
			pElement = xmlExportDoc->createElement(XMLString::transcode("LOADS"));

			pChildElement = xmlExportDoc->createElement(XMLString::transcode("LOAD"));
			pChildElement->setAttribute(XMLString::transcode("Type"), XMLString::transcode("NODAL"));

			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("ID"));
			pChildChildElement->setTextContent(XMLString::transcode("1"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("OnNode"));
			pChildChildElement->setTextContent(XMLString::transcode("4"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("Fx"));
			pChildChildElement->setTextContent(XMLString::transcode("3.000"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("iFx"));
			pChildChildElement->setTextContent(XMLString::transcode("11.11"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("Fy"));
			pChildChildElement->setTextContent(XMLString::transcode("4.000"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("iFy"));
			pChildChildElement->setTextContent(XMLString::transcode("12.11"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("Fz"));
			pChildChildElement->setTextContent(XMLString::transcode("5.000"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("iFz"));
			pChildChildElement->setTextContent(XMLString::transcode("13.11"));
			pChildElement->appendChild(pChildChildElement);

			pElement->appendChild(pChildElement);
			pRoot->appendChild(pElement);
			// --

			// BC --
			pElement = xmlExportDoc->createElement(XMLString::transcode("BC"));

			pChildElement = xmlExportDoc->createElement(XMLString::transcode("DBC"));

			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("ID"));
			pChildChildElement->setTextContent(XMLString::transcode("1"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("NODESET"));
			pChildChildElement->setTextContent(XMLString::transcode("2 4 6 8"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("Ux"));
			pChildChildElement->setTextContent(XMLString::transcode("0"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("Uy"));
			pChildChildElement->setTextContent(XMLString::transcode("0"));
			pChildElement->appendChild(pChildChildElement);
			pChildChildElement = xmlExportDoc->createElement(XMLString::transcode("Uz"));
			pChildChildElement->setTextContent(XMLString::transcode("0"));
			pChildElement->appendChild(pChildChildElement);

			pElement->appendChild(pChildElement);
			pRoot->appendChild(pElement);
			// --

			outputXML(xmlExportDoc, "C:/software/repos/STACCATO/xsd/SavedData_xerces.xml");
			xmlExportDoc->release();
		}
		catch (...) {
			cerr << "Error Occurred while writing XML!";
		}
	}
}

void MetaDatabase::outputXML(xercesc::DOMDocument* _pmyDOMDocument, std::string _filePath)
{
	//Return the first registered implementation that has the desired features. In this case, we are after a DOM implementation that has the LS feature... or Load/Save. 
	DOMImplementation *implementation = DOMImplementationRegistry::getDOMImplementation(XMLString::transcode("LS"));

	// Create a DOMLSSerializer which is used to serialize a DOM tree into an XML document. 
	DOMLSSerializer *serializer = ((DOMImplementationLS*)implementation)->createLSSerializer();

	// Make the output more human readable by inserting line feeds. 
	if (serializer->getDomConfig()->canSetParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true))
		serializer->getDomConfig()->setParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true);

	// The end-of-line sequence of characters to be used in the XML being written out.  
	serializer->setNewLine(XMLString::transcode("\r\n"));

	// Convert the path into Xerces compatible XMLCh*. 
	XMLCh *tempFilePath = XMLString::transcode(_filePath.c_str());

	// Specify the target for the XML output. 
	XMLFormatTarget *formatTarget = new LocalFileFormatTarget(tempFilePath);

	// Create a new empty output destination object. 
	DOMLSOutput *output = ((DOMImplementationLS*)implementation)->createLSOutput();

	// Set the stream to our target. 
	output->setByteStream(formatTarget);

	// Write the serialized output to the destination. 
	serializer->write(_pmyDOMDocument, output);

	// Cleanup. 
	serializer->release();
	XMLString::release(&tempFilePath);
	delete formatTarget;
	output->release();
}

std::string MetaDatabase::getWorkingPath(){
#if defined(_WIN32) || defined(__WIN32__) 
	myWorkingPath = "C:/software/repos/STACCATO/model/";
#endif
#if defined(__linux__) 
	myWorkingPath = "/opt/software/repos/STACCATO/model/";
#endif
	return myWorkingPath;
};
