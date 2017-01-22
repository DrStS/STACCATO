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
#include "SimuliaODB.h"
#include "Message.h"


/// SIMULIA includes
#include <odb_API.h>
#include <odb_Coupling.h>
#include <odb_MPC.h>
#include <odb_ShellSolidCoupling.h>


SimuliaODB::SimuliaODB() {
	odb_initializeAPI();
}

SimuliaODB::~SimuliaODB() {
	odb_finalizeAPI();
}

void SimuliaODB::openODBFile(std::string _obdFilePath) {
	try {
		infoOut << "Open OBD file: " << _obdFilePath << std::endl;
		odb_Odb& odb = openOdb(odb_String(_obdFilePath.c_str()));
		infoOut << odb.name().CStr() << " '__________" << std::endl;
		infoOut << "analysisTitle '" << odb.analysisTitle().CStr() << "'" << std::endl;
		infoOut << "description '" << odb.description().CStr() << "'" << std::endl;
	}
	catch (odb_BaseException& exc) {
		errorOut << "odbBaseException caught" << std::endl;
		errorOut << "Abaqus error message: " << exc.UserReport().CStr() << std::endl;
	}
	catch (...) {
		errorOut << "Unknown Exception." << std::endl;
	}
}

