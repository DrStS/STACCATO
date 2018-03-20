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
#include "Material.h"

#include "MetaDatabase.h"
#include <iostream>

Material::Material() {	
}

Material::Material(std::string _materialName, int _partId) {
	int flag = 0;
	for (int j = 0; j < MetaDatabase::getInstance()->xmlHandle->PARTS().begin()->PART()[_partId].MATERIALS().begin()->MATERIAL().size(); j++) {
		if (std::string(MetaDatabase::getInstance()->xmlHandle->PARTS().begin()->PART()[_partId].MATERIALS().begin()->MATERIAL().at(j).Name()->c_str()) == _materialName) {
			myYoungsModulus = std::stod(MetaDatabase::getInstance()->xmlHandle->PARTS().begin()->PART()[_partId].MATERIALS().begin()->MATERIAL().at(j).E()->data());
			myPoissonsRatio = std::stod(MetaDatabase::getInstance()->xmlHandle->PARTS().begin()->PART()[_partId].MATERIALS().begin()->MATERIAL().at(j).nu()->data());
			myDensity = std::stod(MetaDatabase::getInstance()->xmlHandle->PARTS().begin()->PART()[_partId].MATERIALS().begin()->MATERIAL().at(j).rho()->data());
			myDampingParameter = std::stod(MetaDatabase::getInstance()->xmlHandle->PARTS().begin()->PART()[_partId].MATERIALS().begin()->MATERIAL().at(j).eta()->data());
			flag = 1;
			break;
		}
	}
	if (flag == 0)
		std::cerr << ">> Error while finding Material: MATERIAL " << _materialName << " not Found.\n";
	
}

Material::~Material() {
}