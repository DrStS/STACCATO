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
/*************************************************************************************************
* \file SettingSubject.h
* This file holds the class SettingSubject which form the Subject of Observer Pattern
* \date 8/28/2017
**************************************************************************************************/
#pragma once

#include "VisualizerSettingObserver.h"
#include <vector>

class VisualizerSettingSubject {
public:

	virtual void attachObserver(VisualizerSettingObserver* observer) {
		this->observers.push_back(observer);
	}

	virtual void notify() {
		for (std::vector<VisualizerSettingObserver*>::iterator it = observers.begin();
			it != observers.end();
			++it) {
			(*it)->update();
		}
	}

	// attribute
protected:
	std::vector<VisualizerSettingObserver*> observers;
};
