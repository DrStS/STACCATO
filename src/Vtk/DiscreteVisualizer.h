/*  Copyright &copy; 2018, Stefan Sicklinger, Munich
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
* \file DiscreteVisualizer.h
* This file holds the class DiscreteVisualizer; Base class for a Discrete Visualization
* \date 2/19/2018
**************************************************************************************************/

#ifndef DISCRETEVISUALIZER_H_
#define DISCRETEVISUALIZER_H_

#include "HMesh.h"

/*************************************************************************************************
* \brief Class DiscreteVisualizer
**************************************************************************************************/

class DiscreteVisualizer {
public:
	DiscreteVisualizer ();
	~DiscreteVisualizer ();
	void setHMesh(HMesh &_hMesh) { myHMesh = &_hMesh;	}
	HMesh* getHMesh() { return myHMesh; }

protected:
	/// HMesh object 
	HMesh *myHMesh;
};

#endif /* DISCRETEVISUALIZER_H_ */