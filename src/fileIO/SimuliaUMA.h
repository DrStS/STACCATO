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
* \file SimuliaUMA.h
* This file holds the class SimuliaUMA which adds the capability to read Abaqus SIM files
* \date 2/1/2018
**************************************************************************************************/

#ifndef SIMULIAUMA_H_
#define SIMULIAUMA_H_

#include <string>
#include <assert.h>
#include <Reader.h>
#include <vector>

class HMesh;
/********//**
* \brief This handles the output handling with Abaqus SIM
**************************************************************************************************/
class SimuliaUMA :public Reader {
public:
	/***********************************************************************************************
	* \brief Constructor
	* \param[in] _filePath string which holds the path to the sim file
	* \author Harikrishnan Sreekumar
	***********/
	SimuliaUMA(std::string _fileName, HMesh& _hMesh);
	/***********************************************************************************************
	* \brief Destructor
	*
	* \author Harikrishnan Sreekumar
	***********/
	virtual ~SimuliaUMA(void);
	/***********************************************************************************************
	* \brief Open die sim file
	* \param[in] _filePath string which holds the path to the sim file
	* \author Harikrishnan Sreekumar
	***********/
	void openFile();

private:
	std::string myFileName;
	/// HMesh object 
	HMesh *myHMesh;
	/// Node Label and DoF vector
	std::vector<std::vector<int>> simNodeMap;
	/// Number of nodes
	int numNodes;
	/// Number of DoFs per Node
	int numDoFperNode;
};


#endif /* SIMULIAUMA_H_ */
