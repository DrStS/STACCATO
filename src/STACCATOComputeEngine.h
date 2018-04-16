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
/***********************************************************************************************//**
* \file STACCATOComputeEngine.h
* This file holds the class of ComputeEngine.
* \date 3/14/2018
**************************************************************************************************/
#pragma once
#include <string>

/********//**
* \brief Class ComputeEngine the core of STACCATO
***********/
class HMesh;
class OutputDatabase;
class STACCATOComputeEngine {

public:
	/***********************************************************************************************
	* \brief Constructor
	* \param[in] file name of xml file to call singelton constructor of metadatabase
	* \author Stefan Sicklinger
	***********/
	STACCATOComputeEngine(std::string _xmlFileName);
	/***********************************************************************************************
	* \brief Destructor
	* \author Stefan Sicklinger
	***********/
	~STACCATOComputeEngine();
	/***********************************************************************************************
	* \brief prepare compute engine
	* \author Stefan Sicklinger
	***********/
	void prepare(void);
	/***********************************************************************************************
	* \brief compute engine
	* \author Stefan Sicklinger
	***********/
	void compute(void);
	/***********************************************************************************************
	* \brief clean compute engine free memory
	* \author Stefan Sicklinger
	***********/
	void clean(void);
	/***********************************************************************************************
	* \brief get HMesh handle
	* \author Stefan Sicklinger
	***********/
	HMesh & getHMesh() { return *myHMesh; }
	/***********************************************************************************************
	* \brief get OutputDatabase handle
	* \author Stefan Sicklinger
	***********/
	OutputDatabase * getOutputDatabase(void);
private:
	 HMesh *myHMesh;
};