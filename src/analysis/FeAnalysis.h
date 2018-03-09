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
 * \file FeAnalysis.h
 * This file holds the class FeAnalysis which form the entire FE Analysis
 * \date 8/28/2017
 **************************************************************************************************/
#pragma once

#include <string>
#include <assert.h>
#include <math.h>

#include <MathLibrary.h>

class HMesh;
/********//**
 * \brief Class FeAnalysis holds and builds the whole Fe Analysis
 * Input to this class is a FeMetaDatabase and a HMesh object
 **************************************************************************************************/
class FeAnalysis{
public:
    /***********************************************************************************************
     * \brief Constructor
     * \param[in] _Hmesh reference to HMesh object
	 * \param[in] _FeMetaDatabase reference to FeMetaDatabase object
     * \author Stefan Sicklinger
     ***********/
	FeAnalysis(HMesh& _HMesh);
    /***********************************************************************************************
     * \brief Destructor
     *
     * \author Stefan Sicklinger
     ***********/
	virtual ~FeAnalysis(void);
private:
	/// HMesh object 
	HMesh *myHMesh;
	/// Stiffness Matrix
	MathLibrary::SparseMatrix<double> *AReal;
	MathLibrary::SparseMatrix<MKL_Complex16> *AComplex;
};
