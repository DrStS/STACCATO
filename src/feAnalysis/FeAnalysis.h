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
 * Input to this class is a FeMetaDatabase and a HMesh object
 * \date 8/28/2017
 **************************************************************************************************/

#ifndef FEANALYSIS_H_
#define FEANALYSIS_H_

#include <string>
#include <assert.h>
#include <math.h>

class HMesh;
class FeMetaDatabase;
/********//**
 *
 **************************************************************************************************/
class FeAnalysis{
public:
    /***********************************************************************************************
     * \brief Constructor
     * \param[in] _Hmesh reference to HMesh object
	 * \param[in] _FeMetaDatabase reference to FeMetaDatabase object
     * \author Stefan Sicklinger
     ***********/
	FeAnalysis(HMesh& _HMesh, FeMetaDatabase& _FeMetaDatabase);
    /***********************************************************************************************
     * \brief Destructor
     *
     * \author Stefan Sicklinger
     ***********/
	virtual ~FeAnalysis(void);
private:
	/// HMesh object 
	HMesh *myHMesh;
	/// HMesh object 
	FeMetaDatabase *myFeMetaDatabase;

	/***********************************************************************************************
	* \brief Evalute derivative of local shape functions for bi-linear element
	* \param[in] xi
	* \param[in] eta
	* \param[out] d_N_d_xi_eta
	* \author Stefan Sicklinger
	***********/
	void computeShapeFuncOfQuad(const double xi, const double eta, double *d_N_d_xi_eta)
	{
		d_N_d_xi_eta[0] = -0.25 + 0.25* xi; //d_N_d_xi
		d_N_d_xi_eta[1] = +0.25 - 0.25* xi;
		d_N_d_xi_eta[2] = +0.25 + 0.25* xi;
		d_N_d_xi_eta[3] = -0.25 - 0.25* xi;
		d_N_d_xi_eta[4] = -0.25 - 0.25*eta; //d_N_d_eta
		d_N_d_xi_eta[5] = -0.25 + 0.25*eta;
		d_N_d_xi_eta[6] = +0.25 + 0.25*eta;
		d_N_d_xi_eta[7] = +0.25 - 0.25*eta;
	}
};


#endif /* FEANALYSIS_H_ */
