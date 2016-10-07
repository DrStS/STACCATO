/*  Copyright &copy; 2016, Stefan Sicklinger, Munich
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
* \file ProcessIndicator.h
* This file holds the class of StartWindow.
* \date 10/7/2016
**************************************************************************************************/
#ifndef QTPROCESSINDICATOR_H
#define QTPROCESSINDICATOR_H

// QT5
#include <QProgressDialog>
// OCCT
#include <Standard_Macro.hxx>
#include <Message_ProgressIndicator.hxx>


class QtProcessIndicator : public Message_ProgressIndicator
{
public:
	//! Creates an object.
	Standard_EXPORT QtProcessIndicator(QWidget* theParent,
		int theMinVal = 0, int theMaxVal = 100, Qt::WindowFlags theFlags = 0);

	//! Deletes the object.
	Standard_EXPORT virtual ~QtProcessIndicator();

	//! Updates presentation of the object.
	Standard_EXPORT virtual Standard_Boolean Show(const Standard_Boolean theForce);

	//! Returns True if the user has signaled to cancel the process.
	Standard_EXPORT virtual Standard_Boolean UserBreak();


protected:
	QProgressDialog * myProgress;

public:
//	DEFINE_STANDARD_RTTIEXT(QtProcessIndicator, void)

};

#endif // QTPROCESSINDICATOR_H 