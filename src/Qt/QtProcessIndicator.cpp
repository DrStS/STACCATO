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
#include <QtProcessIndicator.h>
#include <assert.h> 

//QT5
#include <QtWidgets>

//OCCT 7
#include <TCollection_HAsciiString.hxx>
#include <Message_ProgressIndicator.hxx>
/*! Creates a widget using specified paramters to initialize QProgressIndicator.
\a theMin and \a theMax are also used to set the range for \a this progress
indicator.
*/
QtProcessIndicator::QtProcessIndicator(QWidget* theParent,
	int theMinVal, int theMaxVal,
	Qt::WindowFlags theFlags)
{
	assert(theMinVal < theMaxVal);
	myProgress = new QProgressDialog(theParent, theFlags);
	myProgress->setWindowModality(Qt::WindowModal);
	myProgress->setMinimum(theMinVal);
	myProgress->setMaximum(theMaxVal);
	myProgress->setMinimumDuration(500); //the dialog will pop up if operation takes >500ms

	SetScale(theMinVal, theMaxVal, 1); //synch up ranges between Qt and Open CASCADE
}

/*! Destroys the associated progress dialog.*/
QtProcessIndicator::~QtProcessIndicator()
{
	if (myProgress) {
		delete myProgress;
		myProgress = 0;
	}
}

/*! Updates visual presentation according to currently achieved progress.
The text label is updated according to the name of a current step.

Always returns TRUE to signal that the presentation has been updated.
*/
Standard_Boolean QtProcessIndicator::Show(const Standard_Boolean theForce)
{
	Handle(TCollection_HAsciiString) aName = GetScope(1).GetName(); //current step
	if (!aName.IsNull())
		myProgress->setLabelText(aName->ToCString());

	Standard_Real aPc = GetPosition(); //always within [0,1]
	int aVal = myProgress->minimum() + aPc *
		(myProgress->maximum() - myProgress->minimum());
	myProgress->setValue(aVal);
	QApplication::processEvents(); //to let redraw and keep GUI responsive

	return Standard_True;
}

/*! Returns True if the user has clicked the Cancel button in QProgressDialog.
*/
Standard_Boolean QtProcessIndicator::UserBreak()
{
	return myProgress->wasCanceled();
}
