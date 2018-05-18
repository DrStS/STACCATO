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
 * \file Message.h
 * This file holds the class Message
 * \date 1/10/2017
 **************************************************************************************************/
#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>


//QT5
#ifdef USE_QT5
#include <QTextEdit>
#endif // USE_QT5
#undef ERROR
// forward declaration
class QTextEdit;

/********//**
 * \brief This manages the output functions for writing to the teminal
 **************************************************************************************************/
class Message : public std::ostream {
public:
    /// Severity flags for Messages
    enum OutputLevel {
        ERROR,   /// Error
        WARNING, /// Warning of possible problem
        INFO,    /// Info for user
        DEBUG    /// Debugging information in event of error
    };
    ///This variable is shared between all objects of type message
    static OutputLevel userSetOutputLevel;
    /***********************************************************************************************
     * \brief Constructor
     * \param[in] _outputLevel type enum OutputLevel set by user and stored in MetaDataBase
     * \param[in] _outputStream pointer to the ouputStream object
     * \author Stefan Sicklinger
     ***********/
	Message(OutputLevel _outputLevel) : std::ostream(new MessageBuffer(_outputLevel, (*this))) {
		textOutput = NULL;
	}
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	Message() : std::ostream(new MessageBuffer(Message::DEBUG, (*this))) {}
    /***********************************************************************************************
     * \brief Destructor
     *
     * \author Stefan Sicklinger
     ***********/
	virtual ~Message() { 
	 //	delete rdbuf(); 
	}
	/***********************************************************************************************
	* \brief Set the QTextEdit reference for output
	* \param[in] _textOutput pointer to QTextEdit
	* \author Stefan Sicklinger
	***********/
	void setQTextEditReference(QTextEdit* _textOutput) {
		textOutput = _textOutput;
	}
private:
	/// textOutput QT QTextEdit
	QTextEdit* textOutput;
	/***********************************************************************************************
	* \brief Class MessageBuffer
	*
	* \author Stefan Sicklinger
	***********/
	class MessageBuffer : public std::stringbuf {
	private:
		/// outputLevel enum
		OutputLevel outputLevel;
		/// Access parent -> Message
		Message& myParent;

	public:
		/***********************************************************************************************
		* \brief Constructor
		* \param[in] _outputLevel type enum OutputLevel set by user and stored in MetaDataBase
		* \param[in] _myParent reference to parent Message class
		* \author Stefan Sicklinger
		***********/
		MessageBuffer(OutputLevel _outputLevel, Message& _myParent) : outputLevel(_outputLevel), myParent(_myParent){
		}
		/***********************************************************************************************
		* \brief Destructor
		*
		* \author Stefan Sicklinger
		***********/
		~MessageBuffer() { pubsync(); }
		/***********************************************************************************************
		* \brief sync redefine of std::stringbuf
		* 
		* \author Stefan Sicklinger
		***********/
		int sync() {
			if (userSetOutputLevel >= outputLevel){
				if (myParent.textOutput != NULL){

					std::string echoString;
#ifdef USE_QT5
					if (outputLevel == ERROR){
						myParent.textOutput->setTextColor(QColor("red"));
					}
					else if (outputLevel == WARNING){
						myParent.textOutput->setTextColor(QColor("orange"));
					}
					else if (outputLevel == INFO){
						myParent.textOutput->setTextColor(QColor("cyan"));
					}
					else if (outputLevel == DEBUG){
						myParent.textOutput->setTextColor(QColor("black"));
					}
					else{
						assert(false);
					}
					myParent.textOutput->insertPlainText(QString::fromStdString(str()));
#endif // USE_QT5
				}
				str("");
			}
			return std::cout ? 0 : -1;
		}

	};
};



extern Message infoOut;
extern Message debugOut;
extern Message errorOut;
extern Message warningOut;


/**************************************************************************************************!
  Forwards \a string argument to infoOut(string) Message object;
***********/
#define INFO_OUT(string) /*
 */                infoOut(string)
/**************************************************************************************************!
  Forwards \a string argument to debugOut(string) Message object;
***********/
#define DEBUG_OUT(string) /*
 */                debugOut(string)
/**************************************************************************************************!
  Forwards \a string argument to errorOut(string) Message object;
***********/
#define ERROR_OUT(string) /*
 */                errorOut(string)
/**************************************************************************************************!
  Forwards \a string argument to warning(string) Message object;
***********/
#define WARNING_OUT(string) /*
 */                warningOut(string)
