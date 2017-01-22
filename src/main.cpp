/*           This file has been prepared for Doxygen automatic documentation generation.          */
/***********************************************************************************************//**
 * \mainpage
 * \section LICENSE
 *  Copyright &copy; 2016, Dr. Stefan Sicklinger, Munich \n
 *  All rights reserved. \n
 *
 *  This file is part of STACCATO.
 *
 *  STACCATO is free software: you can redistribute it and/or modify \n
 *  it under the terms of the GNU General Public License as published by \n
 *  the Free Software Foundation, either version 3 of the License, or \n
 *  (at your option) any later version. \n
 *
 *  STACCATO is distributed in the hope that it will be useful, \n
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of \n
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the \n
 *  GNU General Public License for more details. \n
 *
 *  You should have received a copy of the GNU General Public License \n
 *  along with STACCATO.  If not, see <a href="http://www.gnu.org/licenses/">http://www.gnu.org/licenses/</a>.
 *
 *  Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it with Intel Math Kernel Libraries(MKL) 
 * (or a modified version of that library), containing parts covered by the terms of the license of the MKL, 
 * the licensors of this Program grant you additional permission to convey the resulting work. 
 *
 * \section DESCRIPTION
 *  This is the main file of STACCATO
 *
 *
 * \section COMPILATION
 *  export CC=icc
 *  export CXX=icpc
 *  cd build
 *  cmake ..
 * There are the following make targets available:
 * - make (compilation and linking)
 * - make clean (remove object files and executable including all folders)
 * - make doc (generates documentation) html main file is  /STACCATO/doc/html/index.html
 * - make cleandoc (removes documentation)
 *
 * \section HOWTO
 * Please find all further information on
 * <a href="https://github.com/DrStS/STACCATO">STACCATO Project</a>
 *
 *
 * <EM> Note: The Makefile suppresses per default all compile and linking command output to the terminal.
 *       You may enable this information by make VEREBOSE=1</EM>
 *
 *
 *
 **************************************************************************************************/
/***********************************************************************************************//**
 * \file main.cpp
 * This file holds the main function of STACCATO.
 * \author Stefan Sicklinger
 * \date 4/2/2016
 * \version alpha
 **************************************************************************************************/

#include <iostream>
//Qt5
#include <QApplication>
//OCC7
#include <OSD_Environment.hxx>
//USER
#include <StartWindow.h>



int main(int argc, char **argv) {
  
    //cout << "Hello STACCATO is fired up!" << endl;
    //cout << "GIT: " << STACCATO::AuxiliaryParameters::gitSHA1 << endl;


	//TODO
	// statusBar coordinate
	// 2D mode
	// interactive points 2D
	// interactive lines 2D

    QApplication a(argc, argv);
    StartWindow* mSW= new StartWindow();
	mSW->show();
    return a.exec();
}
