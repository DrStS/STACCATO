#  Copyright &copy; 2016, Dr. Stefan Sicklinger, Munich
#
#  All rights reserved.
#
#  This file is part of STACCATO.
#
#  STACCATO is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  STACCATO is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with STACCATO.  If not, see http://www.gnu.org/licenses/.
#------------------------------------------------------------------------------------#
project (STACCATO)
#------------------------------------------------------------------------------------#
cmake_minimum_required (VERSION 3.2)
cmake_policy(VERSION 3.2)
set( CMAKE_VERBOSE_MAKEFILE off )
#------------------------------------------------------------------------------------#
# Check OS and host system
IF( (${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")) 
    SET(OS_LINUX_X86_64 TRUE)
    MESSAGE("STACCATO is going to be configured for Linux x86_64!")
ELSEIF( (${CMAKE_SYSTEM_NAME} MATCHES "Windows" AND ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")) 
    SET(OS_WIN_X86_64 TRUE)
    MESSAGE("STACCATO is going to be configured for Windows x64!")	
ELSE() 
    MESSAGE(FATAL_ERROR "STACCATO is supported on Linux x86_64 and Windows x64 only")	
ENDIF()
# Diag output
MESSAGE("Current source dir is" ${STACCATO_SOURCE_DIR})
#------------------------------------------------------------------------------------#
# 1. Options
# 2. Macros
# 3. Compiler
# 4. Subdirectories
# 5. Configure
# 6. MPI
#------------------------------------------------------------------------------------#
######################################################################################
# THESE GLOBAL VARIABLES ARE SET DURING THE CONFIGURATION
######################################################################################
#STACCATO_thirdparty_SOURCES
#STACCATO_thirdparty_INCLUDES
#STACCATO_SOURCES
#STACCATO_INCLUDES
######################################################################################
#1. Option
######################################################################################
option(USE_INTEL_COMPILER      "Use Intel Compilers C/C++"  ON )
option(USE_INTEL_MKL           "Use Intel MKL"  OFF )
option(USE_OCCT                "Use Opencascade"  ON )
option(USE_QT5                 "Use QT5"  ON )
option(USE_SIMULIA_ODB         "Use SIMULIA ODB"  ON )
######################################################################################
#2. Macros
######################################################################################
# Add STACCATO cmake modules
set(CMAKE_MODULE_PATH C:/software/repos/STACCATO/cmake/modules)
include(MacroOutOfSourceBuild)
include(MacroAppendGlobalVariable)
include(ExternalProject)
include(GetGitRevisionDescription)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
######################################################################################
#3. Compiler
######################################################################################

#------------------------------------------------------------------------------------#
# Avoid in source builds
macro_ensure_out_of_source_build("In source build is not allowed for ${PROJECT_NAME}. Please run CMake from your build directory. You may want to remove CMakeCache.txt.")
#------------------------------------------------------------------------------------#
# Check compilers, their versions
# Linux x86_64
IF (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" AND ${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
  # using Intel C/C++
  SET(USE_INTEL_COMPILERS_LINUX ON)
  MESSAGE("C compiler is ${CMAKE_C_COMPILER_ID} and has version ${CMAKE_C_COMPILER_VERSION} \nC++ compiler is ${CMAKE_CXX_COMPILER_ID} and has version ${CMAKE_CXX_COMPILER_VERSION} " ) 
ELSEIF (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
   SET(USE_GCC_COMPILERS_LINUX ON)
   MESSAGE("C compiler is ${CMAKE_C_COMPILER_ID} and has version ${CMAKE_C_COMPILER_VERSION} \nC++ compiler is ${CMAKE_CXX_COMPILER_ID} and has version ${CMAKE_CXX_COMPILER_VERSION} " ) 
ELSEIF (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC AND ${CMAKE_C_COMPILER_ID} STREQUAL MSVC)
  # using MSVC
  SET(USE_MICROSOFT_COMPILERS_WINDOWS ON)
  MESSAGE("C compiler is ${CMAKE_C_COMPILER_ID} and has version ${CMAKE_C_COMPILER_VERSION} \nC++ compiler is ${CMAKE_CXX_COMPILER_ID} and has version ${CMAKE_CXX_COMPILER_VERSION} " ) 
ELSE()
  MESSAGE(FATAL_ERROR "Your CXX compiler is ${CMAKE_CXX_COMPILER_ID} and your C compiler is ${CMAKE_C_COMPILER_ID} they are not supported by the build system yet")	
ENDIF()
#------------------------------------------------------------------------------------#
# Try to find Opencascade
IF (${USE_OCCT})
  find_package(OCCT REQUIRED)
  MESSAGE("OCCT status: ${OCCT_FOUND}")
  IF (${USE_MICROSOFT_COMPILERS_WINDOWS})
	add_definitions(-DWNT)
  ENDIF()
  IF (${USE_GCC_COMPILERS_LINUX})
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -lglut -lGL -lGLU -lX11")
  ENDIF()
ENDIF()
# Try to find Intel MKL 
IF (${USE_INTEL_MKL})
  find_package(MKL REQUIRED)
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
SET(CMAKE_PREFIX_PATH "C:/software/libs/QT/QT561/5.6/msvc2013_64/lib/cmake")
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
SET(CMAKE_PREFIX_PATH "/home/stefan/software/tools/Qt/5.7/gcc_64/lib/cmake")
ENDIF()
# Try to find QT5
IF (${USE_QT5})
  find_package(Qt5Widgets REQUIRED)
  MESSAGE("Qt5Widgets status: ${Qt5Widgets_FOUND}")
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTOUIC ON)
  set(CMAKE_AUTORCC ON)
  set(CMAKE_INCLUDE_CURRENT_DIR ON)
ENDIF()
# Try to find SIMULIA ODB API
IF (${USE_SIMULIA_ODB})
  find_package(SIMULIA REQUIRED)
  MESSAGE("SIMULIA status: ${SIMULIA_FOUND}")
  IF (${USE_MICROSOFT_COMPILERS_WINDOWS})
	SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /nologo /INCREMENTAL:NO /machine:AMD64 /STACK:20000000 /FIXED:NO /LARGEADDRESSAWARE" )
	SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0 /MD /TP /EHsc /DNDEBUG /DWIN32 /DTP_IP /D_CONSOLE /DNTI /DFLT_LIC /DOL_DOC /D__LIB__ /DHKS_NT /D_WINDOWS_SOURCE /DFAR= /D_WINDOWS /DABQ_WIN86_64")
  ENDIF()
  IF (${USE_GCC_COMPILERS_LINUX})
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  ENDIF()
ENDIF()
#------------------------------------------------------------------------------------#
IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
SET (CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -std=c99")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# Try to
ENDIF()
######################################################################################
#4. Configure
######################################################################################
######################################################################################
#5. Subdirectories THE ORDER IS IMPORTANT, BECAUSE THEY ARE COMPILED IN THAT ORDER
######################################################################################
add_subdirectory(src)

