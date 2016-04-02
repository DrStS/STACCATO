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
cmake_minimum_required (VERSION 2.8.10)
set( CMAKE_VERBOSE_MAKEFILE on )
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
option(USE_INTEL_MKL           "Use Intel MKL"  ON )
######################################################################################
#2. Macros
######################################################################################
# Add STACCATO cmake modules
set(CMAKE_MODULE_PATH ${STACCATO_SOURCE_DIR}/cmake/modules)
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
IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" AND "${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C/C++
  SET(USE_INTEL_COMPILERS_LINUX ON)
  MESSAGE("C compiler is ${CMAKE_C_COMPILER_ID} and has version ${CMAKE_C_COMPILER_VERSION} \nC++ compiler is ${CMAKE_CXX_COMPILER_ID} and has version ${CMAKE_CXX_COMPILER_VERSION} " ) 
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
   SET(USE_GCC_COMPILERS_LINUX ON)
   MESSAGE("C compiler is ${CMAKE_C_COMPILER_ID} and has version ${CMAKE_C_COMPILER_VERSION} \nC++ compiler is ${CMAKE_CXX_COMPILER_ID} and has version ${CMAKE_CXX_COMPILER_VERSION} " ) 
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
  # using MSVC
  SET(USE_USE_MICROSOFT_COMPILERS_WINDOWS ON)
  MESSAGE("C compiler is ${CMAKE_C_COMPILER_ID} and has version ${CMAKE_C_COMPILER_VERSION} \nC++ compiler is ${CMAKE_CXX_COMPILER_ID} and has version ${CMAKE_CXX_COMPILER_VERSION} " ) 
ELSE()
  MESSAGE(FATAL_ERROR "Your CXX compiler is ${CMAKE_CXX_COMPILER_ID} and your C compiler is ${CMAKE_C_COMPILER_ID} they are not supported by the build system yet")	
ENDIF()
#------------------------------------------------------------------------------------#
# Try to find Intel MKL 
IF (${USE_INTEL_MKL})
  find_package(MKL REQUIRED)
  add_definitions(-DUSE_INTEL_MKL)
ENDIF()
#------------------------------------------------------------------------------------#
######################################################################################
#4. Configure
######################################################################################
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_describe(GIT_TAG --tags)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/tools/AuxiliaryParameters.cpp.in" 
"${CMAKE_CURRENT_SOURCE_DIR}/src/tools/AuxiliaryParameters.cpp" @ONLY)
######################################################################################
#5. Subdirectories THE ORDER IS IMPORTANT, BECAUSE THEY ARE COMPILED IN THAT ORDER
######################################################################################
add_subdirectory(src)

