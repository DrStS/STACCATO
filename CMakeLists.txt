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
#cmake_policy(SET CMP0074 NEW)
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
option(USE_INTEL_COMPILER      "Use Intel Compilers C/C++"  OFF )
option(USE_INTEL_MKL           "Use Intel MKL"  ON )
option(USE_OCCT                "Use Opencascade"  OFF )
option(USE_VTK                 "Use VTK"  OFF )
option(USE_QT5                 "Use QT5"  OFF )
option(USE_SIMULIA_ODB         "Use SIMULIA ODB"  ON )
option(USE_SIMULIA_UMA         "Use SIMULIA UMA"  ON )
option(USE_XSD                 "Use XSD with XERCES"  ON )
option(USE_HDF5                "Use HDF5"  ON )
option(STACCATO_COMMANDLINE    "Build STACCATO COMMANDLINE"  ON)
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
IF (${STACCATO_COMMANDLINE})
  MESSAGE("Build STACCATO commandline only!")
  add_definitions(-DSTACCATO_COMMANDLINE_ON)
  option(USE_OCCT                "Use Opencascade"  OFF )
  option(USE_VTK                 "Use VTK"  OFF )
  option(USE_QT5                 "Use QT5"  OFF )
ENDIF()
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
  add_definitions(-DOCCT_ON)
  MESSAGE("OCCT status: ${OCCT_FOUND}")
  IF (${USE_MICROSOFT_COMPILERS_WINDOWS})
	add_definitions(-DWNT)
  ENDIF()
  IF (${USE_GCC_COMPILERS_LINUX})
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -lglut -lGL -lGLU -lX11")
  ENDIF()
ENDIF()
# Try to find VTK
IF (${USE_VTK})
 add_definitions(-DUSE_VTK)
 SET(VTK_BASE_DIR "C:/software/libs/VTK-8.1.1")
 IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
   SET(VTK_DIR "${VTK_BASE_DIR}/build")
 ENDIF()
 IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
     SET(VTK_DIR "/opt/software/libs/VTK-8.1.1/gcc/lib/cmake/vtk-8.1")
 ENDIF()
  MESSAGE("VTK_DIR: ${VTK_DIR}")
  find_package(VTK REQUIRED)
  INCLUDE(${VTK_USE_FILE})
  LIST(REMOVE_ITEM VTK_INCLUDE_DIRS ${VTK_BASE_DIR}/ThirdParty/hdf5)
  LIST(REMOVE_ITEM VTK_INCLUDE_DIRS ${VTK_BASE_DIR}/build/ThirdParty/hdf5)
  LIST(REMOVE_ITEM VTK_INCLUDE_DIRS ${VTK_BASE_DIR}/build/ThirdParty/hdf5/vtkhdf5)
  LIST(REMOVE_ITEM VTK_INCLUDE_DIRS ${VTK_BASE_DIR}/ThirdParty/hdf5/vtkhdf5/hl/src)
  LIST(REMOVE_ITEM VTK_INCLUDE_DIRS ${VTK_BASE_DIR}/ThirdParty/hdf5/vtkhdf5/src)
  MESSAGE("VTK status: ${VTK_FOUND} ")#${VTK_INCLUDE_DIRS}
  IF (${USE_MICROSOFT_COMPILERS_WINDOWS})
  ENDIF()
  IF (${USE_GCC_COMPILERS_LINUX})
  ENDIF()
ENDIF()
# Try to find Intel MKL 
IF (${USE_INTEL_MKL})
  find_package(MKL REQUIRED)
	#add_definitions(-DMKL_ILP64)
	add_definitions(-DUSE_INTEL_MKL)
  MESSAGE("MKL status: ${MKL_FOUND}")
  # Get OpenMP activated and use Intel link library
	    IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
		    IF(USE_INTEL_COMPILER)
		      set (CMAKE_C_FLAGS   "/MD /Qopenmp")
		      set (CMAKE_CXX_FLAGS "/MD /Qopenmp")
		    ELSE()
		    	set (CMAKE_C_FLAGS   "/MD /openmp")
		      set (CMAKE_CXX_FLAGS "/MD /openmp")
		      set (CMAKE_EXE_LINKER_FLAGS "/nodefaultlib:vcomp ${MKL_OMP_LIBRARY}")
		    ENDIF()
      ENDIF()
      IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
	    	set (CMAKE_C_FLAGS   "-fopenmp")
	      set (CMAKE_CXX_FLAGS "-fopenmp")
	      set (CMAKE_EXE_LINKER_FLAGS "-L${MKL_OMP_LIBRARY}")
      ENDIF()
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
SET(CMAKE_PREFIX_PATH "C:/software/libs/QT/Qt5101/5.10.1/msvc2017_64/lib/cmake")
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
SET(CMAKE_PREFIX_PATH "/opt/software/libs/Qt/5.10.1/gcc_64/lib/cmake")
ENDIF()
# Try to find QT5
IF (${USE_QT5})
  add_definitions(-DUSE_QT5)
  find_package(Qt5Widgets REQUIRED)
  MESSAGE("Qt5Widgets status: ${Qt5Widgets_FOUND}")
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTOUIC ON)
  set(CMAKE_AUTORCC ON)
  set(CMAKE_INCLUDE_CURRENT_DIR ON)
ENDIF()
# Try to find SIMULIA ODB API
IF (${USE_SIMULIA_ODB})
  add_definitions(-DUSE_SIMULIA_ODB_API)
  find_package(SIMULIA REQUIRED)
  MESSAGE("SIMULIA status: ${SIMULIA_FOUND}")
  IF (${USE_MICROSOFT_COMPILERS_WINDOWS})
		SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /nologo /INCREMENTAL:NO /machine:AMD64 /STACK:20000000 /FIXED:NO /LARGEADDRESSAWARE" )
		SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0 /MD /TP /EHsc /DNDEBUG /DWIN32 /DTP_IP /D_CONSOLE /DNTI /DFLT_LIC /DOL_DOC /D__LIB__ /DHKS_NT /D_WINDOWS_SOURCE /DFAR= /D_WINDOWS /DABQ_WIN86_64")
	ENDIF()
  IF (${USE_GCC_COMPILERS_LINUX})
    SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC -Wl,-Bdynamic -Wl,--add-needed" )
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -w -Wno-deprecated -DTYPENAME=typename -D_LINUX_SOURCE -DABQ_LINUX -DABQ_LNX86_64 -DSMA_GNUC -DFOR_TRAIL -DHAS_BOOL -DASSERT_ENABLED -D_BSD_TYPES -D_BSD_SOURCE -D_GNU_SOURCE -D_POSIX_SOURCE -D_XOPEN_SOURCE_EXTENDED -D_XOPEN_SOURCE -DHAVE_OPENGL -DHKS_OPEN_GL -DGL_GLEXT_PROTOTYPES -DMULTI_THREADING_ENABLED -D_REENTRANT -DABQ_MPI_SUPPORT -DBIT64 -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64")
  ENDIF()
ENDIF()
IF (${USE_SIMULIA_UMA})
  add_definitions(-DUSE_SIMULIA_UMA_API)
ENDIF()
# Try to find XSD with XERCES
IF (${USE_XSD})
  find_package(XSDXERCES REQUIRED)
  MESSAGE("XSDXERCES status: ${XSDXERCES_FOUND}")
  IF (${USE_MICROSOFT_COMPILERS_WINDOWS})
  	SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	ENDIF()
  IF (${USE_GCC_COMPILERS_LINUX})
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  ENDIF()
ENDIF()
# 
# Try to find HDF5
IF (${USE_HDF5})
  add_definitions(-DUSE_HDF5)
  find_package(HDF5 REQUIRED)
  MESSAGE("HDF5 status: ${HDF5_FOUND}")
  IF (${USE_MICROSOFT_COMPILERS_WINDOWS})
  	SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	ENDIF()
  IF (${USE_GCC_COMPILERS_LINUX})
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  ENDIF()
ENDIF()

#------------------------------------------------------------------------------------#
# Global compiler flags
IF (${USE_GCC_COMPILERS_LINUX})
set(CMAKE_C_STANDARD 99) 
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#SET (CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ")
#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
ENDIF()
IF (${USE_MICROSOFT_COMPILERS_WINDOWS})
#SET (CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Tp")
#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Tp")
ENDIF()
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
add_subdirectory(srcPerformanceTest)
