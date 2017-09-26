# A cmake module to find SIMULIA ODB API
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
set(SIMULIAROOT_DIR $ENV{SIMULIAROOT})
IF (NOT SIMULIAROOT_DIR)
   IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
       IF (EXISTS "/home/stefan/software/libs/SIMULIA")
          set(SIMULIAROOT_DIR "/home/stefan/software/libs/SIMULIA")
       ENDIF(EXISTS "/home/stefan/software/libs/SIMULIA")
   ENDIF()
   IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
       IF (EXISTS "C:/software/libs/SIMULIA")
          set(SIMULIAROOT_DIR "C:/software/libs/SIMULIA")
       ENDIF()
   ENDIF()
ENDIF ()
message("SIMULIAROOT_DIR is: ${SIMULIAROOT_DIR}")
#------------------------------------------------------------------------------------#
# Stage 1: find the include directory
#------------------------------------------------------------------------------------#
IF (NOT SIMULIA_INCLUDE_DIR)
    find_path(SIMULIA_INCLUDE_DIR
    odb_API.h
    HINTS ${SIMULIAROOT_DIR}
    PATH_SUFFIXES include
    )  
	IF(SIMULIA_INCLUDE_DIR MATCHES "SIMULIA_INCLUDE_DIR-NOTFOUND")
     unset(SIMULIA_INCLUDE_DIR CACHE)
	ENDIF()
ENDIF ()
message("SIMULIA_INCLUDE_DIR is: ${SIMULIA_INCLUDE_DIR}")
#------------------------------------------------------------------------------------#
# Stage 2: find the lib directory
#------------------------------------------------------------------------------------#	
if (NOT SIMULIA_LIB_DIR)
  if (SIMULIAROOT_DIR)
    IF (OS_LINUX_X86_64)
	   set(EXPECT_SIMULIA_LIBPATH "${SIMULIAROOT_DIR}/lib/")
    endif ()	
    IF (OS_WIN_X86_64)
	   set(EXPECT_SIMULIA_LIBPATH "${SIMULIAROOT_DIR}/lib/")
    endif ()	
    if (IS_DIRECTORY ${EXPECT_SIMULIA_LIBPATH})
      set(SIMULIA_LIB_DIR ${EXPECT_SIMULIA_LIBPATH})
    endif ()
  endif ()
endif ()
message("SIMULIA_LIB_DIR is: ${SIMULIA_LIB_DIR}")
#------------------------------------------------------------------------------------#
# Stage 3: set compiler and linker flags
#------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------#
# Stage 4: find the libraries
#------------------------------------------------------------------------------------#	
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
#set (CMAKE_FIND_LIBRARY_SUFFIXES .a)
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
#set (CMAKE_FIND_LIBRARY_SUFFIXES .dll)
# lib needs to be used even for dynamic linking on windows 
# http://stackoverflow.com/questions/3250467/what-is-inside-lib-file-of-static-library-statically-linked-dynamic-library-an
ENDIF()
if (SIMULIA_LIB_DIR)
find_library(SIMULIA_LIBRARY_ABQSMAOdbDdbOdb          ABQSMAOdbDdbOdb                        ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMAOdbApi             ABQSMAOdbApi                           ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMAOdbCore            ABQSMAOdbCore                          ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMAOdbCoreGeom        ABQSMAOdbCoreGeom                      ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMAOdbAttrEO          ABQSMAOdbAttrEO                        ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMAAbuBasicUtils      ABQSMAAbuBasicUtils                    ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMABasShared          ABQSMABasShared                        ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMABasCoreUtils       ABQSMABasCoreUtils                     ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMABasAlloc           ABQSMABasAlloc                         ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMAAbuGeom            ABQSMAAbuGeom                          ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMARomDiagEx          ABQSMARomDiagEx                        ${SIMULIA_LIB_DIR})
find_library(SIMULIA_LIBRARY_ABQSMASimInterface       ABQSMASimInterface                     ${SIMULIA_LIB_DIR})
 
find_library(SIMULIA_LIBRARY_standardB                standardB                              ${SIMULIA_LIB_DIR}) 
find_library(SIMULIA_LIBRARY_ABQSMASspUmaCore         ABQSMASspUmaCore                       ${SIMULIA_LIB_DIR}) 

#find_library(SIMULIA_LIBRARY_oldnames                 oldnames                              ${SIMULIA_LIB_DIR})
#find_library(SIMULIA_LIBRARY_user32                   user32                                ${SIMULIA_LIB_DIR})
#find_library(SIMULIA_LIBRARY_ws2_32                   ws2_32                                ${SIMULIA_LIB_DIR})
#find_library(SIMULIA_LIBRARY_netapi32                 netapi32                              ${SIMULIA_LIB_DIR})
#find_library(SIMULIA_LIBRARY_advapi32                 advapi32                              ${SIMULIA_LIB_DIR})

#find_library(SIMULIA_LIBRARY_msvcrt                   msvcrt                                ${SIMULIA_LIB_DIR})
#find_library(SIMULIA_LIBRARY_vcruntime                vcruntime                             ${SIMULIA_LIB_DIR})
#find_library(SIMULIA_LIBRARY_ucrt                     ucrt                                  ${SIMULIA_LIB_DIR})



IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set (SIMULIA_LIBRARIES  ${SIMULIA_LIBRARY_ABQSMAOdbDdbOdb} ${SIMULIA_LIBRARY_ABQSMAOdbApi} ${SIMULIA_LIBRARY_ABQSMAOdbCore} ${SIMULIA_LIBRARY_ABQSMAOdbCoreGeom} ${SIMULIA_LIBRARY_ABQSMAOdbAttrEO} ${SIMULIA_LIBRARY_ABQSMAAbuBasicUtils} ${SIMULIA_LIBRARY_ABQSMABasShared} ${SIMULIA_LIBRARY_ABQSMABasCoreUtils} ${SIMULIA_LIBRARY_ABQSMABasAlloc} ${SIMULIA_LIBRARY_ABQSMAAbuGeom} ${SIMULIA_LIBRARY_ABQSMARomDiagEx} ${SIMULIA_LIBRARY_ABQSMASimInterface} ${SIMULIA_LIBRARY_standardB} ${SIMULIA_LIBRARY_ABQSMASspUmaCore} ${SIMULIA_LIBRARY_oldnames} ${SIMULIA_LIBRARY_user32} ${SIMULIA_LIBRARY_ws2_32} ${SIMULIA_LIBRARY_netapi32} ${SIMULIA_LIBRARY_advapi32})
  ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    set (SIMULIA_LIBRARIES  ${SIMULIA_LIBRARY_ABQSMAOdbDdbOdb} ${SIMULIA_LIBRARY_ABQSMAOdbApi} ${SIMULIA_LIBRARY_ABQSMAOdbCore} ${SIMULIA_LIBRARY_ABQSMAOdbCoreGeom} ${SIMULIA_LIBRARY_ABQSMAOdbAttrEO} ${SIMULIA_LIBRARY_ABQSMAAbuBasicUtils} ${SIMULIA_LIBRARY_ABQSMABasShared} ${SIMULIA_LIBRARY_ABQSMABasCoreUtils} ${SIMULIA_LIBRARY_ABQSMABasAlloc} ${SIMULIA_LIBRARY_ABQSMAAbuGeom} ${SIMULIA_LIBRARY_ABQSMARomDiagEx} ${SIMULIA_LIBRARY_ABQSMASimInterface} ${SIMULIA_LIBRARY_standardB} ${SIMULIA_LIBRARY_ABQSMASspUmaCore})
 ELSE()
    set (OCCT_LIBRARIES "")
  ENDIF()
ENDIF()
# message("SIMULIA_LIBRARIES are: ${SIMULIA_LIBRARIES}")
# set OCCT_FOUND
include(FindPackageHandleStandardArgs)
IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
find_package_handle_standard_args(SIMULIA REQUIRED_VARS SIMULIA_LIBRARY_ABQSMAOdbDdbOdb  SIMULIA_LIBRARY_ABQSMAOdbApi  SIMULIA_LIBRARY_ABQSMAOdbCore  SIMULIA_LIBRARY_ABQSMAOdbCoreGeom  SIMULIA_LIBRARY_ABQSMAOdbAttrEO  SIMULIA_LIBRARY_ABQSMAAbuBasicUtils  SIMULIA_LIBRARY_ABQSMABasShared  SIMULIA_LIBRARY_ABQSMABasCoreUtils  SIMULIA_LIBRARY_ABQSMABasAlloc  SIMULIA_LIBRARY_ABQSMAAbuGeom  SIMULIA_LIBRARY_ABQSMARomDiagEx  SIMULIA_LIBRARY_ABQSMASimInterface SIMULIA_LIBRARY_standardB  SIMULIA_LIBRARY_ABQSMASspUmaCore  SIMULIA_LIBRARY_oldnames  SIMULIA_LIBRARY_user32  SIMULIA_LIBRARY_ws2_32  SIMULIA_LIBRARY_netapi32  SIMULIA_LIBRARY_advapi32)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
find_package_handle_standard_args(SIMULIA REQUIRED_VARS SIMULIA_LIBRARY_ABQSMAOdbDdbOdb  SIMULIA_LIBRARY_ABQSMAOdbApi  SIMULIA_LIBRARY_ABQSMAOdbCore  SIMULIA_LIBRARY_ABQSMAOdbCoreGeom  SIMULIA_LIBRARY_ABQSMAOdbAttrEO  SIMULIA_LIBRARY_ABQSMAAbuBasicUtils  SIMULIA_LIBRARY_ABQSMABasShared  SIMULIA_LIBRARY_ABQSMABasCoreUtils  SIMULIA_LIBRARY_ABQSMABasAlloc  SIMULIA_LIBRARY_ABQSMAAbuGeom  SIMULIA_LIBRARY_ABQSMARomDiagEx  SIMULIA_LIBRARY_ABQSMASimInterface  SIMULIA_LIBRARY_standardB  SIMULIA_LIBRARY_ABQSMASspUmaCore)
ENDIF()