# A cmake module to find HDF5
#------------------------------------------------------------------------------------#
# MKL includes and libraries are searched for in HDF5_INCLUDE_DIR and HDF5_LIB_DIR.
# If HDF5_INCLUDE_DIR and HDF5_LIB_DIR are not set, the module searches under 
# the environment variable HDF5ROOT and /opt/intel/mkl subdirectories.
#------------------------------------------------------------------------------------#
set(HDF5ROOT_DIR $ENV{HDF5ROOT})
IF (NOT HDF5ROOT_DIR)
   IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
       IF (EXISTS "/opt/software/libs/mkl")
          set(HDF5ROOT_DIR "/opt/software/libs/mkl/mkl")
       ENDIF()
   ENDIF()
   IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
       IF (EXISTS "C:/software/libs/HDF5")
          set(HDF5ROOT_DIR "C:/software/libs/HDF5")
       ENDIF()
   ENDIF()
ENDIF ()
message("HDF5ROOT_DIR is: ${HDF5ROOT_DIR}")
#------------------------------------------------------------------------------------#
# Stage 1: find the include directory
#------------------------------------------------------------------------------------#
IF (NOT HDF5_INCLUDE_DIR)
  find_path(HDF5_INCLUDE_DIR
    NAMES hdf5.h
    HINTS ${HDF5ROOT_DIR}
    PATH_SUFFIXES include
    )  
	IF(HDF5_INCLUDE_DIR MATCHES "HDF5_INCLUDE_DIR-NOTFOUND")
     unset(HDF5_INCLUDE_DIR CACHE)
	ENDIF()
ENDIF ()
message("HDF5_INCLUDE_DIR is: ${HDF5_INCLUDE_DIR}")
#------------------------------------------------------------------------------------#
# Stage 2: find the lib directory
#------------------------------------------------------------------------------------#	
if (NOT HDF5_LIB_DIR)
  if (HDF5ROOT_DIR)
	set(EXPECT_HDF5_LIBPATH "${HDF5ROOT_DIR}/lib")	
    if (IS_DIRECTORY ${EXPECT_HDF5_LIBPATH})
      set(HDF5_LIB_DIR ${EXPECT_HDF5_LIBPATH})
    endif ()
  endif ()
endif ()
message("HDF5_LIB_DIR is: ${HDF5_LIB_DIR}")
#------------------------------------------------------------------------------------#
# Stage 3: find the libraries
#------------------------------------------------------------------------------------#	
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
set (CMAKE_FIND_LIBRARY_SUFFIXES .a)
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
set (CMAKE_FIND_LIBRARY_SUFFIXES .lib)
ENDIF()
if (HDF5_LIB_DIR)
  find_library(HDF5_LIB_HDF5_CPP     libhdf5_cpp   ${HDF5_LIB_DIR})
  find_library(HDF5_LIB_HDF5_C       libhdf5       ${HDF5_LIB_DIR})
  find_library(HDF5_LIB_SZIP         libszip       ${HDF5_LIB_DIR})
  find_library(HDF5_LIB_ZLIB         libzlib       ${HDF5_LIB_DIR})

  find_library(HDF5_LIB_HDF5_CPP_D    libhdf5_cpp_D   ${HDF5_LIB_DIR})
  find_library(HDF5_LIB_HDF5_C_D      libhdf5_D       ${HDF5_LIB_DIR})
  find_library(HDF5_LIB_SZIP_D        libszip_D       ${HDF5_LIB_DIR})
  find_library(HDF5_LIB_ZLIB_D        libzlib_D       ${HDF5_LIB_DIR})

  
  if (HDF5_LIB_HDF5_C AND HDF5_LIB_HDF5_CPP )
    IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
      set (HDF5_LIBRARIES ${HDF5_LIB_HDF5_C})
    ENDIF()
    IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
      set (HDF5_LIBRARIES ${HDF5_LIB_HDF5_C} ${HDF5_LIB_HDF5_CPP} ${HDF5_LIB_SZIP} ${HDF5_LIB_ZLIB})
      set (HDF5_LIBRARIES_DEBUG ${HDF5_LIB_HDF5_CPP_D} ${HDF5_LIB_HDF5_C_D} ${HDF5_LIB_SZIP_D} ${HDF5_LIB_ZLIB_D})
    ENDIF()
  else ()
    set (HDF5_LIBRARIES "")
  endif ()
endif ()

# set MKL_FOUND
include(FindPackageHandleStandardArgs)
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
find_package_handle_standard_args(HDF5 REQUIRED_VARS HDF5_LIB_HDF5_C HDF5_LIB_HDF5_CPP)
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
find_package_handle_standard_args(HDF5 REQUIRED_VARS HDF5_LIB_HDF5_C HDF5_LIB_HDF5_CPP)
ENDIF()
