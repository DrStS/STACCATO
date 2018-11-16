# A cmake module to find Intel Math Kernel Library
#------------------------------------------------------------------------------------#
# MKL includes and libraries are searched for in MKL_INCLUDE_DIR and MKL_LIB_DIR.
# If MKL_INCLUDE_DIR and MKL_LIB_DIR are not set, the module searches under 
# the environment variable MKLROOT and /opt/intel/mkl subdirectories.
#------------------------------------------------------------------------------------#
set(MKLROOT_DIR $ENV{MKLROOT})
IF (NOT MKLROOT_DIR)
   IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
       IF (EXISTS "/opt/software/libs/mkl")
          set(MKLROOT_DIR "/opt/software/libs/mkl")
       ENDIF()
   ENDIF()
   IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
       IF (EXISTS "C:/software/libs/MKL")
          set(MKLROOT_DIR "C:/software/libs/MKL")
       ENDIF()
   ENDIF()
ENDIF ()
message("MKLROOT_DIR is: ${MKLROOT_DIR}")
#------------------------------------------------------------------------------------#
# Stage 1: find the include directory
#------------------------------------------------------------------------------------#
IF (NOT MKL_INCLUDE_DIR)
  find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    HINTS ${MKLROOT_DIR}
    PATH_SUFFIXES include
    )  
	IF(MKL_INCLUDE_DIR MATCHES "MKL_INCLUDE_DIR-NOTFOUND")
     unset(MKL_INCLUDE_DIR CACHE)
	ENDIF()
ENDIF ()
message("MKL_INCLUDE_DIR is: ${MKL_INCLUDE_DIR}")
#------------------------------------------------------------------------------------#
# Stage 2: find the lib directory
#------------------------------------------------------------------------------------#	
if (NOT MKL_LIB_DIR)
  if (MKLROOT_DIR)
	set(EXPECT_MKL_LIBPATH "${MKLROOT_DIR}/lib/intel64")	
    if (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})
      set(MKL_LIB_DIR ${EXPECT_MKL_LIBPATH})
    endif ()
  endif ()
endif ()
message("MKL_LIB_DIR is: ${MKL_LIB_DIR}")
#------------------------------------------------------------------------------------#
# Stage 3: find the libraries
#------------------------------------------------------------------------------------#	
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
set (CMAKE_FIND_LIBRARY_SUFFIXES .a)
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
set (CMAKE_FIND_LIBRARY_SUFFIXES .lib)
ENDIF()
if (MKL_LIB_DIR)
  find_library(MKL_INTEL_LP64_LIBRARY     mkl_intel_lp64     ${MKL_LIB_DIR})
  find_library(MKL_INTEL_THREAD_LIBRARY   mkl_intel_thread   ${MKL_LIB_DIR})
  find_library(MKL_CORE_LIBRARY           mkl_core           ${MKL_LIB_DIR})
  IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
   find_library(MKL_OMP_LIBRARY           iomp5              ${MKL_LIB_DIR})
  ENDIF()
  IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
   find_library(MKL_OMP_LIBRARY           libiomp5md         ${MKL_LIB_DIR})
  ENDIF()
  
  if (MKL_INTEL_LP64_LIBRARY AND MKL_INTEL_THREAD_LIBRARY AND MKL_CORE_LIBRARY)
    IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
      set (MKL_LIBRARIES "-Wl,--start-group ${MKL_INTEL_LP64_LIBRARY} ${MKL_INTEL_THREAD_LIBRARY} ${MKL_CORE_LIBRARY} -Wl,--end-group ${MKL_OMP_LIBRARY} -lpthread -lm -ldl")
    ENDIF()
    IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
      set (MKL_LIBRARIES ${MKL_INTEL_LP64_LIBRARY} ${MKL_INTEL_THREAD_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_OMP_LIBRARY})
    ENDIF()
  else ()
    set (MKL_LIBRARIES "")
  endif ()
endif ()

# set MKL_FOUND
include(FindPackageHandleStandardArgs)
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
find_package_handle_standard_args(MKL REQUIRED_VARS  MKL_INTEL_LP64_LIBRARY MKL_INTEL_THREAD_LIBRARY MKL_CORE_LIBRARY MKL_OMP_LIBRARY)
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
find_package_handle_standard_args(MKL REQUIRED_VARS  MKL_INTEL_LP64_LIBRARY MKL_INTEL_THREAD_LIBRARY MKL_CORE_LIBRARY MKL_OMP_LIBRARY)
ENDIF()
#mark_as_advanced(LIB_PTHREAD MKL_INCLUDE_DIR MKL_LIBRARY_DIR)
