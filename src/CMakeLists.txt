#------------------------------------------------------------------------------------#
file(GLOB SOURCES *.cpp)
MACRO_APPEND_GLOBAL_VARIABLE(STACCATO_SOURCES "${SOURCES}")
#------------------------------------------------------------------------------------#
MACRO_APPEND_GLOBAL_VARIABLE(STACCATO_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_BINARY_DIR}")
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
FILE(GLOB STACCATO_src_subdirectories "*")
FOREACH(f ${STACCATO_src_subdirectories})
  IF(NOT f MATCHES "\\.txt" AND IS_DIRECTORY ${f} )
    add_subdirectory(${f})
  ENDIF()
ENDFOREACH()
#------------------------------------------------------------------------------------#
get_property(STACCATO_INCLUDES GLOBAL PROPERTY STACCATO_INCLUDES) 
get_property(STACCATO_SOURCES GLOBAL PROPERTY STACCATO_SOURCES)  
#------------------------------------------------------------------------------------#
# SET Intel MKL 
IF (${USE_INTEL_MKL})
  include_directories(${MKL_INCLUDE_DIR})
  SET(STACCATO_LIBS ${STACCATO_LIBS} ${MKL_LIBRARIES})
ENDIF()
#------------------------------------------------------------------------------------#
include_directories(${STACCATO_INCLUDES})
LIST(REMOVE_ITEM STACCATO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(STACCATOLib STATIC ${STACCATO_SOURCES})
#------------------------------------------------------------------------------------#
add_executable(STACCATO ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(STACCATO STACCATOLib ${STACCATO_LIBS})
#------------------------------------------------------------------------------------#
add_dependencies(STACCATO STACCATOLib)
#------------------------------------------------------------------------------------#
MESSAGE("C Compiler Flags are:\t${CMAKE_C_FLAGS} \nC++ Compiler Flags are:\t${CMAKE_CXX_FLAGS}")
MESSAGE("STACCATO_LIBS: ${STACCATO_LIBS}")
MESSAGE("STACCATO_INCLUDES: ${STACCATO_INCLUDES}")
MESSAGE("STACCATO_SOURCES: ${STACCATO_SOURCES}")
#------------------------------------------------------------------------------------#
# Where to put EXECUTABLE
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin64)
#------------------------------------------------------------------------------------#
