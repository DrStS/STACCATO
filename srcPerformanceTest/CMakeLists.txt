#------------------------------------------------------------------------------------#
file(GLOB SOURCES *.cpp)
MACRO_APPEND_GLOBAL_VARIABLE(STACCATO_PerformanceTest_SOURCES "${SOURCES}")
#------------------------------------------------------------------------------------#
MACRO_APPEND_GLOBAL_VARIABLE(STACCATO_PerformanceTest_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_BINARY_DIR}")
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
FILE(GLOB STACCATO_src_subdirectories "*")
FOREACH(f ${STACCATO_src_subdirectories})
  IF(NOT f MATCHES "\\.txt" AND IS_DIRECTORY ${f} )
    add_subdirectory(${f})
  ENDIF()
ENDFOREACH()
#------------------------------------------------------------------------------------#
get_property(STACCATO_PerformanceTest_INCLUDES GLOBAL PROPERTY STACCATO_PerformanceTest_INCLUDES) 
get_property(STACCATO_PerformanceTest_SOURCES  GLOBAL PROPERTY STACCATO_PerformanceTest_SOURCES) 
#------------------------------------------------------------------------------------#
# SET Intel MKL 
IF (${USE_INTEL_MKL})
  include_directories(${MKL_INCLUDE_DIR})
  SET(STACCATO_PerformanceTest_LIBS ${STACCATO_PerformanceTest_LIBS} ${MKL_LIBRARIES})
ENDIF()
#------------------------------------------------------------------------------------#
include_directories(${STACCATO_PerformanceTest_INCLUDES})
LIST(REMOVE_ITEM STACCATO_PerformanceTest_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_definitions(-DMKL_DIRECT_CALL)
#------------------------------------------------------------------------------------#
IF (CMAKE_SYSTEM_NAME MATCHES "Windows") 
# WIN32
add_executable(STACCATO_PerformanceTest ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${STACCATO_PerformanceTest_SOURCES})
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
add_executable(STACCATO_PerformanceTest ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${STACCATO_PerformanceTest_SOURCES})
SET (CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -march=native -fomit-frame-pointer -funroll-loops")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fomit-frame-pointer -funroll-loops")
ENDIF()
target_link_libraries(STACCATO_PerformanceTest ${STACCATO_PerformanceTest_LIBS})
#------------------------------------------------------------------------------------#
#
#------------------------------------------------------------------------------------#
MESSAGE("C Compiler Flags are: ${CMAKE_C_FLAGS}")
MESSAGE("C++ Compiler Flags are: ${CMAKE_CXX_FLAGS} ")
MESSAGE("Linker Flags are: ${CMAKE_EXE_LINKER_FLAGS}")
MESSAGE("STACCATO_PerformanceTest_LIBS: ${STACCATO_PerformanceTest_LIBS}")
MESSAGE("STACCATO_PerformanceTest_INCLUDES: ${STACCATO_PerformanceTest_INCLUDES}")
MESSAGE("STACCATO_PerformanceTest_SOURCES: ${STACCATO_PerformanceTest_SOURCES}")
MESSAGE("MKL_LIBRARIES: ${MKL_LIBRARIES}")
#------------------------------------------------------------------------------------#
# Where to put EXECUTABLE
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin64)
#------------------------------------------------------------------------------------#
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()


 
