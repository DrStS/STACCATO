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
# SET OCCT
IF (${USE_OCCT})
  include_directories(${OCCT_INCLUDE_DIR})
  SET(STACCATO_LIBS ${STACCATO_LIBS}  ${OCCT_LIBRARIES})
ENDIF()
# SET Intel MKL 
IF (${USE_INTEL_MKL})
  include_directories(${MKL_INCLUDE_DIR})
  SET(STACCATO_LIBS ${STACCATO_LIBS} ${MKL_LIBRARIES})
ENDIF()
# SET QT5 
IF (${USE_QT5})
ENDIF()
# SET SIMULIA ODB 
IF (${USE_SIMULIA_ODB})
  include_directories(${SIMULIA_INCLUDE_DIR})
  SET(STACCATO_LIBS ${STACCATO_LIBS} ${SIMULIA_LIBRARIES})
ENDIF()
#------------------------------------------------------------------------------------#
include_directories(${STACCATO_INCLUDES})
LIST(REMOVE_ITEM STACCATO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
#------------------------------------------------------------------------------------#
IF (CMAKE_SYSTEM_NAME MATCHES "Windows")
add_executable(STACCATO ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${STACCATO_SOURCES})
ENDIF()
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
add_executable(STACCATO ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${STACCATO_SOURCES} resources.qrc)
ENDIF()

target_link_libraries(STACCATO ${STACCATO_LIBS} )
qt5_use_modules(STACCATO Widgets OpenGL)
#------------------------------------------------------------------------------------#
#
#------------------------------------------------------------------------------------#
MESSAGE("C Compiler Flags are: ${CMAKE_C_FLAGS}")
MESSAGE("C++ Compiler Flags are: ${CMAKE_CXX_FLAGS} ")
MESSAGE("Linker Flags are: ${CMAKE_EXE_LINKER_FLAGS}")
MESSAGE("STACCATO_LIBS: ${STACCATO_LIBS}")
MESSAGE("STACCATO_INCLUDES: ${STACCATO_INCLUDES}")
MESSAGE("STACCATO_SOURCES: ${STACCATO_SOURCES}")
#------------------------------------------------------------------------------------#
# Where to put EXECUTABLE
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin64)
#------------------------------------------------------------------------------------#
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()