#------------------------------------------------------------------------------------#
file(GLOB SOURCES *.cpp *.c)
MACRO_APPEND_GLOBAL_VARIABLE(STACCATO_SOURCES "${SOURCES}")
#------------------------------------------------------------------------------------#
MACRO_APPEND_GLOBAL_VARIABLE(STACCATO_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_BINARY_DIR}")
#------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------#
FILE(GLOB STACCATO_src_sub_subdirectories "*")
FOREACH(sf ${STACCATO_src_sub_subdirectories})
  IF(NOT sf MATCHES "\\.txt" AND IS_DIRECTORY ${sf} )
    add_subdirectory(${sf})
  ENDIF()
ENDFOREACH()
#------------------------------------------------------------------------------------#