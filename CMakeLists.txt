cmake_minimum_required(VERSION 3.12)
project(Legolas)
set(VERSION 2.0)
enable_language(Fortran)

option(Debug "Debug" OFF)
option(Coverage "Coverage" OFF)

get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
    # require at least gfortran 9
    if (CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 9)
        message(
            WARNING
            "Your gfortran version is ${CMAKE_Fortran_COMPILER_VERSION}, which is below"
            " the minimum recommended version of 9.0. It is HIGHLY recommended to"
            " update your compilers. If you decide to continue with outdated compiler"
            " versions, note that while the compilation of Legolas may be succesful,"
            " it may give rise to unexpected results. Proceed with caution."
        )
    endif()
else()
    message(
        WARNING
        "Compilation has only been tested with gfortran."
        " You are using ${CMAKE_Fortran_COMPILER_ID} ${Fortran_COMPILER_NAME}"
        " version ${CMAKE_Fortran_COMPILER_VERSION}. Please report unexpected"
        " results to the developers."
    )
endif()

if (${Coverage})
    message(STATUS "====================")
    message(STATUS "Building with code coverage enabled.")
    message(STATUS "Default Fortran flags are disabled, optimisation disabled.")
    message(STATUS "====================")
    set(
        CMAKE_Fortran_FLAGS
        "--coverage -o0 -g -cpp -ffree-line-length-none -Wno-unused-dummy-argument"
    )
elseif(Debug)
    set(
        CMAKE_Fortran_FLAGS
        "-fcheck=all \
        -fbounds-check \
        -Wall \
        -ffree-line-length-none \
        -Wextra \
        -Wconversion \
        -pedantic \
        -fbacktrace \
        -cpp \
        -Wno-unused-dummy-argument"
    )
else() # Release
    set(
        CMAKE_Fortran_FLAGS
        "-fcheck=all \
        -Wall \
        -Wextra \
        -ffree-line-length-none \
        -O3 \
        -funroll-loops \
        -ftree-vectorize \
        -Wconversion \
        -pedantic \
        -fbacktrace \
        -cpp \
        -Wno-unused-dummy-argument"
    )
endif()

if (DEFINED ENV{LEGOLASDIR})
    set(SRC $ENV{LEGOLASDIR}/src)
    set(setup_tools $ENV{LEGOLASDIR}/setup_tools)
    set(BUILD $ENV{LEGOLASDIR}/build)
    file(MAKE_DIRECTORY ${BUILD})
else()
    message(WARNING
            "\nEnvironment variable LEGOLASDIR is not defined! \nTrying to use current source directory."
            )
    set(SRC ${CMAKE_SOURCE_DIR}/src)
    set(setup_tools "${CMAKE_SOURCE_DIR}/setup_tools")
    set(BUILD ${CMAKE_CURRENT_BINARY_DIR})
endif()
set(BIN ${BUILD}/bin)
set(OUTPUT ${CMAKE_SOURCE_DIR}/output)
# place .mod files in bin directory
set (CMAKE_Fortran_MODULE_DIRECTORY ${BIN})
# place executable in top directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
# place library in lib directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BUILD}/lib)
# create directories
file(MAKE_DIRECTORY ${OUTPUT})

include(${setup_tools}/findBLAS.cmake)
include(${setup_tools}/findLAPACK.cmake)
# we don't use SCALAPACK or MUMPS at the moment
#include(${setup_tools}/findSCALAPACK.cmake)
#include(${setup_tools}/findMUMPS.cmake)
include(${setup_tools}/findARPACK.cmake)

# for some reason TRUE and FALSE are not expanded to 1 and 0 in the preprocessor,
# so we explicitly do it here
set(_ARPACK_FOUND 0)
set(_PARPACK_FOUND 0)
if (${ARPACK_FOUND})
    set(_ARPACK_FOUND 1)
endif()
if (${PARPACK_FOUND})
    set(_PARPACK_FOUND 1)
endif()

# add preprocessor directives for optional compilation of submodules,
# depending on packages found
add_definitions(
#    -D _SCALAPACK_FOUND=${SCALAPACK_FOUND}
#    -D _MUMPS_FOUND=${MUMPS_FOUND}
    -D _ARPACK_FOUND=${_ARPACK_FOUND}
    -D _PARPACK_FOUND=${_PARPACK_FOUND}
)

# avoid "... has no symbols" warnings from ranlib", from
# https://stackoverflow.com/questions/4929255
if (APPLE)
  SET(CMAKE_Fortran_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
  SET(
    CMAKE_Fortran_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>"
  )
endif()

# name of library
set(LEGOLASLIB lego)
# name of executable
set(LEGOLASEXEC legolas)
# location of default user-defined submodule
set(USR_SMOD_LOC equilibria)
add_subdirectory(${SRC} ${BIN})
