
function(unable_to_find)
    set(ARPACK_FOUND False PARENT_SCOPE)
    set(PARPACK_FOUND False PARENT_SCOPE)
    message(STATUS "Unable to find ARPACK!")
endfunction()


# check if arpack compiles
function(check_if_arpack_links)
    set(CMAKE_REQUIRED_INCLUDES
        ${ARPACK_INCLUDE_DIR}
    )
    set(CMAKE_REQUIRED_LIBRARIES
        ${ARPACK_LIBRARY}
        ${LAPACK_LIBRARIES}
    )

    include(CheckFortranSourceCompiles)
    # we don't have to supply arguments to dnaupd, this will error
    # out but link just fine
    check_fortran_source_compiles("
        program test_arpack_call
            implicit none
            call dnaupd()
        end program"
        ARPACK_calls
        SRC_EXT
        f90
    )
    if(NOT ARPACK_calls)
        set(ARPACK_links False PARENT_SCOPE)
    endif()
endfunction()

message("")
message(STATUS "Finding ARPACK")

set(ARPACK_FOUND False)
set(PARPACK_FOUND False)

if (NOT DEFINED ENV{ARPACK_ROOT})
    message(STATUS "Environment variable $ARPACK_ROOT not set!")
    unable_to_find()
    return()
endif()

# try to find ARPACK include dir
find_path(ARPACK_INCLUDE_DIR
    NAMES arpackdef.h
    HINTS $ENV{ARPACK_ROOT} $ENV{ARPACK_ROOT}/installed
    PATH_SUFFIXES
        include/arpack
        include/arpack-ng
)
if(ARPACK_INCLUDE_DIR)
    message(STATUS "Looking for ARPACK include directory - found")
    message(STATUS "Found in ${ARPACK_INCLUDE_DIR}")
else()
    message(STATUS "Looking for ARPACK include directory - not found")
    unable_to_find()
    return()
endif()

# try to find ARPACK library
find_library(ARPACK_LIBRARY
    NAMES arpack
    HINTS $ENV{ARPACK_ROOT}
    PATH_SUFFIXES lib installed/lib
)
if(ARPACK_LIBRARY)
    message(STATUS "Looking for libarpack - found")
else()
    message(STATUS "Looking for libarpack - not found")
    unable_to_find()
    return()
endif()

# try to find PARPACK library
find_library(PARPACK_LIBRARY
    NAMES parpack
    HINTS $ENV{ARPACK_ROOT}
    PATH_SUFFIXES lib installed/lib
)
if(PARPACK_LIBRARY)
    message(STATUS "Looking for libparpack - found")
else()
    # don't abort if not found
    message(STATUS "Looking for libparpack - not found")
endif()

# check if we are able to link and compile Fortran programs containing ARPACK
set(ARPACK_links True)
check_if_arpack_links()
if(NOT ${ARPACK_links})
    message(STATUS "ARPACK libraries found but linking failed!")
    unable_to_find()
    return()
endif()

# find (P)ARPACK package and set corresponding variables
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ARPACK
    REQUIRED_VARS ARPACK_LIBRARY ARPACK_INCLUDE_DIR
    HANDLE_COMPONENTS
)
find_package_handle_standard_args(PARPACK
    REQUIRED_VARS PARPACK_LIBRARY
    HANDLE_COMPONENTS
)

set(ARPACK_LIBRARIES ${ARPACK_LIBRARY})
set(ARPACK_INCLUDE_DIRS ${ARPACK_INCLUDE_DIR})
if(NOT TARGET ARPACK::ARPACK)
    add_library(ARPACK::ARPACK INTERFACE IMPORTED)
    set_target_properties(ARPACK::ARPACK PROPERTIES
        INTERFACE_LINK_LIBRARIES ${ARPACK_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${ARPACK_INCLUDE_DIR}
    )
endif()
if(PARPACK_FOUND)
    set(PARPACK_LIBRARIES ${PARPACK_LIBRARY})
    if(NOT TARGET PARPACK::PARPACK)
        add_library(PARPACK::PARPACK INTERFACE IMPORTED)
        set_target_properties(PARPACK::PARPACK PROPERTIES
            INTERFACE_LINK_LIBRARIES ${PARPACK_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${ARPACK_INCLUDE_DIR}
        )
    endif()
endif()
