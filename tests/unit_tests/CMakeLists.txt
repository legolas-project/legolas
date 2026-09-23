cmake_minimum_required(VERSION 3.12)

project(Legolas_testing)
enable_language(Fortran)

option(Coverage "Coverage" OFF)

get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
set(
    CMAKE_Fortran_FLAGS
    "-fcheck=all \
    -fbounds-check \
    -Wall \
    -Wextra \
    -Wconversion \
    -pedantic \
    -fbacktrace \
    -cpp \
    -ffree-line-length-256 \
    -Wno-unused-dummy-argument"
)

find_package(PFUNIT REQUIRED)
enable_testing()

if (DEFINED ENV{LEGOLASDIR})
    set(BUILD $ENV{LEGOLASDIR}/build)
    set(LIB ${BUILD}/lib)
    set(setup_tools $ENV{LEGOLASDIR}/setup_tools)
else()
    message(FATAL_ERROR "Environment variable LEGOLASDIR is not defined!")
endif()

# place executable in top directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

include(${setup_tools}/findBLAS.cmake)
include(${setup_tools}/findLAPACK.cmake)
#include(${setup_tools}/findSCALAPACK.cmake)
#include(${setup_tools}/findMUMPS.cmake)
include(${setup_tools}/findARPACK.cmake)

# add default modules to library
set( aux_modules
    mod_suite_utils.f90
)
add_library(suite ${aux_modules})
set_target_properties (suite
        PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
# include compiled legolas file as directory
target_include_directories(suite PUBLIC ${BUILD}/bin)
target_include_directories(suite PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
# link compiled library as well
FILE(GLOB LEGOLASLIB ${BUILD}/lib/*.a)
set(LIBSTOLINK
    ${ARPACK_LIBRARIES}
    ${PARPACK_LIBRARIES}
    #    ${MUMPS_LIBRARIES}
    #    ${SCALAPACK_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
)
target_link_libraries(suite ${LEGOLASLIB} ${LIBSTOLINK})

set( test_sources
        mod_test_variables.pf
        mod_test_logging.pf
        mod_test_banded_matrix.pf
        mod_test_banded_matrix_hermitian.pf
        mod_test_matrix_structure.pf
        mod_test_matrix_elements.pf
        mod_test_transform_matrix.pf
        mod_test_settings.pf
        mod_test_matrices_big.pf
        mod_test_solvers_linear_systems.pf
        mod_test_solvers_lapack.pf
        mod_test_solvers_arpack_type.pf
        mod_test_solvers_arpack_general.pf
        mod_test_solvers_arpack_shift_invert.pf
        mod_test_indices.pf
        mod_test_grid.pf
        mod_test_input.pf
        mod_test_valuechecks.pf
        mod_test_units.pf
        mod_test_eigenfunctions.pf
        mod_test_cooling.pf
        mod_test_cooling_tables.pf
        mod_test_heating.pf
        mod_test_heatloss.pf
        mod_test_resistivity.pf
        mod_test_conduction.pf
        mod_test_boundaries.pf
        mod_test_statevector.pf
        mod_test_splines.pf
        mod_test_quadblock.pf
        mod_test_interpolation.pf
        mod_test_integration.pf
        mod_test_inspections.pf
        mod_test_solar_atmosphere.pf
        mod_test_iv_solver.pf
        mod_test_iv_state_vector.pf
)
add_pfunit_ctest (test_legolas
        TEST_SOURCES ${test_sources}
        OTHER_SOURCES mod_pfunit_exceptions.f90
        LINK_LIBRARIES suite ${LIBSTOLINK}
        EXTRA_USE mod_pfunit_exceptions
        EXTRA_INITIALIZE init_pfunit_raise
)
if (${Coverage})
    target_link_options(test_legolas PUBLIC "--coverage")
endif()
