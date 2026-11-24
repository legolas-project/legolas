#!/bin/bash
set -euo pipefail

if [[ -z "${LEGOLASDIR:-}" || ! -d "$LEGOLASDIR" ]]; then
    echo "error: LEGOLASDIR is not set or is not a directory" >&2
    exit 1
fi
cd "$LEGOLASDIR"

# Pull and compile pFUnit if not yet available
if [[ ! -d tests/pFUnit ]]; then
    cd tests
    git clone https://github.com/Goddard-Fortran-Ecosystem/pFUnit.git
    cd pFUnit
    mkdir build
    cd build
    cmake .. -DSKIP_MPI=YES -DSKIP_OPENMP=YES -DSKIP_FHAMCREST=YES
    make -j 2 tests
    make -j 2 install
    cd ../../..
fi

# Build Legolas
if [[ ! -d build ]]; then
    mkdir build
fi
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DDebug=ON ..
make -j 2
cd ..

# Build and run unit tests
cd tests/unit_tests
if [[ ! -d build ]]; then
    mkdir build
fi
cd build
env PFUNIT_DIR="$LEGOLASDIR/tests/pFUnit/build/" cmake ..
make -j 2
cd ..
./test_legolas

