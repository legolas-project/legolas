#!/bin/bash
set -euo pipefail

if [[ -z "${LEGOLASDIR:-}" || ! -d "$LEGOLASDIR" ]]; then
    echo "error: LEGOLASDIR is not set or is not a directory" >&2
    exit 1
fi
cd "$LEGOLASDIR"

# Build Legolas
if [[ ! -d build ]]; then
    mkdir build
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DDebug=ON
make -j 2
cd ..

# Run regression tests
cd tests/regression_tests
pytest -v regression.py test*

