
![legolas-logo](docs/assets/images/logo_legolas_640x237.png)

**L**arge **E**igensystem **G**enerator for **O**ne-dimensional p**LAS**mas

[![unit tests](https://github.com/legolas-project/legolas/actions/workflows/unit.yml/badge.svg?branch=master)](https://github.com/legolas-project/legolas/actions/workflows/unit.yml)[![regression](https://github.com/legolas-project/legolas/actions/workflows/regression.yml/badge.svg?branch=master)](https://github.com/legolas-project/legolas/actions/workflows/regression.yml) [![pylbo](https://github.com/legolas-project/legolas/actions/workflows/pylbo.yml/badge.svg?branch=master)](https://github.com/legolas-project/legolas/actions/workflows/pylbo.yml) [![docs](https://github.com/legolas-project/legolas/actions/workflows/docs_stable.yml/badge.svg?branch=master)](https://github.com/legolas-project/legolas/actions/workflows/docs_stable.yml) [![codecov](https://codecov.io/gh/legolas-project/legolas/branch/master/graph/badge.svg?token=OVLYGOADS7)](https://codecov.io/gh/legolas-project/legolas) [![GitHub release (latest by date)](https://img.shields.io/github/v/release/legolas-project/legolas?color=blue&label=release&style=flat)](https://github.com/legolas-project/legolas/releases) [![Slack](https://img.shields.io/static/v1?label=Slack&message=join%20us&style=flat&logo=Slack&logoColor=white&color=orange)](https://join.slack.com/t/the-legolas-code/shared_invite/zt-tsb5yaht-LtLWHzVu8Zux~Yt3PBx32Q)

### Table of contents
1. [About](#about)
2. [Documentation](#documentation)
3. [Versions](#versions)
4. [Using Legolas](#using-legolas)

## About
Legolas is a modern and user-friendly large-scale object oriented framework, capable of solving (subsets of) the full set of linearised magnetohydrodynamic equations for a general three-dimensional state with one-dimensional variation in either Cartesian or cylindrical geometries. Fourier modes are imposed in the other coordinates. The code is written in modern Fortran and relies on a Finite Element treatment to reduce the resulting system of differential equations to a large-scale complex, non-Hermitian eigenvalue problem. Legolas interfaces with the BLAS/LAPACK/ARPACK libraries, resulting in detailed eigenspectrum calculations with corresponding eigenfunctions. An efficient treatment of the underlying datastructure allows the code to run on regular laptops at high resolutions, and a complementary post-processing framework is provided for interactive visualisation and analysis of the results.

Legolas supports the inclusion of background flows, external gravity, anisotropic thermal conduction, resistivity, optically thin radiative losses, heating, viscosity, and full Hall MHD; most of these are fully customisable by the user.

Legolas is being developed and maintained at the [Centre for mathematical Plasma-Astrophysics](https://wis.kuleuven.be/CmPA), KU Leuven, Belgium.


## Documentation
We have a dedicated website on [legolas.science](https://legolas.science) which contains a detailed guide on how to install, compile and use the code. Source code documentation is regenerated automatically with each commit on both the `master` and `develop` branches.

## Versions
We (try to) ensure that the `master` branch of this repository is always stable and ready for use.
If you would like to check out the latest development goodies instead you can take a look at the `develop` branch, but note that since there may be some time between major releases it is possible that the development branch differs from the latest stable release. The development version of Legolas has its own dedicated website at [dev.legolas.science](https://dev.legolas.science), for those who want to use the bleeding-edge version of the code.
Note that while everything on the `develop` branch _should_ be stable, it is possible that some features there are still under development and/or not yet fully tested or documented.

## Using Legolas
:warning: **Please note**

Legolas has taken (and still takes up) a lot of effort and time to test, develop and maintain.
Since the code is relatively new we would like to know how it is used and provide guidance if possible.
To that end, we kindly ask that the _**the first published peer-reviewed paper from applying Legolas is done in co-authorship with at least one of the original authors**_. The BibTex citation to our ApJS method paper can be found [here](https://ui.adsabs.harvard.edu/abs/2020arXiv201014148C/exportcitation).
