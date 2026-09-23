---
title: About Legolas
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
last_modified_at: 2025-01-27
---

## Origins
The development of Legolas started due to the need of a new and flexible numerical tool to solve the full system of linearised
MHD equations with various physical effects included. Calculating all eigenvalues for such setups is relatively
straightforward for homogeneous cases since this boils down to calculating the algebraic system of linearised equations,
writing down the matrices and plugging them into your favourite eigenvalue problem solver.
However, this becomes extremely difficult (or even impossible) to do for more general cases with inhomogeneous equilibria,
unless some rather special conditions are applied to the system to make it analytically tractable.

Legolas actually builds on the heritage of early numerical codes, most notably LEDA (Kerner, 1985), meant for studies of ideal and resistive MHD spectra.
The original LEDA code has had several extensions over the past decades, such as LEDA2 (Van Der Linden, 1992) for non-adiabatic spectra and LEDAFLOW (Nijboer, 1997) for non-static spectra.
These codes were not mergeable however, such that a combination of for example non-adiabatic effects, flow and resistivity was not possible without a major undertaking.
This prompted us to start from scratch on a brand new, modern MHD spectral code which we named Legolas,
short for "**<u>L</u>**arge **<u>E</u>**igensystem **<u>G</u>**enerator for **<u>O</u>**ne-dimensional p**<u>las</u>**mas".

Development on Legolas started late 2019 by Niels Claes as part of his PhD studies. He was joined in mid 2020 by Jordi De Jonghe, who currently
maintains and develops the code further.
<img src="/assets/images/erc_logo.png" alt="ERC logo" align="right" width="100" height="100">
The development took place under the supervision of Rony Keppens at the [Centre for
mathematical Plasma-Astrophysics](https://wis.kuleuven.be/CmPA) at the KU Leuven, Belgium, as part of the [ERC PROMINENT](https://erc-prominent.github.io) project.

## Aims
The main goal of Legolas is to provide a new and modern user-friendly code, making use of the most recent developments in linear algebra and
theoretical (linear) MHD. This allows for detailed, high-resolution studies of the full (M)HD linear spectrum through fully realistic setups
and parametric studies, including a myriad of physical effects.

## Core people involved with Legolas

### Active
- **Dr. Jordi De Jonghe** ([<i class="fas fa-envelope" aria-hidden="true"></i>](mailto:jordi.dejonghe@kuleuven.be)): development team.

  Source code contributions to both Legolas and Pylbo, and maintenance. Implementation of additional physics and coupling to [MPI-AMRVAC](https://amrvac.org).

- **Dr. Nicolas Brughmans** ([<i class="fas fa-envelope" aria-hidden="true"></i>](mailto:nicolas.brughmans@kuleuven.be)): development team.

  Pylbo extensions in the scope of accretion disk applications.

- **Drs. Adrian Kelly**  ([<i class="fas fa-envelope" aria-hidden="true"></i>](mailto:adrian.kelly@kuleuven.be)): development team.

  Development of the time-dependent initial-value solver.

- **Prof. Rony Keppens** ([<i class="fas fa-envelope" aria-hidden="true"></i>](mailto:rony.keppens@kuleuven.be)): guidance.

### Resigned
- **Dr. Niels Claes**: creator of the code and main developer.

  <!-- Involved with everything & provides overall guidance. Responsible for maintenance and general development of both Legolas and Pylbo. -->
  General development of both Legolas and Pylbo.

- **Drs. Evert Provoost**: development team.

  Contributions to the linear algebra solvers in the scope of a Master's thesis.


## Features
Legolas is written in modern Fortran and is highly modularised. This allows for easy maintenance and makes the code ready
to be extended with additional physics and modern algorithmic requirements. Currently, Legolas supports
both 3D Cartesian or cylindrical geometries with nontrivial 1D variation.
Various physical effects are implemented:
- background flows, allowing for non-static equilibrium conditions
- external gravity
- resistivity, either constant values or realistic temperature-dependent profiles
- optically thin radiative losses, based on modern cooling curves
- anisotropic thermal conduction, with fully customisable parallel/perpendicular effects with respect to the magnetic field
- viscosity
- Hall MHD

Note that Legolas can calculate spectra in either full MHD or hydrodynamics (no magnetic field).
