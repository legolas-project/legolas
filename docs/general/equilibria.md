---
title: Pre-implemented equilibria
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
last_modified_at: 2023-04-13
---

For a complete list of implemented equilibria we refer to the [Legolas docs](../../ford/lists/modules.html), more specifically
all descendants of the `mod_equilibrium` module. These all follow the naming convention `smod_equil_*` ("submodule equilibrium").
You can take a look at the documentation of every submodule for information on default parameters, which ones you can vary and where
they comes from. Below we give a small overview of which equilibria are implemented together with the default included
physical effects, and links to the source docs.

- [Adiabatic homogeneous medium](../../ford/module/smod_equil_adiabatic_homo.html): Cartesian, adiabatic
  ```fortran
  equilibrium_type = "adiabatic_homo"
  ```
- [Constant axial current](../../ford/module/smod_equil_constant_current.html): cylindrical, adiabatic
  ```fortran
  equilibrium_type = "constant_current_tokamak"
  ```
- [Couette flow](../../ford/module/smod_equil_couette_flow.html): Cartesian, flow, viscosity
  ```fortran
  equilibrium_type = "couette_flow"
  ```
- [Discrete Alfvén waves](../../ford/module/smod_equil_discrete_alfven.html): cylindrical, radiative cooling, parallel thermal conduction
  ```fortran
  equilibrium_type = "discrete_alfven"
  ```
- [Gold Hoyle](../../ford/module/smod_equil_gold_hoyle.html): cylindrical, radiative cooling, parallel thermal conduction
  ```fortran
  equilibrium_type = "gold_hoyle"
  ```
- [Gravito acoustic waves](../../ford/module/smod_equil_gravito_acoustic.html): Cartesian, hydrodynamic, external gravity
  ```fortran
  equilibrium_type = "gravito_acoustic"
  ```
- [Gravito MHD waves](../../ford/module/smod_equil_gravito_mhd.html): Cartesian, external gravity
  ```fortran
  equilibrium_type = "gravito_mhd"
  ```
- [Harris sheet with tearing modes](../../ford/module/smod_equil_harris_sheet.html): Cartesian, resistive
  ```fortran
  equilibrium_type = "harris_sheet"
  ```
- [Flux tube under coronal conditions](../../ford/module/smod_equil_coronal_flux_tube.html): cylindrical, adiabatic
  ```fortran
  equilibrium_type = "coronal_flux_tube"
  ```
- [Flux tube under photospheric conditions](../../ford/module/smod_equil_photospheric_flux_tube.html): cylindrical, adiabatic
  ```fortran
  equilibrium_type = "photospheric_flux_tube"
  ```
- [Interchange modes](../../ford/module/smod_equil_interchange_modes.html): Cartesian, external gravity
  ```fortran
  equilibrium_type = "interchange_modes"
  ```
- [Internal kink modes in a force-free magnetic field](../../ford/module/smod_equil_internal_kink_instability.html): cylindrical, flow
  ```fortran
  equilibrium_type = "internal_kink"
  ```
- [Kelvin-Helmholtz instabilities](../../ford/module/smod_equil_khi.html): Cartesian, hydrodynamic, flow
  ```fortran
  equilibrium_type = "kelvin_helmholtz"
  ```
- [Kelvin-Helmholtz current-driven instabilities](../../ford/module/smod_equil_kelvin_helmholtz_cd.html): cylindrical, flow
  ```fortran
  equilibrium_type = "kelvin_helmholtz_cd"
  ```
- [Magneto-rotational instabilities](../../ford/module/smod_equil_mri.html): cylindrical (accretion disk), flow, external gravity
  ```fortran
  equilibrium_type = "MRI_accretion"
  ```
- [Magnetothermal instabilities](../../ford/module/smod_equil_magnetothermal_instabilities.html): cylindrical, radiative cooling, parallel thermal conduction
  ```fortran
  equilibrium_type = "magnetothermal_instabilities"
  ```
- [Numerical setup](../../ford/module/smod_equil_numerical.html): imports user data from a file generated with pylbo. See [here](../../sphinx/autoapi/pylbo/index.html#pylbo.gimli.NumericalEquilibrium)
  ```fortran
  equilibrium_type = "numerical"
  ```
- [Rayleigh-Taylor instabilities](../../ford/module/smod_equil_rti.html): Cartesian, flow, external gravity
  ```fortran
  equilibrium_type = "rayleigh_taylor"
  ```
- [Rayleigh-Taylor + Kelvin-Helmholtz instabilities](../../ford/module/smod_equil_rti_khi.html): Cartesian, flow, external gravity
  ```fortran
  equilibrium_type = "RTI_KHI"
  ```
- [Rayleigh-Taylor instabilities in a rotating theta-pinch](../../ford/module/smod_equil_rti_theta_pinch.html): cylindrical, flow
  ```fortran
  equilibrium_type = "RTI_theta_pinch"
  ```
- [Resistive homogeneous medium](../../ford/module/smod_equil_resistive_homo.html): Cartesian, constant resistivity
  ```fortran
  equilibrium_type = "resistive_homo"
  ```
- [Resistive tearing modes](../../ford/module/smod_equil_resistive_tearing.html): Cartesian, constant resistivity
  ```fortran
  equilibrium_type = "resistive_tearing"
  ```
- [Resistive tearing modes with flow](../../ford/module/smod_equil_resistive_tearing_flow.html): Cartesian, flow, constant resistivity
  ```fortran
  equilibrium_type = "resistive_tearing_flow"
  ```
- [Resonant absorption](../../ford/module/smod_equil_resonant_absorption.html): Cartesian, constant resistivity
  ```fortran
  equilibrium_type = "resonant_absorption"
  ```
- [Rotating plasma cylinder](../../ford/module/smod_equil_rotating_plasma_cylinder.html): cylindrical, flow
  ```fortran
  equilibrium_type = "rotating_plasma_cylinder"
  ```
- [Stratified solar magnetic atmosphere](../../ford/module/smod_equil_isothermal_atmosphere.html): Cartesian, adiabatic
  ```fortran
  equilibrium_type = "isothermal_atmosphere"
  ```
- [Suydam cluster modes](../../ford/module/smod_equil_suydam_cluster.html): cylindrical, flow
  ```fortran
  equilibrium_type = "suydam_cluster"
  ```
- [Taylor-Couette fluid](../../ford/module/smod_equil_taylor_couette.html): cylindrical (coaxial), hydrodynamic, flow, viscosity
  ```fortran
  equilibrium_type = "taylor_couette"
  ```
- [Taylor-Couette plasma](../../ford/module/smod_equil_tc_pinch.html): cylindrical (coaxial), flow, constant resistivity, viscosity
  ```fortran
  equilibrium_type = "tc_pinch"
  ```
