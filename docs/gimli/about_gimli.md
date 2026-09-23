---
title: About GIMLI
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: false
last_modified_at: 2026-02-03
---

The **G**eneric **I**nterface for **M**PI-AMRVAC / **L**egolas **I**nterconnection, or GIMLI for short, is a Python framework to generate user modules and parfiles for both Legolas and the non-linear simulation code [MPI-AMRVAC](https://amrvac.org), and to facilitate the use of one code's output as input in the other. Concretely, GIMLI consists of three parts:

1. **Legolas setup.** Relying on Python's symbolic package [SymPy](https://www.sympy.org/en/index.html), GIMLI allows you to define a plasma equilibrium analytically with the parameters available in Legolas. Passing this equilibrium along with a dictionary of parameter values and code configuration input such as the solver and which data to save, GIMLI will calculate all the equilibrium's derivatives for you and generate a Legolas user module `smod_user_defined.f08` and accompanying parfile(s). After compilation, it suffices to run Legolas with the generated parfile(s).

2. **MPI-AMRVAC setup.** From the same SymPy-based equilibrium definition, with the appropriate accompanying code configuration details, GIMLI can also generate a user module `mod_usr.t` and parfile for MPI-AMRVAC (v3.3+). The perturbation of this equilibrium is provided by selecting eigenmodes from a Legolas spectrum to construct a superposition of linear solutions with a specified amplitude. This linear solution is then interpolated to the grid in MPI-AMRVAC.

3. **Numerical data in Legolas.** With the introduction of GIMLI, Legolas was extended with the option to define equilibria numerically (see [numerical equilibrium](../../ford/module/smod_equil_numerical.html)). GIMLI provides the functionalities to transform numerical arrays, e.g. one-dimensional slices from MPI-AMRVAC data, to a formatted file that can be read by Legolas. Note though that this requires the use of numerical derivatives, which carries additional risks regarding numerical stability.

**Note:** technically, the configurations defined via GIMLI will work fine in both codes even if they are not in equilibrium. However, if the setup is not force-balanced, care has to be taken when interpreting the linear results.
{: .notice--info}