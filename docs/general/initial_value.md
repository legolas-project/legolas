---
title: Initial-value solver
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: true
toc_icon: "chevron-circle-down"
last_modified_at: 2026-03-09
---

Legolas includes an initial-value solver that integrates perturbations forward in time directly from the FEM matrices, without requiring a separate code.
Given a set of initial perturbation profiles, it evolves the system

$$B \frac{d\mathbf{x}}{dt} = -iA\mathbf{x}$$

using the implicit theta-method (implicit midpoint at the default `alpha = 0.5`), and saves snapshots of the solution at regular intervals alongside the standard eigenvalue output.

This page assumes familiarity with how to set up and run Legolas. If not, see [running your first problem](../../getting-started/running) and [implementing a custom setup](../../general/own_setup).

## Supported physics types

The initial-value solver supports the same physics types as the eigenvalue solver:

| Physics type | Perturbed components |
|---|---|
| `isothermal-1d` | $\rho_1$, $v_1$ |
| `hd-1d` | $\rho_1$, $v_1$, $T_1$ |
| `hd` | $\rho_1$, $v_1$, $v_2$, $v_3$, $T_1$ |
| `mhd` | $\rho_1$, $v_1$, $v_2$, $v_3$, $T_1$, $a_1$, $a_2$, $a_3$ |

The physics type is set in the parfile (not in the Fortran submodule):
```fortran
&physicslist
  physics_type = "isothermal-1d"
/
```

## Configuration

IVP mode is configured through the `ivplist` namelist in the parfile:

```fortran
&ivplist
  enabled         = .true.
  alpha           = 0.5    ! implicitness (0 = forward Euler, 1 = backward Euler, 0.5 = implicit midpoint)
  t_end           = 10.0   ! end time
  n_steps         = 1000   ! number of time steps
  n_snapshots     = 100    ! number of snapshots to save
  snapshot_stride = 10     ! save every n-th step (overrides n_snapshots if set)
/
```

Snapshots are written to the datfile automatically when `enabled = .true.`.

### Skipping the eigenvalue solve

The initial-value solver assembles and uses the same `A` and `B` matrices as the eigenvalue
solver, but it does **not** require the (potentially expensive) eigenvalue problem to be solved
first. If you only want the time evolution, select the `"none"` solver to bypass the eigenvalue
solve entirely:

```fortran
&solvelist
  solver = "none"
/
```

The datfile then contains the IVP snapshots but no eigenvalues or eigenfunctions: selecting
`"none"` also switches off eigenfunction, eigenvector and residual output, since these are
meaningless without eigenvalues. IVP snapshot output is unaffected.
See the [solver settings](../../general/solvers) for more details.

## Specifying initial conditions

Initial conditions are set in the `user_defined_eq` procedure of your `smod_user_defined.f08` file,
exactly like a regular equilibrium (see [implementing a custom setup](../../general/own_setup)).
Alongside the standard `settings`, `grid`, `background`, and `physics` objects, the procedure receives
an `iv_initial_conditions` object (of type `initial_conditions_t`) through host association from the
parent module — you do not need to declare it yourself when using `module procedure`.

You set the perturbation profiles with the type-bound setter routines on `iv_initial_conditions`.
Each setter takes a profile function (and optionally its derivative) matching the interface
`f(x) result(y)`, where `x` and `y` are both `real(dp)` arrays of the same size. This is the same
signature used by the initial-value profiles, and differs from the scalar-valued `background`
functions.

A minimal example with a Gaussian density perturbation on a uniform isothermal background
(this mirrors the bundled `ivp_demo` equilibrium):

```fortran
submodule (mod_equilibrium) smod_user_defined
  implicit none

contains

  module procedure user_defined_eq
    call settings%grid%set_geometry("Cartesian")
    call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)

    ! --- background equilibrium (scalar-valued functions) ---
    call background%set_density_funcs(rho0_func=rho0)
    call background%set_temperature_funcs(T0_func=T0)

    ! --- initial conditions (array-valued profile functions) ---
    call iv_initial_conditions%set_ic_density_funcs( &
      rho_func=gaussian_rho, drho_func=gaussian_drho &
    )
  end procedure user_defined_eq


  real(dp) function rho0()
    rho0 = 1.0_dp
  end function rho0

  real(dp) function T0()
    T0 = 1.0_dp
  end function T0

  !> Gaussian density perturbation centred at x = 0.5.
  pure function gaussian_rho(x) result(rho1)
    real(dp), intent(in) :: x(:)
    real(dp) :: rho1(size(x))
    real(dp), parameter :: x0 = 0.5_dp, sigma = 0.05_dp
    rho1 = exp(-((x - x0) / sigma)**2)
  end function gaussian_rho

  !> Derivative of the Gaussian density perturbation.
  pure function gaussian_drho(x) result(drho1)
    real(dp), intent(in) :: x(:)
    real(dp) :: drho1(size(x))
    real(dp), parameter :: x0 = 0.5_dp, sigma = 0.05_dp
    drho1 = -2.0_dp * (x - x0) / sigma**2 * exp(-((x - x0) / sigma)**2)
  end function gaussian_drho

end submodule smod_user_defined
```

The setter routines available on the `iv_initial_conditions` object are:

| Subroutine | Component | Physics types |
|---|---|---|
| `set_ic_density_funcs(rho_func [, drho_func])` | $\rho_1$ | all |
| `set_ic_velocity_1_funcs(v01_func, dv01_func)` | $v_1$ | all |
| `set_ic_velocity_2_funcs(v02_func, dv02_func)` | $v_2$ | `hd`, `mhd` |
| `set_ic_velocity_3_funcs(v03_func, dv03_func)` | $v_3$ | `hd`, `mhd` |
| `set_ic_temperature_funcs(T_func [, dT_func])` | $T_1$ | `hd-1d`, `hd`, `mhd` |
| `set_ic_a1_funcs(a1_func [, da1_func])` | $a_1$ | `mhd` |
| `set_ic_a2_funcs(a2_func, da2_func)` | $a_2$ | `mhd` |
| `set_ic_a3_funcs(a3_func, da3_func)` | $a_3$ | `mhd` |

Components not set default to zero. Components not present in the chosen physics type are silently ignored.
The derivative argument is optional for components that use a quadratic basis function by default
($\rho_1$, $v_2$, $v_3$, $T_1$, $a_1$) and required for those that use a cubic one ($v_1$, $a_2$, $a_3$).

## Post-processing with Pylbo

Once Legolas has run, load the datfile and retrieve the snapshots:

```python
import pylbo
import matplotlib.pyplot as plt

ds = pylbo.load("output/my_datfile.dat")

if ds.has_iv_snapshots:
    ivp = ds.get_iv_snapshots()
```

The returned `IVPSolution` object provides several methods for inspection:

### Space-time heatmap

```python
fig, ax = plt.subplots()
ivp.plot_space_time_heatmap("rho", ax=ax)
plt.show()
```

### Spatial profiles at selected snapshots

```python
fig, ax = plt.subplots()
ivp.plot_spatial_slices("rho", snap_indices=[0, 25, 50, 99], ax=ax)
plt.show()
```

### Accessing raw data

```python
# shape: (n_snapshots, n_points)
rho_data = ivp.get_component("rho")

# physical times at each snapshot
print(ivp.times)

# spatial coordinate
print(ivp.x_domain)
```
