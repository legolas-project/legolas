---
title: Implementing a custom setup
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: true
toc_icon: "chevron-circle-down"
last_modified_at: 2023-04-13
---

This page explains how to configure the custom user submodule to your needs. We assume that you are already familiar with
how to set up a legolas directory and how to run legolas in a separate directory. If this is not the case see [how to run your first problem](../../getting-started/running).
In what follows we also assume that both a `smod_user_defined.f08` file and parfile are present in the current working directory. You can copy both of these over when running
the setup script. Alternatively, you can use Python to define an analytic (SymPy) or numerical configuration, and use `Pylbo`'s submodule `Gimli` to generate your Legolas files.
More information can be found [here](../../sphinx/autoapi/pylbo/index.html#pylbo.gimli).

## Read before proceeding
When implementing your own setup it is useful to know how Legolas treats the user submodule. The program state at a given time is governed by the following objects (click the links for a full overview of accessible attributes and type-bound procedures):
- [`settings`](../../ford/type/settings_t.html) <br>
  This object governs the overall settings of the code, which contains various attributes corresponding to other dedicated settings objects
  (block dimensions, grid settings, physics settings, etc.). Most of these attributes are accessible through the corresponding type-bound procedures.
- [`grid`](../../ford/type/grid_t.html) <br>
  This object governs the base, Gaussian, and eigenfunction grids, as well as the scale factors. In most cases there is no need to access this object,
  except when you want to customise the grid spacing or use a custom grid.
- [`background`](../../ford/type/background_t.html) <br>
  This object contains the functions defining the background equilibrium state, by default all these functions point to zero. You as a user should set these for your specific
  case (see below).
- [`physics`](../../ford/type/physics_t.html) <br>
  This object contains the various physics functions, by default these are set to the pre-implemented ones (see [here](../../physics/equations)). Most of these can be overridden (see below) if desired.

These objects are passed to the `user_defined_eq` subroutine in the user submodule with `intent(inout)`. Note that all variables should be specified using
the global parameter `dp`, corresponding to the double precision value `real64` from `iso_fortran_env`.

## Structure of the submodule
The general template of the submodule looks like this
```fortran
submodule (mod_equilibrium) smod_user_defined
  implicit none

contains

  module procedure user_defined_eq

    ! ...

  end procedure user_defined_eq
end submodule smod_user_defined
```
The `mod_equilibrium` statement between brackets refers to the [parent module](../../ford/module/mod_equilibrium.html), which contains the interface of the procedures that are implemented in their respective submodules. This also implies that you have access to all public variables/routines/functions that are either defined or used in the parent module through host association, without having to `use` them explicitly.

## Defining the grid
### Using the default grid
To use a default, uniformely spaced base grid, Legolas needs the geometry, gridpoints and grid edges.
This can be done in the submodule itself by accessing the `settings%grid` object as shown below, or through the parfile.
```fortran
module procedure user_defined_eq
  call settings%grid%set_geometry("Cartesian")
  call settings%grid%set_gridpts(100)
  call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)
end module procedure user_defined_eq
```

### Using a custom grid
A second option is defining a grid yourself. There are no restrictions on a custom grid (besides it needing to be monotically increasing), so you can supply any array
you want. Note that the size of your custom array must be consistent with the number of gridpoints you set in the submodule/parfile. Setting a custom grid is done by accessing
the `grid` object:
```fortran
module procedure user_defined_eq
  real(dp), allocatable :: my_custom_grid(:)

  allocate(my_custom_grid(settings%grid%get_gridpts()))
  ! fill the grid however you want
  call grid%set_custom_grid(my_custom_grid)
end module procedure user_defined_eq
```

### Defining mesh accumulation
A third option is modifying the spacing function, which may be more convenient than defining a custom grid. An example is given below, where a Gaussian spacing function is used, resulting in grid accumulation near the peak.
{% capture note %}
<i class="fa fa-lightbulb" aria-hidden="true"></i>
The spacing function governs `dx`, meaning that smaller function values imply a smaller grid spacing, and hence more points. Also, when using this method the number of
gridpoints will vary to accomodate the specified spacing function.
{% endcapture %}
<div class="notice--info">
  {{ note | markdownify }}
</div>
```fortran
module procedure user_defined_eq
  call settings%grid%set_grid_boundaries(-1.0_dp, 2.0_dp)
  call grid%set_spacing_function(gaussian_dx)
end procedure user_defined_eq

real(dp) function gaussian_dx(x)
  real(dp), intent(in) :: x
  real(dp) :: base, low, center, width
  base = 0.15_dp
  low = 0.01_dp
  center = 0.5_dp
  width = 0.2_dp
  gaussian_dx = base - (base - low) * exp(-(x - center)**2 / (2.0_dp * width))
end function gaussian_dx
```
Note that the function is placed outside the `user_defined_eq` procedure.

## Using variables
Legolas uses a dedicated module which contains a multitude of variables that you can use in your setups.
For a comprehensive list we refer to [this page](../../ford/module/mod_equilibrium_params.html#variable-k2),
an example is given below.

```fortran
submodule (mod_equilibrium) smod_user_defined
  use mod_equilibrium_params, only: alpha, beta, p1
  implicit none

contains

  module procedure user_defined_eq
    if (settings%equilibrium%use_defaults) then
      k2 = 0.0_dp
      k3 = 1.0_dp
      alpha = 1.0_dp
      beta = 5.0_dp
      p1 = 1.0_dp / 3.0_dp
    end if
  end procedure user_defined_eq
end submodule smod_user_defined
```
This example uses the variables `alpha`, `beta` and `p1` from the list and wraps them in the `use_defaults` condition.
This means that Legolas will use those values as a default if the corresponding variable is not given in the parfile's `paramlist` namelist.
Note that all these variables are initialised using `NaN`, so you will immediately notice if you forgot to set one.

## Setting units
Specifying the unit normalisations can either be done through the parfile (the `unitslist`) or in the submodule directly.
Setting units is done by accessing the `settings%units` object, take a look at the type-bound procedures [here](../../ford/type/units_t.html)
for an overview of the available options. All units should be in cgs, the units below are the default ones.
```fortran
module procedure user_defined_eq
  call units%set_units_from_temperature( &
    unit_length=1.0e9_dp, &  ! cm
    unit_magneticfield=10.0_dp, &  ! Gauss
    unit_temperature=1.0e6_dp, &  ! K
    mean_molecular_weight=0.5_dp &  ! optional
  )
end procedure user_defined_eq
```

## Defining the equilibrium configuration
<i class="fa fa-file-contract" aria-hidden="true"></i>
**Disclaimer**: due to the finite element representation used by Legolas the equilibrium profiles (or their derivatives) do not have to be
continuous. Do note though that smooth, continuous profiles give the best results, and you should try to achieve this as much as possible. Instead of a discontinuity
you can try to connect different regions with a smooth sine or hyperbolic tangent profile. This is not at all required and Legolas will run just fine with strong jumps or gradients,
but know that this may lead to artifacts in the eigenfunctions and/or spectrum.
{: .notice--danger}

Setting the actual equilibrium configuration is done by calling the corresponding type-bound procedure of the `background` object, and provide a function.
All background functions take either no arguments, or a single `real(dp)` scalar argument indicating position. The example below sets a homogeneous $\rho_0$ profile and a position-dependent $T_0$ and $v_y$ profile.
For a full overview of the available background functions see [here](../../ford/type/background_t.html), or take a look at all the various
pre-implemented equilibria [here](../../general/equilibria).
```fortran
submodule (mod_equilibrium) smod_user_defined
  use mod_equilibrium_params, only: alpha
  implicit none

contains

  module procedure user_defined_eq
    call settings%grid%set_geometry("Cartesian")
    call settings%grid%set_grid_boundaries(0.0_dp, 1.0_dp)

    alpha = 1.0_dp
    k2 = 0.0_dp
    k3 = 1.0_dp

    ! Note: functions that are not set are automatically set to zero.
    call background%set_density_funcs(rho0_func=rho0)
    ! dT0 denotes derivative with respect to position
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0)
    call background%set_velocity_2_funcs(v02_func=v02, dv02_func=dv02, ddv02_func=ddv02)
  end procedure user_defined_eq

  real(dp) function rho0()
    rho0 = alpha
  end function rho0

  real(dp) function v02(x)
    real(dp), intent(in) :: x
    v02 = 2.0_dp * x**2
  end function v02

  real(dp) function dv02(x)
    real(dp), intent(in) :: x
    dv02 = 4.0_dp * x
  end function dv02

  real(dp) function ddv02()
    ddv02 = 4.0_dp
  end function ddv02

  real(dp) function T0(x)
    real(dp), intent(in) :: x
    T0 = x
  end function T0

  real(dp) function dT0()
    dT0 = 1.0_dp
  end function dT0
end submodule smod_user_defined
```
Note that you can specify variables in module scope (like `alpha` in the example above) and use them in the background functions.

<i class="fas fa-lightbulb" aria-hidden="true"></i>
**Tip:** to make sure your implementation is in order you can first run without solving the eigenvalue problem by setting `solver = "none"` in the `solvelist`.
You can then plot the equilibrium profiles using Pylbo.
{: .notice--info}


## Including additional physics
Enabling physics can be done either through the parfile, or in the submodule directly.
In case of the latter this is done by accessing the corresponding type-bound procedures in the `settings%physics` object,
for a full overview see [here](../../ford/type/physics_t.html).

### Resistivity
Below is an overview of the options for resistivity:
```fortran
! activate temperature-dependent resistivity (Spitzer)
call settings%physics%enable_resistivity()
! OR set resistivity to a constant value
call settings%physics%enable_resistivity(fixed_resistivity_value=0.0001_dp)

! additionally you can specify a dropoff profile
settings%physics%resistivity%use_dropoff = .true.
settings%physics%dropoff_edge_dist = 0.05_dp
settings%physics%dropoff_width = 0.2_dp
```
For a user-defined resistivity implementation you can repoint the corresponding function in the `physics` object, see [here](../../ford/type/physics_t.html#boundprocedure-set_resistivity_funcs) for more info.
This is exactly the same as for the background functions, see an example below.

```fortran
submodule (mod_equilibrium) smod_user_defined
  use mod_equilibrium_params, only: alpha
  implicit none

contains

  module procedure user_defined_eq
    call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0)
    call physics%set_resistivity_funcs(eta_func=eta, detadT_func=detadT)
  end procedure user_defined_eq

  real(dp) function T0(x)
    real(dp), intent(in) :: x
    T0 = x
  end function T0

  real(dp) function dT0()
    dT0 = 1.0_dp
  end function dT0

  real(dp) function eta(x)
    real(dp), intent(in) :: x
    eta = T0(x)**(5.0_dp / 2.0_dp)
  end function eta

  real(dp) function detadT(x)
    real(dp), intent(in) :: x
    detadT = (5.0_dp / 2.0_dp) * T0(x)**(3.0_dp / 2.0_dp)
  end function detadT
end submodule smod_user_defined
```
Note that you can always define module-scope variables, set them in the `user_defined_eq` procedure, and use them in the various functions.

### Thermal conduction
Thermal conduction parallel and perpendicular to the magnetic field lines is treated separately.
In the absence of a magnetic field (hydrodynamics) perpendicular conduction uses the same functions as parallel conduction (isotropic).
```fortran
! activate parallel Spitzer conductivity
call settings%physics%enable_parallel_conduction()
! activate perpendicular Spitzer conductivity
call settings%physics%enable_perpendicular_conduction()
! OR set conduction to a constant value
call settings%physics%enable_parallel_conduction(fixed_tc_para_value=0.0001_dp)
call settings%physics%enable_perpendicular_conduction(fixed_tc_perp_value=0.0001_dp)
```
Both the parallel and perpendicular conduction functions can be user-defined as well using [these](../../ford/type/physics_t.html#boundprocedure-set_parallel_conduction_funcs) type-bound procedures.

### Radiative cooling and heating
To have an equilibrium in thermal balance you should enable heating and leave the `force_thermal_balance` option to `.true.`.
This will automatically set the heating in such a way that the thermal balance equation is satisfied.
To run the code outside of thermal balance you can set this flag to false.
```fortran
call settings%physics%enable_cooling(cooling_curve="jc_corona")
! to achieve thermal balance, enable heating
call settings%physics%enable_heating(force_thermal_balance=.true.)
```
Both cooling and heating can be user-specified using [these](../../ford/type/physics_t.html#boundprocedure-set_cooling_funcs) type-bound procedures.

### Flow
```fortran
call settings%physics%enable_flow()
```

### External gravity
Enable gravity and specify the gravity function to be used.
```fortran
submodule (mod_equilibrium) smod_user_defined
  implicit none

contains

  module procedure user_defined_eq
    call settings%physics%enable_gravity()
    call physics%set_gravity_funcs(g0_func=g0)
  end procedure user_defined_eq

  real(dp) function g0(x)
    real(dp), intent(in) :: x
    g0 = 1.0_dp / x**2
  end function g0
end submodule smod_user_defined
```

### Viscosity
```fortran
call settings%physics%enable_viscosity(viscosity_value=0.0001_dp, viscous_heating=.false.)
```
Viscosity uses a constant value and does not accept a user-defined function.

### Hall MHD
```fortran
call settings%physics%enable_hall(electron_inertia=.true., electron_fraction=0.5_dp)
! with optional dropoff profile in the hall factor
settings%physics%hall%use_dropoff = .true.
! with optional dropoff profile in the electron inertia
settings%physics%hall%use_inertia_dropoff
settings%physics%dropoff_edge_dist = 0.05_dp
settings%physics%dropoff_width = 0.2_dp
```
The Hall treatment is fixed and does not accept user-defined functions.
