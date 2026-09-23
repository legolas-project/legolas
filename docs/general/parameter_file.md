---
title: The parfile
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: true
toc_label: "Namelists"
toc_icon: "code"
last_modified_at: 2023-04-13
---

The Legolas parfile allows for a full customisation of all Legolas variables, and most of these have default values.
Legolas explicitly inspects your equilibrium for nonsensical values and will warn you if something is not right.
Whenever the code catches something during its inspection phase it either logs a warning and continues running, or
exits with an error depending on the severity of the issue.

You can create the parfile manually using this guide, or let Pylbo take care of generating one.
More information on generating parfiles with Pylbo can be found [here](../../pylbo/legolas_interface/#generating-parfiles).

## gridlist
This namelist includes all grid-related variables.

<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>
**Warning:** For cylindrical geometries we override the start value with $r = 0.025$ if it is initialised
with zero. Some matrix elements scale with $1/r$ and may become very large when $r \rightarrow 0$, resulting in very large elements which may throw off some solvers.
Note that the matrix elements and equilibria are evaluated in the _Gaussian_ grid, the first point of which may be very small but will never be exactly zero. <br><br>
Setting `force_r0 = .true.` in the gridlist forces the `r = 0` condition, but this is not recommended.
{: .notice--warning}

| Parameter       | Type     | Description | Default value     |
| :---            | :---:    |    :----  |          :---:      |
| geometry        | string   | geometry of the setup   | `"Cartesian"` |
| gridpoints      | int      | number of gridpoints in the base grid  |  50 |
| x_start         | real     | starting point of the base grid  | 0 |
| x_end           | real     | end point of the base grid  | 1 |
| coaxial         | logical  | use a coaxial inner boundary in cylindrical geometry | `.false.` |
| force_r0        | logical  | forces `r=0` in cylindrical geometry | `.false.` |
| symmetric_grid  | logical  | enforces grid symmetry around the centre of the interval (for use with a symmetric spacing function) | `.false.` |
| grid_edge_from_data | logical | sets x_start/x_end based on data imported with `import_equilibrium_data` | `.false.` |

## equilibriumlist
This namelist includes all equilibrium-related variables.

| Parameter    | Type | Description      | Default value     |
| :---         | :---: |   :----        |          :---:      |
| equilibrium_type  | string | the equilibrium configuration to use, see the [various options](../../general/equilibria). Set to `"user_defined"` to use a custom configuration. | `"adiabatic_homo"`    |
| boundary_type |  string |   boundary conditions to use  | `"wall"`  |
| use_defaults  | logical | if `.true.`, uses the default values in the equilibrium submodule | `.true.` |


## physicslist
This namelist includes all physics-related variables.

| Parameter             | Type    | Description      | Default value     |
| :---                  |  :---:  |  :----        |          :---:      |
| physics_type          | string  | physics type to use, can be `{"mhd", "hd", "hd-1d"}` | `"mhd"` |
| basis_functions       | list    | basis functions corresponding to the state vector components, options are `{"quadratic", "cubic"}`. By default ($\rho_1$, $v_2$, $v_3$, $T_1$, $a_1$) are `"quadratic"`, ($v_1$, $a_2$ and $a_3$) are `"cubic"`. | - |
| mhd_gamma             | real    | ratio of specific heats $\gamma$ | $\frac{5}{3}$ |
| incompressible | logical | whether to use the incompressible approximation, this sets $\gamma = 10^{12}$ and eliminates some matrix elements | `.false.` |
| flow  | logical | inclusion of background flow effects | `.false.` |
| radiative_cooling | logical | whether to include optically thin radiative losses | `.false.` |
| ncool | int | number of points used when interpolating cooling curves | 4000 |
| cooling_curve | string | which cooling curve to use, can be `{"nothing", "JCcorona", "DM", "DM_2", "MLsolar1", "SPEX", "SPEX_DM", "Rosner", "Colgan", "Colgan_DM"}`. In the case of `"nothing"` you should define your own $\Lambda(T)$ cooling function and its temperature derivative. | `"nothing"` |
| heating | logical | whether to include background heating | `.false.` |
| force_thermal_balance | logical | whether to set the heating in such a way to enforce thermal equilibrium | `.true.` |
| external_gravity | logical | whether to include external gravity | `.false.` |
| parallel_conduction | logical | whether to include parallel thermal conduction | `.false.` |
| fixed_tc_para_value | real | if given (and not 0) a constant value for the parallel thermal conduction will be used | 0 |
| perpendicular_conduction | logical | whether to include perpendicular thermal conduction | `.false.` |
| fixed_tc_perp_value | real | if given (and not 0) a constant value for the perpendicular thermal conduction will be used | 0 |
| resistivity | logical | whether to include resistivity | `.false.` |
| fixed_resistivity_value | real | if given (and not 0) a constant value for the resistivity will be used | 0 |
| use_eta_dropoff | logical | if `.true.`, smoothly drops off the resistivity profile to zero near the edges using a hyperbolic tangent profile | `.false.` |
| viscosity | logical | whether to include viscosity | `.false.` |
| viscosity_value | real | constant value for the viscosity | 0 |
| viscous_heating | logical | whether to include viscous heating | `.false.` |
| hall_mhd | logical | whether to include Hall MHD effects | `.false.` |
| hall_dropoff | logical | if `.true.` the Hall MHD profile is smoothly dropped off to zero near the edges using a hyperbolic tangent profile | `.false.` |
| elec_inertia | logical | whether to include electron inertia | `.false.` |
| inertia_dropoff | logical | if `.true.` the electron inertia profile is smoothly dropped off to zero near the edges using a hyperbolic tangent profile | `.false.` |
| electron_fraction | real | fraction of electrons in the plasma | 0.5 |
| dropoff_edge_dist | real | distance between the grid edge and the smoothened dropoff profile | 0 |
| dropoff_width | real | the width over which the profile is smoothened | 0 |


## solvelist
This namelist includes all solver-related variables. For more information, see [Solvers](../../general/solvers)

| Parameter     | Type  | Description   | Default value     |
| :---          | :---: | :----         | :---:             |
| solver | string | which solver to use, can be `{"none", "arnoldi", "QR-invert", "QR-cholesky", "QZ-direct", "inverse-iteration"}` | `"QR-invert"` |
| arpack_mode | string | the mode for arpack (Arnoldi) | `"general"` |
| which_eigenvalues | string | which eigenvalues to calculate (Arnoldi). Can be `{"LM", "SM", "LR", "SR", "LI", "SI"}` | `"LM"` |
| number_of_eigenvalues | int | number of eigenvalues ($k$) to calculate (Arnoldi) | 10 |
| maxiter | int | maximum number of iterations (Arnoldi, shift-invert) | `max(100, 10N)`|
| ncv | int | number of basis vectors to use (Arnoldi) | `max(k+1, min(2k, N))` |
| tolerance | real | tolerance for eigenvalue convergence (Arnoldi, shift-invert) | $5 \times 10^{-15}$ |
| sigma | complex | shift in the complex plane for iterative methods (Arnoldi, shift-invert) | `NaN + NaNj` |


## unitslist
This namelist includes all units-related variables, all units are in cgs. Defaults are
set from a unit temperature, unit magnetic field and unit length.

| Parameter    | Type | Description      | Default value     |
| :---         | :---: |  :----        |          :---:      |
| unit_numberdensity | real | sets the unit number density in cm$^{-3}$ | - |
| unit_density  | real | sets the unit density in g cm$^{-3}$ | - |
| unit_temperature  | real | sets the unit temperature in K | $10^6$ |
| unit_magneticfield |  real | sets the unit magnetic field in Gauss   | $10$    |
| unit_length   | real | sets the unit length in cm | $ 10^9$ |
| He_abundance | real | the He abundance fraction (between 0 and 1) | 0 |


## savelist
This namelist includes all variables related to data output.

{% capture notice-info %}
**Note:** Legolas uses a logging-based system to write information to the console, similar to Python's logging module.
The level of output is controlled through the integer `logging_level`:
* **level 0** : only critical errors are printed, everything else is suppressed.
* **level 1** : only critical errors and warnings are printed
* **level 2** : prints info messages, warnings and critical errors (default)
* **level 3** : (or higher) prints debug messages in addition to all of the above
{% endcapture %}
<div class="notice--info">
  {{ notice-info | markdownify }}
</div>

| Parameter    | Type | Description      | Default value     |
| :---         | :---: |   :----        |          :---:      |
| write_matrices | logical | whether to write the matrices to the datfile | `.false.` |
| write_eigenvectors | logical | whether to write the eigenvectors to the datfile | `.false.` |
| write_residuals | logical | whether to write the residuals to the datfile | `.false.` |
| write_background | logical | whether to write the background to the datfile | `.true.` |
| write_eigenfunctions | logical | whether to write the eigenfunctions to the datfile | `.true.` |
| write_derived_eigenfunctions | logical | whether to write the derived eigenfunctions to the datfile | `.false.` |
| write_eigenfunction_subset | logical | if `.true.` only writes eigenfunctions for eigenvalues inside a given radius around a point in the complex plane | `.false.` |
| eigenfunction_subset_radius | real | radius of the subset | `NaN` |
| eigenfunction_subset_center | complex | center of the subset | `NaN + NaNj` |
| show_results | logical | whether to call the Pylbo script after the run | `.true.` |
| basename_datfile | string | basename of the datfile | `"datfile"` |
| output_folder | string | folder to write the datfile to | `"output"` |
| logging_level | int | controls console information, see note above | 2 |


## paramlist
This namelist handles constant parameters which are used in the equilibrium prescriptions. This is meant to be used in parameter studies, for a comprehensive list of all the possible parameters we refer to [this page](../../ford/module/mod_equilibrium_params.html) in the
Legolas source code documentation. To see which parameters are used in which equilibria, take a look
[here](../equilibria/).

<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>
**Note**: setting a variable with the paramlist does **NOT** automatically include it in the equilibrium configuration. For this to have any effect you have to _use_ that
variable in your submodule as well. For example, if you never set a magnetic field in your submodule and never use the `cte_B02` variable,
setting `cte_B02 = 1.0d0` in the paramlist will have no effect. Take a look at the different equilibria to see which parameters are supported.
{: .notice--warning}
