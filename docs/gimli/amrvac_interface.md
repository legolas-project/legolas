---
title: Interfacing with MPI-AMRVAC
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: false
last_modified_at: 2026-06-10
---

Aside from facilitating the setup of Legolas configurations, GIMLI offers two more functionalities, connecting Legolas to non-linear simulations. In the first application, we discuss how [MPI-AMRVAC](https://amrvac.org) simulations can be set up from a combination of a GIMLI equilibrium and a selection of perturbing eigenmodes, calculated with Legolas. A demonstration of this is also available in the [Legolas repository](https://github.com/legolas-project/legolas) in the `notebooks` directory. Secondly, we look at the analysis of one-dimensional numerical data with Legolas, e.g. from one-dimensional simulations or 1D slices of 2D/3D simulations.

## Initialising MPI-AMRVAC with Legolas data
After defining a GIMLI equilibrium as described on the [Legolas file generation](../../gimli/file_generation) page, and running Legolas, the equilibrium configuration and Legolas output can be combined to set up a non-linear simulation with MPI-AMRVAC (v3.3+) by perturbing that equilibrium with a selection of modes from the Legolas spectrum. Note that this requires saving the eigenfunctions to the datfile.

Continuing with the Harris sheet example, we define the simulation parameters in a dictionary, taking care to copy directly from the Legolas setup as much as possible:
```python
lam = 2. * np.pi / legolas_setup.config["parameters"]["k2"]

amrvac_config = {
    "physics_type": "mhd",
    "geometry": legolas_setup.config["geometry"],
    "equilibrium": equilibrium,
    "datfile": "./legolas/output/datfile.dat",
    "ev_guess": [0.1j],
    "quantity": "B02",
    "percentage": 0.01,
    "dim": 2,
    "u1_bounds": [legolas_setup.config["x_start"], legolas_setup.config["x_end"]],
    "u2_bounds": [-lam, lam],
    "parameters": legolas_setup.config["parameters"],
    "parfile": {
        "base_filename": "",
        "ditsave_log": 10,
        "dtsave_dat": 1,
        "dtmin": 1e-10,
        "time_max": 100,
        "time_stepper": "threestep",
        "flux_scheme": [20*["hll"]],
        "limiter": [20*["weno5"]],
        "check_small_values": False,
        "fix_small_values": True,
        "small_values_method": "replace",
        "small_pressure": 1e-14,
        "small_density": 1e-12,
        "typeboundary_min2": [6*["periodic"]],
        "typeboundary_max2": [6*["periodic"]],
        "domain_nx1": 256,
        "domain_nx2": 256,
        "refine_max_level": 1,
        "typecourant": "maxsum",
        "courantpar": 0.8,
        "mhd_eta": legolas_setup.config["fixed_resistivity_value"]
    }
}
```
Let us go over the dictionary in a bit more detail. First, we define the physics, geometry, and pass the `Equilibrium` object. Then, we specify the Legolas-related information, i.e. the path to the Legolas datfile, guesses for the eigenvalues to be included in the perturbation (`ev_guess`), and how to scale the perturbation. Defining `quantity` and `percentage` (factor between 0 and 1), the entire perturbation is scaled in such a way that the maximal perturbation in the specified quantity is a percentage of the maximal equilibrium value of that quantity. Alternatively, we can add `"energy_norm" : True` to scale the perturbation as a percentage of the total equilibrium energy. Lastly, we set the simulation details, i.e. the dimensionality `dim` of the simulation, the boundaries in each direction (here `u1_bounds` and `u2_bounds`; for 3D, add `u3_bounds`), and the equilibrium `parameters`, copied from the Legolas object to ensure consistency. Finally, the `parfile` argument takes a dictionary with any remaining MPI-AMRVAC variables. An overview can be found [here](https://amrvac.org/md_doc_2par.html). Note that we did not specify boundary conditions for the first direction (`typeboundary_{min/max}1`) because these are set to Legolas's wall boundary conditions (or `pole` if `r=0`is included in cylindrical coordinates). For numerical stability, it can be beneficial to apply the well boundary conditions or extrapolation only to the perturbation without the background. Relevant routines to be used together with `usr_special_bc` can be found in the MPI-AMRVAC `mod_gimli` module. Secondly, since we are using periodic boundary conditions in the second direction, we defined `u2_bounds` as a multiple of the wavelength of the perturbation we are introducing.

The following optional keywords provide additional features specific to MPI-AMRVAC:
- `save_analytics = True` enables additional analytics logging in the MPI-AMRVAC output in a file with suffix `_c.log`.
- `parfile["has_equi_rho_and_p"] = True` requests split equilibrium density/pressure handling. As a result, the datfile will contain only the perturbed variables so we set `mhd_dump_full_vars = True` to produce an additional datfile containing the full variables. The `convert_type` will be overridden for this to happen.
- If the equilibrium is in thermal balance, `parfile["mhd_equi_thermal"] = True` is added automatically when appropriate, which is important when split equilibrium fields are combined with heating/cooling.
- `parfile["B0field"] = True` splits off the equilibrium magnetic field and current (which it calculates from the `Equilibrium`). Again, the full variables are saved in an additional datfile.

Rather than providing eigenvalue guesses to `ev_guess`, we can also run
```bash
python pylbo_wrapper.py -i output/datfile.dat
```
in a terminal to select the tearing mode, and save its index in a `.npy` file by pressing `j`. It can then be read with Python (note: you may have to adapt the path) as
```python
import numpy as np
import pylbo as pb

ds = pb.load('./legolas/output/datfile.dat')
indices = np.load('./legolas/datfile.npy')
```
after which we set
```python
"ev_guess": [ds.eigenvalues[ix] for ix in indices]
```
in the dictionary. This also works as-is when selecting multiple modes from the spectrum for a superposition. In that case, after rescaling all modes to the same maximal amplitude in the chosen `quantity`, the modes are added together with equal weights, unless `weights` are specified as an array of length equal to the number of modes, summing to $1$. Similarly, the key `ef_factor` is used to define an array of complex factors (modulus $1$), one for each mode in the superposition.

Now, we pass the dictionary to the `Amrvac` class
```python
amrvac_setup = gl.Amrvac(amrvac_config)
```
and generate the MPI-AMRVAC files. To obtain the user module and parfile, run
``` python
amrvac_setup.user_module()
amrvac_setup.parfile()
```
which will place the user module in the current directory and create a subdirectory for the parfile. The location for each file can be specified with the optional argument `loc`. The user module uses a dedicated `mod_gimli` in MPI-AMRVAC that contains the relevant functions. For the parfile, the creation of a subdirectory can be suppressed by passing `subdir=False`. The last step is to write the Legolas data in a format that MPI-AMRVAC can read,
```python
amrvac_setup.prepare_legolas_data()
```
Again, `loc` is used to set the target directory. Setting the keyword `clean=False` will result in the pure Legolas output being saved. Otherwise, a transformation is applied to the eigenfunctions to determine whether they are real/imaginary or complex. For a purely real/imaginary eigenfunction, the remaining noise in the imaginary/real part is discarded, and then the inverse transformation is applied.

With these three files, we are ready to [compile MPI-AMRVAC](https://amrvac.org/md_doc_2getting__started.html) and run the problem.

## Numerical configurations in Legolas
Until now, all configurations in Legolas had to be defined analytically. However, the code also features a way to analyze one-dimensional numerical data. To achieve this, GIMLI is used to convert a set of numerical arrays to a Legolas-interpretable file. Legolas then imports the data from this file, interpolates to a uniform grid of user-specified resolution, and calculates the derivatives numerically.

{% capture note %}
<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>
**Note:** numerical derivatives are unreliable near sharp transitions, so it is highly recommended to inspect the numerical derivatives before interpreting results.
{% endcapture %}
<div class="notice--warning">
  {{ note | markdownify }}
</div>

As a toy example, consider the Harris current sheet once more. Using NumPy we define it numerically:
```python
import numpy as np

N = 1000
x = np.linspace(-10, 10, N)
temperature = np.ones(N)
magnetic_2 = np.tanh(x)
density = np.ones(N) - magnetic_2**2 / (2. * temperature)
```
This is then passed to the GIMLI class `NumericalEquilibrium` as a dictionary,
```python
import pylbo.gimli as gl

arrays = {
    "x" : x,
    "rho0" : density,
    "T0" : temperature,
    "B02" : magnetic_2
}
equilibrium = gl.NumericalEquilibrium(arrays)
```
The accepted dictionary keys are `x`, `r`, `u1`, `rho0`, `T0`, `v01`, `v02`, `v03`, `B01`, `B02`, `B03`, and `grav`, where exactly one of the first three (`x`, `r`, `u1`) should be present, and `grav` is used for gravity. All arrays should be of equal length, and missing keys are assigned a zero array of that length. To generate the data file for Legolas, call
```python
equilibrium.to_legolas_arrays()
```
to create a `.lar` file (**L**egolas **ar**rays) in the current directory. Alternatively, the location can be specified with the `loc` argument, and `filename` is used to set the name of the file.

To run Legolas on the numerical configuration, set the equilibrium in the parfile to `equilibrium_type = "numerical"`, and specify in `paramlist` both `input_file` (default `input_file = "arrays.lar"`) and `n_input`, the resolution for the interpolation. The predefined numerical equilibrium (`smod_equil_numerical.f08`) does not include `v01`, `B01`, and `grav`, but is easily modified if necessary.

**Note:** a sixth-order accurate central difference stencil is used to calculate the numerical derivatives. Near the edges a sixth-order accurate forward and backward difference stencil is used for the left and right boundary, respectively.
{: .notice--info}

{% capture note %}
<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>
**Note:** the calculation of the numerical derivatives assumes an equally spaced grid. Hence, this method cannot be combined with `set_custom_grid` or `set_spacing_function`.
{% endcapture %}
<div class="notice--warning">
  {{ note | markdownify }}
</div>

## Non-ideal physics
The example `notebooks` page in the [Legolas repository](https://github.com/legolas-project/legolas) contains setups where constant resistivity and heating/cooling with parallel thermal conduction are used in MPI-AMRVAC setups created with GIMLI. 

For background heating and optically thin radiative losses, an optional keyword `heatcool` can be added to the `Equilibrium`object. It is a dictionary containing the following keys: `force_thermal_equilibrium`, `cooling_curve`, and `ncool`. When `heatcool` is included in the equilibrium, the relevant parameters will be automatically added to the Legolas and MPI-AMRVAC parameter files. If not, the Legolas heating/cooling parameters will be added to `Equilibrium` during parfile generation.

Note that some features of Legolas are not present in MPI-AMRVAC:
- User-specified thermal conduction.
- Thermal balance when including perpendicular thermal conduction.
- Spitzer resistivity (when in Legolas `resistivity=.true.` but no value for `fixed_resistivity_value` is specified) or `use_eta_dropoff`.
GIMLI can be used in these cases to generate the Legolas setup, but will raise errors when trying to generate an MPI-AMRVAC setup.