---
title: Legolas file generation
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: false
last_modified_at: 2026-02-04
---

Rather than writing the Legolas user module yourself, including calculating all the required derivatives, GIMLI provides a framework to do that work for you, reducing the room for human error. Relying on Python's symbolic package [SymPy](https://www.sympy.org/en/index.html), the equilibrium can be defined analytically. Supplemented with a dictionary of parameter values and code configuration details, GIMLI will generate the user module and parfile(s) for you. Below, we discuss GIMLI's three important Python objects to achieve this. The API can be found [here](../../sphinx/autoapi/pylbo/gimli/index.html). For an interactive tutorial, Jupyter notebooks are also available in the [Legolas repository](https://github.com/legolas-project/legolas) in the `notebooks` directory.

## Importing `gimli` and creating variables
To use GIMLI, we first import it explicitly from the `pylbo` package, alongside SymPy, by adding
```python
import sympy as sp
import pylbo.gimli as gl
```
to the header of our script. This will introduce 5 new classes: `Variables`, `Equilibrium`, `Legolas`, `Amrvac`, and `NumericalEquilibrium`. In this tutorial, we are only concerned with the first three. For more on the other two, check out the section [Interfacing with MPI-AMRVAC](../../gimli/amrvac_interface).

Though GIMLI is built on SymPy (and thus uses its syntax), its list of accepted symbols is limited to those defined by the `Variables` object, which have direct equivalents in Legolas's source code. All symbols defined in the `Variables` class are listed in the [API](../../sphinx/autoapi/pylbo/gimli/index.html). To start using the `Variables` class, simply assign an instance of the class to a Python variable, e.g.
```python
var = gl.Variables()
```
In a Jupyter notebook you can now test the definition of symbolic parameters with a `display` command, e.g.
```python
display(var.alpha)
```
which should display as $\alpha$.

## Defining the equilibrium
With our set of variables, we can now define an equilibrium. For this demonstration, we define an isothermal Harris sheet, e.g.

$$
\begin{gather}
    T_0 = \text{constant}, \\
    \mathbf{B}_0(x) = B_0 \tanh\left( \frac{x}{\alpha} \right)\,\hat{\mathbf{e}}_y, \\
    \rho_0(x) = \rho(0) - \frac{\mathbf{B}_0^2(x)}{2T_0}.
\end{gather}
$$

Using our `var` object, this becomes
```python
B2field = var.B2c * sp.tanh(var.x / var.alpha)
B3field = 0
temperature = var.Tc
density = var.rhoc - (B2field**2 + B3field**2) / (2.0 * temperature)
v2field = 0
v3field = 0
```
First of all, notice that all parameters that we want to assign a numerical value to later should use variable names available in `var`, e.g. `B2c` for the magnetic field strength, since they correspond to variables in Legolas itself. Secondly, since GIMLI is built on SymPy, it can interpret SymPy-native functions like `tanh` without issue. Finally, the velocity components are required arguments of the `Equilibrium` class, so we specify them to zero explicitly.

With our equilibrium expressions, we now define the `Equilibrium` object:
```python
equilibrium = gl.Equilibrium(var, density, v2field, v3field, temperature, B02=B2field, B03=B3field)
```
Note that the first argument is the `Variables` object used in the definition of the equilibrium expressions, followed in order by the density, the $v_2$ and $v_3$ velocity components, and the temperature. For hydrodynamic configurations this would suffice, however, since we are dealing with an MHD equilibrium, we add the magnetic field components as optional arguments. More profiles can be defined as optional arguments for certain physical effects (see the [API](../../sphinx/autoapi/pylbo/gimli/index.html)).

## Generating Legolas files
With our equilibrium set up, all that is left to do is setting the parameter values and code configuration. This is done through a dictionary following the formatting of the parfile generator described on the [Interfacing with Legolas](../../pylbo/legolas_interface) page. For our Harris sheet, a minimal configuration may look like
```python
legolas_config = {
    "geometry": "Cartesian",
    "x_start": -10,
    "x_end": 10,
    "gridpoints": 100,
    "solver": "QR-cholesky",
    "parameters": {
        "k2": 0.5,
        "k3": 0,
        "cte_rho0": 1,
        "cte_T0": 1,
        "cte_B02": 1,
        "alpha": 1
    },
    "equilibrium_type": "user_defined",
    "physics_type": "mhd",
    "resistivity": True,
    "fixed_resistivity_value": 1e-3,
    "output_folder": './output/'
}
```
where the parameters `cte_rho0`, `cte_T0`, `cte_B02`, and `alpha` correspond to the GIMLI variables `var.rhoc`, `var.Tc`, `var.B2c`, and `var.alpha` respectively (for any discrepancies in variable names, see the [API](../../sphinx/autoapi/pylbo/gimli/index.html)). Note that `equilibrium_type` should always be set to `user_defined` for GIMLI-generated modules.

Now we define a Legolas object using both the equilibrium and the configuration dictionary,
```python
legolas_setup = gl.Legolas(equil, legolas_config)
```
from which a user module, including all derivatives, can be generated in a local `legolas/` directory with
```python
legolas_setup.user_module(loc="./legolas/")
```
as well as a parfile with
```python
parfiles = legolas_setup.parfile(loc="./legolas/")
```
Note that setting `loc` does not create a directory. This should be done beforehand. If `loc` is not specified, the files will be generated in the local directory.

To run your generated problem, simply compile Legolas from the terminal (for this example in the `legolas/` directory) and execute as usual:
```bash
setuplegolas.py
buildlegolas.sh
./legolas -i legolas_config.par
```