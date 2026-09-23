---
title: Interfacing with Legolas
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: true
toc_icon: "chevron-circle-down"
last_modified_at: 2026-01-28
---
{% capture tip %}
<i class="fas fa-lightbulb" aria-hidden="true"></i>
**Tip:** when specifying paths using Python, the standard library package [`pathlib`](https://docs.python.org/3/library/pathlib.html)
is really recommended. This will resolve filepaths on any file system and lets you easily work with relative paths as well.
That way you can specify the relative path to a file, copy the script over to another machine, run it, and the path strings
get automatically updated.
```python
from pathlib import Path
output_dir = Path("output").resolve() # <-- this contains the full path, starting from root
```
{% endcapture %}
<div class="notice--success">
  {{ tip | markdownify }}
</div>

## Generating parfiles
Pylbo comes with a method to generate parfiles for you, something that is extremely handy when doing multiruns
and you want to vary one or more parameters. All that is needed is to specify the namelist items in a dictionary,
and pass them to the [`generate_parfiles`](../../sphinx/autoapi/pylbo/index.html#pylbo.generate_parfiles) method.
You don't have to specify which variable goes where, Pylbo will automatically take care of that and place them in their
corresponding namelist. Additionally, Pylbo will do typechecking on all parameters that you supply to the generator,
such that you're sure that the correct datatypes are passed on to Legolas.
Furthermore, you can ask that filepaths and output folders are automatically resolved and placed
in the parfile as well, see below for an example.

### Generating a single parfile
To generate a single parfile, specify the setup as a dictionary. An example for Suydam cluster modes is given below:
```python
import pylbo

config = {
    "geometry": "cylindrical",
    "x_start": 0,
    "x_end": 1,
    "gridpoints": 101,
    "parameters": {
        "k2": 1.0,
        "k3": -1.2,
        "cte_rho0": 1,
        "cte_v02": 0,
        "cte_v03": 0.14,
        "cte_p0": 0.05,
        "p1": 0.1,
        "alpha": 2.0
    },
    "equilibrium_type": "suydam_cluster",
    "write_eigenfunctions": True,
    "basename_datfile": "suydam_modes",
    "output_folder": "output"
}
parfile = pylbo.generate_parfiles(parfile_dict=config)
```
When the key `parameters` (this should be a dictionary itself) is found, Pylbo automatically sets `use_defaults` to
`False`. No additional arguments are given to the generator, so the above code sample will create a parfile named `parfile.par` by default
in a folder `parfiles` in the current working directory.
The actual parfile looks like this:
```fortran
&equilibriumlist
    equilibrium_type = 'suydam_cluster'
    use_defaults = .false.
/

&gridlist
    geometry = 'cylindrical'
    gridpoints = 101
    x_end = 1
    x_start = 0
/

&paramlist
    alpha = 2.0
    cte_p0 = 0.05
    cte_rho0 = 1
    cte_v02 = 0
    cte_v03 = 0.14
    k2 = 1.0
    k3 = -1.2
    p1 = 0.1
/

&savelist
    basename_datfile = 'suydam_modes'
    output_folder = 'output'
    write_eigenfunctions = .true.
/
```
which is a perfectly formatted Legolas namelist.

### Generating multiple parfiles
You'll probably be able to generate a single parfile faster by hand instead of using Pylbo. However, the method
described above excels in varying parameters and modifying the parfiles accordingly. Say, you want to vary the parameter
`alpha` in the above example between 1 and 4, and do 50 runs. This means you have to create and edit 50 parfiles by hand,
which is a long and painful chore. Instead, simply edit the above `config` dictionary with the following:
```python
import numpy as np

config = {
    ...
    "number_of_runs": 50,
    "parameters": {
        ...
        "alpha": np.linspace(1, 4, 50),
        ...
    },
    ...
},
```
where you have to add the `number_of_runs` entry.
Pylbo will throw an appropriate error if there is an inconsistency between this number and the number of runs you want.

Now you call `generate_parfiles` in exactly the same way, which will create 50 parfiles where a number `xxxx` is prepended
in front of the filename, the same holds true for the datfiles.
You can even combine multiple variations, see the example below:

```python
import numpy as np
import pylbo


config = {
    "geometry": "Cartesian",
    "x_start": 0,
    "x_end": 1,
    "number_of_runs": 50,
    "gridpoints": 100,
    "parameters": {
        "k2": 0.0,
        "k3": np.linspace(1, 5, 50),
        "beta": 0.25,
        "cte_rho0": 1.0,
        "cte_B02": 0.0,
        "cte_B03": 1.0,
    },
    "equilibrium_type": "resistive_homo",
    "resistivity": True,
    "fixed_resistivity_value": np.linspace(0.001, 0.01, 50),
    "logging_level": 0,
    "show_results": False,
    "write_eigenfunctions": True,
    "write_matrices": False,
    "solver": "arnoldi",
    "arpack_mode": "shift-invert",
    "sigma": 0.5 - 3j,
    "number_of_eigenvalues": 100,
},
parfiles = pylbo.generate_parfiles(
    parfile_dict=config, basename="parfile_resistive", output_dir="parfile_output"
)
```
The above example will generate 50 parfiles, in which the parameter `k3` is varied between 1 and 5, together
with a variation of the resistivity `fixed_eta_value`. This means that the first parfile will have `k3 = 1` and
`eta = 0.001`, and the last one will have `k3 = 5` and `eta = 0.01`. The eigenvalue problem will be solved using the Arnoldi shift-invert method,
for 100 eigenvalues around a sigma-value of `0.5 - 3i`.

Note that you can change as many parameters as you want, simply add any namelist item as a dictionary key.
The parfiles will be called `xxxxparfile_resistive.par`due to the fact that we added the `basename` argument.
All parfiles will be placed in a directory called `parfile_output` relative to the current working directory. If this is not desired
you can supply a full path instead, either as a string (e.g.`output_dir="/users/Documents/parfiles"`) or a PathLike object (e.g. `Path("../parfiles").resolve()`).

## Running Legolas with Pylbo
The parfiles generated in the above examples can be passed on to Pylbo, which in turn will pass those on to Legolas.

<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>
Note that Pylbo will always link to the executable in the Legolas home directory by default. It is therefore good practice to explicitly specify the Legolas executable when calling the runner.
{: .notice--danger}

### Single run
To do a single Legolas run you specify the parfile and call the runner, like so:
```python
import pylbo
pylbo.run_legolas("parfiles/my_parfile.par", executable="legolas")
```
This will use `my_parfile.par` from the `parfiles` directory, and the executable `legolas`, both relative
to the current working directory. Parfiles can be supplied either as (a list or array of) PathLike objects or strings.

### Multiple runs
Pylbo can run multiple parfiles at the same time. Say you have a local directory called `parfiles` containing a bunch of
generated parfiles, and would like to run Legolas on all of them.
You can either pass the result from the parfile generator directly to the runner, or you can
do it manually. In case of the latter then [`pathlib`](https://docs.python.org/3/library/pathlib.html) comes in
handy: PathLike objects support [`glob`](https://docs.python.org/3/library/glob.html). This means that you can automatically search for parfiles based on a given string,
and don't have to go through the trouble of adding all paths manually.

In the example below, we first search for all files with a `.par` extension in the `parfiles` directory and sort them name-wise.
When we have the list of parfiles we simply pass them to the runner, which can either be done single-threaded or multi-threaded depending on
whether or not the optional keyword `nb_cpus` is given.
```python
from pathlib import Path
import pylbo

parfiles = sorted(Path("parfiles").glob("*.par"))
# this runs single-threaded
pylbo.run_legolas(parfiles, executable="legolas")
# this runs multi-threaded
pylbo.run_legolas(parfiles, nb_cpus=4, executable="legolas")
```
The second case will use Python's [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module to parallelise the
number of runs across the amount of CPUs requested (4 in this case). Every CPU will have 1 instance of Legolas running, and a progress bar
will be printed to keep track of the progress.

The optional keyword argument `remove_parfiles` can be supplied as well, which is False by default. If this is set to True,
the parfiles will be removed after the runs are completed. Only if the folder containing the parfiles is empty after all parfiles are removed,
the folder is removed as well. This may be handy in case you want to automatically clean up afterwards.
More information on the Legolas runner can be found [here](../../sphinx/autoapi/pylbo/index.html#pylbo.run_legolas).
