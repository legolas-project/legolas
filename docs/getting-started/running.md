---
title: Running Legolas
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: false
last_modified_at: 2026-01-28
---

Once you cloned the repository and installed both Legolas and the Pylbo framework you are set to run your first problem.
As an example here we will run one of the pre-implemented equilibria already present in the source code.

## Setting up a directory
Navigate to any directory of your choosing where you want to place the configuration files and output, and call the setup script:
```bash
cd my_favourite_directory
setuplegolas.py
```
Follow the instructions, this script will do and ask you a couple of things:
1. It copies over a `CMakeLists.txt` file if not already present
2. Ask if you want to copy over `pylbo_wrapper.py`, you need this if you want to plot your results immediately after running.
3. Ask if you want to copy over a parfile if none was found.

For this first run, select `y` (yes) when prompted, twice.

Next call the build script:
```bash
buildlegolas.sh
```
If everything went well you should have a few files in your current working directory, including a `CMakeLists.txt` file, a `build` folder, an `output` folder and the `legolas` executable.

<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>
It is NOT recommended to create custom build or run directories _inside_ the main legolas folder.
Keeping the repository clean ensures that you don't run into merge conflicts when updating the code.
Run the setup scripts in a separate directory and let them take care of linking.
{: .notice--warning}

## Creating the configuration file
Configuring Legolas is done through use of a "parfile", where various options are passed in the form of Fortran namelists. For an overview
of different possibilities you can take a look [here](../../general/parameter_file).
Open the file with the `.par` extension (or create a new one if it was not copied over when setting up), and change its contents to following:
```fortran
&gridlist
  gridpoints = 50
/

&equilibriumlist
  equilibrium_type = "rayleigh_taylor"
/

&savelist
  write_eigenfunctions = .true.
  show_results = .true.
/
```

## Running the code
Finally call `legolas` and supply the parfile you just set up as argument:
```bash
./legolas -i my_setup.par
```
This will run a setup which defines Rayleigh-Taylor instabilities in a Cartesian geometry,
the spectrum itself should correspond to case **a** in Section 13.2, p. 487 of [Magnetohydrodynamics of Laboratory and Astrophysical Plasmas](http://doi.org/10.1017/9781316403679).

When the run is completed (this should only take a few seconds) the code will immediately fire up the post-processing framework since we set `show_results = .true.` in the parfile.
A few interactive figures will pop up, including the spectrum. Click on one of the eigenvalues in the figure and press `Enter`, this will draw the corresponding eigenfunctions.
You can cycle through multiple eigenfunctions using the arrow keys, right-clicking a point unselects it and pressing the `d` key clears all selected points.

The legend shows the various continuum regions, which can also be interactively toggled by clicking on their legend entries.
More information on the interface can be found [here](../../pylbo/using_pylbo/#interactive-continua--eigenfunctions).

This covers the basics of running Legolas on a single equilibrium. You can take a look at the various [pre-implemented equilibria](../../general/equilibria) and
modify the `equilibriumlist` or `paramlist` accordingly to play around with different options.
