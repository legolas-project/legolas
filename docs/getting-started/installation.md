---
title: Installation
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: true
toc_label: "Installation Guide"
toc_icon: "cogs"
last_modified_at: 2023-07-14
---

Due to the heavy use of object-oriented features (Fortran 2003) and submodules (Fortran 2008), Legolas
requires relatively recent Fortran compilers. This page gives a detailed overview of required
and optional dependencies to successfully build and run both Legolas and the post-processing framework
[Pylbo](../../pylbo/about_pylbo/).

# Dependencies
## Compilation
- gfortran v9.x+
- CMake v3.12+
- make

{% capture note %}
<i class="fa fa-exclamation-triangle" aria-hidden="true"></i>
**Note:** starting from Legolas 2.0 support for gfortran versions < 9.x is officially dropped.
While the code may still comple and run with older versions, it is _highly recommended_ to use updated compilers.
{% endcapture %}
<div class="notice--warning">
  {{ note | markdownify }}
</div>

## Post-processing
The post-processing framework Pylbo has a few standard dependencies, all of which will be automatically
installed if you choose the Pylbo package install (see [below](/getting-started/installation/#installing-as-a-package)).
- Python v3.7+
- [Numpy](https://numpy.org), for obvious reasons.
- [Matplotlib](https://matplotlib.org) v3.4+, for plotting.
- [f90nml](https://f90nml.readthedocs.io/en/latest/), to handle reading and writing of Fortran namelists.
- [tqdm](https://tqdm.github.io), used for progress bars.
- [psutil](https://psutil.readthedocs.io/en/latest/), for management of multiprocessing resources during parallel runs.
- [packaging](https://packaging.pypa.io/en/stable/), for handling datfile version control.
- [sympy](https://www.sympy.org/en/index.html), to offer a convenient way of user module configuration.
- [scipy](https://scipy.org), for interacting with Fortran code.

You can alternatively install these manually using
```bash
pip install numpy matplotlib f90nml tqdm psutil packaging sympy scipy
```
or
```bash
conda install numpy matplotlib f90nml tqdm psutil packaging sympy scipy
```
depending on your preference. On Linux it's usually best to use the versions available in your package repository.

**Note:** Python is only needed for Pylbo, not for Legolas itself. You can still run Legolas
if the Python requirements are not satisfied, however you will not be able to immediately
see the results after the run finishes (so set `show_results=.false.` in the parfile).
{: .notice--info}

## BLAS and LAPACK
The [BLAS](http://www.netlib.org/blas/) and [LAPACK](http://www.netlib.org/lapack/)
linear algebra packages are **required** dependencies, and you will
not be able to compile without them. We recommend version 3.5 or higher.
CMake is configured in such a way that both libraries should be found and linked automatically if
they are installed. An easy (but not the only) way to install these packages is as follows
- **Linux** (Debian/Ubuntu)
  ```bash
  sudo apt-get install libblas-dev
  sudo apt-get install liblapack-dev
  ```
- **macOS**, using [HomeBrew](https://brew.sh).
  ```bash
  brew install openblas
  brew install lapack
  ```
   Note that macOS ships with a default BLAS/LAPACK installation as part of the
  [vecLib](https://developer.apple.com/documentation/accelerate/veclib) framework, so a custom installation is optional.
  {: .notice--info}

If you did a manual compilation of the BLAS and LAPACK libraries (if you don't have sudo rights, for example),
CMake may not find the libraries by default. In that case it will throw a warning, and you may have to set the
`$BLAS_LIBRARIES` and `$LAPACK_LIBRARIES` variables which link to the compiled libraries.


## ARPACK
The [ARPACK](https://en.wikipedia.org/wiki/ARPACK) library is an **optional** dependency, so
Legolas will compile and run just fine if you don't have this installed (related modules are
conditionally compiled). Also here CMake will try to automatically find and link the libraries if installed.

We recommend using the actively maintained [arpack-ng](https://github.com/opencollab/arpack-ng). Unfortunately
most operating systems do not have this version readily available, so you will have to install it manually using
the following commands.
```bash
git clone https://github.com/opencollab/arpack-ng.git
cd arpack-ng
mkdir build
mkdir installed
cd build
cmake -DEXAMPLES=ON -DMPI=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=../installed ..
make
make install
```
Afterwards you export the `$ARPACK_ROOT` environment variable, pointing to your `arpack-ng` folder.


# Legolas
Legolas can be obtained by cloning the online repository:
```bash
git clone https://github.com/n-claes/legolas.git
```
which will put it in the local repository `legolas`.
Next you set the environment variable `$LEGOLASDIR` which points to this directory and add the `setup_tools` subdirectory to your PATH.
For example, if you cloned the `legolas` repository to `/users/codes/legolas`, you can edit your `.bashrc` (or `.zshrc` on macOS) as follows
```bash
export LEGOLASDIR="/users/codes/legolas"
PATH="$PATH:$LEGOLASDIR/setup_tools"
```
The last line allows for easy access to the `buildlegolas.sh` and `setuplegolas.py` scripts in the `setup_tools`
folder, such that they can be called from any directory.

## Compilation
To compile Legolas you first navigate to the `legolas` directory:
```bash
cd $LEGOLASDIR
```
Next you have the option of compiling the code manually, or to use the shell script we provided which
automatically takes care of creating build folders and calling the various make commands.
- For an automagic build
  ```bash
  buildlegolas.sh
  ```
- For a manual build
  ```bash
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make
  ```

## Doing a clean build
To do a fresh compilation you can call the build script with an additional argument
```bash
buildlegolas.sh clean
```
This will remove
1. the legolas executable in the current working directory (if found)
2. the `build` folder in the current working directory
3. the `build` folder in the legolas directory

You can also remove these folders manually instead if you prefer.

# Pylbo
The Pylbo framework is automatically included in the `legolas/post_processing` folder when you
clone the repository.

If you configured `$LEGOLASDIR` correctly and have all of [Pylbo's dependencies](/getting-started/installation/#post-processing)
installed, the included `pylbo_wrapper.py` (which is automatically run by Legolas when `show_results=.true.`) will find Pylbo on
its own. If you want to write your own scripts using Pylbo, or do not want to install the dependencies manually, you can follow
the steps below.

## Installing as a package
This is the recommended installation option. To do so, navigate to the `post_processing` folder and do the following:
```bash
cd post_processing
python setup.py develop
```
This will automatically install the listed [dependencies](/getting-started/installation/#post-processing) if they are not already installed. Activate any conda/virtualenv environment
beforehand in order to install Pylbo there. The `develop` argument means that the package will be automatically updated whenever
you update the repository.

## Sourcing the folder
Another possibility is adding the folder to your PYTHONPATH. If the above option is not available -- i.e. if you can not manage
your python installation -- then this is a valid alternative. Note that in this case you will have to install all dependencies yourself.
- **macOS/Linux**
  Edit `.zshrc`/`.bashrc` or similar, we assume that you have already set the `$LEGOLASDIR` environment variable (see [here](/getting-started/installation/#legolas)).
  ```bash
  export PYTHONPATH=$LEGOLASDIR/post_processing:$PYTHONPATH
  ```
- **Windows**
  Go to `Start` > `Control panel` > `Edit the system environment variables` > tab `"advanced"` > `environment variables`.
  Create or edit the PYTHONPATH environment variable and add the _full path_ to the `post_processing` folder.
  Multiple paths are separated by a semicolon `";"`.

# Next steps
Now that you've successfully installed Legolas and Pylbo [you can move on to your first run](/getting-started/running/).
