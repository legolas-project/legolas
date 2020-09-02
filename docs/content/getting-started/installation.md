---
title: Installation guide
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: true
toc_label: "Installing"
toc_icon: "cogs"
last_modified_at: 2020-09-02
---

# Requirements
Due to the heavy use of object-oriented features (Fortran 2003) and submodules (Fortran 2008), Legolas
requires relatively recent Fortran compilers.
Below is a list of prerequisites needed to successfully build and run both Legolas and the post-processing
framework [Pylbo](../../general/data_analysis/). Please note that lower compiler versions have not been
tested, but _might_ work.

- Fortran compiler
    - gfortran v8.x+ (recommended and tested)
    - Intel compilers v18.0+ (not tested)
- CMake v3.12+
- Python v3.6+
- make
- BLAS and LAPACK libraries v3.5+

## Linear algebra tools BLAS and LAPACK
CMake is configured in such a way that the BLAS and LAPACK libraries should be found and linked automatically.
In general, it is sufficient if you have them installed through
```bash
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
```
if you're on Linux, or through
```bash
brew install openblas
brew install lapack
```
using [HomeBrew](https://brew.sh) on macOS. Note that macOS ships with a default BLAS/LAPACK installation
as part of the [vecLib](https://developer.apple.com/documentation/accelerate/veclib) framework.
If you did a manual compilation of the BLAS and LAPACK libraries (if you don't have sudo rights, for example),
CMake may not find the libraries by default. In that case it will throw a warning, and you may have to set the 
`$BLAS_LIBRARIES` and `$LAPACK_LIBRARIES` variables which link to the compiled libraries. 

# Pre-build
Below we explain the stuff you _should_ do before attempting to compile Legolas.
## Obtaining legolas
Legolas can be obtained by cloning the online repository:
```bash
git clone https://github.com/n-claes/legolas.git
```
which will put it in the local repository `legolas`.
CMake has been configured for an out-of-source build, meaning that the repository stays clean.
This allows for easy updates through `git pull`.

## Environment variables
For an easy setup we recommend setting the environment variable `$LEGOLASDIR` and adding
the `setup_tools` folder to your PATH. This can be done by editing your `.bashrc` (or `.zshrc` on macOS)
as follows
```bash
export LEGOLASDIR='path_to_the_legolas_directory'
PATH="$PATH:$LEGOLASDIR/setup_tools"
```
The last line allows for easy access to the `buildlegolas.sh` and `setuplegolas.py` scripts in the `setup_tools`
folder, such that they can be called from any directory.

# Compiling the code
Compiling the code is quite straightforward, and for your convenience we have provided a simple shell script
to do everything at once. Compiling the code can either be done in the repository itself (which is not recommended),
or in a dedicated folder somewhere.

**Note:** whichever of the two options you choose, `Legolas` is _always_ compiled in a directory called `build`
_inside_ the main repository, which is ignored by git. Because we make heavy use of submodules (which prevent
compilation cascades when changes are made), you don't have to recompile the code every time you set up a new problem.
Only the modified user-defined submodule is recompiled, and the "new" executable is placed in the same directory. 
{: .notice--info}

## 1. In-repository build
The first option is building inside the repository. To do so, navigate to the legolas source directory and do
```bash 
sh setup_tools/buildlegolas.sh
```
This will create a directory `build` inside the repository (which is ignored by git), and places the `legolas`
executable in the topmost directory. Option two is doing it manually if you don't trust the automatic script:
```bash
mkdir build
cd build
cmake ..
make
```
Note that an in-repository build means that you'll have to edit source files, which could give rise
to merge conflicts when you're updating the code (and we all hate those!) so we don't recommend doing it like this.

## 2. Out-of-repository build (recommended)
This is the recommended way to build Legolas. This requires you to have set the environment variable and PATH
modification described above. To setup a folder for a Legolas build, call `setuplegolas.py` from the command line and
follow the instructions, which look like this
```
Starting Legolas setup process in directory 'your_directory'. Is this ok? [y/n] y
>> Copying CMakeLists.txt to present directory.
>> You'll need the Python wrapper to plot results after running. Copy over? [y/n] n
>> No parfile found. Copy default template? [y/n] y
>> Copying legolas_config.par to present directory.
>> No user-defined submodule found, copy default template? [y/n] y
>> Copying smod_user_defined.f08 to present directory.

>> Setup complete. 
You can start the build process by calling 'buildlegolas.sh' directly or do a manual CMake install.
```
So first `CMakeLists.txt` is copied over, which is needed for the build, followed by the Pylbo wrapper script if you
need it. If no parfile is found a default template is copied over, the same holds for the user-defined equilibrium.
If the latter is found in the repository, the setup script edits the `CMakeLists.txt` file to include it in the build.
Next you simply build as described before, either through `buildlegolas.sh` which does everything automatically, or
manually through
```bash
mkdir build
cd build
cmake ..
make
```
In both cases the executable is placed in the same directory as the parfile and user submodule.
Note that the automatic buildscript removes the build directory afterwards, leaving only the executable.
If you don't want this, build manually.

# Equivalent of "make clean"
Since we use CMake for compilation there is no `make clean` equivalent as there is with GNU Make.
Instead, you can simply remove the `build` directory inside the repository, which contains the compiled object files.
You can do this automatically from any directory by supplying an additional argument to `buildlegolas.sh`.
Say you just compiled in a local directory and you want to do a fresh compilation, simply call
```bash
buildlegolas.sh clean
```
This will remove the `build` folder in the main legolas directory and the local executable (if one is found).
This requires you to have set the `$LEGOLASDIR` environment variable, if you did not do this you have to remove
the `build` folder manually.