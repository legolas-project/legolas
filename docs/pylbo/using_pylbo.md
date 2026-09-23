---
title: Using Pylbo
layout: single
classes: wide
sidebar:
  nav: "leftcontents"
toc: true
toc_icon: "chevron-circle-down"
last_modified_at: 2026-01-28
---

Using Pylbo is quite straightforward, for a detailed guide on the API we refer to the
[Pylbo documentation](../../sphinx/autoapi/pylbo/index.html).
This page will provide a basic guide on how to use the package. In what follows we assume that Pylbo has been installed
(see [installing Pylbo](../../getting-started/installation#pylbo)) and has been imported.

## Loading Legolas datfiles
You can either load a single datfile or multiple datfiles at the same time. Both loaders accept either strings
or PathLike objects for the datfile paths.

### Loading a single file
```python
ds = pylbo.load("path_to_datfile")
```
The variable `ds` will be a
[LegolasDataSet](../../sphinx/autoapi/pylbo/data_containers/index.html#pylbo.data_containers.LegolasDataSet)
instance, which has a lot of convenient methods and attributes that you can use during analysis.

### Loading a series of files
```python
series = pylbo.load_series(["path1", "path2", "path3"])
```
The variable `series` will be a
[LegolasDataSeries](../../sphinx/autoapi/pylbo/data_containers/index.html#pylbo.data_containers.LegolasDataSeries)
instance. This is an iterable object, meaning you can do something like this:
````python
# iterate over the various datasets
for ds in series:
    ....
# get a specific dataset or slice multiple datasets
ds = series[3]
series_slice = series[3:9]
````

## Analysing a single file
In what follows we assume that your single datfile has been loaded in `ds` as explained above.

### Plotting the spectrum
Pylbo has a built-in method to visualise the spectrum of a dataset:
```python
p = pylbo.plot_spectrum(ds)
p.show()
```
This will create a new figure on which the spectrum is plotted. You can specify a custom figure size through the extra argument
`figsize=(12, 8)`, just as in matplotlib. You can also specify your own figure (so no new window will be created):
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
p = pylbo.plot_spectrum(ds, custom_figure=(fig, ax))
p.show()
```
Changing the markersize, color etc. can be done by supplying the usual matplotlib kwargs to `plot_spectrum`. You can fully customise
the figure by accessing the figure and axes objects through `p.fig` and `p.ax`. The detailed API can be found
[here](../../sphinx/autoapi/pylbo/index.html#pylbo.plot_spectrum).

#### Interactive continua & eigenfunctions
Pylbo can interactively plot the various continua regions and eigenfunctions. The example below first plots the spectrum, and
then attaches the continua and eigenfunctions to the plot:
```python
p = pylbo.plot_spectrum(ds)
p.add_continua()
p.add_eigenfunctions()
```
This is interactive by default, to toggle the continua on or off you can click on their respective legend items. You can supply
`p.add_continua(interactive=False)` to disable interactivity.

When you attach the eigenfunctions through `p.add_eigenfunctions()` Pylbo will modify the geometry of the currently supplied
axis and split it in two. The spectrum will be drawn on the left, the eigenfunctions on the right; simply click on a spectrum point (or multiple)
for which you want to see eigenfunctions. Selected points will be annotated on the plot, right-clicking will deselect them.
Use the arrow keys to cycle through the various eigenfunctions.

Note that every point you select will have a different color, the colors between the eigenfunctions and the selected spectrum points are consistent.
The legend on the eigenfunction panel will contain the index of the selected point in the `ds.eigenvalues` array, with the value printed as well.
Below is a detailed overview of the various commands:

| Key / mouse    | Description     |
| :---: | :----          |
| _left click_ | Select a spectrum point. You can select multiple points at once. |
| _right click_ | Remove point from selection. Picker-based, so you may have to zoom in a bit if two selected points are close together.|
| _enter_ | Plots the eigenfunctions of the currently selected points. |
| _i_ | Swaps between the real and imaginary parts of the eigenfunctions. |
| _up arrow_ | Cycles to the next variable in the list. |
| _down arrow_ | Cycles to the previous variable in the list. |
| _d_ | Clears current selection. |
| _t_ | Retransforms the eigenfunctions using the scale factor, hence has no effect if `geometry = "Cartesian"`. You can use this to cycle between the $rv_r$ and $v_r$ eigenfunctions in cylindrical geometry, for example. |
| _w_ | Prints out the currently selected eigenvalues to the console. This may come in handy if you want to copy the exact values somewhere, to extract eigenfunctions for example. |
| _n_ | Counts and prints out the number of zeroes in the selected eigenfunctions, along with the corresponding eigenvalues. |
| _j_ | Saves the indices of all selected eigenvalues as an array in a `.npy` file, which can be loaded using the `numpy.load` function. |
| _e_ | Toggles a visualisation of the subset of eigenvalues that have their eigenfunctions saved, has no effect if `write_eigenfunction_subset` was set to `.false.`. |

#### Retrieving eigenvalues & eigenfunctions
You can manually retrieve the eigenfunctions from the dataset as well. You can either supply the eigenvalue indices (based on the interactive legend on the eigenfunction panel),
or eigenvalue "guesses". In case of the latter Pylbo will select the eigenvalue closest near your guess, and return its eigenfunctions.
```python
# based on indices
eigenfuncs = ds.get_eigenfunctions(ev_idxs=[20, 123, 451, 613])
# based on guesses
eigenfuncs = ds.get_eigenfunctions(ev_guesses=[3.0, 4.5 + 1j, 5 - 3j])
```
Now, `eigenfuncs` will be a Numpy-array of size 3 (since 3 eigenvalue guesses were provided), and every index corresponds
to the index of your guess, i.e. `eigenfuncs[1]` corresponds to the eigenfunctions of `4.5 + i`, and so on.
Every element of the `eigenfuncs` array is a dictionary, containing all eigenfunctions as well as the eigenvalues.
To retrieve what you need, simply do
```python
# for the first eigenvalue:
rho_ef1 = eigenfuncs[0].get("rho")
# for the second eigenvalue:
rho_ef2 = eigenfuncs[1].get("rho")
# to see which keys are in there:
print(eigenfuncs[0].keys())
>> "rho", "v1", "v2", "v3", "T", "a1", "a2", "a3", "eigenvalue"
```
The names of the keys are self-explanatory. If the derived eigenfunctions were saved to the datfile these will automatically
be contained within the corresponding dictionaries with appropriate keys.

You can also retrieve eigenvalues near guesses, this will return both the indices and corresponding eigenvalues:
```python
idxs, eigenvals = ds.get_nearest_eigenvalues(ev_guesses=[3.0, 4.5 + 1j, 5 - 3j])
```
More information on these methods can be found
[here](../../sphinx/autoapi/pylbo/data_containers/index.html#pylbo.data_containers.LegolasDataSet.get_eigenfunctions).

### Plotting the equilibrium profiles
Pylbo allows for a visual inspection of the equilibrium profiles as well:
```python
p = pylbo.plot_equilibrium(ds)
p.show()
```
See the API [here](../../sphinx/autoapi/pylbo/index.html#pylbo.plot_equilibrium) for more information.
The profiles will be drawn interactively, similar to the continua. Pass the additional kwarg `interactive=False`
to disable this. All equilibrium profiles can be accessed directly through the `ds.equilibria` attribute.

### Plotting the continuum profiles
Analogous to the equilibrium profiles, the continuum profiles can be drawn as well:
```python
p = pylbo.plot_continua(ds)
p.show()
```
See the API [here](../../sphinx/autoapi/pylbo/index.html#pylbo.plot_continua) for
more information. The continua can be accessed directly through the `ds.continua` attribute.

### Plotting the matrices
Pylbo can also plot the matrices as calculated by Legolas, note that this requires the matrices to be saved
to the datfile.
```python
p = pylbo.plot_matrices(ds)
p.show()
```
See the API [here](../../sphinx/autoapi/pylbo/index.html#pylbo.plot_matrices) for more information.
Note that this method is only for inspection purposes on low resolution datasets (10, maybe 20 gridpoints).
This method should not be used for "regular" datasets, since thousands of points will be drawn in that case.

### Plotting the equilibrium balance
In case you are setting up an equilibrium by yourself and Legolas is complaining that the equilibrium
balance equations are not satisfied, you can do a quick inspection of these equations like so:
```python
p = pylbo.plot_equilibrium_balance(ds)
p.show()
```
See the API [here](../../sphinx/autoapi/pylbo/index.html#pylbo.plot_equilibrium_balance) for more information.
The resulting curves should be as close to zero as possible, though keep in mind that results may be _numerically_ zero (e.g. $\sim10^{-15}$ or similar.)

## Analysing multiple files
In what follows we assume that all datfiles have been loaded in `series` as explained above.

### Plotting a multispectrum
Plotting the spectrum of multiple datfiles is done in a similar way as for a single datfile.
The main difference here is that we have to provide an additional variable, `xdata`. This should be
of the same length as the number of datasets in the series, since every point of `xdata` will have a dataset
associated with it.

Say you have loaded 10 datasets and you specify `xdata = "k3"`. This means that you will have 10 columns of datapoints,
plotting the real or imaginary part of the eigenvalues versus the `k3` value for each dataset. `xdata` can be anything, as long
as the array sizes are consistent.

In the example below we plot the real part of the eigenvalues versus the wavenumber squared for each dataset, and divide the eigenvalues
with the maximum Alfvén speed in each setup:
```python
xdata = series.get_k0_squared()
p = pylbo.plot_spectrum_multi(series, xdata=xdata)
p.set_y_scaling(1 / series.get_alfven_speed(which_values="maximum"))
p.show()
```
See the API [here](../../sphinx/autoapi/pylbo/index.html#pylbo.plot_spectrum_multi) for more information.
You can plot the imaginary part of the eigenvalues instead by supplying `use_real_parts=False`, or plot the squared eigenvalues through `use_squared_omega=True`.

#### Interactive continua & eigenfunctions
Multispectra can have associated continua and eigenfunctions as well, and these will be automatically scaled to whatever scaling is supplied
to the eigenvalues. Adding these is done in exactly the same way as for the single datfile case:
```python
xdata = series.get_k0_squared()
p = pylbo.plot_spectrum_multi(series, xdata=xdata)
p.add_continua()
p.add_eigenfunctions()
p.show()
```
Interactive controls are the same as before, except for one addition. When multiple datfiles are loaded it is not necessarily the case that all of them have
eigenfunctions as well, making it difficult to know which spectrum points can be selected and which ones cannot. This can be made clear by pressing the M key, which will
reduce the opacity for points without eigenfunctions. This is a toggle, so pressing M again turns this feature off. Note that this is only relevant for multispectra,
and that this will have no effect if all datasets have eigenfunctions present.

| Key / mouse    | Description     |
| :---: | :----          |
| _m_ | Toggles the opacity for spectrum points that have no eigenfunctions, making them less visible.|

### Plotting a merged spectrum
In some cases it may be useful to merge all datasets of a series into a single 2D figure, for examply when probing the spectrum using multiple $\sigma$-values
with the shift-invert solver. Pylbo supplies a method to do this, and even draw eigenfunctions at the same time:
```python
p = pylbo.plot_merged_spectrum(series)
p.add_eigenfunctions()
p.show()
```
See the API [here](../../sphinx/autoapi/pylbo/index.html#pylbo.plot_merged_spectrum) for more information.
This will create a scatterplot where the points of every dataset in the series will be assigned a different color, and a legend will be created saying
which color corresponds to which dataset. The legend is interactive by default, meaning that similar as to the interactive continua regions you can click on the
legend and show/hide the corresponding dataset. To disable this, supply `interactive=False` as an additional keyword argument.

If a great many datasets are loaded it may be useful to disable the legend alltogether. This can be done by giving the kwarg `legend=False` to `plot_merged_spectrum`.
Furthermore, if you simply want to look at the spectrum in a single color, supply the argument `color="blue"` (or your favourite color),
which will also disable the legend and plot all points in that color.

The eigenfunctions are interactive in exactly the same way as for the other types of plots, but note that it is possible that points from different datasets may overlap,
for example when the same eigenvalue is picked up by multiple datasets. Selecting that point will therefore plot multiple eigenfunctions
(equal to the amount of overlapping points), however, you can circumvent this by hiding the points of the datasets you're not interesting in by clicking on their legend entries.
Similar as to the multispectrum case you can highlight datasets that have eigenfunctions present by pressing the M key.

Drawing continua is not supported for this type of figure.

### Comparing two spectra
Sometimes it may be convenient to plot two similar spectra next to each other and do a direct visual comparison of both.
This can be done as follows:
```python
p = pylbo.plot_spectrum_comparison(ds1, ds2)
p.show()
```
See the API [here](../../sphinx/autoapi/pylbo/index.html#pylbo.plot_spectrum_comparison) for more information.
This will create a 2x1 figure with the spectrum of `ds1` on the left and the one from `ds2` on the right. Note that you can add
continua and eigenfunctions to this as well through `p.add_continua()` (with an optional `interactive=False` flag) and
`p.add_eigenfunctions()`. In the case of eigenfunctions, the figure will be transformed to a 2x2 plot, where the panel below each
spectrum corresponds to the eigenfunctions.

The additional keyword argument `lock_zoom=True` can be supplied to `plot_spectrum_comparison()`, which will lock the zoom on both
spectra (and is off by default). Turning this on means that if you zoom in on one of the two spectra,
the zoom of the other plot is automatically adjusted to match.
