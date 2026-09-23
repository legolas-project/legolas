from pylbo.data_containers import ensure_dataseries, ensure_dataset
from pylbo.utilities.logger import pylboLogger
from pylbo.visualisation.matrices import MatrixFigure
from pylbo.visualisation.profiles import (
    ContinuumProfile,
    EquilibriumBalance,
    EquilibriumProfile,
)
from pylbo.visualisation.spectra.spectrum_comparison import SpectrumComparisonPlot
from pylbo.visualisation.spectra.spectrum_merged import MergedSpectrumPlot
from pylbo.visualisation.spectra.spectrum_multi import MultiSpectrumPlot
from pylbo.visualisation.spectra.spectrum_single import SingleSpectrumPlot

forbidden_args = ["linestyle", "linewidth", "lw"]


def plot_spectrum(
    data, figsize=None, custom_figure=None, use_residuals=False, title=None, **kwargs
):
    """
    Plots the spectrum of a single dataset.

    Parameters
    ----------
    data : ~pylbo.data_containers.LegolasDataSet
        The dataset that should be used.
    figsize : tuple
        Optional figure size like the usual matplotlib (x, x) size.
    custom_figure : tuple
        Optional, in the form (fig, ax). If supplied no new figure will be created
        but this one will be used instead. `fig` refers to the matplotlib figure and
        `ax` to a (single) axes instance, meaning that you can pass a subplot as well.
    use_residuals : bool
        If `True`, colors the spectrum based on the residual in the datfile.

    Returns
    -------
    p : ~pylbo.visualisation.spectra.SingleSpectrumPlot
        The spectrum instance which can be used further to add continua,
        eigenfunctions, etc.
    """
    ensure_dataset(data)
    for arg in forbidden_args:
        if kwargs.pop(arg, None) is not None:
            pylboLogger.warning(f"plot_spectrum does not accept the '{arg}' argument.")
    p = SingleSpectrumPlot(data, figsize, custom_figure, use_residuals, title, **kwargs)
    return p


def plot_spectrum_multi(
    data,
    xdata,
    use_squared_omega=False,
    use_real_parts=True,
    figsize=None,
    custom_figure=None,
    **kwargs,
):
    """
    Plots the spectrum of a dataset series.

    Parameters
    ----------
    data : ~pylbo.data_containers.LegolasDataSeries
        The dataseries that should be used.
    xdata : str, list, numpy.ndarray
        Data to use for the horizontal axis. This can either be a key from the
        parameters dictionary, or a list/numpy array containing actual data.
    use_squared_omega : bool
        If `True`, this will square the eigenvalues when they are plotted on the
        vertical axis. If `False` (default), either the real or imaginary part of the
        eigenvalues will be plotted depending on the value of `use_real_parts`.
    figsize : tuple
        Optional figure size like the usual matplotlib (x, x) size.
    custom_figure : tuple
        Optional, in the form (fig, ax). If supplied no new figure will be created
        but this one will be used instead. `fig` refers to the matplotlib figure and
        `ax` to a (single) axes instance, meaning that you can pass a subplot as well.

    Returns
    -------
    p : ~pylbo.visualisation.spectra.MultiSpectrumPlot
        The spectrum instance which can be used further to add continua,
        eigenfunctions, etc.
    """
    ensure_dataseries(data)
    for arg in forbidden_args:
        if kwargs.pop(arg, None) is not None:
            pylboLogger.warning(
                f"plot_spectrum_multi does not accept the '{arg}' argument."
            )
    p = MultiSpectrumPlot(
        data,
        xdata,
        use_squared_omega,
        use_real_parts,
        figsize,
        custom_figure,
        **kwargs,
    )
    return p


def plot_merged_spectrum(
    data, figsize=None, custom_figure=None, interactive=True, legend=True, **kwargs
):
    """
    Creates a merged spectrum from various datasets, useful in plotting multiple
    results from the shift-invert solver, for example. This draws every dataset
    in a different color by default, and creates a corresponding legend. Supply the
    optional argument `color` to draw all points in the same color.

    Parameters
    ----------
    data : ~pylbo.data_containers.LegolasDataSeries
    figsize : tuple
        Optional figure size like the usual matplotlib (x, x) size.
    custom_figure : tuple
        Optional, in the form (fig, ax). If supplied no new figure will be created
        but this one will be used instead. `fig` refers to the matplotlib figure and
        `ax` to a (single) axes instance, meaning that you can pass a subplot as well.
    interactive : bool
        If `True` (default), enables an interactive legend.
    legend : bool
        If `True` (default), draws a legend.

    Returns
    -------
    p : ~pylbo.visualisation.spectra.MultiSpectrumPlot
        The spectrumfigure instance, containing the plot.
    """
    ensure_dataseries(data)
    p = MergedSpectrumPlot(data, figsize, custom_figure, interactive, legend, **kwargs)
    return p


def plot_spectrum_comparison(
    ds1,
    ds2,
    figsize=None,
    custom_figure=None,
    lock_zoom=False,
    use_residuals=False,
    **kwargs,
):
    """
    Compares two spectra.

    Parameters
    ----------
    ds1 : ~pylbo.data_containers.LegolasDataSet
        The first dataset, this one is put on the left panel.
    ds2 : ~pylbo.data_containers.LegolasDataSet
        The second dataset, this one is put on the right panel.
    figsize : tuple
        Optional figure size like the usual matplotlib (x, x) size.
    custom_figure : tuple
        The custom figure to use in the form (fig, axes).
    lock_zoom : bool
        If `True` (`False` by default), locks the zoom of both axis. When locked,
        zoomin in on one of the axis automatically scales the zoom on the other one
        as well.
    use_residuals : bool
        If `True`, colors the spectrum based on the residual in the datfile.

    Returns
    -------
    p : ~pylbo.visualisation.spectra.SpectrumComparisonPlot
        The figure instance containing the compared spectrum plot.

    """
    ensure_dataset(ds1)
    ensure_dataset(ds2)
    p = SpectrumComparisonPlot(
        ds1, ds2, figsize, custom_figure, lock_zoom, use_residuals, **kwargs
    )
    return p


def plot_equilibrium(data, figsize=None, interactive=True, **kwargs):
    """
    Plots the equilibrium profiles.

    Parameters
    ----------
    data : ~pylbo.data_containers.LegolasDataSet
        The dataset or series that should be used.
    figsize : tuple
        Optional figure size like the usual matplotlib (x, x) size.
    interactive : bool
        If `True` (default), enables an interactive legend.

    Returns
    -------
    p : ~pylbo.visualisation.profiles.EquilibriumProfile
        The profile instance containing the equilibrium plots.
    """
    ensure_dataset(data)
    if not data.has_background:
        pylboLogger.warning(
            "The dataset does not contain a background, equilibrium profiles "
            "can not be plotted."
        )
        return
    p = EquilibriumProfile(data, figsize, interactive, **kwargs)
    return p


def plot_equilibrium_balance(data, figsize=None, interactive=True, **kwargs):
    """
    Creates a plot of the balance equations.
    These should be as close to zero as possible over the entire grid.

    Parameters
    ----------
    data : ~pylbo.data_containers.LegolasDataSet
        The dataset that should be used
    figsize : tuple
        Optional figure size like the usual matplotlib (x, x) size.
    interactive : bool
        If `True` (default), enables an interactive legend.

    Returns
    -------
    p : ~pylbo.visualisation.profiles.EquilibriumBalance
        The profile instance containing the equilibrium balance plots.
    """
    ensure_dataset(data)
    if not data.has_background:
        pylboLogger.warning(
            "The dataset does not contain a background, equilibrium balance "
            "can not be plotted."
        )
        return
    return EquilibriumBalance(data, figsize, interactive, **kwargs)


def plot_continua(data, figsize=None, interactive=True, **kwargs):
    """
    Plots the continua profiles.

    Parameters
    ----------
    data : ~pylbo.data_containers.LegolasDataSet
        The dataset or series that should be used.
    figsize : tuple
        Optional figure size like the usual matplotlib (x, x) size.
    interactive : bool
        If `True` (default), enables an interactive legend.

    Returns
    -------
    p : ~pylbo.visualisation.profiles.ContinuumProfile
        The profile instance containing the continua plots.
    """
    ensure_dataset(data)
    p = ContinuumProfile(data, figsize, interactive, **kwargs)
    return p


def plot_matrices(data, figsize=None, **kwargs):
    """
    Plots the continua profiles.

    Parameters
    ----------
    data : ~pylbo.data_containers.LegolasDataSet
        The dataset that should be used.
    figsize : tuple
        Optional figure size like the usual matplotlib (x, x) size.

    Returns
    -------
    p : ~pylbo.visualisation.matrices.MatrixFigure
        The instance containing the matrix plots.
    """
    ensure_dataset(data)
    p = MatrixFigure(data, figsize, **kwargs)
    return p
