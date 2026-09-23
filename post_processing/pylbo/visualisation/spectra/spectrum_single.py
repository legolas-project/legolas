import matplotlib.colors as mpl_colors
import numpy as np
from pylbo.utilities.toolbox import add_pickradius_to_item
from pylbo.visualisation.continua import ContinuaHandler
from pylbo.visualisation.eigenfunctions.eigfunc_handler import EigenfunctionHandler
from pylbo.visualisation.spectra.spectrum_figure import SpectrumFigure
from pylbo.utilities.logger import pylboLogger


class SingleSpectrumPlot(SpectrumFigure):
    """
    Creates a plot of a single spectrum based on a given dataset.

    Parameters
    ----------
    dataset : ~pylbo.data_containers.LegolasDataSet
        The dataset used to create the spectrum.
    figsize : tuple
        Figure size used when creating a window, analogous to matplotlib.
    custom_figure : tuple
        The custom figure to use in the form (fig, axes).
    use_residuals : bool
        If `True`, colors the spectrum points based on the residuals.

    Attributes
    ----------
    dataset : ~pylbo.data_containers.LegolasDataSet
        The dataset passed as parameter
    w_real : numpy.ndarray(dtype=float, ndim=1)
        Real part of the eigenvalues as a numpy array.
    w_imag : numpy.ndarray(dtype=float, ndim=1)
        Imaginary part of the eigenvalues as a numpy array.
    marker : ~matplotlib.markers
        The marker used to draw the points.
    markersize : int, float
        Size of the marker.
    alpha : int, float
        Alpha value of the points.
    """

    def __init__(
        self, dataset, figsize, custom_figure, use_residuals, title=None, **kwargs
    ):
        super().__init__(
            custom_figure=custom_figure, figlabel="single-spectrum", figsize=figsize
        )
        self.dataset = dataset
        self.custom_title = title
        super()._set_plot_properties(kwargs)

        self._use_residuals = use_residuals
        (self._nonzero_w_idxs,) = np.where(abs(dataset.eigenvalues) > 1e-12)

    def add_spectrum(self):
        """Adds the spectrum to the plot, makes the points pickable."""
        spectrum_points = self.ax.scatter(
            self.dataset.eigenvalues[self._nonzero_w_idxs].real * self.x_scaling,
            self.dataset.eigenvalues[self._nonzero_w_idxs].imag * self.y_scaling,
            marker=self.marker,
            c=self._get_colors(),
            s=10 * self.markersize,
            alpha=self.alpha,
            linestyle="None",
            norm=mpl_colors.LogNorm() if self._use_residuals else None,
            cmap=self.plot_props.pop("cmap", "jet") if self._use_residuals else None,
            **self.plot_props,
        )
        # set dataset associated with this line of points
        setattr(spectrum_points, "dataset", self.dataset)
        add_pickradius_to_item(item=spectrum_points, pickradius=10)
        if self._use_residuals:
            self.cbar = self.fig.colorbar(spectrum_points, ax=self.ax, label="Residual")
        self.ax.axhline(y=0, linestyle="dotted", color="grey", alpha=0.3)
        self.ax.axvline(x=0, linestyle="dotted", color="grey", alpha=0.3)
        self.ax.set_xlabel(r"Re($\omega$)")
        self.ax.set_ylabel(r"Im($\omega$)")
        self.ax.set_title(self.custom_title or self.dataset.eq_type)

    def add_continua(self, interactive=True):
        """
        Adds the continua to the spectrum.

        Parameters
        ----------
        interactive : bool
            If `True`, makes the legend pickable.

        Returns
        -------
        c_handler : ~pylbo.continua.ContinuaHandler
            The legendhandler used to plot the continua.
        """
        if not self.has_valid_continua(self.dataset):
            return
        if self.has_zero_continua(self.dataset):
            pylboLogger.warning("Continua not plotted: all are zero.")
            return
        if self._c_handler is None:
            self._c_handler = ContinuaHandler(interactive=interactive)

        for key, color in zip(
            self._c_handler.continua_names, self._c_handler.continua_colors
        ):
            continuum = self.dataset.continua[key]
            if np.allclose(continuum, 0, atol=1e-12) and key in ["doppler", "thermal"]:
                continue
            # removes duplicates
            continuum = np.array(list(set(continuum)), dtype=complex)
            item = self.ax.scatter(
                continuum.real * self.x_scaling,
                continuum.imag * self.y_scaling,
                marker=self._c_handler.marker,
                linewidth=self._c_handler.markersize,
                c=color,
                alpha=self._c_handler.alpha_point,
                label=key,
            )
            self._c_handler.add(item)
        self._c_handler.legend = self.ax.legend(**self._c_handler.legend_properties)
        if interactive:
            super().make_legend_interactive(self._c_handler)

    def add_eigenfunctions(self):
        """Adds the eigenfunctions to the plot, sets the eigenfunction handler."""
        if self._ef_ax is None:
            self._ef_ax = super().add_subplot_axes(self.ax, loc="right")
        if self._ef_handler is None:
            self._ef_handler = EigenfunctionHandler(self.dataset, self._ef_ax, self.ax)
        super().add_eigenfunction_interface(efhandler=self._ef_handler)

    def draw_resonances(self):
        """
        In case the (derived) eigenfunctions are added to the plot, the locations
        of resonance with the continua will also be drawn.
        Does nothing if the (derived) eigenfunctions are not shown.
        """
        if self._ef_handler is not None:
            self._ef_handler._draw_resonances = True
            self._ef_handler.update_plot()

    def _get_colors(self) -> np.ndarray:
        """Returns the colors for the spectrum points."""
        if self._use_residuals:
            return self.dataset.get_residuals()[self._nonzero_w_idxs]
        return self.color
