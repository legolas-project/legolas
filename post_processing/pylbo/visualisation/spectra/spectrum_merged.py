from pylbo.deprecations import log_deprecation_warning
from pylbo.utilities.toolbox import add_pickradius_to_item
from pylbo.visualisation.eigenfunctions.eigfunc_handler import EigenfunctionHandler
from pylbo.visualisation.legend_handler import LegendHandler
from pylbo.visualisation.spectra.spectrum_figure import SpectrumFigure


def _check_deprecations(kwargs: dict) -> None:
    if "color_dict" in kwargs:
        log_deprecation_warning(
            "The kwarg 'color_dict' is deprecated. Use 'color' instead.",
            since_version="2.1",
        )
    if "color_parameter" in kwargs:
        log_deprecation_warning(
            "The kwarg 'color_parameter' is deprecated. "
            "Use 'color' instead and pass a list or array of colors.",
            since_version="2.1",
        )


class MergedSpectrumPlot(SpectrumFigure):
    """
    Merges the datasets from a given series into a single plot.

    Parameters
    ----------
    data : ~pylbo.data_containers.LegolasDataSeries
        The dataseries which will be merged.
    figsize : tuple
        Figure size used when creating a window, analogous to matplotlib.
    custom_figure : tuple
        The custom figure to use in the form (fig, axes).
    interactive : bool
        If `True` an interactive legend is enabled.
    legend : bool
        If `False` no legend will be drawn.

    Attributes
    ----------
    data : ~pylbo.data_containers.LegolasDataSeries
        The dataseries passed as parameter.
    leg_handle : ~pylbo.visualisation.legend_interface.LegendHandler
        The handler for the legend.
    """

    def __init__(self, data, figsize, custom_figure, interactive, legend, **kwargs):
        super().__init__(
            custom_figure=custom_figure, figlabel="merged_spectrum", figsize=figsize
        )
        self.data = data
        self.leg_handle = LegendHandler(interactive)

        _check_deprecations(kwargs)

        # don't override the color if it's already set
        kwargs.update({"color": kwargs.pop("color", None)})
        super()._set_plot_properties(kwargs)

        has_single_color = isinstance(self.color, str)
        if self.color is None or has_single_color:
            self.color = [self.color] * len(self.data)
        self._use_legend = legend and not has_single_color
        self._interactive = interactive

    def add_spectrum(self):
        """Adds the spectrum to the plot, makes the points pickable."""
        for ds, c in zip(self.data, self.color):
            spectrum_point = self.ax.scatter(
                ds.eigenvalues.real * self.x_scaling,
                ds.eigenvalues.imag * self.y_scaling,
                marker=self.marker,
                s=6 * self.markersize,
                c=c,
                alpha=self.alpha,
                label=ds.datfile.stem,
                **self.plot_props,
            )
            setattr(spectrum_point, "dataset", ds)
            add_pickradius_to_item(item=spectrum_point, pickradius=10)
            self.leg_handle.add(spectrum_point)
        self.ax.axhline(y=0, linestyle="dotted", color="grey", alpha=0.3)
        self.ax.axvline(x=0, linestyle="dotted", color="grey", alpha=0.3)
        self.ax.set_xlabel(r"Re($\omega$)")
        self.ax.set_ylabel(r"Im($\omega$)")

        if self._use_legend:
            self.leg_handle.legend = self.ax.legend(loc="best")
            self.leg_handle._make_visible_by_default = True
            if self._interactive:
                super().make_legend_interactive(self.leg_handle)
        self.fig.tight_layout()

    def add_eigenfunctions(self):
        """Adds the eigenfunctions to the current figure."""
        if self._ef_ax is None:
            self._ef_ax = super().add_subplot_axes(self.ax, loc="right")
        if self._ef_handler is None:
            self._ef_handler = EigenfunctionHandler(self.data, self._ef_ax, self.ax)
        super().add_eigenfunction_interface(efhandler=self._ef_handler)

    def add_continua(self, interactive=True):
        raise NotImplementedError("Continua are not supported for this type of figure.")
