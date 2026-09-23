from pylbo.visualisation.spectra.spectrum_figure import SpectrumFigure
from pylbo.visualisation.spectra.spectrum_single import SingleSpectrumPlot


class SpectrumComparisonPlot(SpectrumFigure):
    """
    Subclass to compare two spectra.

    Parameters
    ----------
    ds1 : ~pylbo.data_containers.LegolasDataSet
        First dataset, will be placed on the left side.
    ds2 : ~pylbo.data_containers.LegolasDataSet
        Second dataset for comparison, will be placed on the right side.
    figsize : tuple
        Figure size used when creating a window, analogous to matplotlib.
    custom_figure : tuple
        The custom figure to use in the form (fig, axes).
    lock_zoom : bool
        If `True`, locks the zoom for both spectrum plots.
    use_residuals : bool
        If `True`, colors the spectrum points based on the residuals.

    Attributes
    ----------
    ax2 : ~matplotlib.axes.Axes
        The (top)right axes object.
    panel1 : ~pylbo.visualisation.spectra.SingleSpectrumPlot
        The spectrum instance associated with the left side.
    panel2 : ~pylbo.visualisation.spectra.SingleSpectrumPlot
        The spectrum instance associated with the right side.
    """

    def __init__(
        self, ds1, ds2, figsize, custom_figure, lock_zoom, use_residuals, **kwargs
    ):
        super().__init__(
            custom_figure=custom_figure, figlabel="compare-spectra", figsize=figsize
        )
        super()._set_plot_properties(kwargs)
        share = None
        if lock_zoom:
            share = "all"
        self.ax2 = super().add_subplot_axes(self.ax, "right", share=share)
        # both panels are essentially single spectra, so create two instances and
        # link that figure with this one
        self.panel1 = SingleSpectrumPlot(
            ds1,
            figsize=figsize,
            custom_figure=(self.fig, self.ax),
            use_residuals=use_residuals,
        )
        self.panel1.ax.set_title(ds1.datfile.stem)
        self.panel2 = SingleSpectrumPlot(
            ds2,
            figsize=figsize,
            custom_figure=(self.fig, self.ax2),
            use_residuals=use_residuals,
        )
        self.panel2.ax.set_title(ds2.datfile.stem)

    def set_x_scaling(self, x_scaling):
        """
        Overloads parent method, calls x scaling setter for each panel.

        Parameters
        ----------
        x_scaling : int, float, complex, numpy.ndarray
            The scaling to apply to the x-axis
        """
        for panel in (self.panel1, self.panel2):
            panel.set_x_scaling(x_scaling)

    def set_y_scaling(self, y_scaling):
        """
        Overloads parent method, calls y scaling setter for each panel.

        Parameters
        ----------
        y_scaling : int, float, complex, numpy.ndarray
            The scaling to apply to the y-axis
        """
        for panel in (self.panel1, self.panel2):
            panel.set_y_scaling(y_scaling)

    def _use_custom_axes(self):
        """Splits the original 1x2 plot into a 2x2 plot."""
        self.panel1._ef_ax = self.panel1.add_subplot_axes(self.panel1.ax, loc="bottom")
        self.panel2._ef_ax = self.panel2.add_subplot_axes(self.panel2.ax, loc="bottom")

    def add_spectrum(self) -> None:
        """Adds the spectra for both datasets"""
        self.panel1.add_spectrum()
        self.panel2.add_spectrum()

    def add_eigenfunctions(self):
        """Adds the eigenfunctions for both datasets and merges the mpl callbacks."""
        self._use_custom_axes()
        for panel in [self.panel1, self.panel2]:
            panel.add_eigenfunctions()

    def add_continua(self, interactive=True):
        """Adds the continua for both datasets and merges the mpl callbacks."""
        for panel in (self.panel1, self.panel2):
            panel.add_continua(interactive)
