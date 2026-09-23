import matplotlib.colors as mpl_colors
import numpy as np
from pylbo.utilities.toolbox import add_pickradius_to_item, transform_to_numpy
from pylbo.visualisation.continua import ContinuaHandler
from pylbo.visualisation.eigenfunctions.eigfunc_handler import EigenfunctionHandler
from pylbo.visualisation.spectra.spectrum_figure import SpectrumFigure
from pylbo.utilities.logger import pylboLogger


class MultiSpectrumPlot(SpectrumFigure):
    """
    Subclass that draws the multispectra.

    Parameters
    ----------
    dataseries : ~pylbo.data_containers.LegolasDataSeries
        The dataseries that should be used.
    xdata : str, list, numpy.ndarray
        Data to use for the horizontal axis. This can either be a key from the
        parameters dictionary, or a list/numpy array containing actual data.
    use_squared_omega : bool
        If `True`, this will square the eigenvalues when they are plotted on the
        vertical axis. If `False` (default), either the real or imaginary part of the
        eigenvalues will be plotted depending on the value of `use_real_parts`.
    use_real_parts : bool
        If `True` (default), this will plot the real part of the eigenvalues on the
        vertical axis. If `False` the imaginary part will be used.
    figsize : tuple
        Optional figure size like the usual matplotlib (x, x) size.
    custom_figure : tuple
        Optional, in the form (fig, ax). If supplied no new figure will be created
        but this one will be used instead. `fig` refers to the matplotlib figure and
        `ax` to a (single) axes instance, meaning that you can pass a subplot as well.
    """

    def __init__(
        self,
        dataseries,
        xdata,
        use_squared_omega,
        use_real_parts,
        figsize,
        custom_figure,
        **kwargs,
    ):
        super().__init__(
            custom_figure=custom_figure, figlabel="multi-spectrum", figsize=figsize
        )
        self.dataseries = dataseries
        self.use_squared_omega = use_squared_omega
        self._w_pow = 1
        if self.use_squared_omega:
            self._w_pow = 2
        self.use_real_parts = use_real_parts
        self.xdata = self._validate_xdata(xdata)
        self.ydata = self._get_ydata()
        self.x_scaling = np.ones_like(self.dataseries, dtype=float)
        self.y_scaling = np.ones_like(self.dataseries, dtype=float)
        super()._set_plot_properties(kwargs)

    def _validate_xdata(self, xdata):
        """
        Validates the xdata passed, does typechecking and necessary casting.
        If a string is passed, this will request the proper values based on the
        parameters.

        Parameters
        ----------
        xdata : str, list, numpy.ndarray
            The xdata used as x values on the spectrum plot.

        Returns
        -------
        xdata_values : numpy.ndarray
            The xdata values of proper length and casted to a Numpy array.
        """
        if isinstance(xdata, str):
            if self.dataseries.parameters.get(xdata, None) is None:
                raise ValueError(
                    f"Provided key xdata='{xdata}' is not in parameters: \n"
                    f"{self.dataseries.parameters.keys()}"
                )
            xdata_values = self.dataseries.parameters[xdata]
        elif isinstance(xdata, (list, np.ndarray)):
            xdata_values = transform_to_numpy(xdata)
            if len(xdata_values) != len(self.dataseries):
                raise ValueError(
                    f"Lengts of xdata do not match: "
                    f"{len(xdata_values)} vs {len(self.dataseries)}"
                )
        else:
            raise TypeError(
                f"xdata should be a string, list or numpy array but got {type(xdata)}."
            )
        return xdata_values

    def _get_ydata(self):
        """
        Gets the y data based on the value of :attr:`use_squared_omega`.

        Returns
        -------
        ydata : numpy.ndarray
            The y data values, either the real or imaginary parts based on
            :attr:`use_real_parts`. Every element is an array in itself corresponding
            to the various datasets, hence depending on the gridpoints in every dataset
            the elements themselves may be of different length.
        """
        ydata = np.empty(len(self.dataseries), dtype=object)
        for i, ds in enumerate(self.dataseries):
            ydata[i] = ds.eigenvalues**self._w_pow
            if self.use_real_parts:
                ydata[i] = ydata[i].real
            else:
                ydata[i] = ydata[i].imag
            ydata[i][np.where(np.isclose(ydata[i], 0, atol=1e-15))] = np.nan
        return ydata

    def set_x_scaling(self, x_scaling):
        """
        Sets the x scaling, properly adjusted to the dataseries length.

        Parameters
        ----------
        x_scaling : int, float, complex, numpy.ndarray
            Values to use for the x-scaling.
        """
        if isinstance(x_scaling, (int, float, complex)):
            x_scaling = np.ones_like(self.dataseries, dtype=float) * x_scaling
        super().set_x_scaling(x_scaling)

    def set_y_scaling(self, y_scaling):
        """
        Sets the y scaling, properly adjusted to the dataseries length.

        Parameters
        ----------
        y_scaling : int, float, complex, numpy.ndarray
            Values to use for the y-scaling.
        """
        if isinstance(y_scaling, (int, float, complex)):
            y_scaling = np.ones_like(self.dataseries, dtype=float) * y_scaling
        super().set_y_scaling(y_scaling)

    def add_spectrum(self):
        """
        Draw method, creates the spectrum.
        """
        for i, ds in enumerate(self.dataseries):
            spectrum_points = self.ax.scatter(
                self.xdata[i]
                * np.ones_like(self.ydata[i], dtype=float)
                * self.x_scaling[i],
                self.ydata[i] * self.y_scaling[i],
                marker=self.marker,
                color=self.color,
                s=10 * self.markersize,
                alpha=self.alpha,
                linestyle="None",
                **self.plot_props,
            )
            add_pickradius_to_item(item=spectrum_points, pickradius=10)
            # set dataset associated with this line of points
            setattr(spectrum_points, "dataset", ds)
        self.ax.axhline(y=0, linestyle="dotted", color="grey", alpha=0.3)
        self.ax.axvline(x=0, linestyle="dotted", color="grey", alpha=0.3)

    def add_continua(self, interactive=True):
        """
        Adds the continua to the plot, either interactive or not.

        Parameters
        ----------
        interactive : bool
            If `True`, makes the legend interactive.
        """
        if not self.has_valid_continua(self.dataseries):
            return
        if self.has_zero_continua(self.dataseries):
            pylboLogger.warning("Continua not plotted: all are zero.")
            return
        if self._c_handler is None:
            self._c_handler = ContinuaHandler(interactive=interactive)

        for key, color in zip(
            self._c_handler.continua_names, self._c_handler.continua_colors
        ):
            # we skip duplicates if eigenvalues are squared
            if self.use_squared_omega and key in ("slow-", "alfven-"):
                continue
            # retrieve continuum, calculate region boundaries
            continuum = self.dataseries.continua[key] ** self._w_pow
            continuum = continuum.real if self.use_real_parts else continuum.imag
            min_values = np.array([np.min(c_ds) for c_ds in continuum])
            max_values = np.array([np.max(c_ds) for c_ds in continuum])
            # skip if continua are all zero
            if all(np.isclose(min_values, 0)) and all(np.isclose(max_values, 0)):
                continue
            # when continua are collapsed then min = max and we draw a line instead
            if all(np.isclose(abs(max_values - min_values), 0)):
                (item,) = self.ax.plot(
                    self.xdata * self.x_scaling,
                    min_values * self.y_scaling,
                    color=color,
                    alpha=self._c_handler.alpha_point,
                    label=key,
                )
            else:
                item = self.ax.fill_between(
                    self.xdata * self.x_scaling,
                    min_values * self.y_scaling,
                    max_values * self.y_scaling,
                    facecolor=mpl_colors.to_rgba(color, self._c_handler.alpha_region),
                    edgecolor=mpl_colors.to_rgba(color, self._c_handler.alpha_point),
                    linewidth=self._c_handler.linewidth,
                    label=key,
                )
            self._c_handler.add(item)
        self._c_handler.legend = self.ax.legend(**self._c_handler.legend_properties)
        if interactive:
            super().make_legend_interactive(self._c_handler)

    def add_eigenfunctions(self):
        """Adds the eigenfunctions to the current figure."""
        if self._ef_ax is None:
            self._ef_ax = super().add_subplot_axes(self.ax, loc="right")
        if self._ef_handler is None:
            self._ef_handler = EigenfunctionHandler(
                self.dataseries, self._ef_ax, self.ax
            )
        super().add_eigenfunction_interface(efhandler=self._ef_handler)
