from __future__ import annotations

from copy import copy

import numpy as np
from matplotlib.axes import Axes as mpl_axes
from matplotlib.figure import Figure as mpl_fig
from pylbo.visualisation.figure_window import InteractiveFigureWindow
from pylbo.visualisation.utils import refresh_plot


class SpectrumFigure(InteractiveFigureWindow):
    """
    Class to handle the creation of a figure window dedicated to different types of
    spectrum figures.

    Parameters
    ----------
    custom_figure : tuple[~matplotlib.figure.Figure, ~matplotlib.axes.Axes]
        A custom figure to use, in the form (fig, ax) corresponding to the figure
        and axis objects from matplotlib.
    figlabel : str
        The label of the figure, used to generate a unique figure id
    figsize : tuple[int, int]
        The size of the figure, default is (10, 6).


    Attributes
    ----------
    ax : ~matplotlib.axes.Axes
        The axes object.
    x_scaling : int, float, complex, np.ndarray
        The scaling of the x-axis.
    y_scaling : int, float, complex, np.ndarray
        The scaling of the y-axis.
    """

    def __init__(
        self,
        custom_figure: tuple[mpl_fig, mpl_axes] = None,
        figlabel: str = None,
        figsize: tuple[int, int] = None,
    ):
        fig, ax = (
            custom_figure
            if custom_figure is not None
            else super().create_default_figure(figlabel=figlabel, figsize=figsize)
        )
        super().__init__(fig)

        self.ax = ax
        self.x_scaling = 1.0
        self.y_scaling = 1.0
        self._c_handler = None
        self._ef_handler = None
        self._ef_ax = None
        self._def_handler = None
        self._def_ax = None

        self.plot_props = None
        self.marker = None
        self.color = None
        self.markersize = None
        self.alpha = None

    def draw(self):
        """
        Called on plot refreshing, the super call clears the figure and then redraws
        everything.
        """
        self.add_spectrum()
        super().draw()

    @refresh_plot
    def set_x_scaling(self, x_scaling):
        """
        Sets the x scaling.

        Parameters
        ----------
        x_scaling : int, float, complex, numpy.ndarray
            The scaling to apply to the x-axis.
        """
        self.x_scaling = x_scaling

    @refresh_plot
    def set_y_scaling(self, y_scaling):
        """
        Sets the y scaling.

        Parameters
        ----------
        y_scaling : int, float, complex, numpy.ndarray
            The scaling to apply to the y-axis
        """
        self.y_scaling = y_scaling

    def _set_plot_properties(self, properties):
        """
        Sets all relevant plot properties.

        Parameters
        ----------
        properties : dict
            Dictionary containing the usual matplotlib properties (marker, color,
            markersize, alpha, etc.)
        """
        plot_props = copy(properties)
        self.marker = plot_props.pop("marker", ".")
        self.color = plot_props.pop("color", "blue")
        self.markersize = plot_props.pop("markersize", 6)
        self.alpha = plot_props.pop("alpha", 0.8)
        self.plot_props = plot_props

    def add_spectrum(self):
        raise NotImplementedError()

    def add_continua(self, interactive):
        raise NotImplementedError()

    def add_eigenfunctions(self):
        raise NotImplementedError()

    def has_valid_continua(self, data):
        continua = getattr(data, "continua", None)
        if isinstance(continua, (list, np.ndarray)):
            return all([c is not None for c in continua])
        return continua is not None

    def has_zero_continua(self, data):
        if isinstance(data.continua, (list, np.ndarray)):
            return all([self._zero_continua(c) for c in data.continua])
        else:
            return self._zero_continua(data.continua)

    def _zero_continua(self, continua):
        for key in continua.keys():
            if not np.all(np.isclose(continua[key], 0, atol=1e-12)):
                return False
        return True

    @property
    def c_handler(self):
        """Property, returns the continua handler."""
        return self._c_handler

    @property
    def ef_handler(self):
        """Property, returns the eigenfunction handler."""
        return self._ef_handler

    @property
    def ef_ax(self):
        """Property, returns the eigenfunction axes."""
        return self._ef_ax
