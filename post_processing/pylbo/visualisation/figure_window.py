from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as mpl_axes
from matplotlib.figure import Figure as mpl_fig
from pylbo.utilities.logger import pylboLogger
from pylbo.utilities.toolbox import get_axis_geometry, transform_to_numpy
from pylbo.visualisation.eigenfunctions.eigfunc_interface import EigenfunctionInterface
from pylbo.visualisation.legend_handler import LegendHandler


class FigureWindow:
    """
    Class to handle the top-level creation of figure windows. Assigns unique figure
    ids and takes care of figure, axes, and gridspec management.

    Parameters
    ----------
    fig : ~matplotlib.figure.Figure
        The figure object.

    Attributes
    ----------
    fig : ~matplotlib.figure.Figure
        The figure object.
    figsize : tuple[int, int]
        The size of the figure in inches.
    figure_id : str
        The unique figure id.
    """

    figure_stack = dict()

    def __init__(self, fig: mpl_fig) -> None:
        self.fig = fig
        self.figsize = fig.get_size_inches()
        self.figure_id = fig.get_label()
        self._figure_drawn = False
        self.add_to_stack()

    @property
    def figure_ids(self) -> list[str]:
        """Returns the list of figure ids."""
        return list(self.figure_stack.keys())

    def create_default_figure(
        self, figlabel: str, figsize: tuple[int, int]
    ) -> tuple[mpl_fig, mpl_axes]:
        """
        Creates a default figure with a 1x1 subplot.

        Parameters
        ----------
        figlabel : str
            The label of the figure.
        figsize : tuple[int, int]
            The size of the figure.

        Returns
        -------
        fig : ~matplotlib.figure.Figure
            The figure on which to draw.
        ax : ~matplotlib.axes.Axes
            The axes on which to draw.
        """
        if figsize is None:
            figsize = (12, 8)
        fig = plt.figure(self._generate_figure_id(figlabel), figsize=figsize)
        ax = fig.add_subplot(111)
        return fig, ax

    def _generate_figure_id(self, figlabel: str) -> str:
        """
        Generates a unique figure id.

        Parameters
        ----------
        figlabel : str
            The label of the figure.

        Returns
        -------
        figure_id : str
            The unique figure id of the form "figure_type-x" where x is an integer.
        """
        # count occurences of this type of id in the list
        occurences = sum(figlabel in fig_id for fig_id in self.figure_ids)
        return f"{figlabel}-{1 + occurences}"

    def make_layout_tight(self) -> None:
        """
        Calls tight_layout() on a figure and captures the userwarning introduced
        in matplotlib 3.5.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.fig.tight_layout()
            for warning in w:
                msg = warning.message
                if not (isinstance(msg, UserWarning) and "tight" in str(msg)):
                    pylboLogger.warning(
                        f"{warning.filename}:{warning.lineno}: "
                        f"{warning.category.__name__}: {msg}"
                    )

    def add_to_stack(self) -> None:
        """
        Adds the figure to the stack.
        """
        self.figure_stack[self.figure_id] = self

    def add_subplot_axes(
        self,
        ax: mpl_axes,
        loc: str = "right",
        share: str = None,
        apply_tight_layout: bool = True,
    ):
        """
        Adds a new subplot to a given matplotlib subplot, essentially "splitting" the
        axis into two. Position and placement depend on the loc argument.
        When called on a more complex subplot layout the overall gridspec remains
        untouched, only the `ax` object has its gridspec modified.
        On return, `tight_layout()` is called by default to prevent overlapping labels.

        Parameters
        ----------
        ax : ~matplotlib.axes.Axes
            The axes object, this will be "split" and a new axes will be added
            to the figure.
        loc : str
            The location of the new axes. Should be one of "left", "right", "top",
            "bottom". Defaults to "right".
        share : str
            Can be "x", "y" or "all". This locks axes zooming between both subplots.
        apply_tight_layout : bool
            Whether to call `tight_layout()` on the figure before return.

        Raises
        ------
        ValueError
            If the loc argument is invalid.

        Returns
        -------
        ~matplotlib.axes.Axes
            The axes instance that was added.
        """
        sharex = ax if share in ("x", "all") else None
        sharey = ax if share in ("y", "all") else None

        if loc == "right":
            subplot_geometry = (1, 2)
            old_new_position = (0, 1)
        elif loc == "left":
            subplot_geometry = (1, 2)
            old_new_position = (1, 0)
        elif loc == "top":
            subplot_geometry = (2, 1)
            old_new_position = (1, 0)
        elif loc == "bottom":
            subplot_geometry = (2, 1)
            old_new_position = (0, 1)
        else:
            raise ValueError(
                f"invalid loc={loc}, expected ['top', 'right', 'bottom', 'left']"
            )

        _geometry = transform_to_numpy(get_axis_geometry(ax))
        subplot_index = _geometry[-1]
        gspec_outer = gridspec.GridSpec(*_geometry[0:2], figure=self.fig)
        gspec_inner = gridspec.GridSpecFromSubplotSpec(
            *subplot_geometry, subplot_spec=gspec_outer[subplot_index]
        )
        ax.set_subplotspec(gspec_inner[old_new_position[0]])
        new_axis = self.fig.add_subplot(
            gspec_inner[old_new_position[1]], sharex=sharex, sharey=sharey
        )
        if apply_tight_layout:
            self.make_layout_tight()
        return new_axis

    def draw(self) -> None:
        self._figure_drawn = True

    def redraw(self) -> None:
        self.ax.cla()
        self.draw()

    def save(self, filename: str, **kwargs) -> None:
        """
        Saves the current figure.

        Parameters
        ----------
        filename : str, ~os.PathLike
            The filename to which the current figure is saved.
        kwargs
            Default keyword arguments passed to :meth:`~matplotlib.pyplot.savefig`.
        """
        filepath = Path(filename).resolve()
        self.fig.savefig(filepath, **kwargs)
        pylboLogger.info(f"figure saved to {filepath}")

    def show(self):
        """Shows the selected figure"""
        if not self._figure_drawn:
            self.draw()
        plt.show()


class InteractiveFigureWindow(FigureWindow):
    "Subclass to handle interactivity in the figure windows."

    def __init__(self, fig: mpl_fig) -> None:
        super().__init__(fig)
        self._mpl_callbacks = []

    def redraw(self) -> None:
        self.disconnect_callbacks()
        super().redraw()
        self.connect_callbacks()

    def connect_callbacks(self) -> None:
        """Connects all callbacks to the canvas"""
        for callback in self._mpl_callbacks:
            self.fig.canvas.mpl_connect(callback["kind"], callback["method"])

    def disconnect_callbacks(self) -> None:
        """Disconnects all callbacks from the canvas"""
        for callback in self._mpl_callbacks:
            self.fig.canvas.mpl_disconnect(callback["cid"])

    def make_legend_interactive(self, legendhandler: LegendHandler) -> None:
        """
        Makes the legend interactive.

        Parameters
        ----------
        legendhandler : ~pylbo.visualization.legend_handler.LegendHandler
            The legend handler.
        """
        legendhandler.make_legend_pickable()
        callback_kind = "pick_event"
        callback_method = legendhandler.on_legend_pick
        self._mpl_callbacks.append(
            {
                "kind": callback_kind,
                "method": callback_method,
                "cid": self.fig.canvas.mpl_connect(callback_kind, callback_method),
            }
        )

    def add_eigenfunction_interface(self, efhandler: EigenfunctionInterface) -> None:
        """
        Adds an eigenfunction interface to the figure.

        Parameters
        ----------
        efhandler : ~pylbo.visualisation.eigenfunctions.eigfunc_interface.
        EigenfunctionInterface
            The eigenfunction interface.
        """
        callback_kinds = ("pick_event", "key_press_event")
        callback_methods = (efhandler.on_point_pick, efhandler.on_key_press)
        for kind, method in zip(callback_kinds, callback_methods):
            self._mpl_callbacks.append(
                {
                    "kind": kind,
                    "method": method,
                    "cid": self.fig.canvas.mpl_connect(kind, method),
                }
            )
