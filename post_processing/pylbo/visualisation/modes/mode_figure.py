from __future__ import annotations

from typing import Union
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pylbo.utilities.logger import pylboLogger
from pylbo.utilities.toolbox import reduce_to_unique_array
from pylbo.visualisation.figure_window import FigureWindow
from pylbo.visualisation.modes.mode_data import ModeVisualisationData
from pylbo.visualisation.utils import add_axis_label, ensure_attr_set


class ModeFigure(FigureWindow):
    """
    Main class to hold the figure, axes and colorbar for eigenmode visualisations.

    Parameters
    ----------
    figsize : tuple[int, int]
        The size of the figure.
    data : ModeVisualisationData
        The data used for eigenmode visualisations.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    axes : dict[str, matplotlib.axes.Axes]
        The axes.
    cbar : matplotlib.colorbar.Colorbar
        The colorbar.
    cbar_ax : matplotlib.axes.Axes
        The axes for the colorbar.
    data : ModeVisualisationData
        Data object containing all data associated with the selected eigenmode.
    u1_data : np.ndarray
        The data for the :math:`u_1` coordinate.
    u2_data : Union[float, np.ndarray]
        The data for the :math:`u_2` coordinate.
    u3_data : Union[float, np.ndarray]
        The data for the :math:`u_3` coordinate.
    ef_data : list[dict]
        The data for the eigenfunction.
    time_data : Union[float, np.ndarray]
        The data for the time.
    omega_txt: matplotlib.text.Text
        The text for the :math:`\\omega` label.
    k2k3_txt: matplotlib.text.Text
        The text for the :math:`k_2-k_3` label.
    u2u3_txt: matplotlib.text.Text
        The text for the :math:`u_2-u_3` label.
    t_txt: matplotlib.text.Text
        The text for the time label.
    """

    def __init__(
        self, figsize: tuple[int, int], data: ModeVisualisationData, show_ef_panel: bool
    ) -> None:
        self.cbar = None
        self._cbar_hspace = 0.01
        self._show_ef_panel = show_ef_panel
        self._annotate = True

        if figsize is None:
            figsize = (14, 8)
        if self._kwargs.get("custom_figure", None) is not None:
            pylboLogger.info("using user-defined figure and axes")
            fig, ax = self._kwargs.pop("custom_figure")
            axes = {"view": ax}
            self._show_ef_panel = False
        else:
            fig, axes = self._create_figure_layout(figsize)
        super().__init__(fig)
        self.axes = axes
        self.cbar_ax = self._create_cbar_axes(width=0.02)

        # Main data object
        self.data = data
        # stuff ploted on the view panel
        self._view = None
        # textbox objects
        [setattr(self, f"{val}_txt", None) for val in ("omega", "k2k3", "u2u3", "t")]
        # data objects
        [setattr(self, f"{val}_data", None) for val in ("u1", "u2", "u3", "time")]
        self.ef_data = []
        self.solution_shape = None

        [ensure_attr_set(self, attr) for attr in ("_u1", "_u2", "_u3", "_time")]

        self.set_plot_arrays()
        for attr in ("u1", "u2", "u3", "time"):
            ensure_attr_set(self, f"{attr}_data")
        ensure_attr_set(self, "solution_shape")

        self._multiple_wavenumbers = (
            len(reduce_to_unique_array(self.data.k2))
            + len(reduce_to_unique_array(self.data.k3))
            > 2
        )

        # don't explicitly create an empty array as this may return a broadcasted view
        self._solutions = 0
        for efdata in self.ef_data:
            self._solutions += self.calculate_mode_solution(
                efdata=efdata,
                u2=self.u2_data,
                u3=self.u3_data,
                t=self.time_data,
            )
        if self.data.add_background:
            self._solutions += self.data.get_background(self._solutions.shape)

        pylboLogger.info(f"eigenmode solution shape {self._solutions.shape}")

    def _check_if_number(self, val: float, attr_name: str) -> float:
        """
        Checks if a given value is a number.

        Parameters
        ----------
        val : float
            The value to check.
        attr_name : str
            The name of the value.

        Raises
        ------
        ValueError
            If the value is not a number.
        """
        if not isinstance(val, (int, np.integer, float)):
            raise ValueError(f"expected a number for {attr_name} but got {type(val)}")
        return val

    def _check_if_array(self, array: np.ndarray, attr_name: str) -> np.ndarray:
        """
        Checks is a given value is a numpy array.

        Parameters
        ----------
        array : np.ndarray
            The value to check.
        attr_name : str
            The name of the value.

        Raises
        ------
        ValueError
            If the value is not a numpy array.
        """
        if not isinstance(array, np.ndarray):
            raise ValueError(
                f"expected a Numpy array for {attr_name} but got {type(array)}"
            )
        return array

    def set_plot_arrays(self) -> None:
        """
        Sets the arrays used for plotting. This should implement setting of
        :attr:`u1_data`, :attr:`u2_data`, :attr:`u3_data`, :attr:`t_data` and
        :attr:`ef_data`.
        """
        raise NotImplementedError()

    def calculate_mode_solution(
        self,
        efdata: dict,
        u2: Union[float, np.ndarray],
        u3: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Calculates the mode solution.

        Parameters
        ----------
        efdata : dict
            The data for the eigenfunction. This should be a dictionary with the
            keys ``'ef'``, ``'omega'`` and ``'factor'``, with ``'ef'``containing
            the eigenfunction, ``'omega'`` the corresponding eigenvalue and
            ``'factor'``the corresponding complex factor to be multiplied with.
        u2 : Union[float, np.ndarray]
            The data for the :math:`u_2` coordinate.
        u3 : Union[float, np.ndarray]
            The data for the :math:`u_3` coordinate.
        t : Union[float, np.ndarray]
            The data for the time.

        Returns
        -------
        np.ndarray
            The mode solution.
        """
        return self.data.get_mode_solution(
            ef=efdata["ef"],
            omega=efdata["omega"],
            complex_factor=efdata["factor"],
            u2=u2,
            u3=u3,
            t=t,
            k2=efdata["k2"],
            k3=efdata["k3"],
        )

    @property
    def ax(self) -> Axes:
        """
        Returns
        -------
        matplotlib.axes.Axes
            Alias for the axes containing the eigenmode solution view.
        """
        return self.axes["view"]

    @property
    def solutions(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The solutions for the eigenmode
        """
        return self._solutions

    def draw(self) -> None:
        self.draw_eigenfunction()
        self.draw_solution()
        if self._annotate:
            self.draw_textboxes()
        self.add_axes_labels()
        super().draw()

    def draw_solution(self) -> None:
        raise NotImplementedError()

    def draw_textboxes(self) -> None:
        u2u3ax = self.axes.get("eigfunc", None) or self.ax
        self.add_u2u3_txt(u2u3ax, loc="top right", outside=True)
        if not self._multiple_wavenumbers:
            self.add_k2k3_txt(self.ax, loc="bottom left", color="white", alpha=0.5)

    def draw_eigenfunction(self) -> None:
        """Draws the eigenfunction(s) to the figure."""
        ax = self.axes.get("eigfunc", None)
        if ax is None:
            return
        grid = self.data.ds_bg.ef_grid
        for efs, omegas, factors in zip(
            self.data.eigenfunction, self.data.omega, self.data.complex_factor
        ):
            for ef, omega, factor in zip(efs, omegas, factors):
                label = rf"$\omega$ = {omega:.5f}"
                ef = getattr(factor * ef, self.data.part_name)
                ax.plot(grid, ef, lw=2, label=label)
        ax.axvline(x=0, color="grey", ls="--", lw=1)
        ax.set_xlim(np.min(grid), np.max(grid))
        ax.set_ylabel(self.data._ef_name_latex)
        if len(list(chain.from_iterable(self.data.omega))) < 6:
            ax.legend(loc="best")

    def add_axes_labels(self) -> None:
        self.ax.set_xlabel(self.get_view_xlabel())
        self.ax.set_ylabel(self.get_view_ylabel())
        self.cbar.set_label(self.get_view_cbar_label())

    def _create_cbar_axes(self, width: float) -> Axes:
        """
        Creates the axes for the colorbar.

        Parameters
        ----------
        width : float
            The width of the colorbar axes.
        Returns
        -------
        matplotlib.axes.Axes
            The axes for the colorbar.
        """
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width - 2.5 * width, box.height])
        # shift main and ef axes to the left to make space
        # update box to reflect the new position
        ax_ef = self.axes.get("eigfunc", None)
        if ax_ef is not None:
            box = ax_ef.get_position()
            ax_ef.set_position([box.x0, box.y0, box.width - 2.5 * width, box.height])
        box = self.ax.get_position()
        position = (box.x0 + box.width, box.y0)
        dims = (width, box.height)
        return self.fig.add_axes([*position, *dims])

    def get_view_xlabel(self) -> str:
        return self.data.ds_bg.u1_str

    def get_view_ylabel(self) -> str:
        return ""

    def get_view_cbar_label(self) -> str:
        return rf"{self.data._ef_name_latex}"

    def add_omega_txt(self, ax, **kwargs) -> None:
        """
        Creates a textbox on the axis with the value of the eigenfrequency.

        Parameters
        ----------
        ax : ~matplotlib.axes.Axes
            The axes to use for the textbox.
        **kwargs
            Additional keyword arguments to pass to :meth:`add_axis_label`.
        """
        omega_labels = list(chain.from_iterable(self.data.omega))
        if self.omega_txt is None:
            self.omega_txt = rf"$\omega$ = {omega_labels:.5f}"
        add_axis_label(ax, self.omega_txt, **kwargs)

    def add_k2k3_txt(self, ax, **kwargs) -> None:
        """
        Creates a textbox on the figure with the value of the k2 and k3 coordinates.

        Parameters
        ----------
        ax : ~matplotlib.axes.Axes
            The axes to use for the textbox.
        **kwargs
            Additional keyword arguments to pass to :meth:`add_axis_label`.
        """
        if self.k2k3_txt is None:
            self.k2k3_txt = "".join(
                [
                    f"{self.data.ds_bg.k2_str} = {self.data.k2[0]} | ",
                    f"{self.data.ds_bg.k3_str} = {self.data.k3[0]}",
                ]
            )
        add_axis_label(ax, self.k2k3_txt, **kwargs)

    def add_u2u3_txt(self, ax, **kwargs) -> None:
        """
        Creates a textbox on the figure with the value of the :math:`u_2-u_3`
        coordinates.

        Parameters
        ----------
        ax : ~matplotlib.axes.Axes
            The axes to use for the textbox.
        **kwargs
            Additional keyword arguments to pass to :meth:`add_axis_label`.
        """
        if self.u2u3_txt is None:
            self.u2u3_txt = "".join(
                [
                    rf"{self.data.ds_bg.u2_str} = {self._u2} | ",
                    rf"{self.data.ds_bg.u3_str} = {self._u3}",
                ]
            )
        add_axis_label(ax, self.u2u3_txt, **kwargs)

    def add_t_txt(self, ax, **kwargs) -> None:
        pass

    def _create_figure_layout(self, figsize: tuple[int, int]) -> tuple[Figure, dict]:
        """
        Create the figure layout for the visualisation. Two panels are created:
        the top one for the eigenfunction and the bottom one for the visualisation.

        Parameters
        ----------
        figsize : tuple[int, int]
            The size of the figure.

        Returns
        -------
        fig : ~matplotlib.figure.Figure
            The figure to use for the visualisation.
        axes : dict
            The axes to use for the visualisation.
        """
        fig = plt.figure(figsize=figsize)
        if not self._show_ef_panel:
            ax2 = fig.add_subplot()
            return fig, {"view": ax2}

        width = 0.75
        height_1 = 0.2
        height_2 = 0.5
        v_space = 0.1
        x = (1 - width) / 2
        y1 = 1 - height_1 - v_space
        y2 = v_space
        # left, bottom, width, height in Figure coordinates
        ax1 = fig.add_axes([x, y1, width, height_1])
        ax2 = fig.add_axes([x, y2, width, height_2])
        return fig, {"eigfunc": ax1, "view": ax2}

    def create_animation(
        self, times: np.ndarray, filename: str, fps: float = 10, dpi: int = 200
    ) -> None:
        """
        Creates an animation of the eigenmode solution over a given time interval.

        Parameters
        ----------
        times : np.ndarray
            The times at which to create the animation.
        filename : str
            The filename of the animation.
        fps : float
            The frames per second of the animation.
        dpi : int
            The resolution of the animation.
        """
        raise ValueError(f"{self.__class__.__name__} does not support animation.")
