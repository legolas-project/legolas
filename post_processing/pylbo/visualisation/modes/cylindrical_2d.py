from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from pylbo._version import _mpl_version
from pylbo.visualisation.modes.cartesian_2d import CartesianSlicePlot2D
from pylbo.visualisation.modes.mode_data import ModeVisualisationData


class CylindricalSlicePlot2D(CartesianSlicePlot2D):
    """
    Class for handling cylindrical 2D plots of the eigenmode solutions.

    Parameters
    ----------
    data : ModeVisualisationData
        The data for the visualisation.
    u2 : float or ndarray
        The :math:`\\theta`  coordinate of the eigenmode solution.
    u3 : float or ndarray
        The :math:`z`  coordinate of the eigenmode solution.
    time : float
        The time at which the eigenmode solution is calculated.
    slicing_axis : str
        The axis along which the eigenmode solution is sliced.
    figsize : tuple[int, int]
        The size of the figure.
    show_ef_panel : bool
        Whether to show the eigenfunction panel.
    polar : bool
        Whether to use polar coordinates for the plot.
    **kwargs
        Additional keyword arguments to be passed to
        :meth:`matplotlib.pyplot.pcolormesh`.
    """

    def __init__(
        self,
        data: ModeVisualisationData,
        u2: Union[float, np.ndarray],
        u3: Union[float, np.ndarray],
        time: float,
        slicing_axis: str,
        figsize: tuple[int, int],
        show_ef_panel: bool,
        polar: bool,
        **kwargs,
    ) -> None:
        self._use_polar_axes = polar
        super().__init__(
            data, u2, u3, time, slicing_axis, figsize, show_ef_panel, **kwargs
        )

    def set_plot_arrays(self) -> None:
        if self.slicing_axis == self._u2axis:
            return super().set_plot_arrays()
        self.solution_shape = (len(self._u1), len(self._u2))
        for efs, omegas, factors, k2, k3 in zip(
            self.data.eigenfunction,
            self.data.omega,
            self.data.complex_factor,
            self.data.k2,
            self.data.k3,
        ):
            for ef, omega, factor in zip(efs, omegas, factors):
                data = np.broadcast_to(
                    ef, shape=reversed(self.solution_shape)
                ).transpose()
                self.ef_data.append(
                    {"ef": data, "omega": omega, "factor": factor, "k2": k2, "k3": k3}
                )
        r_2d, theta_2d = np.meshgrid(self.data.ds_bg.ef_grid, self._u2, indexing="ij")

        self.u1_data = r_2d
        self.u2_data = theta_2d
        self.u3_data = self._u3
        self.time_data = self._time

    def _draw_image(self) -> None:
        if self.slicing_axis == self._u2axis:
            return super()._draw_image()
        if self._use_polar_axes:
            xdata = self.u2_data
            ydata = self.u1_data
            if _mpl_version >= "3.5":
                # auto-grid removal deprecates from MPL 3.5 onwards
                self.ax.grid(False)
        else:
            xdata = self.u1_data * np.cos(self.u2_data)
            ydata = self.u1_data * np.sin(self.u2_data)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self._view = self.ax.pcolormesh(
                xdata, ydata, self.solutions, **self._kwargs
            )
        self.cbar = self.fig.colorbar(
            ScalarMappable(norm=self._view.norm, cmap=self._view.cmap), cax=self.cbar_ax
        )

    def _draw_contours(self) -> None:
        if self.slicing_axis == self._u2axis:
            return super()._draw_contours()
        additional_kwargs = {}
        if self._contour_levels is not None:
            additional_kwargs["levels"] = self._contour_levels
        if self._use_polar_axes:
            xdata = self.u2_data
            ydata = self.u1_data
        else:
            xdata = self.u1_data * np.cos(self.u2_data)
            ydata = self.u1_data * np.sin(self.u2_data)
        self._view = self._contour_recipe(
            xdata,
            ydata,
            self.solutions,
            vmin=self.vmin,
            vmax=self.vmax,
            **additional_kwargs,
            **self._kwargs,
        )
        self.cbar = self.fig.colorbar(
            ScalarMappable(norm=self._view.norm, cmap=self._view.cmap), cax=self.cbar_ax
        )

    def draw_eigenfunction(self) -> None:
        super().draw_eigenfunction()
        if self._show_ef_panel:
            self.axes["eigfunc"].set_xlabel(self.data.ds_bg.u1_str)

    def get_view_xlabel(self) -> str:
        if self._use_polar_axes:
            return ""
        if self.slicing_axis == self._u3axis:
            return "x"
        return super().get_view_xlabel()

    def get_view_ylabel(self) -> str:
        if self._use_polar_axes:
            return ""
        if self.slicing_axis == self._u3axis:
            return r"$\theta$"
        return super().get_view_ylabel()

    def _create_figure_layout(self, figsize: tuple[int, int]) -> tuple[Figure, dict]:
        if self.slicing_axis == self._u2axis:
            return super()._create_figure_layout(figsize)
        fig = plt.figure(figsize=figsize)
        polar = self._use_polar_axes
        if self._show_ef_panel:
            ax1 = fig.add_axes([0.1, 0.7, 0.8, 0.2])
            ax2 = fig.add_axes([0.25, 0.1, 0.5, 0.5], aspect="equal", polar=polar)
        else:
            ax1 = None
            ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], aspect="equal", polar=polar)
        if polar:
            self._cbar_hspace = 0.05
        return fig, {"eigfunc": ax1, "view": ax2}
