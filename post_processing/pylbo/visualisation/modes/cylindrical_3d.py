from __future__ import annotations

import numpy as np
from matplotlib.cm import ScalarMappable
from pylbo.visualisation.modes.cartesian_3d import CartesianSlicePlot3D
from pylbo.visualisation.modes.mode_data import ModeVisualisationData


class CylindricalSlicePlot3D(CartesianSlicePlot3D):
    """
    Class for handling cylindrical 3D plots of the eigenmode solution.

    Parameters
    ----------
    data : ModeVisualisationData
        The data for the visualisation.
    u2 : np.ndarray
        The :math:`\\theta` coordinate of the eigenmode solution.
    u3 : np.ndarray
        The :math:`z` coordinate of the eigenmode solution.
    time : float
        The time at which the eigenmode solution is calculated.
    slicing_axis : str
        The axis along which the eigenmode solution is sliced.
    figsize : tuple[int, int]
        The size of the figure.
    vmin : float
        The minimum value of the colourbar. If None, the minimum value of the
        solution is used.
    vmax : float
        The maximum value of the colourbar. If None, the maximum value of the
        solution is used.
    """

    def __init__(
        self,
        data: ModeVisualisationData,
        u2: np.ndarray,
        u3: np.ndarray,
        time: float,
        slicing_axis: str,
        figsize: tuple[int, int],
        vmin: float = None,
        vmax: float = None,
        **kwargs,
    ) -> None:
        if figsize is None:
            figsize = (8, 8)
        if slicing_axis == "theta":
            raise NotImplementedError(
                "3D slicing is not implemented for a theta slice."
            )
        super().__init__(data, u2, u3, time, slicing_axis, figsize, **kwargs)

        self.vmin = np.min(self._solutions) if vmin is None else vmin
        self.vmax = np.max(self._solutions) if vmax is None else vmax
        self.set_contours(levels=25, fill=True)

    def set_plot_arrays(self) -> None:
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

    def draw_solution(self) -> None:
        level_kwargs = {}
        if self._contour_levels is not None:
            level_kwargs["levels"] = self._contour_levels
        for i, z in enumerate(self._u3):
            self._view[i] = self._contour_recipe(
                self.u1_data * np.cos(self.u2_data),
                self.u1_data * np.sin(self.u2_data),
                self.solutions[..., i],
                zdir="z",
                offset=z,
                alpha=max(0.4, 1 - i * 0.1),
                vmin=self.vmin,
                vmax=self.vmax,
                **level_kwargs,
                **self._kwargs,
            )
        self.cbar = self.fig.colorbar(
            ScalarMappable(norm=self._view[0].norm, cmap=self._view[0].cmap),
            cax=self.cbar_ax,
            orientation="horizontal",
        )
        xmax = np.max(self._u1)
        self.ax.set_xlim(-xmax, xmax)
        self.ax.set_ylim(-xmax, xmax)
        self.ax.set_zlim(np.min(self._u3), np.max(self._u3))

    def get_view_xlabel(self) -> str:
        return "x"

    def get_view_ylabel(self) -> str:
        return "y"
