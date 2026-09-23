from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from pylbo.visualisation.modes.cartesian_2d import CartesianSlicePlot2D
from pylbo.visualisation.modes.mode_data import ModeVisualisationData


class CartesianSlicePlot3D(CartesianSlicePlot2D):
    """
    Class for handling Cartesian 3D plots of the eigenmode solution.

    Parameters
    ----------
    data : ModeVisualisationData
        The data for the visualisation.
    u2 : np.ndarray
        The :math:`y` coordinate of the eigenmode solution.
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
        if slicing_axis == "y":
            raise NotImplementedError("3D slicing is not implemented for y-axis.")
        super().__init__(data, u2, u3, time, slicing_axis, figsize, False, **kwargs)

        self.vmin = np.min(self._solutions) if vmin is None else vmin
        self.vmax = np.max(self._solutions) if vmax is None else vmax
        self._view = [None] * len(self._u3)
        self.set_contours(levels=25, fill=True)

    def _create_figure_layout(self, figsize: tuple[int, int]) -> tuple[Figure, dict]:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_color("none")
        ax.grid(False)
        for attr in "xyz":
            getattr(ax, f"{attr}axis").set_pane_color((1.0, 1.0, 1.0, 1.0))
        return fig, {"view": ax}

    def _create_cbar_axes(self, width: float) -> Axes:
        box = self.ax.get_position()
        position = (box.x0, box.height + 0.02 + 0.1)
        dims = (box.width, width)
        return self.fig.add_axes([*position, *dims])

    def _validate_u2(self, u2: np.ndarray, *args, **kwargs) -> np.ndarray:
        return u2

    def _validate_u3(self, u3: np.ndarray, *args, **kwargs) -> np.ndarray:
        return u3

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
        x_2d, y_2d = np.meshgrid(self.data.ds_bg.ef_grid, self._u2, indexing="ij")
        self.u1_data = x_2d
        self.u2_data = y_2d
        self.u3_data = self._u3
        self.time_data = self._time

    def calculate_mode_solution(
        self, efdata: np.ndarray, u2: np.ndarray, u3: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        solutions = np.zeros(shape=(*self.solution_shape, len(u3)))
        for i, z in enumerate(u3):
            for efdata in self.ef_data:
                solutions[..., i] += super().calculate_mode_solution(efdata, u2, z, t)
        return solutions

    def draw_eigenfunction(self) -> None:
        pass

    def draw_solution(self) -> None:
        level_kwargs = {}
        if self._contour_levels is not None:
            level_kwargs["levels"] = self._contour_levels
        for i, z in enumerate(self._u3):
            self._view[i] = self._contour_recipe(
                self.u1_data,
                self.u2_data,
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
        self.ax.set_xlim(np.min(self._u1), np.max(self._u1))
        self.ax.set_ylim(np.min(self._u2), np.max(self._u2))
        self.ax.set_zlim(np.min(self._u3), np.max(self._u3))

    def add_axes_labels(self) -> None:
        super().add_axes_labels()
        self.ax.set_zlabel("z")

    def draw_textboxes(self) -> None:
        self.t_txt = self.ax.text2D(
            0.9,
            0.9,
            f"t = {self._time:.2f}",
            fontsize=15,
            transform=self.ax.transAxes,
            ha="center",
            bbox=dict(facecolor="grey", alpha=0.2, boxstyle="round", pad=0.2),
        )

    def _clear_contours(self) -> None:
        for view in self._view:
            for coll in view.collections:
                try:
                    coll.remove()
                except ValueError:
                    pass

    def _update_view(self, updated_solution: np.ndarray) -> None:
        super()._update_contour_plot(updated_solution)

    def _update_view_clims(self, solution: np.ndarray) -> None:
        if self.update_colorbar:
            self.vmin, self.vmax = np.min(solution), np.max(solution)

    def _set_t_txt(self, t):
        self.t_txt.set_text(f"t = {t:.2f}")
