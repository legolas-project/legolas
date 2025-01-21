from __future__ import annotations

from typing import Union

import numpy as np
import pylbo
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from pylbo.utilities.toolbox import transform_to_list, is_custom_grid
from pylbo.visualisation.modes.mode_data import ModeVisualisationData
from pylbo.visualisation.modes.mode_figure import ModeFigure
from pylbo.visualisation.modes.vectorplot_handler import VectorplotHandler
from pylbo.visualisation.utils import add_axis_label
from tqdm import tqdm


class CartesianSlicePlot2D(ModeFigure):
    """
    Class for handling Cartesian 2D plots of the eigenmode solutions.

    Parameters
    ----------
    data : ModeVisualisationData
        The data for the visualisation.
    u2 : float or ndarray
        The :math:`y`  coordinate of the eigenmode solution.
    u3 : float or ndarray
        The :math:`z`  coordinate of the eigenmode solution.
    time : float
        The time at which the eigenmode solution is calculated.
    slicing_axis : str
        The axis along which the eigenmode solution is sliced.
    figsize : tuple[int, int]
        The size of the figure.
    show_ef_panel: bool
        Whether to show the eigenfunction panel.
    **kwargs
        Additional keyword arguments to be passed to :meth:`matplotlib.pyplot.imshow`.
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
        **kwargs,
    ) -> None:
        for i in "123":
            _axis = getattr(data.ds_bg, f"u{i}_str")
            setattr(self, f"_u{i}axis", _axis.replace("$", "").replace("\\", ""))
        self.slicing_axis = self._validate_slicing_axis(
            slicing_axis, allowed_axes=[self._u2axis, self._u3axis]
        )
        self.update_colorbar = True
        self._u1 = data.ds_bg.ef_grid
        self._u2 = self._validate_u2(u2, slicing_axis, axis=self._u2axis)
        self._u3 = self._validate_u3(u3, slicing_axis, axis=self._u3axis)
        self._time = self._check_if_number(time, "time")
        self._kwargs = kwargs

        self._use_contour_plot = False
        self._contour_levels = None
        self._contour_recipe = None
        self._view_dot = None
        self._has_streamlines = False
        self._has_quivers = False
        super().__init__(figsize, data, show_ef_panel)

        self.vmin = np.min(self._solutions)
        self.vmax = np.max(self._solutions)
        self.xmin = np.min(self.data.ds.ef_grid) if self.ax.name != "polar" else 0.0
        self.xmax = np.max(self.data.ds.ef_grid)
        self.coordmin = (
            np.min(self._u2) if self.slicing_axis == self._u3axis else np.min(self._u3)
        )
        self.coordmax = (
            np.max(self._u2) if self.slicing_axis == self._u3axis else np.max(self._u3)
        )

    def _validate_slicing_axis(self, slicing_axis: str, allowed_axes: list[str]) -> str:
        """
        Validates the slicing axis.

        Parameters
        ----------
        slicing_axis : str
            The slicing axis.
        allowed_axes : list[str]
            The list of allowed axes.

        Returns
        -------
        str
            The validated slicing axis.
        """
        if slicing_axis not in allowed_axes:
            raise ValueError(f"Slicing axis must be one of {allowed_axes}.")
        return slicing_axis

    def _validate_u2(self, u2: float, slicing_axis: str, axis: str) -> float:
        """
        Validates the combination of u2 and slicing axis.

        Parameters
        ----------
        u2 : float
            The :math:`u_2` coordinate.
        slicing_axis : str
            The slicing axis.
        axis : str
            The coordinate axis corresponding to :math:`u_2`.

        Returns
        -------
        float
            The validated :math:`u_2` coordinate.
        """
        if slicing_axis == axis and not isinstance(u2, (int, np.integer, float)):
            raise ValueError(f"u2 must be a number for slicing axis '{axis}'.")
        return u2

    def _validate_u3(self, u3: float, slicing_axis: str, axis: str) -> float:
        """
        Validates the combination of u3 and slicing axis.

        Parameters
        ----------
        u3 : float
            The :math:`u_3` coordinate.
        slicining_axis : str
            The slicing axis.
        axis : str
            The coordinate axis corresponding to :math:`u_3`.

        Returns
        -------
        float
            The validated :math:`u_3` coordinate.
        """
        if slicing_axis == axis and not isinstance(u3, (int, np.integer, float)):
            raise ValueError(f"u3 must be a number for slicing axis '{axis}'.")
        return u3

    def set_plot_arrays(self) -> None:
        axis = self.slicing_axis
        coord = self._u2 if axis == self._u3axis else self._u3
        self.solution_shape = (len(self._u1), len(coord))
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
        x_2d, coord_2d = np.meshgrid(self.data.ds_bg.ef_grid, coord, indexing="ij")

        self.u1_data = x_2d
        self.u2_data = coord_2d if axis == self._u3axis else self._u2
        self.u3_data = coord_2d if axis == self._u2axis else self._u3
        self.time_data = self._time

    def add_u2u3_txt(self, ax, **kwargs) -> None:
        if self.slicing_axis == self._u3axis:
            txt = rf"{self.data.ds_bg.u3_str} = {self._u3}"
        else:
            txt = rf"{self.data.ds_bg.u2_str} = {self._u2}"
        txt = rf"{txt} | t = {self._time:.2f}"
        self.u2u3_txt = add_axis_label(ax, txt, **kwargs)

    def set_contours(self, levels=None, fill=False) -> None:
        """
        Sets up a contour plot instead of an image plot.

        Parameters
        ----------
        levels : int or list[float]
            The number of levels or the list of levels.
        fill : bool
            Whether to fill the contour plot.
        """
        self._use_contour_plot = True
        self._contour_levels = levels
        self._contour_recipe = self.ax.contour if fill is False else self.ax.contourf

    def draw_solution(self) -> None:
        if self._use_contour_plot:
            self._draw_contours()
        else:
            self._draw_image()

    def _draw_image(self) -> None:
        vertical = self.u2_data if self.slicing_axis == self._u3axis else self.u3_data
        self._view = self.ax.pcolormesh(
            self.u1_data,
            vertical,
            self.solutions,
            **self._kwargs,
        )
        self.cbar = self.fig.colorbar(
            ScalarMappable(norm=self._view.norm, cmap=self._view.cmap), cax=self.cbar_ax
        )

    def _draw_contours(self) -> None:
        vertical = self.u2_data if self.slicing_axis == self._u3axis else self.u3_data
        additional_kwargs = {}
        if self._contour_levels is not None:
            additional_kwargs["levels"] = self._contour_levels
        self._view = self._contour_recipe(
            self.u1_data,
            vertical,
            self.solutions,
            vmin=self.vmin,
            vmax=self.vmax,
            **additional_kwargs,
            **self._kwargs,
        )
        self.cbar = self.fig.colorbar(
            ScalarMappable(norm=self._view.norm, cmap=self._view.cmap), cax=self.cbar_ax
        )

    def get_view_ylabel(self) -> str:
        return (
            self.data.ds_bg.u2_str
            if self.slicing_axis == self._u3axis
            else self.data.ds_bg.u3_str
        )

    def create_animation(
        self,
        times: np.ndarray,
        filename: str,
        fps: float = 10,
        dpi: int = 200,
        draw_dots: bool = False,
        ndots: int = 20,
    ) -> None:
        writer = animation.FFMpegWriter(fps=fps)
        pbar = tqdm(total=len(times), unit="frames", desc=f"Creating '{filename}'")
        self.data._print_bg_info = False
        self._ensure_first_frame_is_drawn()
        initial_solution = self._solutions
        with writer.saving(self.fig, filename, dpi=dpi):
            for t in times:
                solution = 0
                for efdata in self.ef_data:
                    solution += self.calculate_mode_solution(
                        efdata, self.u2_data, self.u3_data, t
                    )
                if self.data.add_background:
                    solution += self.data.get_background(shape=self._solutions.shape)
                self._time = t
                self._update_view(updated_solution=solution)
                if self.update_colorbar:
                    self._update_view_clims(solution)
                else:
                    self._update_view_clims(initial_solution)
                if draw_dots:
                    self._draw_comoving_dot(t, ndots)
                self._set_t_txt(t)
                writer.grab_frame()

                pbar.update()
        self._solutions = initial_solution
        self._time = self.time_data

    def _ensure_first_frame_is_drawn(self) -> None:
        if None in transform_to_list(self._view):
            self.draw()

    def _set_t_txt(self, t):
        if self.u2u3_txt is None:
            return
        txt = self.u2u3_txt.get_text().split("|")[0]
        self.u2u3_txt.set_text(f"{txt}| t = {t:.2f}")

    def _draw_comoving_dot(self, t, ndots):
        """
        Overplots the data in an animation with dots that
        are comoving with the flow.

        Parameters
        ----------
        t : float
            The current time.

        """
        dotcolor = "red"

        if self.slicing_axis == self._u3axis:
            bg = self.data.ds_bg.equilibria["v02"]
            edge_max = np.max(self.u2_data)
            edge_min = np.min(self.u2_data)
        else:
            bg = self.data.ds_bg.equilibria["v03"]
            edge_max = np.max(self.u3_data)
            edge_min = np.min(self.u3_data)
        if np.allclose(bg, 0.0):
            pylbo.logger.warning("Projected velocity is zero.")

        x0 = edge_min

        if not self.ax.name == "polar":
            lims = self.ax.get_xlim()
        else:
            lims = self.ax.get_ylim()
        ymin = max(self.data.ds_bg.x_start, min(lims))
        ymax = min(self.data.ds_bg.x_end, max(lims))
        scaling = 1.0
        yloc = np.linspace(ymin, ymax, ndots + 2)[1:-1]
        if self.data.ds.geometry == "cylindrical":
            ymin = max(self.data.ds_bg.x_start, min(self.ax.get_ylim()))
            ymax = min(self.data.ds_bg.x_end, max(self.ax.get_ylim()))
            yloc = np.linspace(ymin, ymax, ndots + 2)[1:-1]
            scaling = yloc

        xloc = x0 + t * np.interp(yloc, self.data.ds_bg.grid_gauss, bg) / scaling
        for i in range(len(xloc)):  # for periodic reappearance
            if xloc[i] > edge_max:
                xloc[i] = edge_min + xloc[i] - edge_max
            if xloc[i] < edge_min:
                xloc[i] = edge_max - (edge_min - xloc[i])
        if self._view_dot is not None:
            self._view_dot.remove()
        if self.ax.name == "polar":
            self._view_dot = self.ax.scatter(xloc, yloc, marker="o", c=dotcolor)
        else:
            self._view_dot = self.ax.scatter(yloc, xloc, marker="o", c=dotcolor)

    def _update_view(self, updated_solution: np.ndarray) -> None:
        """
        Updates the axes with the new solution. If a contour plot is used, the
        contour lines are cleared and redrawn. If an image plot is used, the image is
        updated.

        Parameters
        ----------
        updated_solution : np.ndarray
            The new solution.
        """
        if self._use_contour_plot:
            self._update_contour_plot(updated_solution)
        else:
            self._view.set_array(updated_solution.ravel())
        if self._has_streamlines:
            self._update_streamplot()
        if self._has_quivers:
            self._update_quiverplot()

    def _update_view_clims(self, solution: np.ndarray) -> None:
        self.vmin, self.vmax = np.min(solution), np.max(solution)
        self._view.set_clim(self.vmin, self.vmax)

    def _clear_contours(self):
        # sadly contour(f) does not support updating the data, so we have to
        # delete the old contours and create new ones every frame...
        for coll in self._view.collections:
            try:
                coll.remove()
            except ValueError:
                pass

    def _update_contour_plot(self, updated_solution: np.ndarray) -> None:
        self._clear_contours()
        self._solutions = updated_solution
        self.draw_solution()
        self.add_axes_labels()

    def add_streamlines(
        self, xgrid=None, coordgrid=None, field="v", add_background=True, **kwargs
    ) -> None:
        if is_custom_grid(self.data.ds_bg.ef_grid):
            raise ValueError("Streamlines are not supported for non-equidistant grids.")
        self._has_streamlines = True
        self._add_streamplot(
            xgrid=xgrid,
            coordgrid=coordgrid,
            field=field,
            add_background=add_background,
            **kwargs,
        )

    def add_quivers(
        self, xgrid=None, coordgrid=None, field="v", add_background=True, **kwargs
    ) -> None:
        self._has_quivers = True
        self._add_quiverplot(
            xgrid=xgrid,
            coordgrid=coordgrid,
            field=field,
            add_background=add_background,
            **kwargs,
        )

    def _add_streamplot(
        self, xgrid=None, coordgrid=None, field="v", add_background=True, **kwargs
    ) -> None:
        self.stream_handler = self._create_vectorplot(
            xgrid=xgrid,
            coordgrid=coordgrid,
            field=field,
            add_background=add_background,
            **kwargs,
        )
        self._draw_vectorplot()

    def _add_quiverplot(
        self, xgrid=None, coordgrid=None, field="v", add_background=True, **kwargs
    ) -> None:
        self.quiver_handler = self._create_vectorplot(
            xgrid=xgrid,
            coordgrid=coordgrid,
            field=field,
            add_background=add_background,
            **kwargs,
        )
        self._draw_vectorplot()

    def _create_vectorplot(
        self, xgrid=None, coordgrid=None, field="v", add_background=True, **kwargs
    ) -> VectorplotHandler:

        if xgrid is None:
            xgrid = self.data.ds_bg.ef_grid
        if coordgrid is None:
            coordgrid = self._u2 if self.slicing_axis == self._u3axis else self._u3
        xgrid = np.asarray(xgrid)
        coordgrid = np.asarray(coordgrid)

        streamline_data = ModeVisualisationData(
            self.data.ds,
            self.data.omega,
            None,
            self.data.use_real_part,
            self.data.complex_factor,
            self.data.add_background,
        )

        if field not in ["v", "B"]:
            raise ValueError("Field must be either 'v' or 'B'.")
        if field == "B" and not self.data.ds_bg.has_derived_efs:
            raise ValueError("No derived B eigenfunctions included.")
        vector_handler = VectorplotHandler(
            xgrid=xgrid,
            coordgrid=coordgrid,
            field=field,
            data=streamline_data,
            axes=self.ax,
            add_background=add_background,
            **kwargs,
        )
        vector_handler._set_slicing_axis(self.slicing_axis, self._u2axis, self._u3axis)
        vector_handler._set_streamplot_arrays(u2=self.u2_data, u3=self.u3_data)
        return vector_handler

    def _draw_vectorplot(self) -> None:
        self._ensure_first_frame_is_drawn()
        if self._has_streamlines:
            self.stream_handler._set_time(self._time)
            self.stream_handler._set_solutions()
            self.stream_handler._draw_streamlines()
        if self._has_quivers:
            self.quiver_handler._set_time(self._time)
            self.quiver_handler._set_solutions()
            self.quiver_handler._draw_quivers()

        xlims = (
            (self.xmin, self.xmax)
            if not self.ax.name == "polar"
            else (self.coordmin, self.coordmax)
        )
        ylims = (
            (self.coordmin, self.coordmax)
            if not self.ax.name == "polar"
            else (self.xmin, self.xmax)
        )
        self.ax.set_xlim(xlims)
        self.ax.set_ylim(ylims)

    def _update_streamplot(self) -> None:
        self.stream_handler._set_time(self._time)
        self.stream_handler._set_solutions()
        self.stream_handler._draw_streamlines()  # sadly, streamlines can't be updated

    def _update_quiverplot(self) -> None:
        self.quiver_handler._set_time(self._time)
        self.quiver_handler._set_solutions()
        self.quiver_handler._update_quivers()
