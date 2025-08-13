import numpy as np
from matplotlib.axes import Axes as mpl_axes
from matplotlib.streamplot import StreamplotSet
from matplotlib.quiver import Quiver
from matplotlib.patches import FancyArrowPatch
from pylbo.visualisation.modes.mode_data import ModeVisualisationData


class VectorplotHandler:
    """
    Main handler for vector-based fields (Streamlines and Quiver).
    """

    def __init__(
        self,
        xgrid: np.ndarray,
        coordgrid: np.ndarray,
        field: str,
        data: ModeVisualisationData,
        axes: mpl_axes,
        add_background: bool,
        **kwargs
    ):

        self.xgrid = xgrid
        self.coordgrid = coordgrid
        self.data = data

        self.field = field
        self.ax = axes
        self.polar = self.ax.name == "polar"
        self.add_background = add_background

        self.coord_dict = {"theta": "3", "z": "2", "y": "3"}
        self.streamlines = None
        self.quivers = None
        self._kwargs = kwargs

    def _draw_streamlines(self) -> StreamplotSet:

        if "density" not in self._kwargs.keys():
            self._kwargs["density"] = 2.0
        if "color" not in self._kwargs.keys():
            self._kwargs["color"] = "w"
        if "broken_streamlines" not in self._kwargs.keys():
            self._kwargs["broken_streamlines"] = True

        if self.polar:
            self.xvec = self._solutions[1]
            self.yvec = self._solutions[0]

        self._clear_streamlines()
        self.streamlines = self.ax.streamplot(
            self.xdata, self.ydata, self.xvec, self.yvec, zorder=2, **self._kwargs
        )
        self.streamlines.lines.set_alpha(0.7)
        self.streamlines.arrows.set_alpha(0.7)

    def _draw_quivers(self) -> Quiver:

        if "color" not in self._kwargs.keys():
            self._kwargs["color"] = "w"
        if "pivot" not in self._kwargs.keys():
            self._kwargs["pivot"] = "mid"
        if "units" not in self._kwargs.keys():
            self._kwargs["units"] = "height"
        if "scale" not in self._kwargs.keys():
            self._kwargs["scale"] = 20
        if "alpha" not in self._kwargs.keys():
            self._kwargs["alpha"] = 0.7
        if "width" not in self._kwargs.keys():
            self._kwargs["width"] = 0.002

        if self.polar:
            self._transform_vectors()
        self.quivers = self.ax.quiver(
            self.xdata, self.ydata, self.xvec, self.yvec, zorder=2, **self._kwargs
        )

    def _update_quivers(self) -> None:
        if self.polar:
            self._transform_vectors()
        self.quivers.set_UVC(self.xvec, self.yvec)

    def _clear_quivers(self) -> None:
        try:
            self.quivers.remove()
        except AttributeError:
            pass

    def _clear_streamlines(self) -> None:
        try:
            self.streamlines.lines.remove()
            for art in self.ax.get_children():
                if not isinstance(art, FancyArrowPatch):
                    continue
                art.remove()  # removes the arrows
            self.streamlines.arrows.set_alpha(0)
        except AttributeError:
            pass

    def _set_slicing_axis(self, slicing_axis, u2axis, u3axis) -> None:
        self.slicing_axis = slicing_axis
        self._u2axis = u2axis
        self._u3axis = u3axis

    def _set_time(self, time) -> None:
        self.time_data = time

    def _set_streamplot_arrays(self, u2, u3) -> None:
        axis = self.slicing_axis
        self.solution_shape = (len(self.xgrid), len(self.coordgrid))
        x_2d, coord_2d = np.meshgrid(self.xgrid, self.coordgrid, indexing="ij")

        self.coord_data = coord_2d
        self.u1_data = x_2d
        self.u2_data = coord_2d if axis == self._u3axis else u2
        self.u3_data = coord_2d if axis == self._u2axis else u3

        if self.polar:
            self.xdata = self.coord_data
            self.ydata = self.u1_data
        else:
            self.xdata = self.u1_data.transpose()
            self.ydata = self.coord_data.transpose()

    def _set_solutions(self) -> np.ndarray:
        """
        Returns the eigenmode solution for a given time.

        Parameters
        ----------
        u2_data : Union[float, np.ndarray]
            The u2 data from the Plot2d.
        u3_data : Union[float, np.ndarray]
            The u3 data from the Plot2d.

        Returns
        -------
        np.ndarray
            The eigenmode solution.
        """
        # name = validate_ef_name(self.data.ds_bg, name)
        fields = self._get_field_names()
        solutions = [np.zeros_like(self.u1_data), np.zeros_like(self.u1_data)]

        for all_efs, factors, k2, k3 in zip(
            self.data._all_efs, self.data.complex_factor, self.data.k2, self.data.k3
        ):
            for ef_highres, factor in zip(all_efs, factors):
                for i in range(2):
                    ef = ef_highres.get(fields[i])
                    ef = np.interp(self.xgrid, self.data.ds_bg.ef_grid, ef)
                    ef = np.broadcast_to(
                        ef, shape=reversed(self.solution_shape)
                    ).transpose()
                    solutions[i] += self.data.get_mode_solution(
                        ef=ef,
                        omega=ef_highres.get("eigenvalue"),
                        complex_factor=factor,
                        u2=self.u2_data,
                        u3=self.u3_data,
                        t=self.time_data,
                        k2=k2,
                        k3=k3,
                    )

        if self.add_background:
            bgs = self._get_bg_names()

            print_bg_temp = self.data._print_bg_info
            self.data._print_bg_info = False

            for i in range(2):
                bg_temp = self.data.get_background(
                    name=bgs[i], shape=self.data.ds_bg.ef_grid.shape
                )
                bg = np.interp(self.xgrid, self.data.ds_bg.ef_grid, bg_temp)
                bg = np.broadcast_to(
                    bg, shape=reversed(self.solution_shape)
                ).transpose()
                solutions[i] += bg

            self.data._print_bg_info = print_bg_temp

        self._solutions = solutions

        norm = np.max(np.sqrt(self._solutions[0] ** 2 + self._solutions[1] ** 2))
        self.xvec = (self._solutions[0] / norm).transpose()
        self.yvec = (self._solutions[1] / norm).transpose()

    def _transform_vectors(self) -> None:
        xvec_val = self.xvec.transpose()
        yvec_val = self.yvec.transpose()
        self.xvec = xvec_val * np.cos(self.coord_data) - yvec_val * np.sin(
            self.coord_data
        )
        self.yvec = xvec_val * np.sin(self.coord_data) + yvec_val * np.cos(
            self.coord_data
        )

    def _get_field_names(self) -> list:
        field1 = self.field + "1"
        field2 = self.field + self.coord_dict[self.slicing_axis]
        return [field1, field2]

    def _get_bg_names(self) -> list:
        bg1 = self.field + "01"
        bg2 = self.field + "0" + self.coord_dict[self.slicing_axis]
        return [bg1, bg2]
