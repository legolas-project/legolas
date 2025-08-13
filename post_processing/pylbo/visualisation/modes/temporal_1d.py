from __future__ import annotations

import numpy as np
from matplotlib.cm import ScalarMappable
from pylbo.visualisation.modes.mode_data import ModeVisualisationData
from pylbo.visualisation.modes.mode_figure import ModeFigure


class TemporalEvolutionPlot1D(ModeFigure):
    """
    Main class for temporal evolutions of the eigenfunction.

    Parameters
    ----------
    data : ModeVisualisationData
        Data object containing all data associated with the selected eigenmode.
    u2 : float
        The data for the :math:`u_2` coordinate.
    u3 : float
        The data for the :math:`u_3` coordinate.
    time : np.ndarray
        The data for the time.
    figsize : tuple[int, int]
        The size of the figure.
    show_ef_panel : bool
        Whether to show the eigenfunction panel.
    **kwargs
        Additional keyword arguments to be passed to :meth:`matplotlib.pyplot.imshow`.
    """

    def __init__(
        self,
        data: ModeVisualisationData,
        u2: float,
        u3: float,
        time: np.ndarray,
        figsize: tuple[int, int],
        show_ef_panel: bool,
        **kwargs,
    ) -> None:
        self._u1 = data.ds_bg.ef_grid
        self._u2 = self._check_if_number(u2, "u2")
        self._u3 = self._check_if_number(u3, "u3")
        self._time = self._check_if_array(time, "time")
        self._kwargs = kwargs
        super().__init__(figsize, data, show_ef_panel)

    def set_plot_arrays(self) -> None:
        self.solution_shape = (len(self._u1), len(self._time))
        for efs, omegas, factors, k2, k3 in zip(
            self.data.eigenfunction,
            self.data.omega,
            self.data.complex_factor,
            self.data.k2,
            self.data.k3,
        ):
            for ef, omega, factor in zip(efs, omegas, factors):
                # transpose here so data[:, i] gives eigenfunction data at time i
                data = np.broadcast_to(
                    ef, shape=reversed(self.solution_shape)
                ).transpose()
                self.ef_data.append(
                    {"ef": data, "omega": omega, "factor": factor, "k2": k2, "k3": k3}
                )
        x_2d, time_2d = np.meshgrid(self._u1, self._time, indexing="ij")
        self.time_data = time_2d
        self.u1_data = x_2d
        self.u2_data = self._u2
        self.u3_data = self._u3

    def draw_solution(self) -> None:
        self._view = self.ax.pcolormesh(
            self.u1_data,
            self.time_data,
            self.solutions,
            **self._kwargs,
        )
        self.cbar = self.fig.colorbar(
            ScalarMappable(norm=self._view.norm, cmap=self._view.cmap),
            cax=self.cbar_ax,
        )

    def get_view_ylabel(self) -> str:
        return "time"
