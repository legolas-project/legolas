import numpy as np
from matplotlib.axes import Axes as mpl_axes
from pylbo.data_containers import LegolasDataContainer, LegolasDataSet
from pylbo.visualisation.eigenfunctions.eigfunc_interface import EigenfunctionInterface
from pylbo.visualisation.utils import ef_name_to_latex


class EigenfunctionHandler(EigenfunctionInterface):
    """
    Main handler for eigenfunctions.
    """

    def __init__(self, data: LegolasDataContainer, ef_ax: mpl_axes, spec_ax: mpl_axes):
        super().__init__(data, ef_ax, spec_ax)
        self.spec_axis.set_title(f"{self.spec_axis.get_title()} -- eigenfunctions")

    def update_plot(self):
        self.axis.clear()
        if not self._selected_idxs:
            self._display_tooltip()
            return
        ef_name = self._function_names[self._selected_name_idx]
        for ds, idxs_dict in self._selected_idxs.items():
            idxs = np.array([idx for idx in idxs_dict.keys()], dtype=int)
            ef_container = ds.get_eigenfunctions(ev_idxs=idxs)
            for ev_idx, efs in zip(idxs, ef_container):
                ef = getattr(
                    efs.get(ef_name), "real" if self._use_real_part else "imag"
                )
                if self._ef_needs_retransform(ef_name):
                    ef = ef * ds.ef_grid
                label = super()._get_label(ds, ev_idx, efs.get("eigenvalue"))
                color = self._selected_idxs.get(ds).get(ev_idx).get_color()
                self.axis.plot(ds.ef_grid, ef, color=color, label=label)
                if self._draw_resonances:
                    self._show_resonances(ds, ev_idx, color)
        self.axis.axhline(y=0, linestyle="dotted", color="grey")
        if isinstance(self.data, LegolasDataSet):
            self.axis.axvline(x=self.data.x_start, linestyle="dotted", color="grey")
        self.axis.set_title(self._get_title(ef_name))
        self.axis.legend(loc="best", fontsize=8)

    def _ef_needs_retransform(self, ef_name: str) -> bool:
        return (
            self._retransform
            and ef_name in ("rho", "v1", "v3", "T", "a2")
            and self.data.geometry == "cylindrical"
        )

    def _get_title(self, ef_name):
        """
        Creates the title of the eigenfunction plot.
        If the eigenfunction is retransformed an 'r' is prepended if appropriate,
        along with Re/Im depending on the real/imaginary part shown.

        Parameters
        ----------
        ef_name : str
            The name of the eigenfunction as present in the datfile.

        Returns
        -------
        name : str
            The 'new' name for the eigenfunction, used as title.
        """
        if self._ef_needs_retransform(ef_name):
            ef_name = f"r{ef_name}"
        ef_name = ef_name_to_latex(
            ef_name, geometry=self.data.geometry, real_part=self._use_real_part
        )
        return rf"{ef_name} eigenfunction"
