import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _fmt(x, pos):
    a, b = "{:.2e}".format(x).split("e")
    b = int(b)
    return r"${} \times 10^{{{}}}$".format(a, b)


class IVPSolution:
    """
    Container for IVP snapshot data returned by Legolas.

    Parameters
    ----------
    times : np.ndarray
        Physical times at each snapshot, shape ``(n_snap,)``.
    data : np.ndarray
        Snapshot data, shape ``(n_snap, n_comp, n_points)``.
    component_names : dict, optional
        Map from integer index to component name, e.g. ``{0: "rho", 1: "v1"}``.
    x_domain : np.ndarray, optional
        Spatial coordinates, shape ``(n_points,)``.
    units : dict, optional
        Unit normalisations from the Legolas datfile header.
    """

    def __init__(self, times, data, component_names=None, x_domain=None, units=None):
        self.times = times
        self.data = data
        self.component_names = component_names or {}
        self.x_domain = x_domain
        self.units = units

    def _scale_component_array(self, component, comp_array):
        """Returns ``comp_array`` scaled to physical (cgs) units."""
        if self.units is None:
            return comp_array, ""
        comp_name = str(component).lower()
        if "rho" in comp_name:
            factor = self.units["unit_density"]
            label = r"[g cm$^{-3}$]"
        elif "v" in comp_name:
            factor = self.units["unit_velocity"]
            label = r"[cm s$^{-1}$]"
        elif "temp" in comp_name or "t" in comp_name:
            factor = self.units["unit_temperature"]
            label = r"[K]"
        elif "p" in comp_name:
            factor = self.units["unit_pressure"]
            label = r"[dyn cm$^{-2}$]"
        else:
            factor = 1.0
            label = ""
        return comp_array * factor, label

    def _scale_time_array(self, times):
        if self.units is None:
            return times
        return times * self.units["unit_time"]

    def _scale_x_domain(self, x_vals):
        if self.units is None:
            return x_vals
        # dimensionless * unit_length [cm] * 1e-8 => Mm
        return x_vals * self.units["unit_length"] * 1e-8

    def _get_component_index(self, component):
        if isinstance(component, int):
            return component
        elif isinstance(component, str):
            for idx, nm in self.component_names.items():
                if nm == component:
                    return idx
            raise ValueError(
                f"No component named '{component}' found in {self.component_names}"
            )
        else:
            raise TypeError("component must be an int or str")

    def get_component(self, component):
        """
        Returns the ``(n_snap, n_points)`` array for the requested component.

        Parameters
        ----------
        component : int or str
            Component index or name (e.g. ``"rho"``).
        """
        return self.data[:, self._get_component_index(component), :]

    def plot_space_time_heatmap(
        self, component, ax=None, cmap="plasma", time_range=None, **imshow_kwargs
    ):
        """
        Plot a space-time heatmap for the given component.

        Parameters
        ----------
        component : int or str
            Which component to plot.
        ax : ~matplotlib.axes.Axes, optional
            Axes to plot on; created if not provided.
        cmap : str
            Colormap passed to ``imshow``.
        time_range : tuple, optional
            ``(t_min, t_max)`` to restrict the plotted time window.
        """
        comp_array = self.get_component(component)
        times_phys = self._scale_time_array(np.array(self.times))

        if time_range is not None:
            t_min, t_max = time_range
            mask = (times_phys >= t_min) & (times_phys <= t_max)
            comp_array = comp_array[mask, :]
            times_phys = times_phys[mask]
        t_min, t_max = times_phys[0], times_phys[-1]

        comp_array_phys, label = self._scale_component_array(component, comp_array)

        if ax is None:
            fig, ax = plt.subplots()

        n_points = comp_array_phys.shape[1]
        if self.x_domain is not None and len(self.x_domain) == n_points:
            x_vals_phys = self._scale_x_domain(self.x_domain)
            x_min, x_max = x_vals_phys[0], x_vals_phys[-1]
        else:
            x_min, x_max = 0, n_points - 1
            x_vals_phys = None

        im = ax.imshow(
            comp_array_phys,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[x_min, x_max, t_min, t_max],
            **imshow_kwargs,
        )
        cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(_fmt))
        cbar.set_label(f"{component} {label}")
        ax.set_xlabel("x [Mm]" if x_vals_phys is not None else "x index")
        ax.set_ylabel("Time [s]")
        ax.set_title(f"{component} space-time")
        return ax

    def plot_spatial_slices(self, component, snap_indices, ax=None, **plot_kwargs):
        """
        Plot the spatial profile of a component at selected snapshot indices.

        Parameters
        ----------
        component : int or str
            Which component to plot.
        snap_indices : list of int
            Snapshot indices to plot, in ``[0, n_snap-1]``.
        ax : ~matplotlib.axes.Axes, optional
            Axes to plot on; created if not provided.
        """
        comp_array = self.get_component(component)
        comp_array_phys, label = self._scale_component_array(component, comp_array)
        times_phys = self._scale_time_array(np.array(self.times))
        n_snap, n_points = comp_array_phys.shape

        if self.x_domain is not None and len(self.x_domain) == n_points:
            x_vals_phys = self._scale_x_domain(self.x_domain)
            xlabel = "x [Mm]"
        else:
            x_vals_phys = np.arange(n_points)
            xlabel = "x index"

        if ax is None:
            fig, ax = plt.subplots()

        for snap_idx in snap_indices:
            if snap_idx < 0 or snap_idx >= n_snap:
                raise IndexError(
                    f"Snapshot index {snap_idx} out of range (0..{n_snap-1})."
                )
            ax.plot(
                x_vals_phys,
                comp_array_phys[snap_idx, :],
                label=f"t={times_phys[snap_idx]:.3f} s",
                **plot_kwargs,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{component} {label}")
        ax.set_title(f"{component} spatial profiles")
        ax.legend()
        return ax

    def plot_growth_vs_time(
        self,
        component,
        mode="centre",
        centre_idx=None,
        region=None,
        logy=True,
        tau=None,
        ax=None,
        data_kw=None,
        fit_kw=None,
    ):
        """
        Plot perturbation amplitude vs. time, with an optional analytic
        exp(t/tau) overlay for comparison against eigenvalue growth rates.

        Parameters
        ----------
        component : str or int
            Which component to track.
        mode : {"centre", "max", "integral"}
            How to reduce the spatial data to a scalar amplitude.
        centre_idx : int, optional
            Spatial index for ``mode="centre"``; defaults to the midpoint.
        region : tuple (i_min, i_max), optional
            Spatial slice for ``mode="max"`` or ``"integral"``.
        logy : bool
            Use a semilog-y axis.
        tau : float, optional
            Analytic e-folding time [s]. Positive => growth, negative => decay.
        ax : ~matplotlib.axes.Axes, optional
            Axes to plot on; created if not provided.
        data_kw, fit_kw : dict, optional
            Extra kwargs forwarded to the data and fit ``plot`` calls.
        """
        data_kw = {} if data_kw is None else data_kw
        fit_kw = {} if fit_kw is None else fit_kw

        comp_arr = self.get_component(component)
        times_phys = self._scale_time_array(np.asarray(self.times))

        sl = slice(*region) if region is not None else slice(None)

        if mode == "centre":
            if centre_idx is None:
                centre_idx = comp_arr.shape[1] // 2
            amp = comp_arr[:, centre_idx]
        elif mode == "max":
            amp = np.max(np.abs(comp_arr[:, sl]), axis=1)
        elif mode == "integral":
            amp = np.sqrt(np.sum(comp_arr[:, sl] ** 2, axis=1))
        else:
            raise ValueError("mode must be 'centre', 'max' or 'integral'.")

        amp_phys, label = self._scale_component_array(component, amp)

        if ax is None:
            fig, ax = plt.subplots()

        plot_fn = ax.semilogy if logy else ax.plot
        plot_fn(times_phys, amp_phys, label=f"{component}", **data_kw)

        if tau is not None and tau != 0.0:
            fit = amp_phys[0] * np.exp((times_phys - times_phys[0]) / tau)
            plot_fn(
                times_phys,
                fit,
                linestyle="--",
                label=f"exp($t/{tau:.0f}$ s)",
                **fit_kw,
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Amplitude {label}")
        ax.set_title(f"{component} amplitude vs. time")

        if mode == "centre":
            ax.annotate(
                f"centre idx = {centre_idx}",
                xy=(0.02, 0.95),
                xycoords="axes fraction",
                fontsize=8,
                va="top",
            )
        elif region is not None:
            ax.annotate(
                f"region = [{sl.start}:{sl.stop}]",
                xy=(0.02, 0.95),
                xycoords="axes fraction",
                fontsize=8,
                va="top",
            )

        ax.legend()
        return ax
