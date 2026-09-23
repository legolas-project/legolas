from __future__ import annotations

import numpy as np
from pylbo.utilities.eq_balance import get_equilibrium_balance
from pylbo.visualisation.continua import ContinuaHandler
from pylbo.visualisation.figure_window import InteractiveFigureWindow
from pylbo.visualisation.legend_handler import LegendHandler
from pylbo.visualisation.utils import background_name_to_latex


class EquilibriumProfile(InteractiveFigureWindow):
    """Subclass responsible for drawing the equilibrium profiles."""

    def __init__(self, data, figsize, interactive, **kwargs):
        fig, ax = super().create_default_figure(
            figlabel="equilibrium-fields", figsize=figsize
        )
        super().__init__(fig)
        self.data = data
        self.kwargs = kwargs
        self.ax = ax
        self.ax2 = None
        # check if we need an additional axis
        for name in self.data.eq_names:
            # check derivative names
            if name.startswith("d"):
                values = self.data.equilibria.get(name)
                if any(values != 0):
                    self.ax2 = super().add_subplot_axes(self.ax, "bottom")
                    break
        self.leg_handle = LegendHandler(interactive)
        self.draw()
        self._figure_drawn = True
        if interactive:
            super().make_legend_interactive(self.leg_handle)
        self.make_layout_tight()

    def draw(self):
        """Adds the equilibria to the figure. Also sets the legend handler items"""
        items = []
        for name in self.data.eq_names:
            if name.startswith("d"):
                axis = self.ax2
            else:
                axis = self.ax
            values = self.data.equilibria.get(name)
            if all(values == 0):
                continue
            (item,) = axis.plot(
                self.data.grid_gauss,
                values,
                label=background_name_to_latex(name),
                alpha=self.leg_handle.alpha_point,
            )
            axis.axhline(y=0, color="grey", linestyle="dotted", alpha=0.6)
            axis.axvline(
                x=self.data.x_start, color="grey", linestyle="dotted", alpha=0.6
            )
            self.leg_handle.add(item)
            items.append(item)
        # we add the plasma beta as well, only for MHD
        if not np.all(np.isclose(self.data.equilibria["B0"], 0)):
            plasma_beta = (
                2 * self.data.equilibria["rho0"] * self.data.equilibria["T0"]
            ) / self.data.equilibria["B0"] ** 2
            (item,) = self.ax.plot(
                self.data.grid_gauss,
                plasma_beta,
                label=r"plasma-$\beta$",
                alpha=self.leg_handle.alpha_point,
            )
            self.leg_handle.add(item)
            items.append(item)
        labels = [l_item.get_label() for l_item in items]
        self.leg_handle.legend = self.ax.legend(
            items,
            labels,
            bbox_to_anchor=(0.0, 1, 1, 0.102),
            loc="lower left",
            ncol=10,
            mode="expand",
        )
        # enable autoscaling when clicking
        self.leg_handle.autoscale = True


class ContinuumProfile(InteractiveFigureWindow):
    """Subclass responsible for drawing the continuum profiles."""

    def __init__(self, data, figsize, interactive, **kwargs):
        fig, ax = super().create_default_figure(figlabel="continua", figsize=figsize)
        super().__init__(fig)
        self.ax = ax
        self.data = data
        self.kwargs = kwargs
        self.handler = ContinuaHandler(interactive)
        self.draw()
        self._figure_drawn = True
        if interactive:
            self.make_legend_interactive(self.handler)
        self.make_layout_tight()

    def draw(self):
        """Adds the continua to the plot, also sets the legend handlers."""
        for color, name in zip(
            self.handler.continua_colors, self.handler.continua_names
        ):
            continuum = self.data.continua[name]
            if np.allclose(abs(continuum), 0, atol=1e-12):
                continue
            # non-adiabatic slow continua have real and imaginary parts
            if np.any(np.iscomplex(continuum)) and "slow" in name:
                (item,) = self.ax.plot(
                    self.data.grid_gauss,
                    self.data.continua[name].real,
                    color=color,
                    label="".join([name, r"$_{Re}$"]),
                )
                self.handler.add(item)
                (item,) = self.ax.plot(
                    self.data.grid_gauss,
                    self.data.continua[name].imag,
                    linestyle="dashed",
                    color=color,
                    label="".join([name, r"$_{Im}$"]),
                )
                self.handler.add(item)
            else:
                cont = self.data.continua[name]
                if name == "thermal":
                    cont = cont.imag
                (item,) = self.ax.plot(
                    self.data.grid_gauss, cont, color=color, label=name
                )
                self.handler.add(item)
        self.handler.legend = self.ax.legend()
        self.handler.autoscale = True
        self.ax.axhline(y=0, color="grey", linestyle="dotted", alpha=0.8)
        self.ax.axvline(
            x=self.data.x_start, color="grey", linestyle="dotted", alpha=0.8
        )
        self.ax.set_xlabel("Grid coordinate")
        self.ax.set_ylabel(r"$\omega$")


class EquilibriumBalance(InteractiveFigureWindow):
    """Subclass responsible for plotting the equilibrium balance equations."""

    def __init__(self, data, figsize, interactive, **kwargs):
        fig, ax = super().create_default_figure(
            figlabel="equilibrium-balance", figsize=figsize
        )
        super().__init__(fig)
        self.data = data
        self.kwargs = kwargs
        self.ax = ax
        self.eq_balance = get_equilibrium_balance(ds=data)
        self.legend_handler = LegendHandler(interactive)
        self.draw()
        self._figure_drawn = True
        if interactive:
            self.make_legend_interactive(self.legend_handler)
        self.make_layout_tight()

    def draw(self):
        """Draws the equilibrium balance equations."""
        for key, values in self.eq_balance.items():
            (item,) = self.ax.plot(self.data.grid_gauss, values, label=key)
            self.legend_handler.add(item)
        self.ax.axhline(y=0, color="grey", linestyle="dotted", alpha=0.8)
        self.legend_handler.legend = self.ax.legend(
            bbox_to_anchor=(0.0, 1, 1, 0.102), loc="lower right", ncol=7
        )
        self.legend_handler.autoscale = True
