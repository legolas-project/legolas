from __future__ import annotations

import abc
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from pylbo.data_containers import (
    LegolasDataContainer,
    LegolasDataSeries,
    LegolasDataSet,
)
from pylbo.exceptions import EigenfunctionsNotPresent
from pylbo.utilities.logger import pylboLogger
from pylbo.utilities.toolbox import (
    add_pickradius_to_item,
    count_zeroes,
    find_resonance_location,
    get_all_eigenfunction_names,
    transform_to_numpy,
)


def get_artist_data(artist: plt.Artist) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the (x, y) coordinates of a given artist.

    Parameters
    ----------
    artist : ~matplotlib.artist.Artist
        The artist to get the data from.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The (x, y) coordinates of the artist.
    """
    if isinstance(artist, PathCollection):
        # this block is entered for points drawn using scatter instead of plot
        xdata, ydata = np.split(artist.get_offsets(), [-1], axis=1)
        xdata = xdata.squeeze(axis=1)
        ydata = ydata.squeeze(axis=1)
    else:
        xdata = artist.get_xdata()
        ydata = artist.get_ydata()
    return xdata, ydata


class EigenfunctionInterface:
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, axis, spec_axis):
        self.data = data
        self.axis = axis
        self.spec_axis = spec_axis
        self._check_data_is_present()
        # holds the points that are currently selected in the form of a "double" dict:
        # {ds instance : {index : line2D instance}}
        self._selected_idxs = {}
        self._use_real_part = True
        self._selected_name_idx = 0
        self._function_names = get_all_eigenfunction_names(self.data)
        self._retransform = False

        self._condition_to_make_transparent = None
        self._transparent_data = False
        self._unmarked_alpha = None

        self._ef_subset_artists = None
        self._display_tooltip()

        self._draw_resonances = False
        self.savedir = None

    def _check_data_is_present(self):
        """
        Checks if the required data is present to draw for example
        eigenfunctions, is overloaded in subclasses.
        """
        if not any(transform_to_numpy(self.data.has_efs)):
            raise EigenfunctionsNotPresent(
                "None of the given datfiles has eigenfunctions written to it."
            )

    def _artist_has_valid_attributes(self, event):
        """
        Checks if a given event has valid attributes, this prevents triggering the
        interface when clicking on legend items, for example.

        Parameters
        ----------
        event : ~matplotlib.backend_bases.PickEvent
            The pick event.

        Returns
        -------
        bool
            `True` if all conditions are met and callbacks can be connected, `False`
            otherwise.
        """
        if not event.mouseevent.inaxes == self.spec_axis:
            return False
        artist = event.artist
        # this attr has been set when adding the legend, return for legend items
        if hasattr(artist, "is_legend_item"):
            return False
        if not hasattr(artist, "dataset"):
            return False
        return True

    def _clear_figure_and_selection(self):
        """
        Clears the current figure, clears the dictionary of selected eigenvalues.
        """
        for ds_artist_dict in self._selected_idxs.values():
            for artist in ds_artist_dict.values():
                artist.remove()
        self._selected_idxs.clear()
        self.axis.clear()
        self._display_tooltip()

    def _switch_real_and_imaginary_part(self):
        """
        Switches between the real and imaginary part of a given function.
        """
        self._use_real_part = not self._use_real_part

    def _select_next_function(self):
        """
        Increments the index of the currently selected function by 1.
        """
        self._selected_name_idx += 1
        if self._selected_name_idx > len(self._function_names) - 1:
            self._selected_name_idx = 0

    def _select_previous_function(self):
        """
        Decrements the index of the currently selected function by 1.
        """
        self._selected_name_idx -= 1
        if self._selected_name_idx < 0:
            self._selected_name_idx = len(self._function_names) - 1

    def _retransform_functions(self):
        """
        Toggles a retransform of a function, for example an eigenfunction
        :math:`v_r \\leftrightarrow rv_r`"
        """
        self._retransform = not self._retransform

    def _print_selected_eigenvalues(self):
        """
        Prints all selected eigenvalues to the console as an array.
        """
        if not self._selected_idxs:
            return
        pylboLogger.info("Currently selected eigenvalues:")
        for ds, points in self._selected_idxs.items():
            idxs = np.array([idx for idx in points.keys()], dtype=int)
            pylboLogger.info(f"{ds.datfile.stem} | {ds.eigenvalues[idxs]}")

    def _save_eigenvalue_selection(self):
        """
        Saves all selected eigenvalues and their eigenfunctions as a list of
        dictionaries in a .npy file. Files can be loaded with the numpy load function.
        """
        if not self._selected_idxs:
            return
        count = 1
        for ds in self._selected_idxs:
            pylboLogger.info(f"Saving selected eigenvalues for dataset {count}...")
            to_store = [ds.ef_grid]
            for point in self._selected_idxs[ds]:
                point_to_store = ds.get_eigenfunctions(ev_idxs=int(point))[0]
                to_store.append(point_to_store)
            to_store = np.asarray(to_store, dtype=object)
            filename = ds.datfile.name.replace(".dat", "")
            if self.savedir is not None:
                filename = Path(self.savedir) / filename
            np.save(filename, to_store)
            pylboLogger.info(
                f"{len(to_store)-1} mode(s) saved to " + str(filename) + ".npy"
            )
            count += 1

    def _save_selection_indices(self):
        """
        Saves the indices of all selected eigenvalues as an array in a .npy file.
        Files can be loaded with the numpy load function.
        """
        if not self._selected_idxs:
            return
        count = 1
        for ds in self._selected_idxs:
            pylboLogger.info(
                f"Saving indices of selected eigenvalues for dataset {count}..."
            )
            to_store = []
            for point in self._selected_idxs[ds]:
                to_store.append(int(point))
            to_store = np.asarray(to_store, dtype=int)
            filename = ds.datfile.name.replace(".dat", "")
            if self.savedir is not None:
                filename = Path(self.savedir) / filename
            np.save(filename, to_store)
            pylboLogger.info(
                f"{len(self._selected_idxs[ds])} indices saved to "
                + str(filename)
                + ".npy"
            )
            count += 1

    def _print_nzeroes(self):
        """
        Counts and prints the number of zeroes of the eigenfunctions for all selected
        eigenvalues on the plot, together with eigvals.
        """
        if not self._selected_idxs:
            return
        pylboLogger.info(
            "Currently selected eigenvalues and number of zeroes "
            "of their eigenfunctions:"
        )
        ef_name = self._function_names[self._selected_name_idx]
        for ds, points in self._selected_idxs.items():
            idxs = np.array([int(idx) for idx in points.keys()])

            ef_container = ds.get_eigenfunctions(ev_idxs=idxs)
            eigfuncs = np.zeros((len(idxs), len(ds.ef_grid)), dtype="complex")
            current_index = 0
            for ev_idx, efs in zip(idxs, ef_container):
                eigfuncs[current_index] = efs.get(ef_name)
                current_index += 1

            nzeroes = count_zeroes(eigfuncs)
            pylboLogger.info(
                f"{ds.datfile.stem} | {dict(zip(ds.eigenvalues[idxs], nzeroes))}"
            )

    def _get_label(self, ds, ev_idx, w):
        """
        Returns the label used in the legend. In case of a data series, the datfile
        name is prepended.

        Parameters
        ----------
        ds : ~pylbo.data_containers.LegolasDataSet
            The current dataset
        ev_idx : int
            The index of the current eigenvalue in the corresponding array
        w : float, complex
            The eigenvalue to use in the label

        Returns
        -------
        str
            The label to use in the legend.
        """
        label = rf"$\omega_{{{ev_idx}}}$ = {w:2.3e}"
        if isinstance(self.data, LegolasDataSeries):
            label = f"{ds.datfile.stem} | {label}"
        return label

    @abc.abstractmethod
    def _get_title(self):
        """
        Creates the title of a given plot, has to be overridden in a subclass.
        """
        pass

    def get_selected_idxs(self) -> dict[LegolasDataContainer, dict[int, plt.Artist]]:
        return self._selected_idxs

    def get_name_of_drawn_eigenfunction(self) -> str:
        return self._function_names[self._selected_name_idx]

    @abc.abstractmethod
    def update_plot(self):
        """
        Updates the plot when an event is triggered, clears and then redraws
        the functions. Rescaling of the axes is done automatically.
        Has to be overridden in a subclass.
        """
        pass

    def on_point_pick(self, event):
        """
        Determines what happens when an eigenvalue is clicked.

        Parameters
        ----------
        event : ~matplotlib.backend_bases.PickEvent
            The pick event.
        """
        if not self._artist_has_valid_attributes(event):
            return
        # retrieve limits to prevent resetting zoom
        ax = event.artist.axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if event.mouseevent.button == 1:
            self.on_left_click(event)
        elif event.mouseevent.button == 3:
            self.on_right_click(event)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        event.artist.figure.canvas.draw()

    def on_key_press(self, event):
        """
        Determines what happens when a key is pressed.

        Parameters
        ----------
        event : ~matplotlib.backend_bases.KeyEvent
            The key event.
        """
        if event.key == "m":
            self._mark_points_without_data_written()
        elif event.key == "d":
            self._clear_figure_and_selection()
        elif event.key == "i":
            self._switch_real_and_imaginary_part()
        elif event.key == "e":
            self._toggle_eigenfunction_subset_radius()
        elif event.key == "up":
            self._select_next_function()
        elif event.key == "down":
            self._select_previous_function()
        elif event.key == "t":
            self._retransform_functions()
        elif event.key == "w":
            self._print_selected_eigenvalues()
        elif event.key == "n":
            self._print_nzeroes()
        elif event.key == "a":
            self._save_eigenvalue_selection()
        elif event.key == "j":
            self._save_selection_indices()
        if event.key in ("enter", "up", "down", "i", "t"):
            self.update_plot()
        event.canvas.draw()

    def on_left_click(self, event):
        """
        Determines what happens when left-clicking an eigenvalue.

        Parameters
        ----------
        event : ~matplotlib.backend_bases.PickEvent
            The pick event.
        """
        idx, xdata, ydata = self._get_clicked_point_data(event)
        associated_ds = event.artist.dataset
        items = self._selected_idxs.get(associated_ds, {})
        # skip if point index is already in list
        if idx in items.keys():
            return
        # skip if point has no eigenfunction due to e.g. subset
        if not self._selected_point_has_eigenfunctions(associated_ds, idx):
            return
        (marked_point,) = event.artist.axes.plot(
            xdata,
            ydata,
            "x",
            markersize=8,
            label="marked_point",
            alpha=0.7,
            markeredgewidth=3,
        )
        add_pickradius_to_item(item=marked_point, pickradius=1)
        items.update({idx: marked_point})
        self._selected_idxs.update({associated_ds: items})
        self.update_plot()

    def on_right_click(self, event):
        """
        Determines what happens when right-clicking an eigenvalue.

        Parameters
        ----------
        event : ~matplotlib.backend_bases.PickEvent
            The pick event.
        """
        idx, _, _ = self._get_clicked_point_data(event)
        # remove selected index from list
        associated_ds = event.artist.dataset
        selected_artist = self._selected_idxs.get(associated_ds, {}).pop(idx, None)
        if selected_artist is not None:
            selected_artist.remove()
            # if no items remaining for this ds, remove key
            if len(self._selected_idxs[associated_ds]) == 0:
                self._selected_idxs.pop(associated_ds)
            self.update_plot()

    def _get_clicked_point_data(self, event):
        """
        Retrieves the index (in the eigenvalue array), x data coordinate and y data
        coordinate of the eigenvalue nearest to the clicked point.

        Parameters
        ----------
        event : ~matplotlib.backend_bases.PickEvent
            The pick event.

        Returns
        -------
        idx : integer
            The index of the selected point in the eigenvalue array
        xdata : float
            The x data coordinate of the selected eigenvalue
        ydata : float
            The y data coordinate of the selected eigenvalue
        """
        artist = event.artist
        idxs = event.ind
        xdata, ydata = get_artist_data(artist)

        # In case of overlap between clicked points then multiple
        # indices were selected, so we check smallest distance to mouse click
        if len(idxs) == 1:
            idx = idxs[0]
        else:
            mouse_x = event.mouseevent.xdata
            mouse_y = event.mouseevent.ydata
            distances = (mouse_x - xdata[idxs]) ** 2 + (mouse_y - ydata[idxs]) ** 2
            idx = idxs[distances.argmin()]

        # turn the artist index into the real dataset index
        associated_ds = event.artist.dataset
        sigma_clicked = xdata[idx] + 1j * ydata[idx]
        ev_idx = np.nanargmin(np.abs(associated_ds.eigenvalues - sigma_clicked))
        return ev_idx, np.real(sigma_clicked), np.imag(sigma_clicked)

    def _selected_point_has_eigenfunctions(self, ds, idx):
        """
        Checks if the selected index has eigenfunctions associated with it, in the
        case of for example eigenfunction subsets this is not guaranteed.

        Parameters
        ----------
        ds : ~pylbo.data_containers.LegolasDataSet
            The dataset associated with the given eigenvalue index
        idx : int
            The index of the selected eigenvalue

        Returns
        -------
        bool
            Returns `True` if `idx` corresponds to an eigenvalue with eigenfunctions,
            `False` otherwise.
        """
        point_has_efs = idx in ds.header["ef_written_idxs"]
        if not point_has_efs:
            pylboLogger.warning(
                f"eigenvalue {ds.eigenvalues[idx]} has no eigenfunctions!"
            )
        return point_has_efs

    def _toggle_eigenfunction_subset_radius(self):
        if not isinstance(self.data, LegolasDataSet):
            return
        if not self.data.has_ef_subset:
            return
        xlim = self.spec_axis.get_xlim()
        ylim = self.spec_axis.get_ylim()
        if self._ef_subset_artists is None:
            center = self.data.header["ef_subset_center"]
            (subset_center,) = self.spec_axis.plot(
                np.real(center), np.imag(center), ".r", markersize=8, alpha=0.8
            )
            subset_center.set_visible(False)
            radius = self.data.header["ef_subset_radius"]
            circle = plt.Circle(
                (np.real(center), np.imag(center)),
                radius,
                edgecolor="red",
                facecolor="none",
                lw=2,
            )
            circle.set_visible(False)
            self.spec_axis.add_patch(circle)
            self._ef_subset_artists = {"center": subset_center, "circle": circle}
        for artist in self._ef_subset_artists.values():
            artist.set_visible(not artist.get_visible())
        self.spec_axis.set_xlim(xlim)
        self.spec_axis.set_ylim(ylim)

    def _mark_points_without_data_written(self):
        """
        For dataseries, it is possible that not all datasets in the series
        have eigenfunctions associated with them. This routine will toggle a change
        in the opacity value for datapoints with no functions, so they are
        clearly distinguishable from those who do have them.
        """
        if not isinstance(self.data, LegolasDataSeries) or all(self.data.has_efs):
            return
        self._transparent_data = not self._transparent_data
        for ax in self.axis.figure.get_axes():
            for child in ax.get_children():
                # the ones with this attribute are all vertical rows of datapoints
                if hasattr(child, "dataset"):
                    if self._unmarked_alpha is None:
                        self._unmarked_alpha = child.get_alpha()
                    if not child.dataset.has_efs:
                        child.set_alpha(0)
                    else:
                        child.set_alpha(self._unmarked_alpha)

    def _display_tooltip(self):
        tooltip_values = [
            [r"$\mathbf{L~click}$", "select spectrum point"],
            [r"$\mathbf{R~click}$", "remove spectrum point"],
            [r"$\mathbf{\hookleftarrow}$", "update plot"],
            [r"$\mathbf{\uparrow}$", "cycle to next eigenfunction"],
            [r"$\mathbf{\downarrow}$", "cycle to previous eigenfunction"],
            [r"$\mathbf{i}$", r"switch real $\leftrightarrow$ imaginary"],
            [r"$\mathbf{d}$", "clear selection"],
            [r"$\mathbf{t}$", r"cylindrical: toggle $v_r \leftrightarrow r v_r$"],
            [r"$\mathbf{w}$", r"print selected $\omega$ to console"],
            [r"$\mathbf{e}$", r"subset: toggle center and radius"],
            [r"$\mathbf{a}$", r"save eigenvalue selection"],
            [r"$\mathbf{j}$", r"save selection indices"],
        ]
        bbox = np.array([0.1, 0.3, 0.8, 0.4])
        rectangle = patches.Rectangle(
            xy=0.95 * bbox[0:2],
            width=1.05 * bbox[2],
            height=1.05 * bbox[3],
            facecolor="lightgrey",
            alpha=0.3,
        )
        self.axis.add_patch(rectangle)

        table = self.axis.table(
            cellText=tooltip_values,
            bbox=bbox,
            cellLoc="center",
            edges="open",
            alpha=0.5,
        )
        table.scale(1, 1.5)
        table.auto_set_column_width(col=list(range(len(tooltip_values[0]))))
        # table title
        self.axis.text(
            0.5,
            1.05 * (bbox[1] + bbox[3]),
            "Interactive eigenfunction plotting",
            ha="center",
            va="center",
            weight="bold",
        )

    def _invert_continua(self, ds, ev_idx):
        """
        Calculates the locations of resonance with the continua for a specific
        eigenmode.

        Parameters
        ----------
        ef_idx : int
            The number of the eigenvalue in the dataset.

        Returns
        -------
        r_inv : dict
            Dictionary of continua names and inverted resonance locations
            (float, or None if not in domain).
        labels : dict
            Dictionary containing the corresponding labels to be printed when drawing
            the locations of resonance.
        """

        CONTINUUM_LABELS = {
            "slow-": r"$r(\Omega_S^-)",
            "slow+": r"$r(\Omega_S^+)",
            "alfven-": r"$r(\Omega_A^-)",
            "alfven+": r"$r(\Omega_A^+)",
            "thermal": r"$r(\Omega_T)",
            "doppler": r"$r(\Omega_0)",
        }

        r_inv = dict()
        labels = dict()

        eigfuncs = ds.get_eigenfunctions(ev_idxs=[ev_idx])
        sigma = eigfuncs[0].get("eigenvalue")

        continua_keys = ds.continua.keys()

        for continuum_key in continua_keys:
            continuum = ds.continua[continuum_key]
            if np.allclose(continuum, 0, atol=1e-12):
                continue
            # removes duplicates
            continuum = np.array(continuum, dtype=complex)

            r_inv_temp = find_resonance_location(continuum, ds.grid_gauss, sigma)

            r_inv[continuum_key] = r_inv_temp
            labels[continuum_key] = CONTINUUM_LABELS[continuum_key]

        if ds.gamma > 1e3:
            # Approximation for incompressibility currently implemented.
            del r_inv["slow-"]
            del r_inv["slow+"]

        return r_inv, labels

    def _show_resonances(self, ds, ev_idx, color):
        """
        Shows the locations of resonance with the continua.
        There is a different linestyle for every continuum.

        """
        RESONANCE_STYLES = {
            "slow-": "dotted",
            "slow+": "dotted",
            "alfven-": "dashed",
            "alfven+": "dashed",
            "thermal": "solid",
            "doppler": "dashdot",
        }

        r_inv, labels = self._invert_continua(ds, ev_idx)
        cont_keys = r_inv.keys()
        for cont_key in cont_keys:
            if r_inv[cont_key] is not None:
                self.axis.axvline(
                    x=r_inv[cont_key],
                    linestyle=RESONANCE_STYLES[cont_key],
                    color=color,
                    alpha=0.4,
                )
