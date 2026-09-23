import matplotlib.collections as mpl_collections
import matplotlib.lines as mpl_lines
import matplotlib.patches as mpl_patches
import numpy as np
from pylbo._version import _mpl_version
from pylbo.utilities.toolbox import add_pickradius_to_item


def _get_legend_handles(legend):
    """
    Returns the legend handles.

    Parameters
    ----------
    legend : ~matplotlib.legend.Legend
        The matplotlib legend to use.

    Returns
    -------
    handles : list
        A list of handles.
    """
    if _mpl_version >= "3.7":
        return legend.legend_handles
    return legend.legendHandles


class LegendHandler:
    """
    Main handler for legend stuff.

    Attributes
    ----------
    legend : ~matplotlib.legend.Legend
        The matplotlib legend to use.
    alpha_point : int, float
        Alpha value for non-hidden lines or points.
    alpha_region : int, float
        Alpha value for non-hidden regions.
    alpha_hidden : int, float
        Alpha value for hidden artists.
    marker : ~matplotlib.markers
        The marker to use for points.
    markersize : int, float
        Size of the marker.
    pickradius : int, float
        Radius around pickable items so pickevents are triggered.
    linewidth : int, float
        Width of drawn lines.
    legend_properties : dict
        Additional properties used when setting the legend.
    interactive : bool
        If `True`, makes the legend interactive
    autoscale : bool
        If `True`, will check if autoscale is needed when clicking the legend.
    """

    def __init__(self, interactive):
        self.legend = None
        self.alpha_point = 0.8
        self.alpha_region = 0.2
        self.alpha_hidden = 0.05

        self.marker = "p"
        self.markersize = 64
        self.pickradius = 10
        self.linewidth = 2
        self.legend_properties = {}

        self.interactive = interactive
        self.autoscale = False
        self._drawn_items = []
        self._legend_mapping = {}
        self._make_visible_by_default = False

    def on_legend_pick(self, event):
        """
        Determines what happens when the legend gets picked.

        Parameters
        ----------
        event : ~matplotlib.backend_bases.PickEvent
            The matplotlib pick event.
        """
        artist = event.artist
        if artist not in self._legend_mapping:
            return
        drawn_item = self._legend_mapping.get(artist)
        visible = not drawn_item.get_visible()
        drawn_item.set_visible(visible)
        if visible:
            if isinstance(artist, (mpl_collections.PathCollection, mpl_lines.Line2D)):
                artist.set_alpha(self.alpha_point)
            else:
                artist.set_alpha(self.alpha_region)
        else:
            artist.set_alpha(self.alpha_hidden)
        self._check_autoscaling()
        artist.figure.canvas.draw()

    def make_legend_pickable(self):
        """Makes the legend pickable, only used if interactive."""
        legend_handles = _get_legend_handles(self.legend)
        handle_labels = [handle.get_label() for handle in legend_handles]
        # we need a mapping of the legend item to the actual item that was drawn
        for i, drawn_item in enumerate(self._drawn_items):
            # try-except needed for fill_between, which returns empty handles
            try:
                idx = handle_labels.index(drawn_item.get_label())
                legend_item = _get_legend_handles(self.legend)[idx]
            except ValueError:
                idx = i
                legend_item = _get_legend_handles(self.legend)[idx]
                # fix empty label
                legend_item.set_label(drawn_item.get_label())
            add_pickradius_to_item(item=legend_item, pickradius=self.pickradius)
            # add an attribute to this artist to tell it's from a legend
            setattr(legend_item, "is_legend_item", True)
            # make sure colourmapping is done properly, only for continua regions
            if isinstance(legend_item, mpl_patches.Rectangle):
                legend_item.set_facecolor(drawn_item.get_facecolor())
                legend_item.set_edgecolor(drawn_item.get_edgecolor())
            # we make the regions invisible until clicked, or set visible as default
            if self._make_visible_by_default:
                legend_item.set_alpha(self.alpha_point)
            else:
                legend_item.set_alpha(self.alpha_hidden)
            drawn_item.set_visible(self._make_visible_by_default)
            self._legend_mapping[legend_item] = drawn_item

    def add(self, item):
        """
        Adds an item to the list of drawn items on the canvas.

        Parameters
        ----------
        item : object
            A single object, usually a return from the matplotlib plot or scatter
            methods.
        """
        if isinstance(item, (list, np.ndarray, tuple)):
            raise ValueError("object expected, not something list-like")
        self._drawn_items.append(item)

    def _check_autoscaling(self):
        """
        Checks if autoscaling is needed and if so, rescales the y-axis to the min-max
        value of the currently visible legend items.
        """
        if not self.autoscale:
            return
        visible_items = [item for item in self._drawn_items if item.get_visible()]
        # check scaling for visible items. This explicitly implements the equilibria,
        # but works for general cases as well. If needed we can simply subclass
        # and override.
        if len(visible_items) == 0:
            return
        ydata = visible_items[0].get_ydata()
        ymin1 = np.min(ydata)
        ymax1 = np.max(ydata)
        for item in visible_items:
            ymin1 = min(ymin1, np.min(item.get_ydata()))
            ymax1 = max(ymax1, np.max(item.get_ydata()))
        item.axes.set_ylim(ymin1 - abs(0.1 * ymin1), ymax1 + abs(0.1 * ymax1))
