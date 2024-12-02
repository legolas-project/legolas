from functools import wraps
from typing import Any

import matplotlib.axes
from pylbo.data_containers import transform_to_dataseries
from pylbo.utilities.toolbox import get_all_eigenfunction_names

_BACKGROUND_NAME_MAPPING = {
    "rho0": r"$\rho_0$",
    "drho0": r"$\partial \rho_0$",
    "T0": r"$T_0$",
    "dT0": r"$\partial T_0$",
    "ddT0": r"$\partial^2 T_0$",
    "B01": r"$B_{01}$",
    "B02": r"$B_{02}$",
    "B03": r"$B_{03}$",
    "dB02": r"$\partial B_{02}$",
    "dB03": r"$\partial B_{03}$",
    "ddB02": r"$\partial^2 B_{02}$",
    "ddB03": r"$\partial^2 B_{03}$",
    "B0": r"$B_0$",
    "v01": r"$v_{01}$",
    "v02": r"$v_{02}$",
    "v03": r"$v_{03}$",
    "dv01": r"$\partial v_{01}$",
    "dv02": r"$\partial v_{02}$",
    "dv03": r"$\partial v_{03}$",
    "ddv01": r"$\partial^2 v_{01}$",
    "ddv02": r"$\partial^2 v_{02}$",
    "ddv03": r"$\partial^2 v_{03}$",
    "L0": r"$\mathcal{L}_0$",
    "dLdT": r"$\partial_T \mathcal{L}$",
    "dLdrho": r"$\partial_\rho \mathcal{L}$",
    "lambdaT": r"$\Lambda(T)$",
    "dlambdadT": r"$\partial_T \Lambda$",
    "H0": r"$\mathcal{H}_0$",
    "dHdT": r"$\partial_T \mathcal{H}$",
    "dHdrho": r"$\partial_\rho \mathcal{H}$",
    "kappa_para": r"$\kappa_\parallel$",
    "kappa_perp": r"$\kappa_\perp$",
    "dkappa_para_dT": r"$\partial_T \kappa_\parallel$",
    "dkappa_para_dr": r"$\partial \kappa_\parallel$",
    "dkappa_perp_drho": r"$\partial_\rho \kappa_\perp$",
    "dkappa_perp_dT": r"$\partial_T \kappa_\perp$",
    "dkappa_perp_dB2": r"$\partial_{B^2} \kappa_\perp$",
    "dkappa_perp_dr": r"$\partial \kappa_\perp$",
    "eta": r"$\eta$",
    "detadT": r"$\partial_T \eta$",
    "detadr": r"$\partial \eta$",
    "gravity": r"$g$",
}


def refresh_plot(f: callable) -> callable:
    """
    Simple decorator, when a routine is wrapped with this the plot will be
    cleared and redrawn on calling it.
    Useful for when the scaling is changed or artists are added/removed.
    """

    @wraps(f)
    def refresh(*args, **kwargs):
        f(*args, **kwargs)
        window = args[0]
        window.redraw()
        return f

    return refresh


def ensure_attr_set(obj: Any, attr: str) -> None:
    """
    Ensures that a given attribute is set.

    Parameters
    ----------
    obj : Any
        The object to check.
    attr : str
        The attribute to check.

    Raises
    ------
    ValueError
        If the attribute is not set.
    """
    if getattr(obj, attr, None) is None:
        raise AttributeError(f"attribute '{attr}' not set for {type(obj)}")


def ef_name_to_latex(
    ef_name: str, geometry: str = "Cartesian", real_part: bool = None
) -> str:
    """
    Converts an eigenfunction name to latex formatting. Numbers are replaced with a
    suffix corresponding to the geometry: :math:`(1, 2, 3)` becomes :math:`(x, y, z)`
    for Cartesian and :math:`(r, \\theta, z)` for cylindrical geometries. Symbols
    and letters are also converted to LaTeX.

    Parameters
    ----------
    ef_name : str
        The name of the eigenfunction.
    geometry : str, optional
        The geometry of the eigenfunction. The default is "Cartesian".
    real_part : bool, optional
        Whether the real part of the eigenfunction is being plotted. The default is
        None.
    """
    part = ""
    if real_part is not None:
        part = "Re" if real_part else "Im"

    if geometry == "cylindrical":
        suffix = ("_r", r"_\theta", "_z")
    else:
        suffix = ("_x", "_y", "_z")
    for i, idx in enumerate("123"):
        ef_name = ef_name.replace(idx, suffix[i])

    ef_name = ef_name.replace("rho", r"\rho")
    ef_name = ef_name.replace("div", "\\nabla\\cdot")
    ef_name = ef_name.replace("curl", "\\nabla\\times")
    ef_name = ef_name.replace("para", "\\parallel")
    ef_name = ef_name.replace("perp", "\\perp")
    ef_name = ef_name.replace("dB", "B^\\prime")
    latex_name = rf"${ef_name}$"
    if part != "":
        latex_name = rf"{part}({latex_name})"
    return latex_name


def background_name_to_latex(bg_name: str) -> str:
    """
    Maps the background name to latex formatting.

    Parameters
    ----------
    bg_name : str
        The name of the background as given by the corresponding dictionary key.

    Returns
    -------
    str
        The latex formatted background name. If the background name has no mapping
        the original name is returned.
    """
    return _BACKGROUND_NAME_MAPPING.get(bg_name, bg_name)


def validate_ef_name(ds, ef_name: str) -> str:
    """
    Returns the validated eigenfunction name.

    Parameters
    ----------
    ds : ~pylbo.data_containers.LegolasDataContainer
        The dataset/series containing the eigenfunctions.
    ef_name : str
        The name of the eigenfunction.

    Raises
    ------
    ValueError
        If the eigenfunction name is not valid.

    Returns
    -------
    str
        The validated eigenfunction name.
    """
    # copy this or we're editing the property itself
    for dataset in transform_to_dataseries(ds).datasets:
        names = get_all_eigenfunction_names(dataset)
        if ef_name not in names:
            raise ValueError(
                f"The eigenfunction '{ef_name}' is not part of the "
                f"eigenfunctions {names}."
            )
    return ef_name


def _validate_textbox_location(loc: str) -> str:
    """
    Validates the location of the textbox.

    Parameters
    ----------
    loc : str
        The location of the textbox.

    Raises
    ------
    ValueError
        If the location is not one of "top left", "top right", "bottom left" or
        "bottom right".

    Returns
    -------
    str
        The validated location.

    """
    allowed_locs = ["top left", "top right", "bottom left", "bottom right"]
    if loc not in allowed_locs:
        raise ValueError(f"Invalid location: {loc}, must be one of {allowed_locs}")
    return loc


def _get_textbox_axes_coords(loc: str, outside: bool, width: float, height: float):
    """
    Returns the coordinates of the textbox.

    Parameters
    ----------
    loc : str
        The location of the textbox.
    outside : bool
        Whether the textbox is outside the axes.
    width : float
        The width of the bounding box.
    height : float
        The height of the bounding box.

    Returns
    -------
    float
        The x-coordinate of the textbox.
    float
        The y-coordinate of the textbox.

    """
    x = 0.5 * width
    y = 0.5 * height
    if loc == "top left":
        y = 1 - y
    elif loc == "top right":
        x = 1 - x
        y = 1 - y
    elif loc == "bottom right":
        x = 1 - x

    if outside:
        if "top" in loc:
            y = y + height
        else:
            y = y - height
    return x, y


def add_textbox_to_axes(
    ax: matplotlib.axes.Axes,
    text: str,
    x: float,
    y: float,
    coords: str = "axes",
    fs: int = 15,
    alpha: float = 0.2,
    halign: str = "center",
    color: str = "grey",
    textcolor: str = "black",
    boxstyle: str = "round",
) -> matplotlib.axes.Axes.text:
    """
    Convenience method to add a textbox to the given axes.

    Parameters
    ----------
    ax : ~matplotlib.axes.Axes
        The axes to add the textbox to.
    text : str
        The text to add to the textbox.
    x : float
        The x-coordinate of the textbox.
    y : float
        The y-coordinate of the textbox.
    coords : str, optional
        The coordinate system of the textbox. The default is "axes", options are
        "axes", "figure", and "data".
    fs : int, optional
        The fontsize of the textbox. The default is 15.
    alpha : float, optional
        The alpha value of the textbox. The default is 0.2.
    halign : str, optional
        The horizontal alignment of the textbox. The default is "center".
    color : str, optional
        The color of the textbox. The default is "grey".
    textcolor : str, optional
        The color of the text. The default is "black".
    boxstyle : str, optional
        The style of the textbox. The default is "round".

    Returns
    -------
    ~matplotlib.axes.Axes.text
        The textbox.
    """
    transform = {
        "data": ax.transData,
        "axes": ax.transAxes,
        "figure": ax.figure.transFigure,
    }
    bbox = {"facecolor": color, "alpha": alpha, "boxstyle": boxstyle, "pad": 0.2}
    return ax.text(
        x,
        y,
        text,
        transform=transform[coords],
        fontsize=fs,
        bbox=bbox,
        horizontalalignment=halign,
        color=textcolor,
    )


def add_axis_label(
    ax: matplotlib.axes.Axes,
    text: str,
    loc: str = "top left",
    fs: int = 15,
    alpha: float = 0.2,
    color: str = "grey",
    textcolor: str = "black",
    boxstyle: str = "round",
    bold: bool = False,
    outside: bool = False,
) -> matplotlib.axes.Axes.text:
    """
    Creates a textbox in one of the corners of the specified axis. This method is meant
    to create panel labels without having to manually specify the coordinates of the
    textbox.

    Parameters
    ----------
    ax : ~matplotlib.axes.Axes
        The axes to add the textbox to.
    text : str
        The text to add to the textbox.
    loc : str, optional
        The location of the textbox. The default is "top left", options are
        "top right", "bottom left" and "bottom right".
    fs : int, optional
        The fontsize of the textbox. The default is 15.
    alpha : float, optional
        The alpha value of the textbox. The default is 0.2.
    color : str, optional
        The color of the textbox. The default is "grey".
    textcolor : str, optional
        The color of the text. The default is "black".
    boxstyle : str, optional
        The style of the textbox. The default is "round". If `None` is passed
        no box will be drawn.
    bold : bool, optional
        Whether to bold the text. The default is False.
    outside : bool, optional
        Whether to place the textbox outside of the axis. The default is False.

    Raises
    ------
    ValueError
        If the location is not one of "top left", "top right", "bottom left" or
        "bottom right".

    Returns
    -------
    ~matplotlib.axes.Axes.text
        The textbox.
    """
    _validate_textbox_location(loc)
    bbox = {"facecolor": "none", "alpha": 0}
    if boxstyle is not None:
        bbox.update(
            {"facecolor": color, "alpha": alpha, "boxstyle": boxstyle, "pad": 0.2}
        )
    va = "center"
    ha = "center"

    # optional kwargs
    kwargs = {}
    if bold:
        kwargs.update({"weight": "bold"})

    transform = ax.transAxes
    # first draw sample to calculate the size of the textbox
    sample = ax.text(
        0.5,
        0.5,
        text,
        transform=transform,
        fontsize=fs,
        bbox=bbox,
        ha=ha,
        va=va,
        color=textcolor,
        **kwargs,
    )
    ax.figure.canvas.draw()
    # get bounding box and make it 2% larger to prevent hugging the axes
    bb = sample.get_bbox_patch().get_extents().transformed(transform.inverted())
    bb_width = 1.02 * (bb.x1 - bb.x0)
    bb_height = 1.02 * (bb.y1 - bb.y0)
    sample.remove()
    x, y = _get_textbox_axes_coords(
        loc=loc, outside=outside, width=bb_width, height=bb_height
    )

    return ax.text(
        x,
        y,
        text,
        transform=transform,
        fontsize=fs,
        bbox=bbox,
        ha=ha,
        va=va,
        color=textcolor,
        **kwargs,
    )
