from __future__ import annotations

from typing import Union

import numpy as np
from pylbo.data_containers import LegolasDataSet, LegolasDataSeries
from pylbo.utilities.toolbox import transform_to_numpy
from pylbo.visualisation.modes.cartesian_2d import CartesianSlicePlot2D
from pylbo.visualisation.modes.cartesian_3d import CartesianSlicePlot3D
from pylbo.visualisation.modes.cylindrical_2d import CylindricalSlicePlot2D
from pylbo.visualisation.modes.cylindrical_3d import CylindricalSlicePlot3D
from pylbo.visualisation.modes.mode_data import ModeVisualisationData
from pylbo.visualisation.modes.temporal_1d import TemporalEvolutionPlot1D
from pylbo.visualisation.modes.vtk_export import VTKCartesianData, VTKCylindricalData


def plot_1d_temporal_evolution(
    ds: Union[LegolasDataSet, LegolasDataSeries],
    omega: Union[
        complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
    ],
    ef_name: str,
    u2: float,
    u3: float,
    time: Union[list, np.ndarray],
    figsize: tuple[int, int] = None,
    add_background: bool = False,
    use_real_part: bool = True,
    complex_factor: Union[
        complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
    ] = None,
    show_ef_panel: bool = True,
    **kwargs,
) -> TemporalEvolutionPlot1D:
    """
    Plot the temporal evolution of a 1D eigenmode solution, i.e.

    :math:`f(u_1, u_2, u_3, t) =
    f_1(u_1) \\exp\\left(ik_2u_2 + ik_3u_3 - i\\omega t\\right)`,

    for a particular set of coordinates :math:`(u_2, u_3)` over a time interval.

    Parameters
    ----------
    ds : LegolasDataSet, LegolasDataSeries
        The data set/series containing the eigenfunctions, having the same equilibria.
    omega : complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
        The (approximate) eigenvalue of the mode(s) to visualise. For multiple data
        series, length of omega and ds should match.
    ef_name : str
        The name of the eigenfunction to visualise.
    u2 : float
        The y or :math:`\\theta` coordinate of the eigenmode solution.
    u3 : float
        The z coordinate of the eigenmode solution.
    time : list or np.ndarray
        The time interval to visualise.
    figsize : tuple[int, int]
        The size of the figure.
    add_background : bool
        Whether to add the equilibrium background to the plot.
    use_real_part : bool
        Whether to use the real part of the eigenmode solution.
    complex_factor : complex, list[complex], np.ndarray, list[list[complex]],
        list[np.ndarray]
        A complex factor to multiply the eigenmode solution with. For multiple data
        series, length of omega and ds should match.
    show_ef_panel : bool
        Whether to show the eigenfunction panel.
    kwargs : dict
        Additional keyword arguments to pass to :meth:`~matplotlib.pyplot.imshow`.

    Returns
    -------
    TemporalEvolutionPlot1D
        The plot object.
    """

    data = ModeVisualisationData(
        ds, omega, ef_name, use_real_part, complex_factor, add_background
    )
    if isinstance(time, list):
        time = transform_to_numpy(time)
    p = TemporalEvolutionPlot1D(data, u2, u3, time, figsize, show_ef_panel, **kwargs)
    return p


def plot_2d_slice(
    ds: Union[LegolasDataSet, LegolasDataSeries],
    omega: Union[
        complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
    ],
    ef_name: str,
    u2: Union[float, np.ndarray],
    u3: Union[float, np.ndarray],
    time: float,
    slicing_axis: str,
    figsize: tuple[int, int] = None,
    add_background: bool = False,
    use_real_part: bool = True,
    complex_factor: Union[
        complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
    ] = None,
    show_ef_panel: bool = True,
    polar: bool = False,
    **kwargs,
) -> CartesianSlicePlot2D:
    """
    Plot a 2D spatial view of the eigenmode solution, i.e.

    :math:`f(u_1, u_2, u_3, t) =
    f_1(u_1) \\exp\\left(ik_2u_2 + ik_3u_3 - i\\omega t\\right)`,

    for a particular set of coordinates. If `slicing_axis = 'z'` then a 2D view is
    created for a given slicing point along the 'z'-axis (hence a 'xy'-view), for
    `slicing_axis = 'y'` a 'xz'-view will be created. The given spatial coordinates
    `u2` and `u3` must be consistent with the slicing axis. For cylindrical geometries
    slices in both Cartesian and polar coordinates can be created.

    Parameters
    ----------
    ds : LegolasDataSet, LegolasDataSeries
        The data set/series containing the eigenfunctions, having the same equilibria.
    omega : complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
        The (approximate) eigenvalue of the mode(s) to visualise. For multiple
        data series, length of omega and ds should match.
    ef_name : str
        The name of the eigenfunction to visualise.
    u2 : float, np.ndarray
        The y or :math:`\\theta` coordinate of the eigenmode solution.
    u3 : float, np.ndarray
        The z coordinate of the eigenmode solution.
    time : float
        The time at which to visualise the eigenmode solution.
    slicing_axis : str
        The axis to slice the 2D view along, either 'z', 'y' or 'theta'
    figsize : tuple[int, int]
        The size of the figure.
    add_background : bool
        Whether to add the equilibrium background to the plot.
    use_real_part : bool
        Whether to use the real part of the eigenmode solution.
    complex_factor : complex, list[complex], np.ndarray, list[list[complex]],
        list[np.ndarray]
        A complex factor to multiply the eigenmode solution with. For multiple data
        series, length of omega and ds should match.
    show_ef_panel : bool
        Whether to show the eigenfunction panel.
    polar : bool
        Whether to use polar coordinates for the 2D view. Only used if the
        dataset geometry is cylindrical. Default is False.
    kwargs : dict
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    p : CartesianSlicePlot2D or CylindricalSlicePlot2D
        The plot object.
    """

    data = ModeVisualisationData(
        ds, omega, ef_name, use_real_part, complex_factor, add_background
    )
    if not polar or ds.geometry == "Cartesian":
        p = CartesianSlicePlot2D(
            data, u2, u3, time, slicing_axis, figsize, show_ef_panel, **kwargs
        )
    else:
        p = CylindricalSlicePlot2D(
            data, u2, u3, time, slicing_axis, figsize, show_ef_panel, polar, **kwargs
        )
    return p


def plot_3d_slice(
    ds: Union[LegolasDataSet, LegolasDataSeries],
    omega: Union[
        complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
    ],
    ef_name: str,
    u2: Union[float, np.ndarray],
    u3: Union[float, np.ndarray],
    time: float,
    slicing_axis: str,
    figsize: tuple[int, int] = None,
    add_background: bool = False,
    use_real_part: bool = True,
    complex_factor: Union[
        complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
    ] = None,
    **kwargs,
) -> CartesianSlicePlot3D:
    """
    Plot a 3D spatial view of the eigenmode solution, i.e.

    :math:`f(u_1, u_2, u_3, t) =
    f_1(u_1) \\exp\\left(ik_2u_2 + ik_3u_3 - i\\omega t\\right)`,

    for a particular set of coordinates. Several 2D slices are superimposed on each
    other for every value of :math:`u_3`.

    Parameters
    ----------
    ds : LegolasDataSet, LegolasDataSeries
        The data set/series containing the eigenfunctions, having the same equilibria.
    omega : complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
        The (approximate) eigenvalue of the mode(s) to visualise. For multiple data
        series, length of omega and ds should match.
    ef_name : str
        The name of the eigenfunction to visualise.
    u2 : float, np.ndarray
        The y or :math:`\\theta` coordinate of the eigenmode solution.
    u3 : float, np.ndarray
        The z coordinate of the eigenmode solution.
    time : float
        The time at which to visualise the eigenmode solution.
    slicing_axis : str
        The axis to slice the 2D view along, either 'z', 'y' or 'theta'
    figsize : tuple[int, int]
        The size of the figure.
    add_background : bool
        Whether to add the equilibrium background to the plot.
    use_real_part : bool
        Whether to use the real part of the eigenmode solution.
    complex_factor : complex, list[complex], np.ndarray, list[list[complex]],
        list[np.ndarray]
        A complex factor to multiply the eigenmode solution with. For multiple data
        series, length of omega and ds should match.
    kwargs : dict
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    p : CartesianSlicePlot3D or CylindricalSlicePlot3D
        The plot object.
    """

    u3 = transform_to_numpy(u3)
    data = ModeVisualisationData(
        ds, omega, ef_name, use_real_part, complex_factor, add_background
    )
    if ds.geometry.lower() == "cartesian":
        p = CartesianSlicePlot3D(data, u2, u3, time, slicing_axis, figsize, **kwargs)
    else:
        p = CylindricalSlicePlot3D(data, u2, u3, time, slicing_axis, figsize, **kwargs)
    return p


def prepare_vtk_export(
    ds: Union[LegolasDataSet, LegolasDataSeries],
    omega: Union[
        complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
    ],
    u2: np.ndarray,
    u3: np.ndarray,
    use_real_part: bool = True,
    complex_factor: Union[
        complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
    ] = None,
) -> Union[VTKCartesianData, VTKCylindricalData]:
    """
    Prepares for a VTK file export of the eigenmode solution in three dimensions.
    Returns a :class:`VTKDataCube3D` object which can be used to generate VTK files.

    Parameters
    ----------
    ds : LegolasDataSet, LegolasDataSeries
        The data set/series containing the eigenfunctions, having the same equilibria.
    omega : complex, list[complex], np.ndarray, list[list[complex]], list[np.ndarray]
        The (approximate) eigenvalue of the mode(s) to visualise. For multiple data
        series, length of omega and ds should match.
    u2 : np.ndarray
        The y or :math:`\\theta` coordinates of the eigenmode solution.
    u3 : np.ndarray
        The z coordinates of the eigenmode solution.
    use_real_part : bool
        Whether to use the real part of the eigenmode solution.
    complex_factor : complex, list[complex], np.ndarray, list[list[complex]],
        list[np.ndarray]
        A complex factor to multiply the eigenmode solution with. For multiple data
        series, length of omega and ds should match.

    Returns
    -------
    VTKCartesianData or VTKCylindricalData
        Object that can be used to generate VTK files.
    """

    data = ModeVisualisationData(
        ds, omega, None, use_real_part, complex_factor, add_background=False
    )
    if ds.geometry.lower() == "cartesian":
        return VTKCartesianData(data, u2, u3)
    else:
        return VTKCylindricalData(data, u2, u3)
