from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Tuple

import matplotlib.lines as mpl_lines
import numpy as np
from pylbo.utilities.logger import pylboLogger

if TYPE_CHECKING:
    from pylbo.data_containers import LegolasDataContainer


def timethis(func):
    @functools.wraps(func)
    def _time_method(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            pylboLogger.debug(
                f"{func.__name__} took {time.perf_counter() - t0} seconds to execute"
            )

    return _time_method


def get_axis_geometry(ax):
    """
    Retrieves the geometry of a given matplotlib axis.

    Parameters
    ----------
    ax : ~matplotlib.axes.Axes
        The axis to retrieve the geometry from.

    Returns
    -------
    tuple
        The geometry of the given matplotlib axis.
    """
    return ax.get_subplotspec().get_geometry()[0:3]


def get_values(array, which_values):
    """
    Determines which values to retrieve from an array.

    Parameters
    ----------
    array : numpy.ndarray
        The array with values.
    which_values : str
        Can be one of the following:

            - "average": returns the average of the array
            - "minimum": returns the minimum of the array
            - "maximum": returns the maximum of the array

        If not supplied or equal to None, simply returns the array.

    Returns
    -------
    array : numpy.ndarray
        Numpy array with values depending on the argument provided.
    """
    if which_values is None:
        return array
    elif which_values == "average":
        return np.average(array)
    elif which_values == "minimum":
        return np.min(array)
    elif which_values == "maximum":
        return np.max(array)
    else:
        raise ValueError(f"unknown argument which_values: {which_values}")


def add_pickradius_to_item(item, pickradius):
    """
    Makes a matplotlib artist pickable and adds a pickradius.
    We have to handle this separately, because for line2D items the method
    :meth:`~matplotlib.axes.Axes.set_picker` is deprecated from version 3.3 onwards.

    Parameters
    ----------
    item : ~matplotlib.artist.Artist
        The artist which will be made pickable
    pickradius : int, float
        Sets the pickradius, which determines if something is "on" the picked point.
    """
    # set_picker is deprecated for line2D from matplotlib 3.3 onwards
    if isinstance(item, mpl_lines.Line2D):
        item.set_picker(True)
        item.pickradius = pickradius
    else:
        item.set_picker(pickradius)


def custom_enumerate(iterable, start=0, step=1):
    """
    Does a custom enumeration with a given stepsize.

    Parameters
    ----------
    iterable : ~typing.Iterable
        The iterable to iterate over.
    start : int
        The starting value for enumerate.
    step : int
        The stepsize between enumerate values.

    Yields
    ------
    start : int
        The current index in `iterable`, incremented with `step`.
    itr : ~typing.Iterable
        The corresponding entry of `iterable`.
    """
    for itr in iterable:
        yield start, itr
        start += step


def transform_to_list(obj: any) -> list:
    """
    Transforms a given input argument `obj` to a list. If `obj`
    is a Numpy-array or tuple, a cast to `list()` is invoked.

    Parameters
    ----------
    obj : any
        The object to transform.

    Returns
    -------
    list
        The object converted to a list.
    """
    if obj is None:
        return [obj]
    elif isinstance(obj, (tuple, np.ndarray)):
        return list(obj)
    elif isinstance(obj, list):
        return obj
    return [obj]


def transform_to_numpy(obj: any) -> np.ndarray:
    """
    Transforms a given input argument `obj` to a numpy array.

    Parameters
    ----------
    obj : any
        The object to transform.

    Returns
    -------
    numpy.ndarray
        The object transformed to a numpy array.

    """
    if obj is None:
        return np.asarray([obj])
    elif isinstance(obj, (tuple, list)):
        return np.asarray(obj)
    elif isinstance(obj, np.ndarray):
        return np.atleast_1d(obj) if obj.shape == () else obj
    return np.asarray([obj])


def reduce_to_unique_array(array: np.ndarray) -> np.ndarray:
    """
    Reduces a given array to its unique values, preserving the order.

    Parameters
    ----------
    array : numpy.ndarray
        The array to reduce.

    Returns
    -------
    numpy.ndarray
        The array with unique values.
    """
    objs, idxs = np.unique(array, return_index=True)
    return objs[np.argsort(idxs)]


def get_all_eigenfunction_names(data: LegolasDataContainer) -> np.ndarray[str]:
    """
    Merges the regular and derived eigenfunction names into a unique array,
    preserving order.

    Parameters
    ----------
    data : LegolasDataContainer
        The data container containing the eigenfunction names.

    Returns
    -------
    numpy.ndarray
        The array with unique eigenfunction names.
    """
    names = reduce_to_unique_array(data.ef_names)
    if any(transform_to_numpy(data.has_derived_efs)):
        names = np.concatenate((names, reduce_to_unique_array(data.derived_ef_names)))
    return names


def get_maximum_eigenvalue(
    eigenvalues: np.ndarray[complex],
    real: bool = True,
    re_range: Tuple[float, float] = None,
) -> complex:
    """
    Calculates the maximum eigenvalue of a given array of eigenvalues.
    The real or imaginary part is used, depending on the `real` argument.
    If a range is specified, the maximum eigenvalue is calculated within
    that range on the real axis.

    Parameters
    ----------
    eigenvalues : numpy.ndarray(dtype=complex)
        The array of eigenvalues.
    real : bool
        If `True`, the real part of the eigenvalues is used. Imaginary part otherwise.
    re_range : tuple(float, float)
        The range on the real axis to calculate the maximum eigenvalue.
        Defaults to None, which means all eigenvalues are considered.

    Returns
    -------
    complex
        The maximum eigenvalue.
    """
    func = np.real if real else np.imag
    evs = eigenvalues.copy()
    if re_range is not None:
        # set eigenvalues outside of range to NaN
        evs[np.logical_or(func(evs) < re_range[0], func(evs) > re_range[1])] = np.nan
    if np.all(np.isnan(func(evs))):
        raise ValueError("get_maximum_eigenvalue: no eigenvalues found within range")
    return eigenvalues[np.nanargmax(func(evs))]


def solve_cubic_exact(a, b, c, d):
    """
    Solves a given cubic polynomial of the form
    :math:`ax^3 + bx^2 + cx + d = 0` using the analytical cubic root formula
    instead of the general `numpy.roots` routine.
    From `StackOverflow <https://math.stackexchange.com/questions
    15865why-not-write-the-solutions-of-a-cubic-this-way/18873#18873/>`_.

    Parameters
    ----------
    a : int, float, complex
        Cubic coefficient.
    b : int, float, complex
        Quadratic coefficient.
    c : int, float, complex
        Linear coefficient.
    d : int, float, complex
        Constant term

    Returns
    -------
    roots : np.ndarray(ndim=3, dtype=complex)
        The three roots of the cubic polynomial as a Numpy array.
    """

    if a == 0:
        raise ValueError("cubic coefficient may not be zero")
    p = b / a
    q = c / a
    r = d / a
    Aterm = (
        -2 * p**3
        + 9 * p * q
        - 27 * r
        + 3
        * np.sqrt(3)
        * np.sqrt(-(p**2) * q**2 + 4 * q**3 + 4 * p**3 * r - 18 * p * q * r + 27 * r**2)
    ) ** (1 / 3) / (3 * 2 ** (1 / 3))
    Bterm = (-(p**2) + 3 * q) / (9 * Aterm)
    cterm_min = (-1 - np.sqrt(3) * 1j) / 2
    cterm_pos = (-1 + np.sqrt(3) * 1j) / 2
    x1 = -p / 3 + Aterm - Bterm
    x2 = -p / 3 + cterm_min * Aterm - cterm_pos * Bterm
    x3 = -p / 3 + cterm_pos * Aterm - cterm_min * Bterm
    return np.array([x1, x2, x3], dtype=complex)


def count_zeroes(eigfuncs, real=True):
    """
    Counts the number of zeroes of an array of complex eigenfunctions by looking at
    sign changes of the real and imaginary part of the eigenfunctions. Excludes
    the eigenfunction boundaries.

    Parameters
    ----------
    eigfuncs : numpy.ndarray(dtype=complex)
        Array of eigenfunction arrays of complex numbers.
    real : bool
        If `True`, counts the number of zeroes of the real part of the eigenfunctions.
        If `False`, counts the number of zeroes of the imaginary part.

    Returns
    -------
    np.ndarray(dtype=int)
        The number of zeroes of each eigenfunction.
    """
    eigfuncs = np.array([ef[1:-1] for ef in eigfuncs], dtype=complex)
    func = np.real if real else np.imag
    return np.sum(np.diff(np.sign(func(eigfuncs)), axis=1) != 0, axis=1)


def find_resonance_location(continuum, grid, sigma):
    """
    Finds the resonance location between sigma and the continuum. For example, if
    the continuum is given by [5, 6, 7, 8, 9, 10] and the grid is equal to
    [0, 1, 2, 3, 4, 5], then for a sigma = 9 the resonance location is 4. For a sigma
    equal to 8.5 the resonance location is 3.5. For a sigma outside of the continuum
    the resonance location is None. If the continuum array is not monotone, then
    the resonance location is interpolated between the first matched interval.

    Parameters
    ----------
    continuum : numpy.ndarray(dtype=complex)
        Array containing the range of a specific continuum. Can be complex, but only
        the resonance with the real part is calculated.
    grid : numpy.ndarray
        The grid on which the continuum is defined.
    sigma : complex
        A given eigenvalue.

    Returns
    -------
    None, np.ndarray(float)
        The position where there is resonance between the eigenmode and the continuum.
        Returns None if there is no resonance with the specified continuum.
    """
    if np.min(continuum.real) > sigma or np.max(continuum.real) < sigma:
        return None
    # if continuum is monotone then do simple interpolation
    if np.all(np.diff(continuum.real) > 0):
        return np.array([np.interp(sigma, continuum.real, grid)], dtype=float)
    # otherwise find intervals and handle multiple matches
    locs = []
    c = continuum.real
    for idx in range(len(continuum) - 1):
        if c[idx] <= sigma <= c[idx + 1]:
            locs.append(
                np.interp(sigma, [c[idx], c[idx + 1]], [grid[idx], grid[idx + 1]])
            )
        elif c[idx + 1] <= sigma <= c[idx]:
            locs.append(
                np.interp(sigma, [c[idx + 1], c[idx]], [grid[idx + 1], grid[idx]])
            )
    return np.array(list(set(locs)), dtype=float)
