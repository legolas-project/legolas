import functools
import time

import matplotlib.lines as mpl_lines
import numpy as np
from pylbo._version import _mpl_version
from pylbo.utilities.logger import pylboLogger


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
    if _mpl_version >= "3.4":
        axis_geometry = ax.get_subplotspec().get_geometry()[0:3]
    else:
        # this is 1-based indexing by default, use 0-based here for consistency
        # with subplotspec in matplotlib 3.4+
        axis_geometry = transform_to_numpy(ax.get_geometry())
        axis_geometry[-1] -= 1
        axis_geometry = tuple(axis_geometry)
    return axis_geometry


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
    if isinstance(item, mpl_lines.Line2D) and _mpl_version >= "3.3":
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
        return obj
    return np.asarray([obj])


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


def count_zeroes(eigfuncs):
    """
    Counts the number of zeroes of an array of complex eigenfunctions by looking at
    sign changes of the real and imaginary part of the eigenfunctions.
    Doesn't include the grid endpoints in the count, since the boundary conditions are
    automatically satisfied. This only becomes accurate for eigenfunctions with enough
    oscillations and is resolution dependent. Therefore, we take the minimum
    of the number of zeroes of the real and imaginary part.

    Parameters
    ----------
    eigfuncs : numpy.ndarray
        Array of eigenfunction arrays of complex numbers.

    Returns
    -------
    nzeroes : np.ndarray(dtype=int)
        Counter array containing the number of zeroes of the real or imaginary part
        of each input eigenfunction array.
    """

    nzeroes = np.array([], dtype=int)

    for eigfunc in eigfuncs:
        counter_real = 0
        counter_imag = 0
        sign_real_eigfunc = np.sign(np.real(eigfunc))
        sign_imag_eigfunc = np.sign(np.imag(eigfunc))

        for i in range(1, len(sign_real_eigfunc) - 1):
            if sign_real_eigfunc[i - 1] * sign_real_eigfunc[i] == -1:
                counter_real += 1
            if sign_real_eigfunc[i - 1] * sign_real_eigfunc[i] == 0:
                if sign_real_eigfunc[i - 2] * sign_real_eigfunc[i - 1] == 0:
                    counter_real += 1

            if sign_imag_eigfunc[i - 1] * sign_imag_eigfunc[i] == -1:
                counter_imag += 1
            if sign_imag_eigfunc[i - 1] * sign_imag_eigfunc[i] == 0:
                if sign_imag_eigfunc[i - 2] * sign_imag_eigfunc[i - 1] == 0:
                    counter_imag += 1

        counter = min(counter_real, counter_imag)
        nzeroes = np.append(nzeroes, counter)

    return nzeroes


def invert_continuum_array(cont, r_gauss, sigma):
    """
    Finds the location of resonance for eigenmode solutions having a real part that
    might overlap with a continuum range.

    Parameters
    ----------
    cont : numpy.ndarray
        Array containing the range of a specific continuum. Automatically has the same
        length as r_gauss, since it has the same shape as the equilibrium fields used
        to calculate the continua. Can be complex, but only the resonance with the real
        part is calculated.
    r_gauss : numpy.ndarray
        Array containing the grid on which equilibrium fields are defined.
    sigma : complex
        An eigenvalue solution of the generalized eigenvalue problem.

    Returns
    -------
    r_inv : None, float
        The location where there is resonance between the eigenmode and the continuum.
        Returns None if there is no resonance with the specified continuum.
    """

    diff = np.sign(np.real(cont) - np.real(sigma))

    if len(np.unique(diff)) < 2:
        # There is no sign change, value is not contained in array.
        return None

    for i in range(1, len(diff) - 1):
        if diff[i] * diff[i - 1] < 0:
            # Linear interpolation between the points where the sign change occurs.
            r_inv = (np.real(sigma) - np.real(cont[i - 1])) / (
                np.real(cont[i]) - np.real(cont[i - 1])
            ) * (r_gauss[i] - r_gauss[i - 1]) + r_gauss[i - 1]
            return r_inv
        elif diff[i] * diff[i - 1] == 0:
            # The exact same value is in the continuum array, return it.
            return r_gauss[i]
