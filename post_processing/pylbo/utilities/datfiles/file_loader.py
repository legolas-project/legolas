import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
from pylbo.data_containers import LegolasDataSeries, LegolasDataSet
from pylbo.exceptions import InvalidLegolasFile
from pylbo.utilities.logger import pylboLogger
from pylbo.utilities.toolbox import transform_to_list


def _validate_file(file):
    """
    Checks the file validity of a given logfile or datfile.

    Parameters
    ----------
    file : str, ~os.PathLike
        The path to the datfile, either as a :class:`str` or
        :class:`~os.PathLike` object.

    Raises
    -------
    FileNotFoundError
        If the datfile can not be found.
    InvalidLegolasFile
        If the datfile is not a valid Legolas output file.
    """
    path_to_file = Path(file).resolve()
    if not path_to_file.is_file():
        raise FileNotFoundError(path_to_file)
    if path_to_file.suffix not in (".dat", ".log"):
        raise InvalidLegolasFile(path_to_file)


def load(datfile):
    """
    Loads a single Legolas datfile.

    Parameters
    ----------
    datfile : str, ~os.PathLike
        Path to the datfile.

    Raises
    ------
    ValueError
        If `datfile` is not a single file.

    Returns
    -------
    ds : ~pylbo.data_containers.LegolasDataSet
        A dataset instance for the given datfile.
    """
    if not isinstance(datfile, (str, os.PathLike)):
        raise ValueError("load() takes a single datfile.")
    _validate_file(datfile)
    ds = LegolasDataSet(datfile)
    pylboLogger.info(f"Legolas v{ds.legolas_version}")
    pylboLogger.info(f"file loaded : {ds.datfile.parent}/{ds.datfile.name}")
    pylboLogger.info(f"gridpoints  : {ds.gridpoints}")
    pylboLogger.info(f"geometry    : {ds.geometry} in {ds.x_start, ds.x_end}")
    pylboLogger.info(f"equilibrium : {ds.eq_type}")
    pylboLogger.info(f"state vector: {ds.header.get('state_vector', 'not present')}")
    pylboLogger.info(
        f"basis functions: {ds.header.get('basis_functions', 'not present')}"
    )
    if ds.has_matrices:
        pylboLogger.info("matrices present in datfile")
    if ds.has_eigenvectors:
        pylboLogger.info("eigenvectors present in datfile")
    if ds.has_efs:
        pylboLogger.info("eigenfunctions present in datfile")
    if ds.has_derived_efs:
        pylboLogger.info("derived eigenfunctions present in datfile")
    if ds.has_ef_subset:
        saved_efs = len(ds.header["ef_written_idxs"])
        total_efs = len(ds.eigenvalues)
        pylboLogger.info(
            f"subset saved: {saved_efs}/{total_efs} eigenvalues have eigenfunctions"
        )
    pylboLogger.info("-" * 75)
    return ds


def load_series(datfiles, sorting_par=None):
    """
    Loads multiple Legolas datfiles.

    Parameters
    ----------
    datfiles : list, numpy.ndarray
        Paths to the datfiles that should be loaded, in list/array form. Every element
        should be a string or a ~os.PathLike object.
    sorting_par : str, optional
        The parameter to sort the datfiles by.

    Raises
    ------
    ValueError
        If an empty list or array is supplied.

    Returns
    -------
    series : ~pylbo.data_containers.LegolasDataSeries
        A dataseries instance for the given datfiles.
    """
    transform_to_list(datfiles)
    if not datfiles:
        raise ValueError("load_series: supplied an empty list")
    for datfile in datfiles:
        _validate_file(datfile)
    series = LegolasDataSeries(datfiles)

    # sorting based on parameter
    if sorting_par is not None:
        datfiles_sorted = np.array(
            [
                datfile
                for _, datfile in sorted(zip(series.parameters[sorting_par], datfiles))
            ]
        )
        series = LegolasDataSeries(datfiles_sorted)

    # handle version printing
    versions = [ds.legolas_version.parse() for ds in series.datasets]
    minversion, maxversion = min(versions), max(versions)
    if minversion == maxversion:
        info_msg = str(minversion)
    else:
        info_msg = f"{minversion} --> {maxversion}"
    pylboLogger.info(f"Legolas_version  : {info_msg}")

    # handle file information printing
    names = sorted([ds.datfile.name for ds in series.datasets])
    pylboLogger.info(f"files loaded     : {names[0]} --> {names[-1]}")

    # handle gridpoints printing
    pts = [ds.gridpoints for ds in series.datasets]
    minpts, maxpts = min(pts), max(pts)
    if minpts == maxpts:
        info_msg = str(minpts)
    else:
        info_msg = f"{minpts} --> {maxpts}"
    pylboLogger.info(f"gridpoints       : {info_msg}")

    # handle geometry printing
    if not isinstance(series.geometry, str) and len(series.geometry) > 1:
        pylboLogger.warning("multiple geometries detected!")
    else:
        pylboLogger.info(f"geometries       : {series.geometry}")

    # handle equilibrium printing
    equils = set([ds.eq_type for ds in series.datasets])
    if len(equils) > 1:
        pylboLogger.error(f"multiple equilibria detected! -- {equils}")
        raise ValueError
    else:
        pylboLogger.info(f"equilibria       : {equils.pop()}")

    # check presence of matrices
    matrices_present = set(series.has_matrices)
    if len(matrices_present) > 1:
        pylboLogger.info("matrices present in some datfiles, but not all")
    else:
        if matrices_present.pop():
            pylboLogger.info("matrices present in all datfiles")

    # check presence of eigenfunctions
    efs_present = set(series.has_efs)
    if len(efs_present) > 1:
        pylboLogger.info("eigenfunctions present in some datfiles, but not all")
    else:
        if efs_present.pop():
            pylboLogger.info("eigenfunctions present in all datfiles")

    # check presence of derived eigenfunctions
    defs_present = set(series.has_derived_efs)
    if len(defs_present) == 0:
        pylboLogger.info("no derived eigenfunctions present")
    elif len(defs_present) > 1:
        pylboLogger.info("derived eigenfunctions present in some datfiles, but not all")
    else:
        if defs_present.pop():
            pylboLogger.info("derived eigenfunctions present in all datfiles")
    pylboLogger.info("-" * 75)
    return series


def load_logfile(logfile, sort=False):
    """
    Reads a Legolas log file.

    Parameters
    ----------
    logfile : str, ~os.PathLike
       The path to the logfile.
    sort : bool
       If `True`, sorts the eigenvalues in the logfile. Sorting is done first
       on the real part, then on the imaginary part.

    Returns
    -------
    eigenvalues : numpy.ndarray
       Array containing the eigenvalues from the logfile.
    """
    _validate_file(logfile)
    filepath = Path(logfile).resolve()
    eigenvalues = []
    with open(filepath, "r") as logfile:
        for line in logfile:
            line = line.strip().split(",")
            x, y = line
            eigenvalues.append(complex(float(x), float(y)))
    eigenvalues = np.asarray(eigenvalues)
    if sort:
        eigenvalues = np.sort(eigenvalues)
    return eigenvalues


def select_files():
    """
    Opens an interactive window to select files to open.
    Requires a visual interface.

    Returns
    -------
    files : list
        A list containing the paths to the files selected.
    """
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.focus_set()
    files = list(
        filedialog.askopenfilenames(parent=root, title="Select Legolas file(s)")
    )
    root.destroy()
    if not files:
        exit()
    return files
