from __future__ import annotations

import copy
from pathlib import Path
from typing import Union

import f90nml
from pylbo.automation.defaults import legolas_namelist_items, amrvac_namelist_items
from pylbo.exceptions import ParfileGenerationError
from pylbo.utilities.logger import pylboLogger
from pylbo.utilities.toolbox import transform_to_list

KEYS_EXPECTED_AS_LIST = ["basis_functions", "w_convert_factor", "w_write", "writelevel", "writespshift", "flux_scheme", "limiter", "loglimit", "entropycoef", "small_values_fix_iw", "typeboundary_min1", "typeboundary_max1", "typeboundary_min2", "typeboundary_max2", "typeboundary_min3", "typeboundary_max3", "amr_wavefilter", "refine_threshold", "derefine_ratio", "w_refine_weight", "logflag", "stretch_dim", "boundary_divbfix"]


def _ensure_nb_names_and_nb_runs_matches(
    names: Union[str, list[str]], nb_runs: int
) -> list[str]:
    """
    Ensures that the number of names matches the number of runs.

    Parameters
    ----------
    names : str, list[str]
        The basename(s) for the parfile(s).
    nb_runs : int
        Number of runs for which the parfiles are generated.

    Returns
    -------
    names : list[str]
        The basename(s) for the parfile(s).

    """
    names = transform_to_list(names)
    if len(names) == 1:
        names = names * nb_runs
    elif len(names) != nb_runs:
        raise ValueError(
            f"Number of parfile names ({len(names)}) does not match number of runs "
            f"({nb_runs})."
        )
    return names


def _validate_basenames(basenames: list[str]) -> list[str]:
    """
    Validates the basenames for given parfiles.

    Parameters
    ----------
    basename : list[str]
        The basenames for the parfiles. If not given, defaults to "parfile[i]".

    Returns
    -------
    basename : list[str]
        The basename for a parfile.

    """
    if len(basenames) == 1:
        return ["parfile"] if basenames[0] is None else basenames
    for i, name in enumerate(basenames):
        basenames[i] = f"parfile{i + 1:04d}" if name is None else name
    return basenames


def _validate_output_dir(output_dir, subdir):
    """
    Validates and returns the output directory for the parfiles.

    Parameters
    ----------
    output_dir : str, ~os.PathLike
        The output directory to store the parfiles in. If not given, defaults to
        the current working directory.
    subdir : boolean
        If `.true.`, creates a subdirectory called `parfiles`.

    Raises
    ------
    NotADirectoryError
        If the output directory is not found.

    Returns
    -------
    output : ~os.PathLike
        The resolved path to the output directory with "parfiles" appended.

    """
    if output_dir is None:
        output_dir = Path.cwd()
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise NotADirectoryError(output)
    if subdir:
        output = (output / "parfiles").resolve()
        if not output.is_dir():
            Path.mkdir(output)
    return output


def _handle_expected_item_list(item):
    # check that item is a list containing additional lists
    if not isinstance(item, list):
        raise TypeError(f"Expected item {item} to be a list of lists.")
    for obj in item:
        if not isinstance(obj, list):
            raise TypeError(f"Expected item '{obj}' in {item} to be a list itself.")
    return item


class ParfileGenerator:
    """
    Handles parfile generation.

    Parameters
    ----------
    parfile_dict : dict
        Dictionary containing the keys to be placed in the parfile.
    basename : str, list[str]
        The basename for the parfile, the `.par` suffix is added automatically and is
        not needed. If not provided, the basename will default to `parfile`.
        Can be a list of names as well if multiple parfiles are being generated.
    output_dir : str, ~os.PathLike
        Output directory where the parfiles are saved, defaults to the current
        working directory if not specified.
    subdir : boolean
        If `True` (default), creates a subdirectory under `output_dir` called
         `parfiles` in which the parfiles will be saved.
    prefix_numbers : boolean
        If `True` prepends the `basename` by a n-digit number (e.g. xxxxmyparfile.par).
        The number of digits is specified by `nb_prefix_digits`.
    nb_prefix_digits : int
        Number of digits to prepend to the `basename` if `prefix_numbers` is `True`.
        Defaults to 4.
    code : str
        Code for which to generate a parfile. Defaults to 'legolas'.
    """

    def __init__(
        self,
        parfile_dict,
        basename=None,
        output_dir=None,
        subdir=True,
        prefix_numbers=True,
        nb_prefix_digits=4,
        code="legolas",
    ):
        self.parfile_dict = copy.deepcopy(parfile_dict)
        self.nb_runs = self.parfile_dict.pop("number_of_runs", 1)

        names = _ensure_nb_names_and_nb_runs_matches(basename, self.nb_runs)
        self.basenames = _validate_basenames(transform_to_list(names))
        self.output_dir = _validate_output_dir(output_dir, subdir)
        self._use_prefix = prefix_numbers
        self._nb_prefix_digits = nb_prefix_digits
        self.parfiles = []
        self.container = {}
        self.code = code.lower()

    def _get_and_check_item(self, namelist, name, allowed_dtypes):
        """
        Does typechecking on the various dictionary keys supplied to the parfile
        generator. Pops the key from the dictionary.

        Parameters
        ----------
        namelist : str
            One of the namelists ("gridlist", "savelist", etc.)
        name : str
            The key to check.
        allowed_dtypes : class, tuple
            Allowed types for that particular key. Either a single value or a tuple.

        Raises
        ------
        TypeError
            If the types do not match, e.g. if "gridpoints" is specified as a float
            value when it should be an integer.

        Returns
        -------
        item : any
            The item popped from the dictionary corresponding to the given key.
        """
        item = self.parfile_dict.pop(name, None)
        if item is not None:
            item_aslist = (
                transform_to_list(item)
                if name not in KEYS_EXPECTED_AS_LIST
                else _handle_expected_item_list(item)
            )
            for obj in item_aslist:
                if not isinstance(obj, allowed_dtypes):
                    raise TypeError(
                        f"namelist '{namelist}' expected item '{name}' to be of "
                        f"type {allowed_dtypes} but got {type(obj)}. \n"
                        f"item value = {item}"
                    )
        return item

    def create_namelist_from_dict(self):
        """
        Creates one major namelist from the given dictionary.

        Raises
        ------
        ParfileGenerationError
            - If the original dictionary is not empty after everything should be popped
            - If there is an inconsistency between array sizes of dictionary items
        """
        if self.code == "legolas":
            namelist_items = legolas_namelist_items
        elif self.code == "amrvac":
            namelist_items = amrvac_namelist_items
        else:
            raise ValueError("Code for parfile generation not recognized.")

        for namelist, items in namelist_items.items():
            # update container
            self.container.update({namelist: {}})
            # loop over names for this specific namelist
            for name, dtypes in items:
                obj = self._get_and_check_item(namelist, name, dtypes)
                if obj is not None:
                    obj = transform_to_list(obj)
                    self.container[namelist].update({name: obj})
            # if this namelist is still empty, remove it
            if self.container[namelist] == {}:
                self.container.pop(namelist)
        # account for parameters separately
        params = self.parfile_dict.pop("parameters", {})
        if len(params) != 0:
            self.container["equilibriumlist"].update({"use_defaults": [False]})
            for name, param in params.items():
                params.update({name: transform_to_list(param)})
            self.container.update({"paramlist": params})

        # here the original dictionary should be empty,
        # something went wrong if it isn't
        if len(self.parfile_dict) != 0:
            raise ParfileGenerationError(self.parfile_dict)

        # update container for number of runs, all items are lists
        for namelist, items in self.container.items():
            for key, values in items.items():
                if len(values) == 1:
                    values_list = values * self.nb_runs
                elif len(values) == self.nb_runs:
                    values_list = values
                else:
                    raise ParfileGenerationError(items, self.nb_runs, key)
                # make sure that sigma has complex values
                if key == "sigma":
                    values_list = [complex(value) for value in values_list]
                self.container.get(namelist).update({key: values_list})

    def generate_parfiles(self):
        """
        Creates separate parfiles from the main namelist container and writes
        the individual parfiles to disk.

        Returns
        -------
        parfiles : list of str
            List containing the paths of the parfiles, can be passed to the legolas
            runner.

        """
        run_dict = {key: {} for key in self.container.keys()}
        # savelist must be present
        try:
            run_dict["savelist"]
        except KeyError:
            run_dict.update({"savelist": {}})

        for current_run in range(self.nb_runs):
            prefix = (
                f"{current_run + 1:0{self._nb_prefix_digits}d}"
                if self._use_prefix and self.nb_runs > 1
                else ""
            )
            # generate dictionary for this specific run
            for namelist, items in self.container.items():
                for key, values in items.items():
                    run_dict[namelist].update({key: values[current_run]})
            basename = self.basenames[current_run]
            parfile_name = f"{prefix}{basename}.par"
            # datfile name (no extension .dat needed)
            datfile_name = (
                f"{prefix}{run_dict['savelist'].get('basename_datfile', basename)}"
            )
            run_dict["savelist"].update({"basename_datfile": datfile_name})
            # set paths and write parfile
            parfile_path = (self.output_dir / parfile_name).resolve()
            self.parfiles.append(str(parfile_path))
            f90nml.write(run_dict, parfile_path, force=True)
            # clear dictionary but keep keys
            run_dict.update({key: {} for key in run_dict})
        pylboLogger.info(f"parfiles generated and saved to {self.output_dir}")
        return self.parfiles
