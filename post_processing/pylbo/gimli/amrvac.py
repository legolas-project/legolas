import numpy as np
import sympy as sp
from scipy.io import FortranFile
from sympy.printing.fortran import fcode
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, dblquad, tplquad
from numpy.polynomial.polynomial import Polynomial

from pylbo.utilities.datfiles.file_loader import load
from pylbo.utilities.logger import pylboLogger
from pylbo.gimli.utils import (
    create_file,
    write_pad,
    get_equilibrium_parameters,
    is_sympy_number,
    validate_output_dir,
)


def write_equilibrium_functions(file, eq):
    """
    Iterates over all equilibrium quantities and writes them to the MPI-AMRVAC user module.

    Parameters
    ----------
    file : file
        The file object to write to.
    equilibrium : Equilibrium
        The equilibrium object containing the user-defined equilibrium functions.
    """
    translation = eq.variables.fkey
    translation["x_v"] = "w(ixI^S, 1)"
    xv = sp.Symbol('x_v')
    varlist = {
        "rho_": (eq.rho0).subs(eq.variables.x, xv),
        "mom(1)": None,
        "mom(2)": (eq.v02).subs(eq.variables.x, xv),
        "mom(3)": (eq.v03).subs(eq.variables.x, xv),
        "p_": (eq.rho0 * eq.T0).subs(eq.variables.x, xv),
        "mag(1)": None,
        "mag(2)": (eq.B02).subs(eq.variables.x, xv),
        "mag(3)": (eq.B03).subs(eq.variables.x, xv),
    }
    for key in list(varlist.keys()):
        expr = varlist[key]
        if expr is None:
            write_pad(file, f"w(ixI^S, {key}) = 0.0d0", 2)
        elif is_sympy_number(expr):
            write_pad(
                file,
                fcode(sp.sympify(float(expr)), assign_to=f"w(ixI^S, {key})").lstrip(),
                2
            )
        else:
            func = fcode(expr, assign_to=f"w(ixI^S, {key})").lstrip()
            for key in list(translation.keys()):
                func = func.replace(key, translation[key])
            func = func.replace("\n", " &\n")
            func = func.replace("@", "")
            write_pad(file, func, 2)
    return


class Amrvac:
    """
    Class to prepare Legolas data for use in MPI-AMRVAC (https://amrvac.org).

    Parameters
    ----------
    config : dict
        The configuration dictionary detailing everything needed for the desired functionalities.
    """

    def __init__(self, config):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """
        Validates the presence and value of `physics_type` in the configuration
        dictionary.

        Raises
        ------
        KeyError
            If `physics_type` is missing.
        ValueError
            If `physics_type` is invalid.
        """
        if "physics_type" not in self.config.keys():
            raise KeyError('"physics_type" ("hd" / "mhd") not specified.')
        elif self.config["physics_type"] == "mhd":
            self.ef_list = ["rho", "v1", "v2", "v3", "p", "B1", "B2", "B3"]
            self.eq_list = [
                "rho0",
                "v01",
                "v02",
                "v03",
                "rho0 * T0",
                "B01",
                "B02",
                "B03",
            ]
            self.units = [
                "unit_length",
                "unit_numberdensity",
                "unit_temperature",
                "unit_density",
                "unit_pressure",
                "unit_velocity",
                "unit_magneticfield",
                "unit_time",
            ]
        elif self.config["physics_type"] == "hd":
            self.ef_list = ["rho", "v1", "v2", "v3", "p"]
            self.eq_list = ["rho0", "v01", "v02", "v03", "rho0 * T0"]
            self.units = [
                "unit_length",
                "unit_numberdensity",
                "unit_temperature",
                "unit_density",
                "unit_pressure",
                "unit_velocity",
                "unit_time",
            ]
        else:
            raise ValueError("Unknown physics type.")

        if "u1_bounds" in self.config.keys():
            assert self.config["u1_bounds"][0] < self.config["u1_bounds"][1]
        if "u2_bounds" in self.config.keys():
            assert self.config["u2_bounds"][0] < self.config["u2_bounds"][1]
        if "u3_bounds" in self.config.keys():
            assert self.config["u3_bounds"][0] < self.config["u3_bounds"][1]

        return

    def _validate_datfile(self):
        """
        Validates whether a valid Legolas data file was specified in the configuration.
        Further checks whether all necessary parameters are present in the configuration
        to prepare Legolas data for use with MPI-AMRVAC.

        Raises
        ------
        AssertionError
            If the length of `weights` is not equal to the number of eigenvalues or if
            the elements of the weights do not add up to 1; if `ef_factor` does not have
            modulus 1; if `norm_range` does not have length 2; if `norm_range`'s first
            element is larger than the second.
        KeyError
            If no datfile is specified; if no initial guess for the eigenvalue is
            specified.
        TypeError
            If `ev_guess` is not a single float/complex number or a list/NumPy array of
            float/complex numbers; if `weights` is not a list or NumPy array; if
            `ef_factor` is not a list with length equal to the number of eigenvalues, or
            an integer, float, or complex number; if `quantity` is not a string; if
            `percentage` is not a float; if `norm_range` is not a NumPy array.
        ValueError
            If `quantity` is not in the list of equilibrium quantities.
        Exception
            If the datfile is invalid.
        """
        if "datfile" not in self.config.keys():
            raise KeyError("No datfile specified.")
        else:
            try:
                self.ds = load(self.config["datfile"])
            except Exception:
                pylboLogger.error("Invalid datfile specified.")

        if "ev_guess" not in self.config.keys():
            raise KeyError("Initial guess for eigenvalue not specified.")
        elif not isinstance(
            self.config["ev_guess"], (float, complex, list, np.ndarray)
        ):
            raise TypeError(
                '"ev_guess" must be a single float/complex number or a list/NumPy array'
                " of float/complex numbers."
            )
        elif isinstance(self.config["ev_guess"], (float, complex)):
            self.config["ev_guess"] = [self.config["ev_guess"]]

        if "weights" in self.config.keys():
            if len(self.config["ev_guess"]) > 1 and not isinstance(
                self.config["weights"], (list, np.ndarray)
            ):
                raise TypeError(
                    '"weights" must be a list with length equal to the number of'
                    " eigenvalues and elements adding up to 1."
                )
            elif len(self.config["ev_guess"]) != len(self.config["weights"]):
                raise AssertionError(
                    'Length of "weights" must be equal to the number of eigenvalues.'
                )
            elif abs(np.sum(self.config["weights"])) > 1e-12:
                raise AssertionError('Elements of "weights" must add up to 1.')
        else:
            pylboLogger.warning('No "weights" specified, defaulting to equal weights.')
            self.config["weights"] = np.ones(len(self.config["ev_guess"])) / len(
                self.config["ev_guess"]
            )

        if "ef_factor" in self.config.keys():
            if len(self.config["ev_guess"]) > 1 and not isinstance(
                self.config["ef_factor"], (list, np.ndarray)
            ):
                raise TypeError(
                    '"ef_factor" must be a list with length equal to the number of'
                    " eigenvalues."
                )
            elif not isinstance(self.config["ef_factor"], (float, int, complex)):
                raise TypeError(
                    '"ef_factor" must be an integer, a float, or a complex number.'
                )
            elif abs(self.config["ef_factor"] - 1) > 1e-12:
                raise AssertionError('"ef_factor" must have modulus 1.')
            else:
                self.config["ef_factor"] = [self.config["ef_factor"]]
        else:
            pylboLogger.warning(
                'No "ef_factor" specified, defaulting to 1 for all eigenvalues.'
            )
            self.config["ef_factor"] = np.ones(len(self.config["ev_guess"]))

        if "quantity" not in self.config.keys():
            pylboLogger.warning(
                'No "quantity" specified for normalisation, defaulting to "rho0".'
            )
            self.config["quantity"] = "rho0"
        elif not isinstance(self.config["quantity"], str):
            raise TypeError('"quantity" must be a string.')
        elif self.config["quantity"] not in self.eq_list:
            raise ValueError(f'Unknown quantity "{self.config["quantity"]}" specified.')

        if "percentage" not in self.config.keys():
            pylboLogger.warning('No "percentage" specified, defaulting to 0.01.')
            self.config["percentage"] = 0.01
        elif not isinstance(self.config["percentage"], float):
            raise TypeError('"percentage" must be a float.')

        if "norm_range" in self.config.keys():
            if not isinstance(self.config["norm_range"], (list, np.ndarray)):
                raise TypeError('"norm_range" must be a list or NumPy array.')
            elif len(self.config["norm_range"]) != 2:
                raise AssertionError('"norm_range" must have length 2.')
            elif self.config["norm_range"][0] >= self.config["norm_range"][1]:
                raise AssertionError(
                    'First element of "norm_range" must be smaller than the second'
                    " element."
                )

        if "energy_norm" not in self.config.keys():
            self.config["energy_norm"] = False
        else:
            assert isinstance(self.config["energy_norm"], bool)
            if self.config["energy_norm"]:
                if "dim" not in self.config.keys():
                    raise KeyError("Must specify 'dim' when using energy to scale.")
                elif not isinstance(self.config["dim"], (int, float)):
                    raise TypeError("'dim' should be an integer or a float.")
                elif self.config["dim"] < 1 or self.config["dim"] > 3:
                    raise ValueError("'dim' must lie between 1 and 3.")
                if "u1_bounds" not in self.config.keys():
                    raise KeyError("Must specify 'u1_bounds' when scaling with energy.")
                elif self.config["dim"] == 3:
                    if "u2_bounds" not in self.config.keys():
                        raise KeyError("Must specify 'u2_bounds'.")
                    if "u3_bounds" not in self.config.keys():
                        raise KeyError("Must specify 'u3_bounds'.")
                elif self.config["dim"] >= 2:
                    if not (
                        "u2_bounds" in self.config.keys()
                        or "u3_bounds" in self.config.keys()
                    ):
                        raise KeyError("Must specify bounds matching largest k-value.")

        return
    
    def _validate_config_for_mod_usr(self):
        """
        Validates whether the configuration dictionary contains all the arguments to
        generate a mod_usr.t file for use with MPI-AMRVAC.
        """
        if "geometry" not in self.config.keys():
            raise KeyError("Geometry (Cartesian / cylindrical) not specified.")
        elif not isinstance(self.config["geometry"], str):
            raise TypeError(
                "'geometry' must be a string ('Cartesian' / 'cylindrical')."
            )
        elif self.config["geometry"].lower() == "cartesian":
            self.config["geometry"] = "Cartesian"
        elif self.config["geometry"].lower() == "cylindrical":
            self.config["geometry"] = "cylindrical"
        else:
            raise ValueError("'geometry' must be 'Cartesian' or 'cylindrical'.")
        
        if "dim" not in self.config.keys():
            raise KeyError("'dim' required to setup MPI-AMRVAC files.")
        elif not isinstance(self.config["dim"], (int, float)):
            raise TypeError("'dim' must be an integer or float.")
        elif self.config["dim"] not in [2, 2.5, 3]:
            raise ValueError("Specified dimenisionality not supported (2, 2.5, 3).")
        
        if "ldatfile" not in self.config.keys():
            raise KeyError("'ldatfile' not specified.")
        elif not isinstance(self.config["ldatfile"], str):
            raise TypeError("'ldatfile' must be a string.")
        elif len(self.config["ldatfile"]) > 5:
            if self.config["ldatfile"][-5:] == ".ldat":
                self.config["ldatfile"] = self.config["ldatfile"][:-5]

        if "parameters" not in self.config.keys():
            raise KeyError("'parameters' (including k2 and k3) not specified.")
        elif not isinstance(self.config["parameters"], dict):
            raise TypeError("'parameters' must be a dictionary.")
        elif not ("k2" in self.config["parameters"].keys() and "k3" in self.config["parameters"].keys()):
            raise KeyError("'parameters' must contain 'k2' and 'k3'.")
        else:
            for key in self.config["parameters"].keys():
                if not isinstance(self.config["parameters"][key], (int, float)):
                    raise TypeError(f"Parameter {key} must be an integer or float.")
        return

    def _get_combined_perturbation(self, ef):
        """
        Takes Legolas's perturbations of different eigenvalues and adds them up to a
        single perturbation.

        Parameters
        ----------
        ef : str
            The eigenfunction to combine.

        Returns
        -------
        np.ndarray
            The combined perturbation.
        """
        ef_data = self.ds.get_eigenfunctions(ev_guesses=self.config["ev_guess"])
        perturbation = np.zeros(self.ds.ef_gridpoints, dtype=np.complex128)
        for ii in range(len(ef_data)):
            fac = self.config["ef_factor"][ii]
            w = self.config["weights"][ii]
            raw = ef_data[ii][ef]
            scaling = ef_data[ii][self.config["quantity"].replace("0", "")]
            perturbation += w * fac * (raw / np.nanmax(np.abs(scaling)))
        return perturbation

    def _get_total_perturbation(self, ef_type):
        """
        Combines the perturbations of different eigenvalues into a single perturbation.
        Derives the pressure perturbation from the density and temperature
        perturbations.

        Parameters
        ----------
        ef_type : str
            The eigenfunction to calculate.

        Returns
        -------
        np.ndarray
            The total perturbation.
        """
        if ef_type == "p":
            rho1 = self._get_combined_perturbation("rho")
            T1 = self._get_combined_perturbation("T")
            rho0 = np.interp(
                self.ds.ef_grid, self.ds.grid_gauss, self.ds.equilibria["rho0"]
            )
            T0 = np.interp(
                self.ds.ef_grid, self.ds.grid_gauss, self.ds.equilibria["T0"]
            )
            perturbation = rho1 * T0 + rho0 * T1
        else:
            perturbation = self._get_combined_perturbation(ef_type)
        return perturbation

    def _integrate_energy_term(self, array, order):
        k2 = self.ds.parameters["k2"]
        k3 = self.ds.parameters["k3"]

        interp_r = CubicSpline(self.ds.ef_grid, array.real)
        interp_i = CubicSpline(self.ds.ef_grid, array.imag)

        if self.config["dim"] == 3:

            def integrand(u3, u2, u1):
                value = (
                    (interp_r(u1) + 1j * interp_i(u1))
                    * np.exp(1j * order * (k2 * u2 + k3 * u3))
                ).real
                return value

            integral = tplquad(
                integrand,
                *self.config["u1_bounds"],
                *self.config["u2_bounds"],
                *self.config["u3_bounds"],
            )
        elif self.config["dim"] >= 2:
            kvec = np.array([k2, k3])
            arg = np.argmax(abs(kvec))

            def integrand(u2, u1):
                value = (
                    (interp_r(u1) + 1j * interp_i(u1))
                    * np.exp(1j * order * (kvec[arg] * u2))
                ).real
                return value

            integral = dblquad(
                integrand,
                *self.config["u1_bounds"],
                *self.config[f"u{int(arg+2)}_bounds"],
            )
        elif self.config["dim"] >= 1:

            def integrand(u1):
                return interp_r(u1) + 1j * interp_i(u1)

            integral = quad(integrand, *self.config["u1_bounds"])

        return integral[0]

    def _get_ef_normalisation(self):
        """
        Normalises the perturbation of the specified quantity by the maximum background
        value.

        Returns
        -------
        float
            The normalisation factor.
        """
        ef_match = self.config["quantity"].replace("0", "")
        max_bg = np.nanmax(np.abs(self.ds.equilibria[self.config["quantity"]]))
        perturbation = self._get_total_perturbation(ef_match)
        if np.nanmax(np.abs(perturbation)) < 1e-10:
            raise AssertionError(
                f"{self.config['quantity']} is not perturbed by the specified mode(s)."
                " Select another quantity, please."
            )
        else:
            if "norm_range" in self.config.keys():
                idx1 = np.where(self.ds.ef_grid > self.config["norm_range"][0])[0][0]
                idx2 = np.where(self.ds.ef_grid > self.config["norm_range"][1])[0][0]
                perturbation = perturbation[idx1:idx2]
            norm = self.config["percentage"] * max_bg / np.nanmax(np.abs(perturbation))
        return norm

    def _get_energy_normalisation(self):
        """
        Normalises the perturbation eigenfunctions by the energy.

        Returns
        -------
        float
            The normalisation factor.
        """
        u1 = self.ds.ef_grid
        u1_gauss = self.ds.grid_gauss
        gamma_1 = self.ds.gamma - 1

        eq_list = self.eq_list
        idx = eq_list.index("rho0 * T0")
        eq_list[idx] = "T0"
        ef_list = self.ef_list
        idx = ef_list.index("p")
        ef_list[idx] = "T"

        eq = {}
        for key in eq_list:
            eq[key] = np.interp(u1, u1_gauss, self.ds.equilibria[key])
        efs = {}
        for key in ef_list:
            efs[key] = self._get_total_perturbation(key)

        e0 = (
            eq["rho0"] * (eq["v01"] ** 2 + eq["v02"] ** 2 + eq["v03"] ** 2) / 2
            + eq["rho0"] * eq["T0"] / gamma_1
        )
        e1 = (
            efs["rho"] * (eq["v01"] ** 2 + eq["v02"] ** 2 + eq["v03"] ** 2) / 2
            + eq["rho0"]
            * (eq["v01"] * efs["v1"] + eq["v02"] * efs["v2"] + eq["v03"] * efs["v3"])
            + (eq["rho0"] * efs["T"] + efs["rho"] * eq["T0"]) / gamma_1
        )
        e2 = (
            eq["rho0"] * (efs["v1"] ** 2 + efs["v2"] ** 2 + efs["v3"] ** 2) / 2
            + efs["rho"]
            * (eq["v01"] * efs["v1"] + eq["v02"] * efs["v2"] + eq["v03"] * efs["v3"])
            + efs["rho"] * efs["T"] / gamma_1
        )
        e3 = efs["rho"] * (efs["v1"] ** 2 + efs["v2"] ** 2 + efs["v3"] ** 2) / 2

        if self.config["physics_type"] == "mhd":
            e0 += (eq["B01"] ** 2 + eq["B02"] ** 2 + eq["B03"] ** 2) / 2
            e1 += eq["B01"] * efs["B1"] + eq["B02"] * efs["B2"] + eq["B03"] * efs["B3"]
            e2 += (efs["B1"] ** 2 + efs["B2"] ** 2 + efs["B3"] ** 2) / 2

        coeff0 = self._integrate_energy_term(e0, 0)
        coeff1 = self._integrate_energy_term(e1, 1)
        coeff2 = self._integrate_energy_term(e2, 2)
        coeff3 = self._integrate_energy_term(e3, 3)

        p = Polynomial([-self.config["percentage"] * coeff0, coeff1, coeff2, coeff3])
        roots = p.roots()
        index = np.argmin(abs(roots))
        norm = roots[index]
        if abs(norm) > 1:
            pylboLogger.warning(
                "Normalization factor larger than 1. Perturbation may "
                "be larger than or comparable to equilibrium quantity."
            )
        else:
            pylboLogger.info(f"Normalization factor = {norm}")
        return norm

    def _get_normalisation(self):
        """
        Selects which procedure to follow for the normalisation.

        Returns
        -------
        float
            The normalisation factor.
        """
        if self.config["energy_norm"]:
            norm = self._get_energy_normalisation()
        else:
            norm = self._get_ef_normalisation()
        return norm

    def prepare_legolas_data(self, name=None, loc=None):
        """
        Prepares a file (.ldat) from the Legolas data for use with MPI-AMRVAC.

        Parameters
        ----------
        name : str
            Name of the .ldat file
        loc : str, ~os.PathLike
            Path to the directory where the .ldat file will be stored. Default is the
            current directory.

        Raises
        ------
        ValueError
            If the datfile is invalid.

        Examples
        --------
        >>> from pylbo.gimli import Amrvac
        >>> amrvac_config = {
        >>>     "datfile": "./datfile.dat",
        >>>     "physics_type": "mhd",
        >>>     "ev_guess": [-0.1, 0.1],
        >>>     "percentage": 0.01,
        >>>     "quantity": "rho0"
        >>> }
        >>> amrvac = gimli.Amrvac(amrvac_config)
        >>> amrvac.prepare_legolas_data()
        """
        loc = validate_output_dir(loc)
        self._validate_datfile()
        datfile = self.config["datfile"]
        if name is None:
            file = str(datfile).rsplit("/")[-1]
            name = file[:-4]
        self.config["ldatfile"] = name
        f = FortranFile(loc + "/" + name + ".ldat", "w")
        f.write_record(np.array([self.ds.ef_gridpoints], dtype=np.int32))
        f.write_record(
            np.array(
                [self.ds.parameters["k2"], self.ds.parameters["k3"]], dtype=np.float64
            )
        )
        f.write_record(self.ds.ef_grid)

        norm = self._get_normalisation()
        for ix in range(len(self.ef_list)):
            pert = self._get_total_perturbation(self.ef_list[ix]) * norm
            f.write_record(pert)

        u = []
        for ix in range(len(self.units)):
            u.append(self.ds.units[self.units[ix]])
        f.write_record(np.array(u, dtype=np.float64))

        f.close()
        return name

    def user_module(self, filename="mod_usr", loc=None):
        """
        Writes the user module for MPI-AMRVAC.

        Parameters
        ----------
        filename : str
            Name of the user module file, defaults to mod_usr
        loc : str, ~os.PathLike
            Path to the directory where the user module will be stored. Default is the
            current directory.
        """
        self._validate_config_for_mod_usr()
        quantities = ["density", "v1", "v2", "v3", "pressure"]
        keyring = ["rho_", "mom(1)", "mom(2)", "mom(3)", "p_"]
        if self.config["physics_type"] == "mhd":
            quantities = quantities + ["B1", "B2", "B3"]
            keyring = keyring + ["mag(1)", "mag(2)", "mag(3)"]

        loc = validate_output_dir(loc)
        path = loc + "/" + filename + ".t"
        create_file(path)
        file = open(path, "a")

        write_pad(file, "!> User module for perturbation with Legolas eigenmodes.", 0)
        write_pad(file, "!! Generated with GIMLI.", 0)
        write_pad(file, "module mod_usr", 0)
        write_pad(file, "use, intrinsic :: iso_fortran_env", 1)
        write_pad(file, f"use mod_{self.config["physics_type"]}", 1)
        write_pad(file, "use mod_global_parameters", 1)
        write_pad(file, "implicit none", 1)
        file.write("\n")
        write_pad(
            file,
            f"character(len=100), parameter :: legolas_file = '{self.config["ldatfile"]+".ldat"}'",
            1
        )
        file.write("\n")
        write_pad(file, "integer, parameter :: dp = real64", 1)
        write_pad(file, "integer, parameter :: file_id = 123", 1)
        file.write("\n")
        eqparam = get_equilibrium_parameters(self.config)
        write_pad(file, f"real(dp) :: {eqparam}", 1)
        file.write("\n")
        write_pad(file, "complex(dp), parameter :: ic = (0.0d0, 1.0d0)", 1)
        file.write("\n")
        write_pad(file, "integer :: ef_gridpts", 1)
        write_pad(file, "real(dp) :: k2, k3", 1)
        write_pad(file, "real(dp), allocatable :: ef_grid(:)", 1)
        for ii in range(len(quantities)):
            write_pad(file, f"complex(dp), allocatable :: {quantities[ii]}(:)", 1)
        file.write("\n")

        write_pad(file, "contains", 0)
        file.write("\n")

        write_pad(file, "subroutine usr_init()", 1)
        write_pad(
            file,
            f"call set_coordinate_system('{self.config["geometry"]}_{self.config["dim"]}D')",
            2
        )
        write_pad(file, "call read_legolas_data()", 2)
        file.write("\n")
        write_pad(file, "usr_set_parameters => initialise_constants", 2)
        write_pad(file, "usr_init_one_grid  => initialise_grid", 2)
        file.write("\n")
        write_pad(file, f"call {self.config["physics_type"]}_activate()", 2)
        write_pad(file, "end subroutine usr_init", 1)
        file.write("\n")

        write_pad(file, "subroutine initialise_constants()", 1)
        for key in eqparam.split(", "):
            write_pad(file, f"{key} = {self.config["parameters"][key]}", 2)
        write_pad(file, "end subroutine initialise_constants", 1)
        file.write("\n")

        write_pad(file, "subroutine initialise_grid(ixI^L, ixO^L, w, x)", 1)
        write_pad(file, "integer, intent(in)     :: ixI^L, ixO^L", 2)
        write_pad(file, "real(dp), intent(in)    :: x(ixI^S, ndim)", 2)
        write_pad(file, "real(dp), intent(inout) :: w(ixI^S, nw)", 2)
        file.write("\n")

        write_equilibrium_functions(file, self.config["equilibrium"])
        file.write("\n")

        for key in keyring:
            write_pad(
                file,
                f"call add_perturbation_to_w_array(ixI^L, ixO^L, w, {key}, x)",
                2
            )
        file.write("\n")

        write_pad(
            file,
            f"call {self.config["physics_type"]}_to_conserved(ixI^L, ixO^L, w, x)",
            2
        )
        write_pad(file, "end subroutine initialise_grid", 1)
        file.write("\n")

        write_pad(file, "subroutine read_legolas_data()", 1)
        write_pad(file, "open( &", 2)
        write_pad(file, "unit=file_id+mype, &", 3)
        write_pad(file, "file=legolas_file, &", 3)
        write_pad(file, "form='unformatted' &", 3)
        write_pad(file, ")", 2)
        file.write("\n")

        write_pad(file, "read(file_id+mype) ef_gridpts", 2)
        write_pad(file, "read(file_id+mype) k2, k3", 2)
        file.write("\n")

        write_pad(file, "call allocate_arrays(ef_gridpts)", 2)
        write_pad(file, "read(file_id+mype) ef_grid", 2)
        for ii in range(len(quantities)):
            write_pad(file, f"read(file_id+mype) {quantities[ii]}", 2)
        file.write("\n")

        write_pad(
            file,
            "read(file_id+mype) unit_length, unit_numberdensity, unit_temperature, &",
            2
        )
        if self.config["physics_type"] == "mhd":
            write_pad(
                file,
                "unit_density, unit_pressure, unit_velocity, unit_magneticfield, unit_time",
                3
            )
        else:
            write_pad(
                file,
                "unit_density, unit_pressure, unit_velocity, unit_time",
                3
            )
        file.write("\n")

        write_pad(file, "close(file_id+mype)", 2)
        write_pad(file, "end subroutine read_legolas_data", 1)
        file.write("\n")

        write_pad(file, "subroutine allocate_arrays(gridpts)", 1)
        write_pad(file, "integer, intent(in) :: gridpts", 2)
        file.write("\n")
        write_pad(file, "allocate(ef_grid(gridpts))", 2)
        write_pad(file, "allocate(density(gridpts))", 2)
        write_pad(file, f"allocate({", ".join(quantities[1:])}, mold=density)", 2)
        write_pad(file, "end subroutine allocate_arrays", 1)
        file.write("\n")

        write_pad(
            file,
            "subroutine add_perturbation_to_w_array(ixI^L, ixO^L, w, w_index, x)",
            1    
        )
        write_pad(file, "integer, intent(in)     :: ixI^L, ixO^L", 2)
        write_pad(file, "real(dp), intent(inout) :: w(ixI^S, nw)", 2)
        write_pad(file, "integer, intent(in)     :: w_index", 2)
        write_pad(file, "real(dp), intent(in)    :: x(ixI^S, ndim)", 2)
        write_pad(
            file,
            "complex(dp) :: amplitude, exp_factor, quantity, values(ef_gridpts)",
            2
        )
        write_pad(file, "integer  :: ix^D, ib^D", 2)
        write_pad(file, "real(dp) :: k1 = 0.0d0", 2)
        file.write("\n")
        write_pad(file, "call w_index_to_array(w_index, values)", 2)
        file.write("\n")
        write_pad(file, "{ib^D = ixImax^D - ixImin^D + 1|\\}", 2)
        write_pad(file, "{do ix^D = 1, ib^D|\\}", 2)
        write_pad(file, "exp_factor = {exp(ic * k^D * x({ix^D}, ^D))|*}", 3)
        write_pad(
            file,
            "call ef_amplitude(x(ix^D, 1), ef_grid, values, amplitude)",
            3
        )
        write_pad(file, "quantity = amplitude * exp_factor", 3)
        write_pad(file, "w(ix^D, w_index) = w(ix^D, w_index) + realpart(quantity)", 3)
        write_pad(file, "{end do|\\}", 2)
        write_pad(file, "end subroutine add_perturbation_to_w_array", 1)
        file.write("\n")

        write_pad(file, "subroutine w_index_to_array(w_index, array)", 1)
        write_pad(file, "integer, intent(in) :: w_index", 2)
        write_pad(file, "complex(dp), intent(inout) :: array(ef_gridpts)", 2)
        file.write("\n")
        write_pad(file, f"if (w_index == {keyring[0]}) then", 2)
        write_pad(file, f"array = {quantities[0]}", 2)
        for ii in range(1, len(quantities)):
            write_pad(file, f"else if (w_index == {keyring[ii]}) then", 2)
            write_pad(file, f"array = {quantities[ii]}", 3)
        write_pad(file, "end if", 2)
        write_pad(file, "end subroutine w_index_to_array", 1)
        file.write("\n")

        write_pad(file, "subroutine ef_amplitude(x, grid, array, amplitude)", 1)
        write_pad(file, "real(dp), intent(in) :: x, grid(ef_gridpts)", 2)
        write_pad(file, "complex(dp), intent(in) :: array(ef_gridpts)", 2)
        write_pad(file, "complex(dp), intent(out) :: amplitude", 2)
        write_pad(file, "integer :: idl, idu", 2)
        file.write("\n")
        write_pad(file, "if (x <= grid(1)) then", 2)
        write_pad(file, "amplitude = array(1)", 3)
        write_pad(file, "else if (x >= grid(size(grid))) then", 2)
        write_pad(file, "amplitude = array(size(grid))", 3)
        write_pad(file, "else", 2)
        write_pad(file, "idl = maxloc(grid, mask=(grid < x), dim=1)", 3)
        write_pad(file, "idu = minloc(grid, mask=(grid > x), dim=1)", 3)
        write_pad(file, "amplitude = array(idl) + (x - grid(idl)) * &", 3)
        write_pad(file, "(array(idu) - array(idl)) / (grid(idu) - grid(idl))", 4)
        write_pad(file, "end if", 2)
        write_pad(file, "end subroutine ef_amplitude", 1)

        write_pad(file, "end module", 0)
        file.write("!")
        file.close()
        return
