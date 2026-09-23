import copy
import numpy as np
import sympy as sp
from scipy.io import FortranFile
from sympy.printing.fortran import fcode
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, dblquad, tplquad
from numpy.polynomial.polynomial import Polynomial

from pylbo.automation.generator import ParfileGenerator
from pylbo.utilities.datfiles.file_loader import load
from pylbo.utilities.logger import pylboLogger
from pylbo.gimli.utils import (
    create_file,
    write_pad,
    get_equilibrium_parameters,
    is_sympy_number,
    # is_symbol_dependent,
    validate_output_dir,
)
from pylbo.gimli.equilibrium import Equilibrium


def write_equilibrium_functions(file, eq, to_fetch_total, to_fetch_split):
    """
    Writes a subroutine where all equilibrium functions are defined for reuse
    in other routines. `get_equilibrium' then returns the total equilibrium functions
    plus the current if needed (if not, this part of the array is zero).

    Parameters
    ----------
    file : file
        The file object to write to.
    eq : Equilibrium
        The equilibrium object containing the user-defined equilibria.
    to_fetch_total : list
        The list of equilibrium functions to write to the file.
    to_fetch_split : list
        The list of split equilibrium functions to write to the file.
    """
    translation = eq.variables.fkey
    translation["x_v"] = "x(ixI^S, 1)"
    xv = sp.Symbol("x_v")
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
    jlist = []
    if any(idx in to_fetch_split for idx in ["j2", "j3"]):
        varlist["j1"] = None
        varlist["j2"] = (eq.J02).subs(eq.variables.x, xv)
        varlist["j3"] = (eq.J03).subs(eq.variables.x, xv)
        jlist = [idx for idx in to_fetch_split if idx[0] == "j"]

    write_pad(file, "subroutine get_equilibrium(ixI^L, ixO^L, x, equil)", 1)
    write_pad(file, "integer, intent(in)     :: ixI^L, ixO^L", 2)
    write_pad(file, "double precision, intent(in)    :: x(ixI^S, ndim)", 2)
    write_pad(file, "double precision                :: equil(ixI^S, nw+3)", 2)
    if any(idx in to_fetch_split for idx in ["j2", "j3"]):
        write_pad(file, "j1 = nw+1", 2)
        write_pad(file, "j2 = nw+2", 2)
        write_pad(file, "j3 = nw+3", 2)
    file.write("\n")
    write_pad(file, "equil(ixI^S, :) = 0.0d0", 2)
    file.write("\n")
    for key in to_fetch_total + jlist:
        expr = varlist[key]
        if expr is None:
            write_pad(file, f"equil(ixI^S, {key}) = 0.0d0", 2)
        elif is_sympy_number(expr):
            write_pad(
                file,
                fcode(
                    sp.sympify(float(expr)),
                    assign_to=f"equil(ixI^S, {key})",
                    source_format="free",
                ).lstrip(),
                2,
            )
        else:
            func = fcode(
                expr, assign_to=f"equil(ixI^S, {key})", source_format="free"
            ).lstrip()
            for key in list(translation.keys()):
                func = func.replace(key, translation[key])
            func = func.replace("@", "")
            write_pad(file, func, 2)
    write_pad(file, "end subroutine get_equilibrium", 1)
    file.write("\n")

    # writing the split functions
    if len(to_fetch_split) > 0:
        fields = []
        if any(idx in to_fetch_split for idx in ["mag(2)", "mag(3)"]):
            fields = fields + ["B0", "J0"]
        if any(idx in to_fetch_split for idx in ["rho_", "p_"]):
            fields.append("equi_vars")
        for field in fields:
            if field == "equi_vars":
                keys_temp = ["rho_", "p_"]
            elif field == "B0":
                keys_temp = ["mag(1)", "mag(2)", "mag(3)"]
            elif field == "J0":
                keys_temp = ["j1", "j2", "j3"]
            keys = [key for key in keys_temp if key in to_fetch_split]
            write_split_equilibrium_functions(file, keys, field)
    return


def write_split_equilibrium_functions(file, to_fetch, field):
    translation = {
        "mag(1)": 1,
        "mag(2)": 2,
        "mag(3)": 3,
        "j1": 1,
        "j2": 2,
        "j3": 3,
        "rho_": "equi_rho0_",
        "p_": "equi_pe0_",
    }

    write_pad(file, f"subroutine special_set_{field}(ixI^L, ixO^L, x, w0)", 1)
    write_pad(file, "integer, intent(in) :: ixI^L, ixO^L", 2)
    write_pad(file, "double precision, intent(in) :: x(ixI^S, 1:ndim)", 2)
    len_w0 = "number_equi_vars" if field == "equi_vars" else "ndir"
    write_pad(file, f"double precision, intent(inout) :: w0(ixI^S, 1:{len_w0})", 2)
    write_pad(file, "double precision                :: equil(ixI^S, nw+3)", 2)
    file.write("\n")
    write_pad(file, "call get_equilibrium(ixI^L, ixO^L, x, equil)", 2)
    for key in to_fetch:
        write_pad(file, f"w0(ixI^S, {translation[key]}) = equil(ixI^S, {key})", 2)
    write_pad(file, f"end subroutine special_set_{field}", 1)
    file.write("\n")


def write_physics_pointers(file, eq):
    vac_names = {
        "gravity": "gravity",
        "parallel_conduction": "",
        "perpendicular_conduction": "",
        "cooling": "",
        "heating": "source",
        "resistivity": "special_resistivity",
    }

    for key in vac_names.keys():
        if eq._dict_phys[key][0] is not None:
            if len(vac_names[key]) > 0:
                write_pad(file, f"usr_{vac_names[key]} => set_{key}", 2)
            else:
                pylboLogger.warning(
                    f"Automated definition of {key} is not yet implemented."
                )
    if eq.heatcool is not None:
        if eq.heatcool["force_thermal_balance"] and not eq.heatcool.get(
            "mhd_equi_thermal", False
        ):
            write_pad(file, "usr_source => set_heating", 2)

    return


def write_physics_subroutines(file, eq):
    translation = eq.variables.fkey
    translation["x_v"] = "x(ixI^S, 1)"
    translation["rho_0"] = "equil(ixI^S, rho_)"
    translation["T_0"] = "(equil(ixI^S, p_)/equil(ixI^S, rho_))"
    xv = sp.Symbol("x_v")

    if eq._dict_phys["gravity"][0] is not None:
        expr = eq._dict_phys["gravity"][0].subs(eq.variables.x, xv)
        write_pad(
            file, "subroutine set_gravity(ixI^L, ixO^L, wCT, x, gravity_field)", 1
        )
        write_pad(file, "use mod_global_parameters", 2)
        write_pad(file, "integer, intent(in)             :: ixI^L, ixO^L", 2)
        write_pad(file, "double precision, intent(in)    :: x(ixI^S,1:ndim)", 2)
        write_pad(file, "double precision, intent(in)    :: wCT(ixI^S,1:nw)", 2)
        write_pad(
            file, "double precision, intent(out)   :: gravity_field(ixI^S,ndim)", 2
        )
        file.write("\n")
        write_pad(file, "gravity_field           = 0.d0", 2)

        func = get_code_expression(-expr, translation, "gravity_field(ixI^S, 1)")
        write_pad(file, func, 2)

        write_pad(file, "end subroutine set_gravity", 1)
        file.write("\n")

    add_heating = False
    if eq.heatcool is None:
        pass
    elif eq.heatcool.get("mhd_equi_thermal", False):
        pass
    else:
        add_heating = True
    if eq._dict_phys["heating"][0] is None:
        pass
    else:
        add_heating = True

    if add_heating:
        write_pad(
            file, "subroutine set_heating(qdt,ixI^L,ixO^L,iw^LIM,qtC,wCT,qt,w,x)", 1
        )
        write_pad(file, "integer, intent(in)             :: ixI^L, ixO^L, iw^LIM", 2)
        write_pad(file, "double precision, intent(in)    :: qdt, qtC, qt", 2)
        write_pad(file, "double precision, intent(in)    :: x(ixI^S,1:ndim)", 2)
        write_pad(file, "double precision, intent(in)    :: wCT(ixI^S,1:nw)", 2)
        write_pad(file, "double precision, intent(inout) :: w(ixI^S,1:nw)", 2)
        write_pad(file, "double precision                :: bQgrid(ixI^S)", 2)
        file.write("\n")
        write_pad(file, "call getbQ(bQgrid,ixI^L,ixO^L,qtC,wCT,x)", 2)
        write_pad(file, "w(ixO^S,e_)=w(ixO^S,e_)+(qdt*bQgrid(ixO^S))", 2)

        write_pad(file, "end subroutine set_heating", 1)
        file.write("\n")

        write_pad(file, "subroutine getbQ(bQgrid,ixI^L,ixO^L,qt,w,x)", 1)
        if eq.heatcool is not None:
            if eq.heatcool.get("force_thermal_balance", True):
                write_pad(
                    file,
                    "use mod_radiative_cooling, only: getvar_cooling, findL, "
                    "calc_l_extended",
                    2,
                )
        write_pad(file, "integer, intent(in) :: ixI^L, ixO^L", 2)
        write_pad(file, "double precision, intent(in) :: qt", 2)
        write_pad(
            file, "double precision, intent(in) :: x(ixI^S,1:ndim), w(ixI^S,1:nw)", 2
        )
        write_pad(file, "double precision, intent (inout) :: bQgrid(ixI^S)", 2)
        write_pad(file, "double precision :: equil(ixI^S, nw+3)", 2)

        if eq.heatcool is not None:
            if eq.heatcool.get("force_thermal_balance", True):
                # assume force_thermal_balance
                write_pad(file, "integer :: idx^D", 2)
                write_pad(
                    file, "double precision :: T0(ixI^S), l_temp, l_tot(ixI^S)", 2
                )
                file.write("\n")

                write_pad(file, "call get_equilibrium(ixI^L, ixO^L, x, equil)", 2)
                write_pad(file, "T0(ixI^S) = equil(ixI^S, p_) / equil(ixI^S, rho_)", 2)
                file.write("\n")

                write_pad(file, "do idx1 = ixOmin1, ixOmax1", 2)
                write_pad(
                    file, "if(T0(idx1, ixOmin2, ixOmin3) <= rc_fl%tcoolmin) then", 3
                )
                write_pad(file, "l_temp = zero", 4)
                write_pad(
                    file, "else if(T0(idx1, ixOmin2, ixOmin3) >= rc_fl%tcoolmax)then", 3
                )
                write_pad(
                    file,
                    "call calc_l_extended(T0(idx1, ixOmin2, ixOmin3),l_temp,rc_fl)",
                    4,
                )
                write_pad(file, "else", 3)
                write_pad(
                    file, "call findL(T0(idx1, ixOmin2, ixOmin3),l_temp,rc_fl)", 4
                )
                write_pad(file, "end if", 3)
                write_pad(
                    file, "l_tot(idx1, ixOmin2:ixOmax2, ixOmin3:ixOmax3) = l_temp", 3
                )
                write_pad(file, "end do", 2)
                file.write("\n")
                write_pad(
                    file, "bQgrid(ixO^S) = equil(ixO^S, rho_)**2 * l_tot(ixO^S)", 2
                )

        elif eq._dict_phys.get("heating")[0] is not None:
            expr = eq._dict_phys["heating"][0].subs(eq.variables.x, xv)
            file.write("\n")
            write_pad(file, "call get_equilibrium(ixI^L, ixO^L, x, equil)", 2)

            func = get_code_expression(expr, translation, "bQgrid(ixI^S)")
            func = func + " * equil(ixI^S, rho_)"
            write_pad(file, func, 2)
            write_pad(
                file, "! MPI-AMRVAC adds rho mathcal{L} to the energy equation", 2
            )

        write_pad(file, "end subroutine getbQ", 1)
        file.write("\n")

    if eq._dict_phys["resistivity"][0] is not None:
        write_pad(
            file, "subroutine set_resistivity(w,ixI^L,ixO^L,idirmin,x,current,eta)", 1
        )
        write_pad(file, "use mod_global_parameters", 2)
        write_pad(file, "integer, intent(in)              :: ixI^L, ixO^L, idirmin", 2)
        write_pad(
            file, "double precision, intent(in)     :: w(ixI^S,nw), x(ixI^S,1:ndim)", 2
        )
        write_pad(
            file,
            "double precision                 :: current(ixI^S,7-2*ndir:3), eta(ixI^S)",
            2,
        )
        write_pad(file, "double precision                 :: equil(ixI^S, nw+3)", 2)
        file.write("\n")
        write_pad(file, "call get_equilibrium(ixI^L, ixO^L, x, equil)", 2)

        expr = eq._dict_phys["resistivity"][0].subs(eq.variables.x, xv)

        func = get_code_expression(expr, translation, "eta(ixI^S)")
        write_pad(file, func, 2)

        write_pad(file, "end subroutine set_resistivity", 1)
        file.write("\n")


def get_code_expression(expr, translation, assign):

    if is_sympy_number(expr):
        func = fcode(
            sp.sympify(float(expr)),
            assign_to=assign,
            source_format="free",
        ).lstrip()
    else:
        func = fcode(expr, assign_to=assign, source_format="free").lstrip()
        for key in list(translation.keys()):
            func = func.replace(key, translation[key])
        func = func.replace("\n", " &\n")
        func = func.replace("@", "")

    return func


class Amrvac:
    """
    Class to prepare Legolas data for use in MPI-AMRVAC (https://amrvac.org).

    Parameters
    ----------
    config : dict
        The configuration dictionary detailing everything needed for the desired
        functionalities.
    """

    def __init__(self, config):
        self.config = copy.deepcopy(config)
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
                "p0",
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
            self.eq_list = ["rho0", "v01", "v02", "v03", "p0"]
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

        for bounds in ["u1_bounds", "u2_bounds", "u3_bounds"]:
            if (
                bounds in self.config.keys()
                and self.config[bounds][0] >= self.config[bounds][1]
            ):
                raise AssertionError(f"{bounds} should be a positive domain")

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
            self.ds = load(self.config["datfile"])

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
            elif abs(1.0 - np.sum(self.config["weights"])) > 1e-12:
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

        if "save_analytics" not in self.config.keys():
            self.config["save_analytics"] = False
        else:
            assert isinstance(self.config["save_analytics"], bool)
            if self.config["save_analytics"]:
                self.config["parfile"]["typefilelog"] = "special"

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
            self.config["geometry"] = "polar"
        else:
            raise ValueError("'geometry' must be 'Cartesian' or 'cylindrical'.")

        if "equilibrium" not in self.config.keys():
            raise KeyError("Equilibrium not defined.")
        elif not isinstance(self.config["equilibrium"], Equilibrium):
            raise TypeError("'equilibrium' must be an Equilibrium class object.")

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
        elif not (
            "k2" in self.config["parameters"].keys()
            and "k3" in self.config["parameters"].keys()
        ):
            raise KeyError("'parameters' must contain 'k2' and 'k3'.")
        else:
            for key in self.config["parameters"].keys():
                if not isinstance(self.config["parameters"][key], (int, float)):
                    raise TypeError(f"Parameter {key} must be an integer or float.")
        return

    def _validate_simulation_dict(self):
        if "parfile" not in self.config.keys():
            raise KeyError("No 'parfile' provided.")
        elif not isinstance(self.config["parfile"], dict):
            raise TypeError("'parfile' must be a dictionary.")

        if "u1_bounds" in self.config.keys():
            if "xprobmin1" in self.config["parfile"].keys():
                assert (
                    abs(
                        self.config["u1_bounds"][0]
                        - self.config["parfile"]["xprobmin1"]
                    )
                    < 1e-12
                )
            else:
                self.config["parfile"]["xprobmin1"] = self.config["u1_bounds"][0]

            if "xprobmax1" in self.config["parfile"].keys():
                assert (
                    abs(
                        self.config["u1_bounds"][1]
                        - self.config["parfile"]["xprobmax1"]
                    )
                    < 1e-12
                )
            else:
                self.config["parfile"]["xprobmax1"] = self.config["u1_bounds"][1]

        if "u2_bounds" in self.config.keys():
            if "xprobmin2" in self.config["parfile"].keys():
                assert (
                    abs(
                        self.config["u2_bounds"][0]
                        - self.config["parfile"]["xprobmin2"]
                    )
                    < 1e-12
                )
            else:
                self.config["parfile"]["xprobmin2"] = self.config["u2_bounds"][0]

            if "xprobmax2" in self.config["parfile"].keys():
                assert (
                    abs(
                        self.config["u2_bounds"][1]
                        - self.config["parfile"]["xprobmax2"]
                    )
                    < 1e-12
                )
            else:
                self.config["parfile"]["xprobmax2"] = self.config["u2_bounds"][1]

        if "u3_bounds" in self.config.keys():
            if "xprobmin3" in self.config["parfile"].keys():
                assert (
                    abs(
                        self.config["u3_bounds"][0]
                        - self.config["parfile"]["xprobmin3"]
                    )
                    < 1e-12
                )
            else:
                self.config["parfile"]["xprobmin3"] = self.config["u3_bounds"][0]

            if "xprobmax3" in self.config["parfile"].keys():
                assert (
                    abs(
                        self.config["u3_bounds"][1]
                        - self.config["parfile"]["xprobmax3"]
                    )
                    < 1e-12
                )
            else:
                self.config["parfile"]["xprobmax3"] = self.config["u3_bounds"][1]

        if "typeboundary_min1" not in self.config["parfile"].keys():  # Cartesian only
            logger_msg = "default wall"
            bc = ["symm", "asymm", "symm", "symm"]
            if self.config["dim"] > 2:
                bc.append("symm")
            if self.config["physics_type"] == "mhd":
                bc = bc + ["asymm", "symm"]
                if self.config["dim"] > 2:
                    bc.append("symm")
            if (
                self.config["geometry"] == "polar"
                and self.config["u1_bounds"][0] < 1e-12
            ):
                bc = ["pole" for _ in bc]
                logger_msg = "pole"
            pylboLogger.info(
                f"'typeboundary_min1' not provided. Adding {logger_msg} boundary "
                "conditions."
            )
            self.config["parfile"]["typeboundary_min1"] = [bc]

        if "typeboundary_max1" not in self.config["parfile"].keys():  # Cartesian only
            pylboLogger.info(
                "'typeboundary_max1' not provided. Adding default wall boundary "
                "conditions."
            )
            bc = ["symm", "asymm", "symm", "symm"]
            if self.config["dim"] > 2:
                bc.append("symm")
            if self.config["physics_type"] == "mhd":
                bc = bc + ["asymm", "symm"]
                if self.config["dim"] > 2:
                    bc.append("symm")
            self.config["parfile"]["typeboundary_max1"] = [bc]

        if self.config["physics_type"] == "hd":
            if self.config["parfile"].get("has_equi_rho_and_p", False):
                raise AssertionError(
                    "Split rho and p not supported for physics type 'hd'."
                )

        if self.config["parfile"].get("has_equi_rho_and_p", False) or self.config[
            "parfile"
        ].get("B0field", False):
            self.config["parfile"]["mhd_dump_full_vars"] = True
            self.config["parfile"]["autoconvert"] = True
            if "convert_type" in self.config["parfile"].keys():
                pylboLogger.warning(
                    "Overriding 'convert_type' to 'dat_generic_mpi' to enable "
                    + "full variable saving. Use the 'aiconvert' option to convert"
                    + " to another format after the simulation."
                )
            self.config["parfile"]["convert_type"] = "dat_generic_mpi"

        if self.config["equilibrium"].heatcool is not None:
            if (
                self.config["equilibrium"].heatcool.get("force_thermal_balance", True)
                and self.config["equilibrium"]._dict_phys["heating"][0] is not None
            ):
                pylboLogger.warning(
                    "Custom heating is overridden by 'force_thermal_balance'."
                    "The thermal-balance heating source will be "
                    "used instead."
                )

            if self.config["equilibrium"].heatcool is not None:
                if self.config["equilibrium"].heatcool["cooling_curve"] != self.config[
                    "parfile"
                ].get("coolcurve", None):
                    pylboLogger.warning(
                        "'coolcurve' is overridden by value in 'heatcool'."
                    )
                self.config["parfile"]["coolcurve"] = self.config[
                    "equilibrium"
                ].heatcool["cooling_curve"]

                if "ncool" in self.config["parfile"].keys():
                    if self.config["equilibrium"].heatcool["ncool"] != self.config[
                        "parfile"
                    ].get("ncool", 4000):
                        pylboLogger.warning(
                            "'ncool' is overridden by value in 'heatcool'."
                        )
                self.config["parfile"]["ncool"] = self.config["equilibrium"].heatcool[
                    "ncool"
                ]

        if self.config["parfile"].get("has_equi_rho_and_p", False):
            if self.config["parfile"].get("mhd_equi_thermal", False):
                if not self.config["equilibrium"].heatcool.get(
                    "force_thermal_balance", True
                ):
                    raise AssertionError(
                        "Keyword 'mhd_equi_thermal' present, "
                        "but equilibrium is not in thermal balance."
                    )
                self.config["equilibrium"].heatcool["mhd_equi_thermal"] = True
            elif self.config["equilibrium"].heatcool is not None:
                if self.config["equilibrium"].heatcool.get(
                    "force_thermal_balance", True
                ):
                    self.config["parfile"][
                        "mhd_equi_thermal"
                    ] = True  # split rho and p implies mhd
                    self.config["equilibrium"].heatcool["mhd_equi_thermal"] = True
                    pylboLogger.warning(
                        "Equilibrium is in thermal balance. "
                        "Adding 'mhd_equi_thermal=.true.' to parfile."
                    )

        if (
            self.config.get("tc_perpendicular", False)
            and self.config["equilibrium"].heatcool is not None
        ):
            if self.config["equilibrium"].heatcool["force_thermal_balance"]:
                raise NotImplementedError(
                    "Exact thermal balance with perpendicular thermal conduction "
                    "not implemented. "
                    "Provide a custom heating function in the equilibrium."
                )

        if (
            self.config["equilibrium"]._dict_phys["resistivity"][0] is not None
            and self.config["parfile"].get("mhd_eta", 10) > 0
        ):
            pylboLogger.warning("Overriding 'mhd_eta' to use custom resistivity.")
            self.config["parfile"]["mhd_eta"] = -1.0

        for key in self.config["equilibrium"]._dict_phys.keys():
            if self.config["equilibrium"]._dict_phys[key][0] is not None:
                if key not in ["gravity", "heating", "resistivity"]:
                    raise NotImplementedError(
                        f"MPI-AMRVAC does not support user-implemented {key}."
                    )

        if "He_abundance" not in self.config["parfile"].keys():
            self.config["parfile"]["He_abundance"] = 0.0
            pylboLogger.warning(
                "'He_abundance' set to 0. Overriding MPI-AMRVAC default of 0.1."
            )

    def _get_combined_perturbation(self, ef, clean=True):
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
            if self.config["quantity"] == "p0":
                rho1 = ef_data[ii]["rho"]
                T1 = ef_data[ii]["T"]
                rho0 = np.interp(
                    self.ds.ef_grid, self.ds.grid_gauss, self.ds.equilibria["rho0"]
                )
                T0 = np.interp(
                    self.ds.ef_grid, self.ds.grid_gauss, self.ds.equilibria["T0"]
                )
                scaling = rho1 * T0 + rho0 * T1
            else:
                scaling = ef_data[ii][self.config["quantity"].replace("0", "")]
            if clean:
                # rotate the eigenfunction so that at the grid point where the real
                # part is largest the value becomes purely real to remove
                # arbitrary phase rotations (e.g. from shift-invert)
                idx_max = np.argmax(np.abs(np.real(raw)))
                phase = np.angle(raw[idx_max])
                raw = raw * np.exp(-1j * phase)
                # absolute check for efs that are almost zero
                if np.allclose(np.abs(raw), 0, atol=1e-9):
                    raw = 0.0
                    pylboLogger.warning(
                        f"Perturbation of {ef} is almost zero."
                        " Setting perturbation to zero to avoid numerical issues."
                    )
                # relative check between real/imag parts
                rel_tol = 1e4
                if np.max(np.abs(np.real(raw))) > rel_tol * np.max(
                    np.abs(np.imag(raw))
                ):
                    raw = np.real(raw)
                    pylboLogger.warning(
                        f"Perturbation of {ef} is almost purely real."
                        " Taking real part to avoid numerical issues."
                    )
            raw = raw * np.exp(1j * phase)  # reapply the phase
            perturbation += w * fac * (raw / np.nanmax(np.abs(scaling)))
        return perturbation

    def _get_total_perturbation(self, ef_type, clean=True):
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
            perturbation = self._get_combined_perturbation(ef_type, clean=clean)
        return perturbation

    def _integrate_energy_term(self, array, order):
        k2 = self.ds.parameters["k2"]
        k3 = self.ds.parameters["k3"]

        interp_r = CubicSpline(self.ds.ef_grid, array.real)
        interp_i = CubicSpline(self.ds.ef_grid, array.imag)
        deps = 1 if np.allclose(self.ds.scale_factor, self.ds.grid_gauss) else 0

        if self.config["dim"] == 3:

            def integrand(u3, u2, u1):
                value = (
                    (interp_r(u1) + 1j * interp_i(u1))
                    * np.exp(1j * order * (k2 * u2 + k3 * u3))
                ).real * (u1**deps)
                return value

            integral = tplquad(
                integrand,
                *self.config["u1_bounds"],
                *self.config["u2_bounds"],
                *self.config["u3_bounds"],
            )
        elif self.config["dim"] >= 2:
            kvec = np.array([k3, k2])  # order needed below
            arg = np.argmax(abs(kvec))

            def integrand(u2, u1):
                value = (
                    (interp_r(u1) + 1j * interp_i(u1))
                    * np.exp(1j * order * (kvec[arg] * u2))
                ).real * (
                    u1**deps
                ) ** arg  # needed for cylindrical
                return value

            integral = dblquad(
                integrand,
                *self.config["u1_bounds"],
                *self.config[f"u{int(3-arg)}_bounds"],
            )
        elif self.config["dim"] >= 1:

            def integrand(u1):
                return interp_r(u1) + 1j * interp_i(u1)

            integral = quad(integrand, *self.config["u1_bounds"])

        return integral[0]

    def _get_ef_normalisation(self, clean=True):
        """
        Normalises the perturbation of the specified quantity by the maximum background
        value.

        Returns
        -------
        float
            The normalisation factor.
        """
        ef_match = self.config["quantity"].replace("0", "")
        if self.config["quantity"] == "p0":
            max_bg = np.nanmax(
                np.abs(self.ds.equilibria["rho0"] * self.ds.equilibria["T0"])
            )
        else:
            max_bg = np.nanmax(np.abs(self.ds.equilibria[self.config["quantity"]]))
        perturbation = self._get_total_perturbation(ef_match, clean=clean)
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

    def _get_energy_normalisation(self, clean=True):
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
        idx = eq_list.index("p0")
        eq_list[idx] = "T0"
        ef_list = copy.deepcopy(self.ef_list)
        idx = ef_list.index("p")
        ef_list[idx] = "T"

        eq = {}
        for key in eq_list:
            eq[key] = np.interp(u1, u1_gauss, self.ds.equilibria[key])
        efs = {}
        for key in ef_list:
            efs[key] = self._get_total_perturbation(key, clean=clean)

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

    def _get_normalisation(self, clean=True):
        """
        Selects which procedure to follow for the normalisation.

        Returns
        -------
        float
            The normalisation factor.
        """
        if self.config["energy_norm"]:
            norm = self._get_energy_normalisation(clean=clean)
        else:
            norm = self._get_ef_normalisation(clean=clean)
        return norm

    def _check_physical_perturbation(self, ef_name, pert):
        if ef_name not in ["rho", "p"]:
            return
        bg = np.interp(self.ds.ef_grid, self.ds.grid_gauss, self.ds.equilibria["rho0"])
        if ef_name == "p":
            bg = bg * np.interp(
                self.ds.ef_grid, self.ds.grid_gauss, self.ds.equilibria["T0"]
            )
        x2_test = np.linspace(
            self.config["u2_bounds"][0], self.config["u2_bounds"][1], 50
        )
        x3_test = np.linspace(
            self.config["u3_bounds"][0], self.config["u3_bounds"][1], 50
        )

        for i in range(len(pert)):
            for x2 in x2_test:
                for x3 in x3_test:
                    pert_total = pert[i] * np.exp(
                        1j * self.ds.parameters["k2"] * x2
                        + 1j * self.ds.parameters["k3"] * x3
                    )

                    if np.real(pert_total) >= bg[i]:
                        raise ValueError(
                            f"Perturbation of '{ef_name}' is bigger than background."
                        )

    def prepare_legolas_data(self, name=None, loc=None, clean=True):
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
        f.write_record(
            np.array([int(self.config["physics_type"] == "mhd")], dtype=np.int32)
        )
        f.write_record(np.array([self.ds.ef_gridpoints], dtype=np.int32))
        f.write_record(
            np.array(
                [self.ds.parameters["k2"], self.ds.parameters["k3"]], dtype=np.float64
            )
        )
        f.write_record(self.ds.ef_grid)

        norm = self._get_normalisation(clean=clean)
        for ix in range(len(self.ef_list)):
            pert = self._get_total_perturbation(self.ef_list[ix], clean=clean) * norm
            self._check_physical_perturbation(self.ef_list[ix], pert)
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
        self._validate_simulation_dict()
        self._validate_config_for_mod_usr()
        quantities = copy.deepcopy(self.ef_list)
        if self.config["dim"] <= 2:
            quantities.remove("v3")
            if "B3" in quantities:
                quantities.remove("B3")

        keyring_total = ["rho_", "mom(1)", "mom(2)"]
        if self.config["dim"] > 2:
            keyring_total.append("mom(3)")
        keyring_total.append("p_")
        if self.config["physics_type"] == "mhd":
            keyring_total = keyring_total + ["mag(1)", "mag(2)"]
            if self.config["dim"] > 2:
                keyring_total.append("mag(3)")

        keyring_split = []
        if self.config["parfile"].get("has_equi_rho_and_p", False):
            keyring_split = keyring_split + ["rho_", "p_"]
        if self.config["parfile"].get("B0field", False):
            keyring_split = keyring_split + ["mag(1)", "mag(2)", "j1", "j2"]
            if self.config["dim"] > 2:
                keyring_split.insert(-2, "mag(3)")
                keyring_split.append("j3")

            geometry = self.config["geometry"]
            self.config["equilibrium"].add_current(geometry, self.config["dim"])
            forcefree = self.config["equilibrium"].Bfield_forcefree(
                geometry, self.config["dim"]
            )
            if forcefree != self.config["parfile"].get("B0field_forcefree", False):
                pylboLogger.warning(
                    "Specified B0field_forcefree does not match the actual "
                    + "force-freeness of the equilibrium. Check your configuration."
                )
            self.config["parfile"]["B0field_forcefree"] = forcefree

        keyring_unsplit = [key for key in keyring_total if key not in keyring_split]

        loc = validate_output_dir(loc)
        path = loc + "/" + filename + ".t"
        create_file(path)
        file = open(path, "a")

        write_pad(file, "!> User module for perturbation with Legolas eigenmodes.", 0)
        write_pad(file, "!! Generated with GIMLI.", 0)
        write_pad(file, "module mod_usr", 0)
        write_pad(file, "use, intrinsic :: iso_fortran_env", 1)
        write_pad(file, f"use mod_{self.config['physics_type']}", 1)
        write_pad(file, "use mod_global_parameters", 1)
        write_pad(file, "use mod_gimli", 1)
        write_pad(file, "implicit none", 1)
        file.write("\n")
        write_pad(
            file,
            "character(len=100), parameter :: legolas_file = '"
            + self.config["ldatfile"]
            + ".ldat'",
            1,
        )
        file.write("\n")
        write_pad(file, "integer, parameter :: file_id = 123", 1)
        file.write("\n")
        eqparam = get_equilibrium_parameters(self.config)
        if eqparam:
            write_pad(file, f"double precision :: {eqparam}", 1)
        if self.config["parfile"].get("B0field", False):
            write_pad(file, "integer :: j1, j2, j3", 1)
        file.write("\n")

        write_pad(file, "contains", 0)
        file.write("\n")

        write_pad(file, "subroutine usr_init()", 1)
        write_pad(
            file,
            "call set_coordinate_system('"
            + self.config["geometry"]
            + "_"
            + str(self.config["dim"])
            + "D')",
            2,
        )
        write_pad(file, "call read_legolas_data(legolas_file, file_id)", 2)
        file.write("\n")
        if eqparam:
            write_pad(file, "usr_set_parameters => initglobaldata_usr", 2)
        write_pad(file, "usr_init_one_grid  => initialise_grid", 2)
        write_physics_pointers(file, self.config["equilibrium"])
        if self.config["parfile"].get("B0field", False):
            write_pad(file, "usr_set_B0 => special_set_B0", 2)
            write_pad(file, "usr_set_J0 => special_set_J0", 2)
        if self.config["parfile"].get("has_equi_rho_and_p", False):
            write_pad(file, "usr_set_equi_vars => special_set_equi_vars", 2)
        if self.config.get("save_analytics", False):
            write_pad(file, "usr_print_log => analytics_log", 2)
        file.write("\n")
        write_pad(file, f"call {self.config['physics_type']}_activate()", 2)
        write_pad(file, "end subroutine usr_init", 1)
        file.write("\n")

        if eqparam:
            write_pad(file, "subroutine initglobaldata_usr()", 1)
            for key in eqparam.split(", "):
                write_pad(file, f"{key} = {self.config['parameters'][key]}", 2)
            write_pad(file, "end subroutine initglobaldata_usr", 1)
            file.write("\n")

        write_pad(file, "subroutine initialise_grid(ixI^L, ixO^L, w, x)", 1)
        write_pad(file, "integer, intent(in)             :: ixI^L, ixO^L", 2)
        write_pad(file, "double precision, intent(in)    :: x(ixI^S, ndim)", 2)
        write_pad(file, "double precision, intent(inout) :: w(ixI^S, nw)", 2)
        write_pad(file, "integer                         :: idx", 2)
        write_pad(file, "double precision                :: equil(ixI^S, nw+3)", 2)
        file.write("\n")

        write_pad(file, "call get_equilibrium(ixI^L, ixO^L, x, equil)", 2)
        for key in keyring_unsplit:
            write_pad(file, f"w(ixI^S, {key}) = equil(ixI^S, {key})", 2)
        file.write("\n")

        write_pad(file, "do idx = 1,nw", 2)
        write_pad(file, "call add_perturbation_to_w_array(ixI^L, ixO^L, w, idx, x)", 3)
        write_pad(file, "end do", 2)
        file.write("\n")

        write_pad(
            file,
            f"call {self.config['physics_type']}_to_conserved(ixI^L, ixO^L, w, x)",
            2,
        )
        write_pad(file, "end subroutine initialise_grid", 1)
        file.write("\n")
        write_equilibrium_functions(
            file, self.config["equilibrium"], keyring_total, keyring_split
        )

        write_physics_subroutines(file, self.config["equilibrium"])

        write_pad(file, "end module", 0)
        file.write("!")
        file.close()
        return

    def parfile(
        self,
        basename="amrvac_config",
        loc=None,
        subdir=True,
        prefix_numbers=False,
        nb_prefix_digits=4,
    ):
        """
        Generates parfiles based on the `parfile` dictionary.
        The separate namelists do not have to be taken into account, and a normal
        dictionary should be supplied where the keys correspond to the namelist
        items that are required. Typechecking is done automatically during parfile
        generation.

        Parameters
        ----------
        basename : str
            The basename for the parfile, the `.par` suffix is added automatically and
            is not needed. If multiple parfiles are generated, these
            will be prepended by a 4-digit number (e.g. 0003myparfile.par).
            If not provided, the basename will default to `amrvac_config`.
        output_dir : str, ~os.PathLike
            Output directory where the parfiles are saved, defaults to the current
            working directory if not specified. A subdirectory called `parfiles` will be
            created in which the parfiles will be saved.
        subdir : boolean
            If `True` (default), creates a subdirectory `parfiles` in the output folder.
        prefix_numbers : boolean
            If `True` prepends the `basename` by a n-digit number (e.g.
            xxxxmyparfile.par). The number of digits is specified by `nb_prefix_digits`.
        nb_prefix_digits : int
            Number of digits to prepend to the `basename` if `prefix_numbers` is `True`.
            Defaults to 4.

        Returns
        -------
        parfiles : list
            A list with the paths to the parfiles that were generated.
        """
        loc = validate_output_dir(loc)
        self._validate_simulation_dict()
        pfgen = ParfileGenerator(
            parfile_dict=self.config["parfile"],
            basename=basename,
            output_dir=loc,
            subdir=subdir,
            prefix_numbers=prefix_numbers,
            nb_prefix_digits=nb_prefix_digits,
            code="amrvac",
        )
        pfgen.create_namelist_from_dict()
        return pfgen.generate_parfiles()
