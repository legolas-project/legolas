import numpy as np
from scipy.io import FortranFile
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, dblquad, tplquad
from numpy.polynomial.polynomial import Polynomial

from pylbo.utilities.datfiles.file_loader import load
from pylbo.utilities.logger import pylboLogger
from pylbo.gimli.utils import validate_output_dir


class Amrvac:
    """
    Class to prepare Legolas data for use in MPI-AMRVAC (https://amrvac.org).

    Parameters
    ----------
    config : dict
        The configuration dictionary detailing which Legolas file and selection of
        eigenmodes to use.
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

        if self.config["energy_norm"] not in self.config.keys():
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
            e2 += efs["B1"] ** 2 + efs["B2"] ** 2 + efs["B3"] ** 2

        coeff0 = self._integrate_energy_term(e0, 0)
        coeff1 = self._integrate_energy_term(e1, 1)
        coeff2 = self._integrate_energy_term(e2, 2)
        coeff3 = self._integrate_energy_term(e3, 3)

        p = Polynomial([-self.config["percentage"] * coeff0, coeff1, coeff2, coeff3])
        roots = p.roots()
        index = np.argmin(abs(roots))
        norm = roots[index]
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
        return
