import sympy as sp
import numpy as np
from scipy.io import FortranFile

from pylbo.gimli.utils import is_symbol_dependent
from pylbo.utilities.logger import pylboLogger


class Variables:
    """
    Defines a set of variables and constants to be used in defining an Equilibrium
    object.

    Attributes
    ----------
    x, y, z : sympy symbols
        Coordinates.
    rho0, T0, B0sq : sympy symbols
        Density, temperature, and magnetic field squared for use in expressions
        depending on these quantities.
    k2, k3 : sympy symbols
        Wavenumbers.
    gamma : sympy symbol
        Adiabatic index.
    rhoc, Tc, B2c, B3c, v2c, v3c, pc : sympy symbols
        Constants typically used for amplitudes or uniform terms in their corresponding
        equilibrium quantities. The corresponding Legolas variable names are cte_rho0,
        cte_T0, cte_B02, cte_B03, cte_v02, cte_v03, and cte_p0.
    p1, p2, p3, p4, p5, p6, p7, p8 : sympy symbols
        Additional free-use constants.
    alpha, beta, delta, theta, tau, lam, nu : sympy symbols
        Additional free-use constants. (Note that 'lam' is used instead of 'lambda' to
        avoid conflict with the reserved keyword. The corresponding Legolas variable
        name is 'lambda'.)
    r0, rc, rj, Bth0, Bz0, V, j0, g : sympy symbols
        Additional constants, originally used in cylindrical coordinates.
    fkey : dict
        Dictionary translating LaTeX notation to Legolas variable names.

    Examples
    --------
    >>> from pylbo.gimli import Variables
    >>> var = Variables()
    """

    def __init__(self):
        self.x, self.y, self.z = sp.symbols("x,y,z")
        self.rho0, self.T0, self.B0sq = sp.symbols("rho_0,T_0,B_0^2")

        self.k2, self.k3 = sp.symbols("k_2,k_3")
        self.gamma = sp.symbols("gamma")
        self.rhoc, self.Tc, self.B2c, self.B3c, self.v2c, self.v3c, self.pc = (
            sp.symbols("rho_c,T_c,B_2,B_3,v_2,v_3,p_c")
        )
        self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8 = (
            sp.symbols("p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8")
        )
        self.alpha, self.beta, self.delta, self.theta, self.tau, self.lam, self.nu = (
            sp.symbols("alpha,beta,delta,theta,tau,lambda,nu")
        )
        self.r0, self.rc, self.rj, self.Bth0, self.Bz0, self.V, self.j0, self.g = (
            sp.symbols("r_0,r_c,r_j,B_theta,B_z,V,j_0,g")
        )
        self.fkey = {
            "rho_c": "cte_rho0",
            "T_c": "cte_T0",
            "B_2": "cte_B02",
            "B_3": "cte_B03",
            "v_2": "cte_v02",
            "v_3": "cte_v03",
            "p_c": "cte_p0",
            "p_1": "p1",
            "p_2": "p2",
            "p_3": "p3",
            "p_4": "p4",
            "p_5": "p5",
            "p_6": "p6",
            "p_7": "p7",
            "p_8": "p8",
            "r_0": "r0",
            "r_c": "rc",
            "r_j": "rj",
            "B_theta": "Bth0",
            "B_z": "Bz0",
            "j_0": "j0",
        }


class Equilibrium:
    """
    Class containing all equilibrium expressions and initialisation functions.
    This object is a required argument when generating user files with the Legolas and
    Amrvac classes.

    Parameters
    ----------
    var : :class:`Variables`
        The Variables object containing the symbols to be used in the equilibrium
        expressions.
    rho0 : sympy expression
        The equilibrium density expression.
    v02, v03 : sympy expressions
        The equilibrium velocity expressions.
    T0 : sympy expression
        The equilibrium temperature expression.
    B02, B03 : sympy expressions
        The equilibrium magnetic field expressions.
    resistivity : sympy expression
        The resistivity expression.
    gravity : constant
        The gravitational acceleration.
    condpara : sympy expression
        The parallel conduction prescription.
    condperp : sympy expression
        The perpendicular conduction prescription.
    cooling : sympy expression
        The cooling prescription.
    heating : sympy expression
        The heating prescription.
    heatcool : dict
        Parameters for cooling and heating, including 'force_thermal_balance'.

    Attributes
    ----------
    variables : Variables object
        Variables object from which all expressions are constructed.
    rho0 : sympy expression
        The equilibrium density expression.
    v02, v03 : sympy expressions
        The equilibrium velocity expressions.
    T0 : sympy expression
        The equilibrium temperature expression.
    B02, B03 : sympy expressions
        The equilibrium magnetic field expressions.

    Examples
    --------
    The example below defines a homogeneous hydrodynamic equilibrium with constant
    density and temperature. Their values can be set later when passing this equilibrium
    to the Legolas or Amrvac class along with a dictionary.

    >>> from pylbo.gimli import Equilibrium, Variables
    >>> var = Variables()
    >>> eq = Equilibrium(var, rho0=var.rhoc, v02=0, v03=0, T0=var.Tc)
    """

    def __init__(
        self,
        var,
        rho0,
        v02,
        v03,
        T0,
        B02=None,
        B03=None,
        resistivity=None,
        gravity=None,
        condpara=None,
        condperp=None,
        cooling=None,
        heating=None,
        legolas_grid_spacing=None,
        heatcool=None,
    ):
        self.variables = var
        self.rho0 = sp.sympify(rho0)
        self.v02, self.v03 = sp.sympify(v02), sp.sympify(v03)
        self.T0 = sp.sympify(T0)
        self.B02, self.B03 = sp.sympify(B02), sp.sympify(B03)

        self.heatcool = heatcool

        self.grid_spacing = sp.sympify(legolas_grid_spacing)

        self._dict_phys = {
            "resistivity": [
                sp.sympify(resistivity),
                ["eta", "detadT", "detadr"],
                [self.variables.T0, self.variables.x],
            ],
            "gravity": [sp.sympify(gravity), ["g0"], [self.variables.x]],
            "parallel_conduction": [
                sp.sympify(condpara),
                ["tcpara", "dtcparadT"],
                [self.variables.T0],
            ],
            "perpendicular_conduction": [
                sp.sympify(condperp),
                ["tcperp", "dtcperpdT", "dtcperpdrho", " dtcperpdB2"],
                [self.variables.T0, self.variables.rho0, self.variables.B0sq],
            ],
            "cooling": [
                sp.sympify(cooling),
                ["lambdaT", "dlambdadT"],
                [self.variables.T0],
            ],
            "heating": [
                sp.sympify(heating),
                ["H", "dHdT", "dHdrho"],
                [self.variables.T0, self.variables.rho0],
            ],
        }
        self._validate_equil()

    def _validate_equil(self):
        for key in self._dict_phys.keys():
            if self._dict_phys[key][0] is not None:
                if key not in ["gravity", "heating", "resistivity"]:
                    pylboLogger.warning(
                        f"MPI-AMRVAC does not support user-implemented {key} "
                        "but Legolas does."
                    )
                if key == "heating":
                    if "force_thermal_balance" not in self.heatcool.keys():
                        self.heatcool["force_thermal_balance"] = False
                    elif self.heatcool["force_thermal_balance"]:
                        pylboLogger.warning(
                            "'force_thermal_balance' overrides "
                            "user-set heating function."
                        )

        if self.heatcool is not None and not isinstance(self.heatcool, dict):
            raise TypeError("heatcool must be a dictionary.")
        elif self.heatcool is not None:
            if "force_thermal_balance" not in self.heatcool.keys():
                self.heatcool["force_thermal_balance"] = True
            if (
                self.heatcool["force_thermal_balance"]
                and "heating" not in self.heatcool.keys()
            ):
                self.heatcool["heating"] = True
            if "cooling_curve" not in self.heatcool.keys():
                self.heatcool["cooling_curve"] = None
            if "ncool" not in self.heatcool.keys():
                self.heatcool["ncool"] = 4000

    def get_physics(self):
        """
        Returns a dictionary containing the physics expressions and the dependencies to
        check for.
        """
        return self._dict_phys

    def get_dependencies(self):
        """
        Checks for dependencies on other equilibrium quantities.
        Returns a dictionary with the replacement expressions for use in Fortran files.
        """
        dep_rho = "x" if is_symbol_dependent([self.variables.x], self.rho0) else ""

        dep_T = "x" if is_symbol_dependent([self.variables.x], self.T0) else ""

        dep_B2 = "x" if is_symbol_dependent([self.variables.x], self.B02) else ""
        dep_B3 = "x" if is_symbol_dependent([self.variables.x], self.B03) else ""

        dict_dependencies = {
            "rho_0": f"(rho0({dep_rho}))",
            "T_0": f"(T0({dep_T}))",
            "B_0^2": f"(B02({dep_B2})**2+B03({dep_B3})**2)",
        }
        return dict_dependencies

    def get_current(self, geometry, dim=3):
        """
        Determines the current density of the equilibrium magnetic field.
        Parameters
        ----------
        geometry : str
            Either 'Cartesian' or 'cylindrical'.
        dim : int
            Dimension of the desired setup (currently 2 or 3).
        """
        scale_factor = 1 if geometry == "Cartesian" else self.variables.x

        B02 = self.B02 if self.B02 is not None else 0
        B03 = self.B03 if self.B03 is not None else 0

        J02 = -B03.diff(self.variables.x)
        J03 = B02 * sp.diff(scale_factor * B02, self.variables.x) / scale_factor

        if dim == 2:
            return 0, sp.simplify(J02), 0
        return 0, sp.simplify(J02), sp.simplify(J03)

    def add_current(self, geometry, dim=3):
        """
        Adds the current density of the equilibrium magnetic field to the Equilibrium
        object as attributes J02 and J03.
        Parameters
        ----------
        geometry : str
            Either 'Cartesian' or 'cylindrical'.
        dim : int
            Dimension of the desired setup (currently 2 or 3).
        """
        _, J02, J03 = self.get_current(geometry, dim=dim)
        self.J02 = J02
        self.J03 = J03

    def Bfield_forcefree(self, geometry, dim=3):
        """
        Determines whether the equilibrium magnetic field is force-free.
        Parameters
        ----------
        geometry : str
            Either 'Cartesian' or 'cylindrical'.
        dim : int
            Dimension of the desired setup (currently 2 or 3).
        """
        _, J02, J03 = self.get_current(geometry, dim=dim)

        B02 = self.B02 if self.B02 is not None else 0
        B03 = self.B03 if self.B03 is not None else 0

        JxB1 = J03 * B03 - J02 * B02
        return sp.simplify(JxB1) == 0


class NumericalEquilibrium:
    """
    Class to convert numerical arrays to a Legolas-readable format.

    Parameters
    ----------
    arrays : dict
        A dictionary linking key/header to a numerical array.
        Must contain "rho0" and "T0" and one of ("u1", "x", "r"). Optional arrays are
        "v01", "v02", "v03", "B01", "B02", "B03", and "grav".

    Attributes
    ----------
    arrays : dict
        Dictionary with specified arrays.

    Examples
    --------
    The example below defines a homogeneous hydrodynamic equilibrium with constant
    density and temperature.

    >>> import numpy as np
    >>> from pylbo.gimli import NumericalEquilibrium
    >>> dictionary = {
    >>>     "x" : np.linspace(0, 1, 100),
    >>>     "rho0": 2 * np.ones(100),
    >>>     "T0" : 0.5 * np.ones(100)
    >>> }
    >>> equil = NumericalEquilibrium(dictionary)
    >>> equil.to_legolas_arrays(filename="homogeneous")
    """

    def __init__(self, arrays):
        self.arrays = arrays
        if isinstance(self.arrays, dict):
            self._validate()
        else:
            raise TypeError("Provided object is not a dictionary.")

    def _validate(self):
        keyring = self.arrays.keys()
        if not ("rho0" in keyring and "T0" in keyring):
            raise KeyError("Must include rho0 and T0 arrays.")

        count = len({"u1", "x", "r"} & set(keyring))
        if count == 0:
            raise KeyError("No u1, x, or r array specified.")
        elif count > 1:
            raise RuntimeError("Combination of u1, x, and r encountered. Keep only 1.")

        if not isinstance(self.arrays["rho0"], (list, np.ndarray)):
            raise TypeError("rho0 must be a list or np.ndarray.")
        length = len(self.arrays["rho0"])
        for key in keyring:
            if not isinstance(self.arrays[key], (list, np.ndarray)):
                raise TypeError(f"{key} must be a list of np.ndarray.")
            if not len(self.arrays[key]) == length:
                raise ValueError(f"Resolution of {key} does not match rho0.")

    def to_legolas_arrays(self, filename="arrays", loc="./"):
        """
        Prepares a numerical arrays file (.lar) for use with Legolas.

        Parameters
        ----------
        filename : str, optional
            Name of the .lar file. Default is 'arrays'.
        loc : str, optional
            The location to save the .lar file. Default is the current directory.
        """
        if loc[-1] != "/":
            loc = loc + "/"
        f = FortranFile(loc + filename + ".lar", "w")

        to_write = [
            "u1",
            "x",
            "r",
            "rho0",
            "v01",
            "v02",
            "v03",
            "T0",
            "B01",
            "B02",
            "B03",
            "grav",
        ]

        length = len(self.arrays["rho0"])
        f.write_record(np.array([length], dtype=np.int32))

        for ii in range(len(to_write)):
            key = to_write[ii]
            if key in self.arrays.keys():
                f.write_record(np.array(self.arrays[key], dtype=np.float64))
            elif ii > 2:
                f.write_record(np.zeros(length, dtype=np.float64))

        f.close()
