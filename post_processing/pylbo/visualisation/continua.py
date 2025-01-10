from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from pylbo.data_containers import LegolasDataSet

import numpy as np
from pylbo.utilities.logger import pylboLogger
from pylbo.visualisation.legend_handler import LegendHandler

SLOW_MIN = "slow-"
SLOW_PLUS = "slow+"
ALFVEN_MIN = "alfven-"
ALFVEN_PLUS = "alfven+"
THERMAL = "thermal"
DOPPLER = "doppler"

CONTINUA_NAMES = {
    SLOW_MIN: r"$\Omega_S^-",
    SLOW_PLUS: r"$\Omega_S^+",
    ALFVEN_MIN: r"$\Omega_A^-",
    ALFVEN_PLUS: r"$\Omega_A^+",
    THERMAL: r"$\Omega_T",
    DOPPLER: r"$\Omega_0",
}
CONTINUA_COLORS = ["red", "red", "cyan", "cyan", "green", "grey"]

_DEFAULT_ZERO_TOL = 1e-12


def _is_zero(values: np.ndarray, zero_tol: float = _DEFAULT_ZERO_TOL) -> bool:
    return np.all(np.isclose(values, 0, atol=zero_tol))


def _is_nonadiabatic(ds: LegolasDataSet) -> bool:
    zeroes = np.zeros_like(ds.grid_gauss)
    dLdT = ds.equilibria.get("dLdT", zeroes)
    dLdrho = ds.equilibria.get("dLdrho", zeroes)
    has_cooling = not _is_zero(dLdT) or not _is_zero(dLdrho)
    has_conduction = not _is_zero(ds.equilibria.get("kappa_para", zeroes))
    return has_cooling or has_conduction


def _get_parallel_wave_vector(ds: LegolasDataSet) -> float:
    bg = ds.equilibria
    k2 = ds.parameters.get("k2", 0)
    k3 = ds.parameters.get("k3", 0)
    B0 = bg["B0"]
    eps = ds.scale_factor
    return 0 if _is_zero(B0) else (k2 * bg["B02"] / eps + k3 * bg["B03"]) / B0


def _get_squared_sound_speed(ds: LegolasDataSet) -> np.ndarray:
    return ds.gamma * ds.equilibria["T0"]


def _get_squared_isothermal_sound_speed(ds: LegolasDataSet) -> np.ndarray:
    return ds.equilibria["T0"]


def _get_squared_Alfven_speed(ds: LegolasDataSet) -> np.ndarray:
    return ds.equilibria["B0"] ** 2 / ds.equilibria["rho0"]


def calculate_continua(ds: LegolasDataSet) -> dict:
    """
    Calculates the different continua for a given dataset.
    The Alfvén and flow continua are always analytical. Depending on the background
    and physical effects the slow and thermal continua are either all analytical, or
    coupled through a third-order polynomial.
    In case of the latter this polynomical is numerically solved through numpy.roots.

    Parameters
    ----------
    ds : ~pylbo.data_containers.LegolasDataSet
        The Legolas dataset.

    Returns
    -------
    dict, None
        Dictionary containing the various continua. The keys are the names of the
        continua, the values are the corresponding frequencies as numpy arrays.
        Returns `None` if the dataset has no background.
    """
    if not ds.has_background:
        return None

    doppler = get_doppler_shift(ds)
    alfven2 = get_squared_alfven_continuum(ds)
    slowneg, slowpos, thermal = _get_thermal_and_slow_continua(ds)
    continua = {
        DOPPLER: doppler,
        SLOW_MIN: slowneg,
        SLOW_PLUS: slowpos,
        THERMAL: thermal,
        ALFVEN_MIN: -np.sqrt(alfven2),
        ALFVEN_PLUS: np.sqrt(alfven2),
    }
    # correct for doppler shift
    for name in CONTINUA_NAMES.keys():
        if name == DOPPLER:
            continue
        if not _is_zero(continua[name]):
            continua[name] += doppler
    return continua


def get_squared_alfven_continuum(ds: LegolasDataSet) -> np.ndarray:
    """
    Calculates the squared Alfvén continuum.

    Returns
    -------
    np.ndarray
        The squared Alfvén continuum.
    """
    bg = ds.equilibria
    k2 = ds.parameters.get("k2", 0)
    k3 = ds.parameters.get("k3", 0)
    eps = ds.scale_factor
    return (1 / bg["rho0"]) * (k2 * bg["B02"] / eps + k3 * bg["B03"]) ** 2


def get_doppler_shift(ds: LegolasDataSet) -> np.ndarray:
    """
    Calculates the Doppler shift as the dot product between the wave vector and the
    background velocity.

    Returns
    -------
    np.ndarray
        The Doppler shift.
    """
    bg = ds.equilibria
    k2 = ds.parameters.get("k2", 0)
    k3 = ds.parameters.get("k3", 0)
    eps = ds.scale_factor
    return k2 * bg["v02"] / eps + k3 * bg["v03"]


def get_squared_slow_continuum(ds: LegolasDataSet) -> np.ndarray:
    """
    Calculates the squared slow continuum.

    Returns
    -------
    np.ndarray
        The squared slow continuum.
    """
    bg = ds.equilibria
    p = bg["rho0"] * bg["T0"]
    alfven_sq = get_squared_alfven_continuum(ds)
    return (ds.gamma * p / (ds.gamma * p + bg["B0"] ** 2)) * alfven_sq


def _get_thermal_and_slow_continua(
    ds: LegolasDataSet,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the thermal and slow continua.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the slow-, slow+ and thermal continua, in this order.
    """
    zeroes = np.zeros_like(ds.grid_gauss)

    bg = ds.equilibria
    T0 = bg["T0"]
    if _is_zero(T0):
        # for pressureless cases (no T) there is no slow/thermal continuum
        return (zeroes, zeroes, zeroes)
    slow_sq = get_squared_slow_continuum(ds)
    eta = bg["eta"]
    if not _is_nonadiabatic(ds):
        return (-np.sqrt(slow_sq), np.sqrt(slow_sq), zeroes)
    if (eta != 0).any():
        thermal = _get_resistive_thermal_continuum(ds)
        return (-np.sqrt(slow_sq), np.sqrt(slow_sq), thermal)
    if _is_zero(slow_sq):
        # if slow continuum vanishes, thermal continuum is analytical
        thermal = _get_thermal_continuum_analytical(ds)
        return (zeroes, zeroes, thermal)
    # for standard cases we have a third-order polynomial
    return _get_slow_and_thermal_continuum_coupled(ds)


def _get_resistive_thermal_continuum(ds: LegolasDataSet) -> np.ndarray:
    """
    Calculates the thermoresistive (quasi-)continuum analytically.

    Returns
    -------
    np.ndarray
        The thermoresistive (quasi-)continuum.
    """
    kpara = _get_parallel_wave_vector(ds)
    gamma = ds.gamma
    zeroes = np.zeros_like(ds.grid_gauss)
    L0 = ds.equilibria.get("L0", zeroes)
    dLdT = ds.equilibria.get("dLdT", zeroes)
    dLdrho = ds.equilibria.get("dLdrho", zeroes)
    kappa_para = ds.equilibria.get("kappa_para", zeroes)
    detadT = ds.equilibria.get("detadT", zeroes)
    rho0 = ds.equilibria["rho0"]
    T0 = ds.equilibria["T0"]
    eps = ds.scale_factor
    deps = 0
    if ds.geometry == "cylindrical":
        deps = 1
    drB02 = deps * ds.equilibria["B02"] + eps * ds.equilibria["dB02"]
    dB03 = ds.equilibria["dB03"]

    return (
        -1j
        * (gamma - 1)
        * (
            (kappa_para * kpara**2 + detadT * ((drB02 / eps) ** 2 + dB03**2)) / rho0
            + dLdT
            - (L0 + rho0 * dLdrho) / T0
        )
        / gamma
    )


def _get_thermal_continuum_analytical(ds: LegolasDataSet) -> np.ndarray:
    """
    Calculates the thermal continuum analytically, when the slow continuum is zero.

    Returns
    -------
    np.ndarray
        The thermal continuum.
    """
    zeroes = np.zeros_like(ds.grid_gauss)
    bg = ds.equilibria
    L0 = bg.get("L0", zeroes)
    dLdT = bg.get("dLdT", zeroes)
    dLdrho = bg.get("dLdrho", zeroes)
    rho0 = bg["rho0"]
    cs2 = _get_squared_sound_speed(ds)
    ca2 = _get_squared_Alfven_speed(ds)
    ci2 = _get_squared_isothermal_sound_speed(ds)
    gamma_1 = ds.gamma - 1
    return 1j * gamma_1 * (L0 + rho0 * dLdrho - (ca2 + ci2) * dLdT) / (cs2 + ca2)


def _get_slow_and_thermal_continuum_coupled(
    ds: LegolasDataSet,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cs2 = _get_squared_sound_speed(ds)
    ca2 = _get_squared_Alfven_speed(ds)
    ci2 = _get_squared_isothermal_sound_speed(ds)
    kpara = _get_parallel_wave_vector(ds)
    gamma_1 = ds.gamma - 1
    zeroes = np.zeros_like(ds.grid_gauss)
    L0 = ds.equilibria.get("L0", zeroes)
    dLdT = ds.equilibria.get("dLdT", zeroes)
    dLdrho = ds.equilibria.get("dLdrho", zeroes)
    kappa_para = ds.equilibria.get("kappa_para", zeroes)
    rho0 = ds.equilibria["rho0"]

    # coeffi means the coefficient corresponding to the term omega^i
    coeff3 = rho0 * (cs2 + ca2) * 1j / gamma_1
    coeff2 = rho0 * (L0 + rho0 * dLdrho) - (kappa_para * kpara**2 + rho0 * dLdT) * (
        ci2 + ca2
    )
    coeff1 = -rho0 * cs2 * ca2 * kpara**2 * 1j / gamma_1
    coeff0 = -(
        (rho0 * (L0 + rho0 * dLdrho) - (kappa_para * kpara**2 + rho0 * dLdT) * ci2)
        * ca2
        * kpara**2
    )
    return _solve_coupled_continuum_polynomial(coeff3, coeff2, coeff1, coeff0)


def _solve_coupled_continuum_polynomial(
    coeff3: np.ndarray, coeff2: np.ndarray, coeff1: np.ndarray, coeff0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the third-order polynomial that couples the slow and thermal continua.
    The thermal continuum corresponds to the purely imaginary solution.

    Parameters
    ----------
    coeff3 : np.ndarray
        The coefficient corresponding to the term :math:`\\omega^3`.
    coeff2 : np.ndarray
        The coefficient corresponding to the term :math:`\\omega^2`.
    coeff1 : np.ndarray
        The coefficient corresponding to the term :math:`\\omega^1`.
    coeff0 : np.ndarray
        The coefficient corresponding to the constant term.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the slow-, slow+ and thermal continua, in this order.
    """
    thermal = np.zeros_like(coeff3, dtype=complex)
    slowneg = np.zeros_like(thermal, dtype=complex)
    slowpos = np.zeros_like(thermal, dtype=complex)
    for i, (c3, c2, c1, c0) in enumerate(zip(coeff3, coeff2, coeff1, coeff0)):
        roots = np.roots([c3, c2, c1, c0])
        slowneg[i], slowpos[i], thermal[i] = _extract_solutions_from_roots(roots, i)
    return slowneg, slowpos, thermal


def _extract_solutions_from_roots(
    roots: np.ndarray, i: int = 0
) -> Tuple(complex, complex, complex):
    # create mask for purely imaginary roots
    mask = np.isclose(abs(roots.real), 0, atol=_DEFAULT_ZERO_TOL)
    if np.count_nonzero(mask) == 1:
        # if mask only has one True, it is the thermal continuum and others are slow
        ws_neg, ws_pos = np.sort_complex(roots[np.invert(mask)])
        (thermal,) = roots[mask]
        return ws_neg, ws_pos, thermal
    _log_slow_continuum_zero_warning(roots, i)
    # here we have multiple purely imaginary roots, so it's not clear which one
    # is the thermal continuum. We assume that it is the largest one.
    mask = np.array([False] * 3, dtype=bool)
    mask[roots.imag.argmax()] = True
    (thermal,) = roots[mask]
    _log_assumed_thermal_continuum(root=thermal)
    ws_neg, ws_pos = np.sort_complex(roots[np.invert(mask)])
    return ws_neg, ws_pos, thermal


def _log_slow_continuum_zero_warning(roots: np.ndarray, i: int):
    pylboLogger.warning(
        f"encountered index = {i} where the slow continuum has a "
        f"real value close to zero. \nFound thermal-slow roots: {roots}"
    )


def _log_assumed_thermal_continuum(root: complex):
    pylboLogger.warning(
        f"Assuming that the largest imaginary root {root} is the thermal continuum."
    )


class ContinuaHandler(LegendHandler):
    """
    Handler to draw continua regions on the plots and make them interactive.

    Parameters
    ----------
    interactive : bool
        If `True`, makes the legend pickable and continuum plotting interactive.

    Attributes
    ----------
    continua_names : list
        The list of continua names
    """

    def __init__(self, interactive):
        super().__init__(interactive)
        self.continua_names = list(CONTINUA_NAMES.keys())
        self.continua_latex = list(CONTINUA_NAMES.values())
        self._continua_colors = CONTINUA_COLORS
        self.marker = "."
        self.markersize = 6

    @property
    def continua_colors(self):
        """
        Returns the list of continua colors.

        Returns
        -------
        The continua colors as a list.
        """
        return self._continua_colors

    @continua_colors.setter
    def continua_colors(self, colors):
        """
        Setter for the continua colors attribute.

        Parameters
        ----------
        colors : list, numpy.ndarray
            The colors to use when plotting the continua as a list of strings.

        Raises
        ------
        ValueError
            If a wrong argument is passed or if it is of improper length.
        """
        if colors is None:
            return
        if not isinstance(colors, (list, np.ndarray)):
            raise ValueError(
                f"continua_colors should be an array/list but got {type(colors)}"
            )
        if not len(colors) == len(CONTINUA_COLORS):
            raise ValueError(
                f"continua_colors should be of length {len(CONTINUA_COLORS)}"
            )
        self._continua_colors = colors
