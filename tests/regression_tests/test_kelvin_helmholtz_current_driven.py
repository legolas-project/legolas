import pytest

from .regression import RegressionTest


class KelvinHelmholtzCurrentDriven(RegressionTest):
    equilibrium = "kelvin_helmholtz_cd"
    geometry = "cylindrical"

    parameters = {
        "k2": -1.0,
        "V": 1.63,
        "cte_rho0": 1.0,
        "cte_p0": 1.0,
        "Bz0": 0.25,
        "rc": 0.5,
        "rj": 1.0,
    }
    physics_settings = {"flow": True}


class TestKelvinHelmholtzCurrentDrivenQR(KelvinHelmholtzCurrentDriven):
    name = "Kelvin-Helmholtz current driven k2=-1 k3=pi QR"
    filename = "kelvin_helmholtz_current_driven_QR_k2_-1_k3_pi"

    spectrum_limits = [
        {"xlim": (-200, 200), "ylim": (-1, 1), "RMS_TOLERANCE": 2.9},
        {"xlim": (-20, 20), "ylim": (-1, 1)},
        {"xlim": (-4, 4), "ylim": (-1, 1), "RMS_TOLERANCE": 2.6},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)


class TestKelvinHelmholtzCurrentDrivenSI(KelvinHelmholtzCurrentDriven):
    name = "Kelvin-Helmholtz current driven k2=-1 k3=pi shift-invert"
    filename = "kelvin_helmholtz_current_driven_SI_k2_-1_k3_pi"
    solver_settings = {
        "solver": "arnoldi",
        "arpack_mode": "shift-invert",
        "number_of_eigenvalues": 6,
        "which_eigenvalues": "LM",
        "sigma": 2.5 + 0.5j,
        "maxiter": 500,
    }

    spectrum_limits = [
        {"xlim": (2.4, 2.65), "ylim": (-0.2, 1)},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)
