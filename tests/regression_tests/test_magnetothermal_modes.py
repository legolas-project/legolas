import numpy as np
import pytest

from .regression import RegressionTest

spectrum_limits = [
    {"xlim": (-650, 650), "ylim": (-0.15, 0.12)},
    {"xlim": (-40, 40), "ylim": (-0.15, 0.12)},
    {"xlim": (-5, 5), "ylim": (-0.15, 0.12)},
    {"xlim": (-0.1, 0.1), "ylim": (-0.15, 0.12)},
    {"xlim": (-0.025, 0.025), "ylim": (-0.015, 0.12), "RMS_TOLERANCE": 2.2},
]


class MagnetoThermalModes(RegressionTest):
    equilibrium = "magnetothermal_instabilities"
    geometry = "cylindrical"

    parameters = {"k2": 0.0, "k3": 1.0, "cte_T0": 1.0, "eq_bool": True}
    physics_settings = {
        "radiative_cooling": True,
        "heating": True,
        "force_thermal_balance": True,
        "cooling_curve": "Rosner",
        "parallel_conduction": True,
        "perpendicular_conduction": False,
        "unit_temperature": 2.6e6,
        "unit_magneticfield": 10.0,
        "unit_length": 1.0e8,
    }


class TestMagnetoThermalModesQR(MagnetoThermalModes):
    name = "magnetothermal modes k2=0 k3=1 QR"
    filename = "magnetothermal_QR_k2_0_k3_1"

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)

    def test_perp_conduction(self, ds_test):
        assert np.all(ds_test.equilibria.get("kappa_perp") == pytest.approx(0))

    def test_para_conduction(self, ds_test, ds_base):
        tc_para = 1.98901013
        assert np.all(ds_test.equilibria.get("kappa_para") == pytest.approx(tc_para))
        assert np.all(ds_base.equilibria.get("kappa_para") == pytest.approx(tc_para))

    def test_units(self, ds_test):
        assert ds_test.cgs
        for val in ("temperature", "magneticfield", "length"):
            assert (
                ds_test.units.get(f"unit_{val}") == self.physics_settings[f"unit_{val}"]
            )


class TestMagnetoThermalModesQZ(MagnetoThermalModes):
    name = "magnetothermal modes k2=0 k3=1 QZ"
    filename = "magnetothermal_QZ_k2_0_k3_1"
    use_custom_baseline = "magnetothermal_QR_k2_0_k3_1"
    solver_settings = {"solver": "QZ-direct"}

    spectrum_limits = spectrum_limits[:-1]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)


class TestMagnetoThermalModesQRCholesky(MagnetoThermalModes):
    name = "magnetothermal modes k2=0 k3=1 QR Cholesky"
    filename = "magnetothermal_QR_cholesky_k2_0_k3_1"
    use_custom_baseline = "magnetothermal_QR_k2_0_k3_1"
    solver_settings = {"solver": "QR-cholesky"}

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)


class TestMagnetoThermalModesSI(MagnetoThermalModes):
    name = "magnetothermal modes k2=0 k3=1 shift-invert"
    filename = "magnetothermal_SI_k2_0_k3_1"
    solver_settings = {
        "solver": "arnoldi",
        "arpack_mode": "shift-invert",
        "number_of_eigenvalues": 15,
        "which_eigenvalues": "LM",
        "sigma": 0.01 + 0.04j,
    }

    spectrum_limits = [
        {"xlim": (-0.001, 0.022), "ylim": (-0.001, 0.05)},
    ]
    eigenfunction_settings = {"write_eigenfunctions": True}
    eigenfunctions = [
        {"eigenvalue": 2.12871e-13 + 0.02799j},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)

    @pytest.mark.parametrize("eigenfunction", eigenfunctions)
    def test_eigenfunction(self, eigenfunction, ds_test, ds_base):
        super().run_eigenfunction_test(eigenfunction, ds_test, ds_base)
