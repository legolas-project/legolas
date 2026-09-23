import numpy as np
import pytest

from .regression import RegressionTest


class TestDiscreteAlfvenQR(RegressionTest):
    name = "Discrete Alfven k2=1 k3=0.05"
    filename = "discrete_alfven_QR_k2_1_k3_0.05"
    equilibrium = "discrete_alfven"
    geometry = "cylindrical"

    parameters = {"k2": 1.0, "k3": 0.05, "j0": 0.125, "delta": 0.2, "eq_bool": True}
    physics_settings = {
        "radiative_cooling": True,
        "heating": True,
        "force_thermal_balance": True,
        "cooling_curve": "Rosner",
        "parallel_conduction": True,
        "perpendicular_conduction": False,
        "unit_density": 1.5e-15,
        "unit_magneticfield": 50.0,
        "unit_length": 1.0e10,
    }

    spectrum_limits = [
        {"xlim": (-700, 700), "ylim": (-0.9, 0.4)},
        {"xlim": (-100, 100), "ylim": (-0.9, 0.4)},
        {"xlim": (-0.2, 0.2), "ylim": (-0.9, 0.4)},
        {"xlim": (-0.01, 0.01), "ylim": (-0.004, 0.009), "RMS_TOLERANCE": 2.1},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)

    def test_perp_conduction(self, ds_test):
        assert np.all(ds_test.equilibria.get("kappa_perp") == pytest.approx(0))

    def test_para_conduction(self, ds_test):
        assert np.any(ds_test.equilibria.get("kappa_para") > 0.003)

    def test_units(self, ds_test):
        assert ds_test.cgs
        for val in ("density", "magneticfield", "length"):
            assert ds_test.units.get(f"unit_{val}") == pytest.approx(
                self.physics_settings[f"unit_{val}"]
            )
