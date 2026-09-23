import numpy as np
import pytest

from .regression import RegressionTest


class TestGoldHoyleQR(RegressionTest):
    name = "Gold Hoyle k2=1 k3=1"
    filename = "gold_hoyle_QR_k2_1_k3_1"
    equilibrium = "gold_hoyle"
    geometry = "cylindrical"

    parameters = {
        "k2": 1.0,
        "k3": 1.0,
        "cte_rho0": 1.0,
        "cte_T0": 0.001,
        "alpha": 20.0,
        "eq_bool": True,
    }
    physics_settings = {
        "radiative_cooling": True,
        "heating": True,
        "force_thermal_balance": True,
        "cooling_curve": "Rosner",
        "parallel_conduction": True,
        "perpendicular_conduction": False,
        "unit_density": 1.6727e-15,
        "unit_magneticfield": 22.5,
        "unit_length": 1.0e10,
    }

    spectrum_limits = [
        {"xlim": (-275, 275), "ylim": (-0.05, 1.2)},
        {"xlim": (-0.1, 0.1), "ylim": (0.92, 1)},
        {"xlim": (-150, 150), "ylim": (-0.005, 0.02)},
        {"xlim": (-20, 20), "ylim": (-0.0025, 0.018)},
        {"xlim": (-1.5, 1.5), "ylim": (-0.001, 0.018)},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)

    def test_perp_conduction(self, ds_test):
        assert np.all(ds_test.equilibria.get("kappa_perp") == pytest.approx(0))

    def test_para_conduction(self, ds_test):
        assert np.all(
            ds_test.equilibria.get("kappa_para")
            == pytest.approx(0.00017559285606016614, abs=1e-10)
        )

    def test_units(self, ds_test):
        assert ds_test.cgs
        for val in ("density", "magneticfield", "length"):
            assert ds_test.units.get(f"unit_{val}") == pytest.approx(
                self.physics_settings[f"unit_{val}"]
            )

    def test_temperature(self, ds_test):
        assert np.all(
            ds_test.equilibria["T0"] * ds_test.units["unit_temperature"]
            == pytest.approx(2.9e5, rel=0.0065)
        )
