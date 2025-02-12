from .regression import RegressionTest
import pytest


class TestNumericalQR(RegressionTest):
    name = "Numerical"
    filename = "numerical_QR"
    equilibrium = "numerical"
    geometry = "Cartesian"

    parameters = {
        "k2": 1.0,
        "k3": 0,
        "eq_bool": True,
        "input_file": "../pylbo_tests/utility_files/test_numerical.lar",
        "n_input": 2000
    }

    physics_settings = {"physics_type": "hd"}

    spectrum_limits = [
        {"xlim": (-70, 70), "ylim": (-5e-8, 5e-8)},
        {"xlim": (-2.5, 2.5), "ylim": (-5e-8, 5e-8)},
    ]

    @pytest.mark.parametrize("limits", spectrum_limits)
    def test_spectrum(self, limits, ds_test, ds_base):
        super().run_spectrum_test(limits, ds_test, ds_base)
