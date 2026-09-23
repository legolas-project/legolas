import numpy as np
import pylbo
import pytest

from .viz_modes import ModeVizTest


class TemporalTest(ModeVizTest):
    @property
    def omega(self):
        pass

    @property
    def time(self):
        pass

    @pytest.fixture(scope="function")
    def view(self, ds):
        p = pylbo.plot_1d_temporal_evolution(
            ds,
            omega=self.omega,
            ef_name="rho",
            u2=1,
            u3=1,
            time=self.time,
        )
        p.draw()
        return p

    def test_invalid_time(self, ds):
        with pytest.raises(ValueError):
            pylbo.plot_1d_temporal_evolution(ds, self.omega, "rho", u2=1, u3=1, time=0)

    def test_time_as_list(self, ds, mode_solution):
        view = pylbo.plot_1d_temporal_evolution(
            ds, omega=self.omega, ef_name="rho", u2=1, u3=1, time=list(self.time)
        )
        view.draw()
        assert self.cbar_matches(view, mode_solution)
        assert np.allclose(view.solutions, mode_solution)

    def test_1d_mode_cbar_lims(self, view, mode_solution):
        assert self.cbar_matches(view, mode_solution)

    def test_1d_ylabel(self, view):
        assert view.ax.get_ylabel() == "time"

    def test_1d_cbar_label(self, view):
        assert view.cbar.ax.get_ylabel() == "Re($\\rho$)"


class TestTemporal1DCart1Mode(TemporalTest):
    filename = "temporal_1d_cart_1mode_rho.npy"
    omega = 1.19029136 + 3.75969744j
    time = np.linspace(0, 2, 100)

    @pytest.fixture(scope="class")
    def ds(self, ds_v121_rti_khi):
        return ds_v121_rti_khi

    def test_invalid_u2(self, ds):
        with pytest.raises(ValueError):
            pylbo.plot_1d_temporal_evolution(
                ds, self.omega, "rho", u2=[0], u3=1, time=0
            )

    def test_invalid_u3(self, ds):
        with pytest.raises(ValueError):
            pylbo.plot_1d_temporal_evolution(
                ds, self.omega, "rho", u2=1, u3=[0], time=0
            )

    def test_animation_fail(self, view):
        with pytest.raises(ValueError):
            view.create_animation(times=np.linspace(0, 1, 10), filename="test.mp4")


class TestTemporal1dCyl1Mode(TemporalTest):
    filename = "temporal_1d_cyl_1mode_rho.npy"
    omega = 0.01746995 + 0.02195201j
    time = np.linspace(0, 100, 100)

    @pytest.fixture(scope="class")
    def ds(self, ds_v121_magth):
        return ds_v121_magth


class TestTemporal1dCart3Modes(TestTemporal1DCart1Mode):
    filename = "temporal_1d_cart_3modes_rho.npy"
    omega = [1.19029 + 3.75969j, 0.65179 + 1.32900j, 0.16193 + 0.45623j]


class TestTemporal1dCart3Modes3Factors(TestTemporal1DCart1Mode):
    filename = "temporal_1d_cart_3modes_3factors_rho.npy"
    omega = [1.19029 + 3.75969j, 0.65179 + 1.32900j, 0.16193 + 0.45623j]
    factor = [1.0, 1.0j, 1.0]


class TestTemporal1dCyl6Datasets6Modes(TemporalTest):
    filename = "temporal_1d_cyl_6datasets_6modes_rho.npy"
    omega = np.linspace(0.2, 0.6, 6)
    time = np.linspace(0, 100, 100)

    @pytest.fixture(scope="function")
    def ds(self, series_v211_mri_parameter):
        return series_v211_mri_parameter


class TestTemporal1dCyl6Datasets8Modes(TestTemporal1dCyl6Datasets6Modes):
    filename = "temporal_1d_cyl_6datasets_8modes_rho.npy"
    omega = list(np.linspace(0.2, 0.6, 5))
    omega.append([0.6, 0.5])
