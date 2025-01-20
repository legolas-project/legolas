import pytest
from pylbo.visualisation.modes.mode_data import ModeVisualisationData


class ModeDataTest:
    @property
    def omega(self):
        pass

    @property
    def complex_factor(self):
        pass


class TestModeDataDataseriesResolution(ModeDataTest):
    omega = 0.6j

    @pytest.fixture(scope="function")
    def ds(self, series_v211_mixed_res):
        return series_v211_mixed_res

    def test_resolution(self, ds):
        with pytest.raises(ValueError):
            ModeVisualisationData(ds, [self.omega] * len(ds), "rho")


class TestModeDataDataseriesFormat(ModeDataTest):
    omega = 0.6j
    complex_factor = 1.0j

    @pytest.fixture(scope="function")
    def ds(self, series_v200_mri_efs):
        return series_v200_mri_efs

    def test_omega_format(self, ds):
        omegas = [self.omega] * len(ds)
        with pytest.raises(ValueError):
            ModeVisualisationData(ds, [omegas], "rho")

    def test_omega_len(self, ds):
        with pytest.raises(ValueError):
            ModeVisualisationData(ds, [self.omega] * (len(ds) - 1), "rho")

    def test_different_format(self, ds):
        factors = [self.complex_factor] * (len(ds) - 1)
        factors.append([self.complex_factor] * 2)
        with pytest.raises(ValueError):
            ModeVisualisationData(
                ds,
                [self.omega] * len(ds),
                "rho",
                complex_factor=factors,
            )


class TestModeDataDataseriesEfs(ModeDataTest):
    omega = 0.6j

    @pytest.fixture(scope="function")
    def ds(self, series_v200_mixed_efs):
        return series_v200_mixed_efs

    def test_efs(self, ds):
        with pytest.raises(ValueError):
            ModeVisualisationData(ds, [self.omega] * len(ds), "rho")


class TestModeDataDataseriesDerivedEfs(ModeDataTest):
    omega = 0.6j

    @pytest.fixture(scope="function")
    def ds(self, series_v211_mixed_derived_efs):
        return series_v211_mixed_derived_efs

    def test_derived_efs(self, ds):
        with pytest.raises(ValueError):
            ModeVisualisationData(ds, [self.omega] * len(ds), "B1")
