import numpy as np
import pytest
from pylbo.data_containers import LegolasDataSeries, LegolasDataSet
from pylbo.exceptions import BackgroundNotPresent


def test_series_iterable(series_v112):
    for ds in series_v112:
        assert isinstance(ds, LegolasDataSet)


def test_series_getitem(series_v112):
    ds = series_v112[1]
    assert isinstance(ds, LegolasDataSet)


def test_series_getslice(series_v112):
    series_part = series_v112[0:2]
    assert isinstance(series_part, LegolasDataSeries)
    assert len(series_part) == 2
    for ds in series_part:
        assert isinstance(ds, LegolasDataSet)


def test_series_efs_none(series_v100):
    assert isinstance(series_v100.has_efs, np.ndarray)
    assert not np.all(series_v100.has_efs)


def test_series_has_efs(series_v112):
    assert isinstance(series_v112.has_efs, np.ndarray)
    assert np.all(series_v112.has_efs)


def test_series_ef_names_not_present(series_v100):
    assert np.all(series_v100.ef_names == [None] * len(series_v100))


def test_series_ef_names(series_v112):
    assert isinstance(series_v112.ef_names, np.ndarray)


def test_series_ef_grid(series_v112):
    grids = series_v112.ef_grid
    assert isinstance(grids, np.ndarray)
    assert len(grids) == len(series_v112)
    for grid in grids:
        assert isinstance(grid, np.ndarray)
        assert np.all(np.isreal(grid))


def test_series_ef_grid_none(series_v100):
    grids = series_v100.ef_grid
    assert isinstance(grids, np.ndarray)
    assert len(grids) == len(series_v100)
    assert all(grid is None for grid in grids)


def test_series_sound_speed(series_v112):
    cs_avg = series_v112.get_sound_speed(which_values="average")
    assert np.all(np.isclose(cs_avg, [0.35804914] * len(series_v112)))


def test_series_alfven_speed(series_v112):
    ca_avg = series_v112.get_alfven_speed(which_values="average")
    assert np.all(np.isclose(ca_avg, [0.82264676] * len(series_v112)))


def test_series_tube_speed(series_v112):
    ct_avg = series_v112.get_tube_speed(which_values="average")
    assert np.all(np.isclose(ct_avg, [0.3282505] * len(series_v112)))


def test_series_tube_speed_cartesian(series_v100):
    ct = series_v100.get_tube_speed()
    assert isinstance(ct, np.ndarray)
    assert all(val is None for val in ct)


def test_series_reynolds_nb_no_eta(series_v112):
    reynolds = series_v112.get_reynolds_nb()
    assert isinstance(reynolds, np.ndarray)
    assert all(re is None for re in reynolds)


def test_series_reynolds_nb(series_v112_eta):
    reynolds = series_v112_eta.get_reynolds_nb(which_values="average")
    assert isinstance(reynolds, np.ndarray)
    assert np.all(np.isclose(reynolds, [3535.53390593] * len(series_v112_eta)))


def test_series_magnetic_reynolds_nb_no_eta(series_v112):
    magnetic_reynolds = series_v112.get_magnetic_reynolds_nb()
    assert isinstance(magnetic_reynolds, np.ndarray)
    assert all(val is None for val in magnetic_reynolds)


def test_series_magnetic_reynolds_nb(series_v112_eta):
    magnetic_reynolds = series_v112_eta.get_magnetic_reynolds_nb(which_values="average")
    assert isinstance(magnetic_reynolds, np.ndarray)
    assert np.all(np.isclose(magnetic_reynolds, [1e4] * len(series_v112_eta)))


def test_series_get_k0_squared(series_v112):
    k02 = series_v112.get_k0_squared()
    assert isinstance(k02, np.ndarray)
    assert np.all(np.isclose(k02, 1 + (-1.2) ** 2))


def test_series_get_continua(series_v112):
    continua = series_v112.continua
    assert isinstance(continua, dict)
    for value in continua.values():
        assert isinstance(value, np.ndarray)
        assert len(value) == len(series_v112)


def test_series_get_parameters(series_v112):
    params = series_v112.parameters
    assert isinstance(params, dict)
    for value in params.values():
        assert isinstance(value, np.ndarray)
        assert len(value) == len(series_v112)


def test_series_no_bg(series_v200_nobg):
    assert not any(series_v200_nobg.has_background)


def test_series_nobg_continua(series_v200_nobg):
    continua = series_v200_nobg.continua
    assert isinstance(continua, np.ndarray)
    assert all(val is None for val in continua)


def test_series_nobg_soundspeed(series_v200_nobg):
    with pytest.raises(BackgroundNotPresent):
        series_v200_nobg.get_sound_speed()


def test_series_nobg_alfven_speed(series_v200_nobg):
    with pytest.raises(BackgroundNotPresent):
        series_v200_nobg.get_alfven_speed()


def test_series_nobg_tube_speed(series_v200_nobg):
    with pytest.raises(BackgroundNotPresent):
        series_v200_nobg.get_tube_speed()


def test_series_derived_ef_names(series_v200_mri_efs):
    names = series_v200_mri_efs.derived_ef_names
    expected = series_v200_mri_efs[0].derived_ef_names
    assert names.shape == (len(series_v200_mri_efs), len(expected))
    assert np.all(names == expected)


def test_series_has_subset(series_v200_mixed_efs):
    has_subset = series_v200_mixed_efs.has_ef_subset
    assert isinstance(has_subset, np.ndarray)
    assert np.all(has_subset == [True, False, True, False, True, False])


def test_series_has_eigenvecs(series_v200_mixed_efs):
    has_eigenvecs = series_v200_mixed_efs.has_eigenvectors
    assert isinstance(has_eigenvecs, np.ndarray)
    assert np.all(has_eigenvecs == [True, False, True, False, True, False])


def test_series_has_residuals(series_v200_mixed_efs):
    has_residuals = series_v200_mixed_efs.has_residuals
    assert isinstance(has_residuals, np.ndarray)
    assert np.all(has_residuals == [True, False, True, False, True, False])


def test_series_max_eigenvalue(series_v200_mri_efs):
    evs = series_v200_mri_efs.get_omega_max(real=True)
    expected = np.array([11.184804375] * len(series_v200_mri_efs), dtype=complex)
    assert np.all(np.isclose(expected, evs))
    ev = series_v200_mri_efs.get_omega_max(real=False)
    expected = np.array([-0.001919987625 + 0.6214319226j] * len(series_v200_mri_efs))
    assert np.all(np.isclose(expected, ev))
