import numpy as np
import pylbo
import pytest

DEFAULT_MHD_BASIS = {
    "rho": "quadratic",
    "v1": "cubic",
    "v2": "quadratic",
    "v3": "quadratic",
    "T": "quadratic",
    "a1": "quadratic",
    "a2": "cubic",
    "a3": "cubic",
}


def test_invalid_file():
    with pytest.raises(FileNotFoundError):
        pylbo.load("unknown_file")


def test_invalid_suffix(tmpdir):
    from pylbo.exceptions import InvalidLegolasFile

    parfile = tmpdir / "parfile.par"
    parfile.write_text("content")
    with pytest.raises(InvalidLegolasFile):
        pylbo.load(parfile)


def test_load_invalid(datv0):
    with pytest.raises(ValueError):
        pylbo.load([datv0, datv0])


def test_load_no_version(datv0):
    ds = pylbo.load(datv0)
    assert ds.legolas_version == "0.0.0"


def test_load_series_empty():
    with pytest.raises(ValueError):
        pylbo.load_series([])


def test_load_multiple_equilibria(datv1, datv112_eta):
    with pytest.raises(ValueError):
        pylbo.load_series([datv1, datv112_eta])


def test_load_logfile(logv0):
    eigenvals = pylbo.load_logfile(logv0)
    assert isinstance(eigenvals, np.ndarray)


def test_load_logfile_and_sort(logv0):
    eigenvals = pylbo.load_logfile(logv0, sort=True)
    assert isinstance(eigenvals, np.ndarray)
    assert np.all(eigenvals[:-1] <= eigenvals[1:])


def test_load_v200_matrices(ds_v200_mri_matrix):
    assert ds_v200_mri_matrix.legolas_version >= "2.0"
    assert ds_v200_mri_matrix.has_matrices


def test_load_v200_matrices_get_B(ds_v200_mri_matrix):
    result = ds_v200_mri_matrix.get_matrix_B()
    assert len(result) == 3
    assert np.all(result is not None)


def test_load_v200_matrices_get_A(ds_v200_mri_matrix):
    result = ds_v200_mri_matrix.get_matrix_A()
    assert len(result) == 3
    assert np.all(result is not None)


def test_load_v200_efs(ds_v200_mri_efs):
    assert ds_v200_mri_efs.legolas_version >= "2.0"
    assert ds_v200_mri_efs.has_efs
    assert ds_v200_mri_efs.has_derived_efs
    assert ds_v200_mri_efs.has_ef_subset


def test_load_v200_efs_get_eigenvectors(ds_v200_mri_efs):
    assert ds_v200_mri_efs.has_eigenvectors
    result = ds_v200_mri_efs.get_eigenvectors()
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_load_v200_efs_get_residuals(ds_v200_mri_efs):
    assert ds_v200_mri_efs.has_residuals
    result = ds_v200_mri_efs.get_residuals()
    assert result is not None
    assert isinstance(result, np.ndarray)


def test_load_v210_default_basis(ds_v210_rotcyl_efs_default_basis):
    ds = ds_v210_rotcyl_efs_default_basis
    assert ds.legolas_version >= "2.1"
    assert ds.header.get("basis_functions") == DEFAULT_MHD_BASIS


def test_load_v210_custom_basis(ds_v210_rotcyl_efs_custom_basis):
    ds = ds_v210_rotcyl_efs_custom_basis
    assert ds.legolas_version >= "2.1"
    expected = {
        "rho": "quadratic",
        "v1": "cubic",
        "v2": "cubic",
        "v3": "quadratic",
        "T": "cubic",
        "a1": "quadratic",
        "a2": "cubic",
        "a3": "quadratic",
    }
    assert ds.header.get("basis_functions") == expected


def test_load_v211_parameter(series_v211_mri_parameter):
    ds = series_v211_mri_parameter
    assert np.allclose(
        ds.parameters["k3"],
        np.array(sorted([50.0, 80.0] * 3)),
    )
