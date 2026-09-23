import numpy as np

SUBSET_FLAGS_KEY = "ef_written_flags"
SUBSET_IDXS_KEY = "ef_written_idxs"
SUBSET_CENTER_KEY = "ef_subset_center"
SUBSET_RADIUS_KEY = "ef_subset_radius"


def test_subset_present(ds_v114_subset):
    assert ds_v114_subset.has_ef_subset


def test_subset_center(ds_v114_subset):
    center = ds_v114_subset.header[SUBSET_CENTER_KEY]
    assert isinstance(center, complex)
    assert np.isclose(center, 20 + 1j)


def test_subset_radius(ds_v114_subset):
    radius = ds_v114_subset.header[SUBSET_RADIUS_KEY]
    assert isinstance(radius, float)
    assert np.isclose(radius, 15)


def test_subset_eigenfunction_flags(ds_v114_subset):
    flags = ds_v114_subset.header[SUBSET_FLAGS_KEY]
    assert len(flags) == len(ds_v114_subset.eigenvalues)
    assert len(*np.where(flags)) == 6


def test_subset_eigenfunction_flags_full(ds_v112):
    flags = ds_v112.header[SUBSET_FLAGS_KEY]
    assert len(flags) == len(ds_v112.eigenvalues)
    assert len(*np.where(flags)) == len(ds_v112.eigenvalues)


def test_subset_eigenfuncion_idxs(ds_v114_subset):
    idxs = ds_v114_subset.header[SUBSET_IDXS_KEY]
    assert len(idxs) == 6


def test_subset_eigenfunction_idxs_full(ds_v112):
    idxs = ds_v112.header[SUBSET_IDXS_KEY]
    assert len(idxs) == len(ds_v112.eigenvalues)


def test_subset_eigenfunction_retrieval_inside(ds_v114_subset):
    guess = 18 + 0.01j
    _, ev = ds_v114_subset.get_nearest_eigenvalues(ev_guesses=guess)
    (efs,) = ds_v114_subset.get_eigenfunctions(ev_guesses=guess)
    assert ev is not None
    assert efs is not None
    assert isinstance(efs, dict)
    assert np.isclose(efs["eigenvalue"], ev)
    assert set(ds_v114_subset.ef_names).issubset(efs.keys())


def test_subset_eigenfunction_retrieval_outside(ds_v114_subset):
    guess = 50 - 0.002j
    _, ev = ds_v114_subset.get_nearest_eigenvalues(ev_guesses=guess)
    (efs,) = ds_v114_subset.get_eigenfunctions(ev_guesses=guess)
    assert ev is not None
    assert not np.isnan(ev)
    assert efs is None


def test_subset_derived_efs_present(ds_v114_subset_defs):
    assert ds_v114_subset_defs.has_ef_subset
    assert ds_v114_subset_defs.has_derived_efs


def test_subset_derived_efs_retrieval_inside(ds_v114_subset_defs):
    guess = 10.0 - 0.1j
    _, ev = ds_v114_subset_defs.get_nearest_eigenvalues(ev_guesses=guess)
    (efs,) = ds_v114_subset_defs.get_eigenfunctions(ev_guesses=guess)
    assert ev is not None
    assert not np.isnan(ev)
    assert efs is not None
    assert isinstance(efs, dict)
    assert np.isclose(efs["eigenvalue"], ev)
    assert set(ds_v114_subset_defs.derived_ef_names).issubset(efs.keys())


def test_subset_derived_efs_retrieval_outside(ds_v114_subset_defs):
    guess = 35.0 - 0.01j
    _, ev = ds_v114_subset_defs.get_nearest_eigenvalues(ev_guesses=guess)
    (efs,) = ds_v114_subset_defs.get_eigenfunctions(ev_guesses=guess)
    assert ev is not None
    assert not np.isnan(ev)
    assert efs is None
