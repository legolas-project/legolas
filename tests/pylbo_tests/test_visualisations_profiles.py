import numpy as np
import pylbo


def test_plot_equilibrium_profile(ds_v112):
    p = pylbo.plot_equilibrium(ds_v112)
    assert p is not None
    p.draw()


def test_plot_continuum_profile(ds_v112):
    p = pylbo.plot_continua(ds_v112)
    assert p is not None
    p.draw()


def test_plot_matrices(ds_v100):
    p = pylbo.plot_matrices(ds_v100)
    assert p is not None
    p.draw()


def test_plot_equilibrium_profile_no_bg(ds_v200_tear_nobg):
    p = pylbo.plot_equilibrium(ds_v200_tear_nobg)
    assert p is None


def test_plot_equilibrium_balance_no_bg(ds_v200_tear_nobg):
    p = pylbo.plot_equilibrium_balance(ds_v200_tear_nobg)
    assert p is None


def test_plot_equilibrium_balance(ds_v200_mri_efs):
    p = pylbo.plot_equilibrium_balance(ds_v200_mri_efs)
    assert p is not None
    p.draw()


def test_plot_equilibrium_balance_hd(ds_v200_hd_khi):
    p = pylbo.plot_equilibrium_balance(ds_v200_hd_khi)
    assert p is not None
    p.draw()


def test_plot_continua_allreal(ds_v200_mri_efs):
    p = pylbo.plot_continua(ds_v200_mri_efs)
    assert p is not None
    p.draw()


def test_plot_continua_slow_imag(fake_ds, monkeypatch):
    monkeypatch.setitem(
        fake_ds.continua,
        "slow+",
        np.random.rand(fake_ds.gauss_gridpoints, 2).view(complex).flatten(),
    )
    p = pylbo.plot_continua(fake_ds)
    assert p is not None
    p.draw()


def test_plot_continua_thermal_imag(fake_ds, monkeypatch):
    # fill with random complex data
    monkeypatch.setitem(
        fake_ds.continua,
        "thermal",
        np.random.rand(fake_ds.gauss_gridpoints) * 1j,
    )
    p = pylbo.plot_continua(fake_ds)
    assert p is not None
    p.draw()


def test_eq_balance(monkeypatch, ds_v200_mri_efs):
    monkeypatch.setitem(
        ds_v200_mri_efs.equilibria, "v01", np.ones_like(ds_v200_mri_efs.grid_gauss)
    )
    balance = pylbo.get_equilibrium_balance(ds_v200_mri_efs)
    assert balance.get("continuity", None) is not None
    assert balance.get("induction 1", None) is not None
