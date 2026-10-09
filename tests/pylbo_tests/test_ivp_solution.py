import numpy as np
import pylbo
import pytest
from pylbo.ivp_solution import IVPSolution

from .conftest import utils


@pytest.fixture
def plane_wave():
    """F(x) = g(x) e^{i phi0} for rho and a different profile for v1;
    k2 = 0.5, k3 = 2."""
    x = np.linspace(0, 1, 11)
    times = np.array([0.0, 0.1, 0.2])
    phi0 = 0.3
    g = np.cos(np.pi * x)
    rho = np.array([g * np.exp(1j * phi0) * (1 + i) for i in range(3)])
    v1 = np.array([1j * x * (1 + i) for i in range(3)])
    data = np.stack([rho, v1], axis=1)
    sol = IVPSolution(
        times,
        data,
        {0: "rho", 1: "v1"},
        x_domain=x,
        k2=0.5,
        k3=2.0,
        geometry="Cartesian",
    )
    return sol, x, g, phi0


def test_component_by_name_is_complex(plane_wave):
    sol, x, g, phi0 = plane_wave
    rho = sol.get_component("rho")
    assert rho.shape == (3, 11)
    assert np.iscomplexobj(rho)
    assert np.allclose(rho[0], g * np.exp(1j * phi0))
    assert np.allclose(sol.get_component(1), sol.get_component("v1"))


def test_unknown_component_raises(plane_wave):
    sol = plane_wave[0]
    with pytest.raises(ValueError):
        sol.get_component("B1")


def test_snapshot_index_from_time(plane_wave):
    sol = plane_wave[0]
    assert sol.get_snapshot_index(0.12) == 1
    assert sol.get_snapshot_index(5.0) == 2


def test_reconstruct_at_origin_is_real_part(plane_wave):
    sol, x, g, phi0 = plane_wave
    field = sol.reconstruct("rho", 0)
    assert field.shape == (11,)
    assert np.allclose(field, g * np.cos(phi0))


def test_reconstruct_follows_the_fourier_ansatz(plane_wave):
    sol, x, g, phi0 = plane_wave
    y = np.linspace(0, 2 * np.pi, 7)
    z = np.array([0.0, 0.25, 1.0])
    field = sol.reconstruct("rho", 2, coord2=y, coord3=z)
    assert field.shape == (11, 7, 3)
    expected = (
        3
        * g[:, None, None]
        * np.cos(0.5 * y[None, :, None] + 2.0 * z[None, None, :] + phi0)
    )
    assert np.allclose(field, expected)


def test_reconstruct_imaginary_profile_is_in_quadrature(plane_wave):
    sol, x, g, phi0 = plane_wave
    z = np.array([0.0, np.pi / 4])  # k3 z = 0 and pi/2
    field = sol.reconstruct("v1", 0, coord3=z)
    assert field.shape == (11, 2)
    assert np.allclose(field[:, 0], 0.0)
    assert np.allclose(field[:, 1], -x)


def test_reconstruct_accepts_a_time_and_keeps_axes(plane_wave):
    sol = plane_wave[0]
    field = sol.reconstruct("rho", 0.21, coord2=1.0, coord3=np.zeros(4), squeeze=False)
    assert field.shape == (11, 1, 4)
    assert np.allclose(
        field[:, 0, :], sol.reconstruct("rho", 2, coord2=1.0, coord3=np.zeros(4))
    )


def test_snapshots_from_datfile():
    ds = pylbo.load(utils / "ivp_demo_snapshots.dat")
    sol = ds.get_iv_snapshots()
    assert list(sol.component_names.values()) == ["rho", "v1"]
    assert sol.k2 == pytest.approx(0.5)
    assert sol.k3 == pytest.approx(1.0)
    rho = sol.get_component("rho")
    assert np.iscomplexobj(rho)
    assert rho.shape == (len(sol.times), len(sol.x_domain))
    assert np.allclose(sol.x_domain[::2], ds.grid)
    assert len(sol.times) == 5
    assert sol.times[-1] == pytest.approx(0.5)
    assert np.all(np.isfinite(rho))
    field = sol.reconstruct(
        "rho", 0.5, coord2=np.linspace(0, 1, 3), coord3=np.linspace(0, 1, 4)
    )
    assert field.shape == (len(sol.x_domain), 3, 4)
    assert np.allclose(field[:, 0, 0], rho[-1].real)
