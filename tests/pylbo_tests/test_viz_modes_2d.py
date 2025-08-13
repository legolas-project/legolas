import numpy as np
import pylbo
import pytest
from pathlib import Path

from .viz_modes import ModeVizTest
from matplotlib.testing.compare import compare_images


class Slice2D(ModeVizTest):
    @property
    def xlabel(self):
        pass

    @property
    def ylabel(self):
        pass

    @property
    def u2vals(self):
        pass

    @property
    def u3vals(self):
        pass

    @property
    def omega(self):
        pass

    @property
    def background(self):
        return False

    @property
    def ef_name(self):
        return "rho"

    @pytest.fixture(scope="function")  # function scope to avoid caching
    def view(self, ds):
        p = pylbo.plot_2d_slice(
            ds,
            omega=self.omega,
            ef_name=self.ef_name,
            u2=self.u2vals,
            u3=self.u3vals,
            time=0,
            slicing_axis=self.slicing_axis,
            add_background=self.background,
        )
        p.draw()
        return p

    def test_cbar_lims(self, view, mode_solution):
        assert self.cbar_matches(view, mode_solution)

    def test_labels(self, view):
        assert view.ax.get_xlabel() == self.xlabel
        assert view.ax.get_ylabel() == self.ylabel
        assert view.cbar.ax.get_ylabel() == "Re($\\rho$)"


class TestSliceZ_2DCart(Slice2D):
    filename = "slice_2d_z_cart_rho.npy"
    omega = 1.19029 + 3.75969j
    slicing_axis = "z"
    u2vals = np.linspace(0, 2, 50)
    u3vals = 0
    xlabel = "x"
    ylabel = "y"

    @pytest.fixture(scope="class")
    def ds(self, ds_v121_rti_khi):
        return ds_v121_rti_khi

    def test_invalid_coords(self, ds):
        with pytest.raises(ValueError):
            pylbo.plot_2d_slice(ds, self.omega, "rho", self.u2vals, self.u2vals, 0, "z")

    def test_invalid_time(self, ds):
        with pytest.raises(ValueError):
            pylbo.plot_2d_slice(
                ds, self.omega, "rho", self.u2vals, self.u3vals, [1, 2, 3], "z"
            )

    def test_contour_empty(self, view, mode_solution):
        view.set_contours(levels=20, fill=False)
        view.draw()
        assert self.cbar_matches(view, mode_solution)

    def test_contour_filled(self, view, mode_solution):
        view.set_contours(levels=20, fill=True)
        view.draw()
        assert self.cbar_matches(view, mode_solution)

    def test_invalid_slicing_axis(self, ds):
        with pytest.raises(ValueError):
            pylbo.plot_2d_slice(ds, self.omega, "rho", self.u2vals, self.u3vals, 0, "x")

    def test_no_top_panel(self, ds, view):
        assert len(view.fig.get_axes()) == 3
        view = pylbo.plot_2d_slice(
            ds, self.omega, "rho", self.u2vals, self.u3vals, 0, "z", show_ef_panel=False
        )
        view.draw()
        assert len(view.fig.get_axes()) == 2

    def test_animation(self, view, tmpdir, mode_solution):
        view.create_animation(
            times=np.arange(5), filename=tmpdir / "test_2d.mp4", fps=1
        )
        assert view.update_colorbar is True
        # check this did not modify solutions
        assert np.allclose(view.solutions, mode_solution)

    def test_animation_contour(self, view, tmpdir, mode_solution):
        view.set_contours(20, fill=False)
        view.draw()
        view.create_animation(
            times=np.arange(5), filename=tmpdir / "test_contour.mp4", fps=1
        )
        assert np.allclose(view.solutions, mode_solution)

    def test_animation_cbar_lock(self, view, tmpdir, mode_solution):
        view.update_colorbar = False
        view.create_animation(
            times=np.arange(5), filename=tmpdir / "test_2d_cbar_lock.mp4", fps=1
        )
        assert view.update_colorbar is False
        assert self.cbar_matches(view, mode_solution)
        assert np.allclose(view.solutions, mode_solution)

    def test_quivers_v(self, view, ds):
        view.add_quivers(
            xgrid=np.linspace(ds.ef_grid[0], ds.ef_grid[-1], 10),
            coordgrid=np.linspace(np.min(self.u2vals), np.max(self.u2vals), 8),
            field="v",
        )
        assert view.quiver_handler.quivers.U.size == 80

    def test_quivers_b(self, view, ds):
        view.add_quivers(
            xgrid=np.linspace(ds.ef_grid[0], ds.ef_grid[-1], 10),
            coordgrid=np.linspace(np.min(self.u2vals), np.max(self.u2vals), 8),
            field="B",
        )
        assert view.quiver_handler.quivers.U.size == 80

    def test_quivers_invalid_field(self, view):
        with pytest.raises(ValueError):
            view.add_quivers(
                xgrid=np.linspace(0, 1, 2), coordgrid=np.linspace(0, 1, 2), field="a"
            )

    def test_quivers_nogrids(self, view, ds):
        view.add_quivers(field="v")
        assert view.quiver_handler.quivers.U.size == len(ds.ef_grid) * len(self.u2vals)
        assert np.allclose(view.quiver_handler.quivers.X[: len(ds.ef_grid)], ds.ef_grid)

    def test_streamlines_invalid_field(self, view):
        with pytest.raises(ValueError):
            view.add_quivers(
                xgrid=np.linspace(0, 1, 2), coordgrid=np.linspace(0, 1, 2), field="a"
            )


class TestSliceZ_2DCartBackground(Slice2D):
    filename = "slice_2d_z_cart_rho_bg.npy"
    omega = 1.19029 + 3.75969j
    slicing_axis = "z"
    u2vals = np.linspace(0, 2, 50)
    u3vals = 0
    xlabel = "x"
    ylabel = "y"
    background = True

    @pytest.fixture(scope="class")
    def ds(self, ds_v121_rti_khi):
        return ds_v121_rti_khi

    def test_bg_with_vector_potential(self, ds):
        with pytest.raises(ValueError):
            pylbo.plot_2d_slice(
                ds,
                self.omega,
                "a1",
                self.u2vals,
                self.u3vals,
                0,
                "z",
                add_background=True,
            )

    def test_bg_with_derived(self, ds):
        with pytest.raises(ValueError):
            pylbo.plot_2d_slice(
                ds,
                self.omega,
                "S",
                self.u2vals,
                self.u3vals,
                0,
                "z",
                add_background=True,
            )

    def test_animation_with_bg(self, view, tmpdir, mode_solution):
        view.create_animation(
            times=np.arange(5), filename=tmpdir / "test_2d.mp4", fps=1
        )
        assert view.update_colorbar is True
        assert np.allclose(view.solutions, mode_solution)


class TestSliceZ_2DCartDerivedEigenfunctions(Slice2D):
    filename = "slice_2d_z_cart_b2.npy"
    omega = 1.19029 + 3.75969j
    slicing_axis = "z"
    u2vals = np.linspace(0, 2, 50)
    u3vals = 0
    xlabel = "x"
    ylabel = "y"
    ef_name = "B2"

    @pytest.fixture(scope="class")
    def ds(self, ds_v121_rti_khi):
        return ds_v121_rti_khi

    def test_labels(self, view):
        assert view.ax.get_xlabel() == self.xlabel
        assert view.ax.get_ylabel() == self.ylabel
        assert view.cbar.ax.get_ylabel() == "Re($B_y$)"


class TestSliceY_2DCart(Slice2D):
    filename = "slice_2d_y_cart_rho.npy"
    omega = 1.19029 + 3.75969j
    slicing_axis = "y"
    u2vals = 1
    u3vals = np.linspace(0, 2, 50)
    xlabel = "x"
    ylabel = "z"

    @pytest.fixture(scope="class")
    def ds(self, ds_v121_rti_khi):
        return ds_v121_rti_khi

    def test_invalid_coords(self, ds):
        with pytest.raises(ValueError):
            pylbo.plot_2d_slice(ds, self.omega, "rho", self.u3vals, self.u3vals, 0, "y")

    def test_contour_empty(self, view, mode_solution):
        view.set_contours(levels=20, fill=False)
        view.draw()
        assert self.cbar_matches(view, mode_solution)

    def test_animation_dots(self, view, tmpdir, mode_solution):
        view.create_animation(
            times=np.arange(5),
            filename=tmpdir / "test_2d_dots.mp4",
            fps=1,
            draw_dots=True,
            ndots=6,
        )
        assert view._view_dot is not None
        xy = view._view_dot.get_offsets()
        x, _ = np.array(xy).transpose()
        assert len(x) == 6
        assert view.update_colorbar is True
        assert np.allclose(view.solutions, mode_solution)

    def test_streamlines_nogrids(self, view, tmpdir, modebaselinedir):
        view.add_streamlines(field="v")
        image_test = tmpdir / "slice_2d_y_cart_streamlines.png"
        view.save(filename=image_test)

        image_baseline = modebaselinedir / "slice_2d_y_cart_streamlines.png"
        result = compare_images(str(image_baseline), str(image_test), tol=2)
        if result is not None:
            pytest.fail(result, pytrace=False)
        # test succeeded if result = None
        if result is None:
            Path(image_test).unlink()


class TestSliceZ_2DCyl(Slice2D):
    filename = "slice_2d_z_cyl_rho.npy"
    omega = 0.01746995 + 0.02195201j
    slicing_axis = "z"
    u2vals = np.linspace(0, 2 * np.pi, 50)
    u3vals = 1
    xlabel = "r"
    ylabel = r"$\theta$"

    @pytest.fixture(scope="class")
    def ds(self, ds_v121_magth):
        return ds_v121_magth

    def test_contour_empty(self, view, mode_solution):
        view.set_contours(levels=20, fill=False)
        view.draw()
        assert self.cbar_matches(view, mode_solution)

    def test_contour_filled(self, view, mode_solution):
        view.set_contours(levels=20, fill=True)
        view.draw()
        assert self.cbar_matches(view, mode_solution)

    def test_polar_plot(self, ds, mode_solution):
        view = pylbo.plot_2d_slice(
            ds, self.omega, "rho", self.u2vals, self.u3vals, 0, "z", polar=True
        )
        view.draw()
        assert self.cbar_matches(view, mode_solution)

    def test_no_top_panel(self, ds, view):
        assert len(view.fig.get_axes()) == 3
        view = pylbo.plot_2d_slice(
            ds, self.omega, "rho", self.u2vals, self.u3vals, 0, "z", show_ef_panel=False
        )
        assert len(view.fig.get_axes()) == 2

    def test_polar_contour(self, ds, mode_solution):
        view = pylbo.plot_2d_slice(
            ds, self.omega, "rho", self.u2vals, self.u3vals, 0, "z", polar=True
        )
        view.set_contours(levels=20, fill=False)
        view.draw()
        assert self.cbar_matches(view, mode_solution)

    def test_animation(self, view, tmpdir, mode_solution):
        view.create_animation(
            times=np.arange(5), filename=tmpdir / "test_2d.mp4", fps=1
        )
        assert view.update_colorbar is True
        assert np.allclose(view.solutions, mode_solution)

    def test_animation_contour(self, view, tmpdir, mode_solution):
        view.set_contours(20, fill=False)
        view.draw()
        view.create_animation(
            times=np.arange(5), filename=tmpdir / "test_contour.mp4", fps=1
        )
        assert np.allclose(view.solutions, mode_solution)

    def test_quivers_invalid_field(self, view):
        with pytest.raises(ValueError):
            view.add_quivers(
                xgrid=np.linspace(0, 1, 2), coordgrid=np.linspace(0, 1, 2), field="B"
            )

    def test_quivers_polar(self, view, ds):
        view = pylbo.plot_2d_slice(
            ds, self.omega, "rho", self.u2vals, self.u3vals, 0, "z", polar=True
        )
        view.add_quivers(field="v")
        assert view.quiver_handler.quivers.U.size == len(ds.ef_grid) * len(self.u2vals)
        assert np.allclose(
            view.quiver_handler.quivers.X[: len(self.u2vals)], self.u2vals
        )


class TestSliceTheta_2DCyl(Slice2D):
    filename = "slice_2d_theta_cyl_rho.npy"
    omega = 0.01746995 + 0.02195201j
    slicing_axis = "theta"
    u2vals = np.pi
    u3vals = np.linspace(0, 2, 50)
    xlabel = "r"
    ylabel = "z"

    @pytest.fixture(scope="class")
    def ds(self, ds_v121_magth):
        return ds_v121_magth

    def test_contour_filled(self, view, mode_solution):
        view.set_contours(levels=25, fill=True)
        view.draw()
        assert self.cbar_matches(view, mode_solution)

    def test_streamlines_polar(self, view, tmpdir, modebaselinedir):
        view.add_streamlines(field="v")
        image_test = tmpdir / "slice_2d_theta_cyl_streamlines.png"
        view.save(filename=image_test)

        image_baseline = modebaselinedir / "slice_2d_theta_cyl_streamlines.png"
        result = compare_images(str(image_baseline), str(image_test), tol=2)
        if result is not None:
            pytest.fail(result, pytrace=False)
        # test succeeded if result = None
        if result is None:
            Path(image_test).unlink()
