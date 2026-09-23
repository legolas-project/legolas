import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pylbo
import pylbo.testing
import pytest
from matplotlib.testing.compare import compare_images

testlog = logging.getLogger("test_logger")

only_for_baseline_generation = pytest.mark.skipif(
    condition="not config.getoption('generate_baseline')",
    reason="'--generate' option not passed",
)

SOLVERS_WITHOUT_BASELINE_GENERATION = ["QZ-direct", "QR-cholesky"]
DEFAULT_LEGOLAS_EXEC = Path(os.environ["LEGOLASDIR"]) / "legolas"


def use_existing_baseline(capturemanager, baseline):
    use_existing = False
    if baseline.is_file():
        testlog.info(f"baseline file '{baseline.name}' is already present!")
        capturemanager.suspend_global_capture(in_=True)
        use_existing = input("Regenerate this file? ").lower() not in ("yes", "y")
        capturemanager.resume_global_capture()
    return use_existing


def validate_eigenfunctions_present(ds, present, which_efs):
    if not present:
        pytest.fail(f"{which_efs} are not present in {ds.datfile.name}")


def validate_subplot_sizes(ef_names, axes):
    nb_axs = len(axes.flatten())
    nb_efs = len(ef_names)
    if nb_efs > nb_axs:
        pytest.fail(
            f"number of eigenfunctions ({nb_efs}) is larger than "
            f"number of axes ({nb_axs})"
        )


class TestCase:
    SAVEFIG_KWARGS = {"dpi": 200, "transparent": True}
    RMS_TOLERANCE = 2
    executable = DEFAULT_LEGOLAS_EXEC

    gridpoints = 51
    logging_level = 1
    show_results = False

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def filename(self):
        raise NotImplementedError()

    @property
    def parameters(self):
        raise NotImplementedError()

    @property
    def equilibrium(self):
        raise NotImplementedError()

    @property
    def geometry(self):
        raise NotImplementedError()

    @property
    def number_of_runs(self):
        return 1

    @property
    def eigenfunction_settings(self):
        return {"write_eigenfunctions": False}

    @property
    def physics_settings(self):
        return {}

    @property
    def solver_settings(self):
        return {}

    @property
    def eigenvalues_are_real(self):
        return False

    def run_settings(self, outputdir):
        _setup = {
            "geometry": self.geometry,
            "x_start": getattr(self, "x_start", 0),
            "x_end": getattr(self, "x_end", 1),
            "gridpoints": self.gridpoints,
            "parameters": self.parameters,
            "equilibrium_type": self.equilibrium,
            "logging_level": self.logging_level,
            "show_results": self.show_results,
            "basename_datfile": self.filename,
            "output_folder": str(outputdir),
        }
        _setup.update(self.eigenfunction_settings)
        _setup.update(self.physics_settings)
        _setup.update(self.solver_settings)
        _setup.update({"number_of_runs": self.number_of_runs})
        return _setup

    def get_spectrum_image_filenames(self, limits):
        figname_test = f"{self.filename}"
        if limits is not None:
            xlim = limits["xlim"]
            ylim = limits["ylim"]
            figname_test = f"{figname_test}_Re{xlim[0]}-{xlim[1]}_Im{ylim[0]}-{ylim[1]}"
        figname_base = f"{figname_test}-baseline"
        return (
            self._spectradir / f"{figname_test}.png",
            self._spectradir / f"{figname_base}.png",
        )

    def get_eigenfunction_image_filenames(self, eigenvalue, prefix):
        figname_test = f"{self.filename}_{prefix}efs_w_{eigenvalue:.8f}"
        figname_base = f"{figname_test}-baseline"
        return (
            self._eigfuncdir / f"{figname_test}.png",
            self._eigfuncdir / f"{figname_base}.png",
        )

    def compare_test_images(self, image_test, image_baseline, tol):
        result = compare_images(str(image_baseline), str(image_test), tol=tol)
        if result is not None:
            pytest.fail(result, pytrace=False)
        # test succeeded if result = None, check if files are kept
        if result is None and not self._keep_files:
            Path(image_baseline).unlink()
            Path(image_test).unlink()

    @pytest.fixture(scope="class")
    def file_base(self, baselinedir):
        custom_baseline = getattr(self, "use_custom_baseline", None)
        if custom_baseline is not None:
            return baselinedir / f"BASE_{custom_baseline}.dat"
        return baselinedir / f"BASE_{self.filename}.dat"

    @pytest.fixture(scope="class")
    def file_test(self, datfiledir):
        return datfiledir / f"{self.filename}.dat"

    @pytest.fixture(scope="session")
    def capturemanager(self, pytestconfig):
        return pytestconfig.pluginmanager.getplugin("capturemanager")


class RegressionTest(TestCase):
    @only_for_baseline_generation
    def test_generate_baseline(self, capturemanager, file_base):
        setup = self.run_settings(outputdir=file_base.parent)
        setup.update({"basename_datfile": file_base.stem})
        if setup.get("solver") in SOLVERS_WITHOUT_BASELINE_GENERATION:
            pytest.skip(f"solver '{setup.get('solver')}' uses the QR-invert baseline")
        if use_existing_baseline(capturemanager, file_base):
            pytest.skip("using existing file")
        self.generate_test_dataset(setup)

    @pytest.fixture(scope="class")
    def ds_base(self, file_base):
        return pylbo.load(file_base)

    @pytest.fixture(scope="class")
    def ds_test(self, file_test, datfiledir):
        setup = self.run_settings(datfiledir)
        self.generate_test_dataset(setup)
        return pylbo.load(file_test)

    def generate_test_dataset(self, setup):
        testlog.info(f"generating dataset: {setup['basename_datfile']}.dat")
        parfile = pylbo.generate_parfiles(
            parfile_dict=setup,
            basename=setup["basename_datfile"],
            output_dir=self._datfiledir,
            subdir=False,
        )
        pylbo.run_legolas(parfile, executable=self.executable)

    def generate_spectrum_images(self, limits, ds_test, ds_base):
        p_test = pylbo.plot_spectrum(ds_test)
        p_base = pylbo.plot_spectrum(ds_base)
        figname_test, figname_base = self.get_spectrum_image_filenames(limits)
        xlim = limits["xlim"]
        ylim = limits["ylim"]
        for pp, name in [(p_test, figname_test), (p_base, figname_base)]:
            pp.ax.set_xlim(xlim)
            pp.ax.set_ylim(ylim)
            pp.ax.set_title(self.name)
            pp.draw()
            pp.fig.savefig(name, **self.SAVEFIG_KWARGS)
            plt.close(pp.fig)
        return (figname_test, figname_base)

    def generate_eigenfunction_images(
        self,
        eigenvalue,
        ds_test,
        ds_base,
        names_attr,
        figname_prefix="",
        nb_plots=(3, 3),
        figsize=(10, 10),
        set_to_zero_tol=1e-10,
    ):
        fig_test, ax_test = plt.subplots(*nb_plots, figsize=figsize, sharex="all")
        fig_base, ax_base = plt.subplots(*nb_plots, figsize=figsize, sharex="all")
        figname_test, figname_base = self.get_eigenfunction_image_filenames(
            eigenvalue, prefix=figname_prefix
        )

        for ds, ax in [(ds_test, ax_test), (ds_base, ax_base)]:
            (efs,) = ds.get_eigenfunctions(eigenvalue)
            ef_names = getattr(ds, names_attr)
            validate_subplot_sizes(ef_names, ax)
            for panel, ef_name in zip(ax.flatten(), ef_names):
                result = abs(efs[ef_name].real + efs[ef_name].imag)
                # small values
                result[np.where(result < set_to_zero_tol)] = 0
                panel.plot(ds.ef_grid, result, lw=3)
                panel.set_yticks([])
                panel.set_title(ef_name)
        for fig, name in [(fig_test, figname_test), (fig_base, figname_base)]:
            fig.suptitle(f"eigenvalue = {eigenvalue:.9f}")
            fig.tight_layout()
            fig.savefig(name, **self.SAVEFIG_KWARGS)
            plt.close(fig)
        return (figname_test, figname_base)

    @pytest.mark.required
    def test_generate_ds(self, ds_test):
        assert ds_test is not None

    @pytest.mark.required
    def test_file_base_exists(self, file_base):
        assert file_base.is_file()

    @pytest.mark.required
    def test_ds_base_exists(self, ds_base):
        assert ds_base is not None

    @pytest.mark.required
    def test_geometry(self, ds_test, ds_base):
        assert self.geometry == ds_test.geometry == ds_base.geometry

    def test_resolution(self, ds_test, ds_base):
        assert ds_test.gridpoints == ds_base.gridpoints

    def test_eigenvalue_types(self, ds_test, ds_base):
        if not self.eigenvalues_are_real:
            return
        tol = getattr(self, "custom_evs_all_real_tol", None)

        for ds, name in zip((ds_test, ds_base), ("test", "base")):
            if not np.all(ds.eigenvalues.imag == pytest.approx(0, abs=tol)):
                pytest.fail(
                    f"{name} ds: eigenvalues are not all real, found largest non-zero "
                    f"imaginary part abs(Im) = {np.max(np.abs(ds.eigenvalues.imag))} "
                    f"which is not within tolerance {1e-12 if tol is None else tol:.1e}"
                )

    def run_spectrum_test(self, limits, ds_test, ds_base):
        image_test, image_baseline = self.generate_spectrum_images(
            limits, ds_test, ds_base
        )
        super().compare_test_images(
            image_test,
            image_baseline,
            tol=limits.get("RMS_TOLERANCE", self.RMS_TOLERANCE),
        )

    def run_eigenfunction_test(self, eigenfunction, ds_test, ds_base):
        eigenvalue = eigenfunction["eigenvalue"]
        for ds in (ds_test, ds_base):
            validate_eigenfunctions_present(ds, ds.has_efs, "eigenfunctions")
        image_test, image_baseline = self.generate_eigenfunction_images(
            eigenvalue,
            ds_test,
            ds_base,
            names_attr="ef_names",
        )
        super().compare_test_images(
            image_test,
            image_baseline,
            tol=eigenfunction.get("RMS_TOLERANCE", self.RMS_TOLERANCE),
        )

    def run_derived_eigenfunction_test(self, derived_eigenfunction, ds_test, ds_base):
        eigenvalue = derived_eigenfunction["eigenvalue"]
        for ds in (ds_test, ds_base):
            validate_eigenfunctions_present(
                ds, ds.has_derived_efs, "derived eigenfunctions"
            )
        image_test, image_baseline = self.generate_eigenfunction_images(
            eigenvalue,
            ds_test,
            ds_base,
            names_attr="derived_ef_names",
            figname_prefix="derived_",
            nb_plots=(5, 5),
            figsize=(23, 10),
            set_to_zero_tol=1e-8,
        )
        super().compare_test_images(
            image_test,
            image_baseline,
            tol=derived_eigenfunction.get("RMS_TOLERANCE", self.RMS_TOLERANCE),
        )


class MultiRegressionTest(TestCase):
    @property
    def multispectrum_settings(self):
        raise NotImplementedError()

    @only_for_baseline_generation
    def test_generate_baseline(self, capturemanager, file_base):
        if use_existing_baseline(capturemanager, file_base.with_suffix(".pickle")):
            pytest.skip("using existing file")
        setup = self.run_settings(outputdir=file_base.parent)
        setup.update({"basename_datfile": file_base.stem})
        self.generate_test_dataseries(setup, capturemanager)
        # get generated files
        base_files = sorted(file_base.parent.glob(f"*{file_base.name}"))
        series = pylbo.load_series(base_files)
        # create pickled datadump
        pylbo.testing.pickle_dataseries_to_file(
            series, file_base.with_suffix(".pickle")
        )
        # remove tmp files
        [file.unlink() for file in base_files]

    @pytest.fixture(scope="class")
    def series_base(self, file_base):
        return pylbo.testing.load_pickled_dataseries(file_base.with_suffix(".pickle"))

    @pytest.fixture(scope="class")
    def series_test(self, capturemanager, datfiledir):
        setup = self.run_settings(datfiledir)
        setup.update({"number_of_runs": self.number_of_runs})
        self.generate_test_dataseries(setup, capturemanager)
        return pylbo.load_series(sorted(datfiledir.glob(f"*{self.filename}.dat")))

    def generate_test_dataseries(self, setup, capturemanager):
        testlog.info(f"generating dataseries: {setup['basename_datfile']}")
        parfiles = pylbo.generate_parfiles(
            parfile_dict=setup,
            basename=setup["basename_datfile"],
            output_dir=self._datfiledir,
            subdir=False,
        )
        capturemanager.suspend_global_capture()
        pylbo.run_legolas(parfiles, nb_cpus=2, executable=self.executable)
        capturemanager.resume_global_capture()

    def generate_multispectrum_images(self, series_test, series_base):
        settings = self.multispectrum_settings
        p_test = pylbo.plot_spectrum_multi(
            series_test,
            xdata=settings["xdata"],
            use_squared_omega=settings.get("use_squared_omega", True),
        )
        p_base = pylbo.plot_spectrum_multi(
            series_base,
            xdata=settings["xdata"],
            use_squared_omega=settings.get("use_squared_omega", True),
        )
        figname_test, figname_base = self.get_spectrum_image_filenames(limits=None)
        xlim = settings["xlim"]
        ylim = settings["ylim"]
        for pp, name in [(p_test, figname_test), (p_base, figname_base)]:
            pp.set_x_scaling(settings.get("x_scaling", 1))
            pp.set_y_scaling(settings.get("y_scaling", 1))
            if settings.get("symlog", None) is not None:
                pp.ax.set_yscale("symlog", linthresh=settings["symlog"])
            if settings.get("xlog"):
                pp.ax.set_xscale("log")
            if settings.get("ylog"):
                pp.ax.set_yscale("log")
            pp.ax.set_xlim(xlim)
            pp.ax.set_ylim(ylim)
            pp.ax.set_title(self.name)
            pp.draw()
            pp.fig.savefig(name, **self.SAVEFIG_KWARGS)
            plt.close(pp.fig)
        return (figname_test, figname_base)

    @pytest.mark.required
    def test_generate_series(self, series_test):
        assert series_test is not None

    @pytest.mark.required
    def test_file_base_exists(self, file_base):
        assert file_base.with_suffix(".pickle").is_file()

    @pytest.mark.required
    def test_geometry(self, series_test, series_base):
        assert np.all(self.geometry == series_test.geometry == series_base.geometry)

    @pytest.mark.required
    def test_resolution(self, series_test, series_base):
        for ds_test, ds_base in zip(series_test, series_base):
            assert ds_test.gridpoints == ds_base.gridpoints

    def test_multispectrum(self, series_test, series_base):
        image_test, image_baseline = self.generate_multispectrum_images(
            series_test, series_base
        )
        super().compare_test_images(
            image_test,
            image_baseline,
            tol=self.multispectrum_settings.get("RMS_TOLERANCE", self.RMS_TOLERANCE),
        )
