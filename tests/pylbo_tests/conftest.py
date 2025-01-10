import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pylbo
import pylbo.testing
import pytest
from pylbo.visualisation.continua import ContinuaHandler

pylbo.set_loglevel("error")

KEEP_FILES_OPTION = "--keep-files"
GENERATE_BASELINE_OPTION = "--generate"
tmpdir_path = Path(__file__).resolve().parent / "tmp"
utils = Path(__file__).resolve().parent / "utility_files"
mode_baseline = Path(__file__).resolve().parent / "mode_baseline"


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def remove_tmp_dir():
        if tmpdir_path.is_dir():
            if not request.config.getoption(KEEP_FILES_OPTION):
                shutil.rmtree(tmpdir_path)

    request.addfinalizer(remove_tmp_dir)


@pytest.fixture(autouse=True)
def close_figures_after_test():
    yield
    plt.close("all")


def pytest_addoption(parser):
    parser.addoption(
        KEEP_FILES_OPTION,
        action="store_true",
        help="if supplied, does not remove files after test completion.",
    )
    parser.addoption(
        GENERATE_BASELINE_OPTION,
        action="store_true",
        dest="generate_baseline",
        help="if set to true, (re)generates baseline data.",
    )


def pytest_runtest_makereport(item, call):
    if "required" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._requiredfailed = item


def pytest_runtest_setup(item):
    requiredfailed = getattr(item.parent, "_requiredfailed", False)
    if requiredfailed:
        pytest.xfail(f"required test failed ({requiredfailed.name})")


@pytest.fixture
def keep_files(request):
    return request.config.getoption(KEEP_FILES_OPTION)


@pytest.fixture(scope="session")
def modebaselinedir():
    return mode_baseline


@pytest.fixture
def tmpdir():
    if not tmpdir_path.is_dir():
        tmpdir_path.mkdir()
    yield tmpdir_path


@pytest.fixture
def default_pf_dict(tmpdir):
    config = {
        "gridpoints": 10,
        "equilibrium_type": "suydam_cluster",
        "show_results": False,
        "write_eigenfunctions": False,
        "basename_datfile": "default_ds",
        "output_folder": str(tmpdir),
        "logging_level": 0,
    }
    return config


@pytest.fixture
def default_parfile(tmpdir, default_pf_dict):
    return pylbo.generate_parfiles(default_pf_dict, output_dir=tmpdir)


@pytest.fixture
def default_ds(tmpdir, default_pf_dict):
    filepath = (tmpdir / "default_ds.dat").resolve()
    if not filepath.is_file():
        parfile = pylbo.generate_parfiles(default_pf_dict, output_dir=tmpdir)
        pylbo.run_legolas(parfile)
    return pylbo.load(filepath)


@pytest.fixture
def fake_ds():
    datfile = utils / "v1.1.2_datfile_efs.dat"
    return pylbo.testing.FakeDataSet(datfile=datfile, seed=20210715)


@pytest.fixture
def c_handle():
    return ContinuaHandler(interactive=True)


@pytest.fixture
def logv0():
    return utils / "v0_logfile_efs.log"


@pytest.fixture
def datv0():
    return utils / "v0_datfile_efs.dat"


@pytest.fixture
def datv1():
    return utils / "v1_datfile_matrices.dat"


@pytest.fixture
def datv112_eta():
    return utils / "v1.1.2_datfile_eta.dat"


@pytest.fixture
def mod_usr():
    return utils / "v2.1.1_mod_usr.f08"


@pytest.fixture
def datv211_harris():
    return utils / "v2.1.1_harris.dat"


@pytest.fixture
def vacv211_harris():
    return utils / "v2.1.1_harris.ldat"


@pytest.mark.timeout(5)
@pytest.fixture
def ds_v090():
    return pylbo.load(utils / "v0.9.0_datfile.dat")


@pytest.mark.timeout(5)
@pytest.fixture
def ds_v100():
    return pylbo.load(utils / "v1_datfile_matrices.dat")


@pytest.fixture
def ds_v112():
    return pylbo.load(utils / "v1.1.2_datfile_efs.dat")


@pytest.mark.timeout(5)
@pytest.fixture
def ds_v112_eta():
    return pylbo.load(utils / "v1.1.2_datfile_eta.dat")


@pytest.mark.timeout(5)
@pytest.fixture
def ds_v114_subset():
    return pylbo.load(utils / "v1.1.4_datfile_subset.dat")


@pytest.mark.timeout(5)
@pytest.fixture
def ds_v114_subset_defs():
    return pylbo.load(utils / "v1.1.4_datfile_subset_defs.dat")


@pytest.mark.timeout(5)
@pytest.fixture
def ds_v114():
    return pylbo.load(utils / "v1.1.4_datfile.dat")


@pytest.mark.timeout(5)
@pytest.fixture(scope="session")
def ds_v121_rti_khi():
    return pylbo.load(utils / "v1.2.1_rti_khi.dat")


@pytest.mark.timeout(5)
@pytest.fixture(scope="session")
def ds_v121_magth():
    return pylbo.load(utils / "v1.2.1_magth.dat")


@pytest.mark.timeout(5)
@pytest.fixture(scope="session")
def ds_v200_mri_matrix():
    return pylbo.load(utils / "v2.0.0_mri_matrix.dat")


@pytest.mark.timeout(5)
@pytest.fixture(scope="session")
def ds_v200_mri_efs():
    return pylbo.load(utils / "v2.0.0_mri_subset_efs.dat")


@pytest.mark.timeout(5)
@pytest.fixture(scope="session")
def ds_v200_tear_nobg():
    return pylbo.load(utils / "v2.0.0_tear_nobg.dat")


@pytest.mark.timeout(5)
@pytest.fixture(scope="session")
def ds_v200_hd_khi():
    return pylbo.load(utils / "v2.0.0_hd_khi.dat")


@pytest.mark.timeout(5)
@pytest.fixture
def series_v100():
    return pylbo.load_series([utils / "v1_datfile_matrices.dat"] * 3)


@pytest.mark.timeout(5)
@pytest.fixture
def series_v112():
    return pylbo.load_series([utils / "v1.1.2_datfile_efs.dat"] * 3)


@pytest.mark.timeout(5)
@pytest.fixture
def series_v112_eta():
    return pylbo.load_series([utils / "v1.1.2_datfile_eta.dat"] * 5)


@pytest.mark.timeout(5)
@pytest.fixture
def ds_v130_suydam_efs_vecs_res():
    return pylbo.load(utils / "v1.3.0_suydam_efs_vecs_res.dat")


@pytest.mark.timeout(5)
@pytest.fixture
def ds_v210_rotcyl_efs_default_basis():
    return pylbo.load(utils / "v2.1.0_rot_cyl_efs_default_basis.dat")


@pytest.mark.timeout(5)
@pytest.fixture
def ds_v210_rotcyl_efs_custom_basis():
    return pylbo.load(utils / "v2.1.0_rot_cyl_efs_custom_basis.dat")


@pytest.mark.timeout(5)
@pytest.fixture
def series_v200_nobg():
    return pylbo.load_series([utils / "v2.0.0_tear_nobg.dat"] * 7)


@pytest.mark.timeout(5)
@pytest.fixture
def series_v200_mri_efs():
    return pylbo.load_series([utils / "v2.0.0_mri_subset_efs.dat"] * 6)


@pytest.mark.timeout(5)
@pytest.fixture
def series_v200_mixed_efs():
    files_with_efs = [utils / "v2.0.0_mri_subset_efs.dat"] * 3
    files_without_efs = [utils / "v2.0.0_mri_matrix.dat"] * 3
    # alternate between files with and without efs
    files = [file for pair in zip(files_with_efs, files_without_efs) for file in pair]
    return pylbo.load_series(files)
