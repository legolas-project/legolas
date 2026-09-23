import copy
import multiprocessing
import os
from pathlib import Path

import pylbo
import pytest
from pylbo.automation.api import run_legolas
from pylbo.automation.runner import LegolasRunner

# get executable from LEGOLASDIR environment variable
DEFAULT_EXEC = Path(os.environ["LEGOLASDIR"]) / "legolas"


def test_invalid_parfiles():
    with pytest.raises(FileNotFoundError):
        run_legolas(["unknown_parfile.par"])


def test_invalid_executable(tmpdir):
    # create temporary parfile
    parfile = tmpdir / "parfile.par"
    parfile.write_text("content")
    with pytest.raises(FileNotFoundError):
        run_legolas(parfile, executable="unknown_exe")


def test_no_executable(tmpdir):
    # create temporary parfile
    parfile = tmpdir / "parfile.par"
    parfile.write_text("content")
    with pytest.raises(TypeError):
        run_legolas(parfile)


def test_cpu_count(default_parfile):
    cpus_available = multiprocessing.cpu_count()
    run = LegolasRunner(
        parfiles=default_parfile,
        remove_parfiles=True,
        nb_cpus=cpus_available + 1,
        executable=DEFAULT_EXEC,
    )
    assert run.nb_cpus == cpus_available


def test_parfile_folder_deletion_oserr(tmpdir, default_pf_dict):
    pf_dict = copy.deepcopy(default_pf_dict)
    pylbo.generate_parfiles(pf_dict, basename="test1", output_dir=tmpdir)
    parfiles2 = pylbo.generate_parfiles(pf_dict, basename="test2", output_dir=tmpdir)
    pylbo.run_legolas(parfiles2, remove_parfiles=True, executable=DEFAULT_EXEC)


def test_single_run(default_pf_dict):
    pf_dict = copy.deepcopy(default_pf_dict)
    parfile = pylbo.generate_parfiles(pf_dict)
    assert len(parfile) == 1
    pylbo.run_legolas(parfile, remove_parfiles=True, executable=DEFAULT_EXEC)
    filepath = Path(pf_dict["output_folder"]) / pf_dict["basename_datfile"]
    assert filepath.with_suffix(".dat").is_file()


def test_multirun(default_pf_dict):
    pf_dict = copy.deepcopy(default_pf_dict)
    pf_dict["number_of_runs"] = 4
    pf_dict["gridpoints"] = [10] * 4
    parfiles = pylbo.generate_parfiles(pf_dict)
    pylbo.run_legolas(
        parfiles, nb_cpus=2, remove_parfiles=True, executable=DEFAULT_EXEC
    )
    for i in range(1, 5):
        filepath = (
            Path(pf_dict["output_folder"]) / f"{i:04d}{pf_dict['basename_datfile']}.dat"
        )
        assert filepath.is_file()
