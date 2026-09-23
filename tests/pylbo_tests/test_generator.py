from pathlib import Path

import f90nml
import numpy as np
import pylbo
import pytest
from pylbo.automation.generator import ParfileGenerator


def test_basename_none(tmpdir):
    gen = ParfileGenerator({}, output_dir=tmpdir)
    assert gen.basenames == ["parfile"]


def test_basename_none_multiple(tmpdir):
    gen = ParfileGenerator({"number_of_runs": 5}, output_dir=tmpdir)
    assert gen.basenames == [f"parfile{i:04d}" for i in range(1, 6)]


def test_basename_multiple():
    names = [f"test{i}" for i in range(1, 6)]
    gen = ParfileGenerator({"number_of_runs": 5}, basename=names)
    assert gen.basenames == names


def test_basename_multiple_inconsistent_nbruns():
    names = [f"test{i}" for i in range(1, 6)]
    with pytest.raises(ValueError):
        ParfileGenerator({"number_of_runs": 4}, basename=names)


def test_outputdir_none():
    parfiledir = Path.cwd() / "parfiles"
    gen = ParfileGenerator({})
    assert gen.output_dir == parfiledir
    parfiledir.rmdir()


def test_outputdir_invalid():
    with pytest.raises(NotADirectoryError):
        ParfileGenerator({}, output_dir="invalid_dir")


def test_invalid_dict_type():
    gen = ParfileGenerator({"gridpoints": 100.5})
    with pytest.raises(TypeError):
        gen.create_namelist_from_dict()


def test_unknown_namelist_item():
    from pylbo.exceptions import ParfileGenerationError

    with pytest.raises(ParfileGenerationError):
        pylbo.generate_parfiles(parfile_dict={"gridpoints": 10, "unknown_item": 5})


def test_invalid_nb_runs():
    from pylbo.exceptions import ParfileGenerationError

    with pytest.raises(ParfileGenerationError):
        pylbo.generate_parfiles(
            parfile_dict={"gridpoints": [50, 100], "number_of_runs": 1}
        )


def test_basenames_with_prefix(tmpdir):
    names = [f"test{i}" for i in range(1, 6)]
    parfiles = pylbo.generate_parfiles(
        parfile_dict={"gridpoints": 100, "number_of_runs": len(names)},
        basename=names,
        output_dir=tmpdir,
    )
    assert len(parfiles) == len(names)
    expected_names = [f"{i:04d}{name}" for i, name in enumerate(names, start=1)]
    for parfile, expected_name in zip(parfiles, expected_names):
        assert Path(parfile).stem == expected_name


def test_basenames_with_custom_prefix(tmpdir):
    names = [f"test{i}" for i in range(1, 6)]
    parfiles = pylbo.generate_parfiles(
        parfile_dict={"gridpoints": 100, "number_of_runs": len(names)},
        basename=names,
        output_dir=tmpdir,
        nb_prefix_digits=6,
    )
    assert len(parfiles) == len(names)
    expected_names = [f"{i:06d}{name}" for i, name in enumerate(names, start=1)]
    for parfile, expected_name in zip(parfiles, expected_names):
        assert Path(parfile).stem == expected_name


def test_basenames_without_prefix(tmpdir):
    names = [f"test{i}" for i in range(1, 6)]
    parfiles = pylbo.generate_parfiles(
        parfile_dict={"gridpoints": 100, "number_of_runs": len(names)},
        basename=names,
        output_dir=tmpdir,
        prefix_numbers=False,
    )
    assert len(parfiles) == len(names)
    for parfile, expected_name in zip(parfiles, names):
        assert Path(parfile).stem == expected_name


def test_savelist_not_present(tmpdir):
    parfile = pylbo.generate_parfiles(
        parfile_dict={"gridpoints": 100, "equilibrium_type": "adiabatic_homo"},
        output_dir=tmpdir,
    )
    parfile_dict = f90nml.read(*parfile)
    assert "savelist" in parfile_dict.keys()


def test_paramlist_present():
    gen = ParfileGenerator(
        {
            "gridpoints": 50,
            "equilibrium_type": "adiabatic_homo",
            "parameters": {"k2": 2, "k3": 0, "cte_T0": 5},
        },
    )
    gen.create_namelist_from_dict()
    assert "paramlist" in gen.container.keys()
    for key, value in zip(("k2", "k3", "cte_T0"), ([2], [0], [5])):
        assert gen.container["paramlist"].get(key) == value


def test_nb_runs():
    gen = ParfileGenerator(
        {
            "gridpoints": [50, 80, 150],
            "number_of_runs": 3,
            "equilibrium_type": "adiabatic_homo",
        },
    )
    gen.create_namelist_from_dict()
    assert gen.container["gridlist"].get("gridpoints") == [50, 80, 150]
    assert (
        gen.container["equilibriumlist"].get("equilibrium_type")
        == ["adiabatic_homo"] * 3
    )


def test_sigma_complex():
    gen = ParfileGenerator({"solver": "arnoldi", "sigma": 5.0})
    gen.create_namelist_from_dict()
    sigma = gen.container["solvelist"].get("sigma")[0]
    assert isinstance(sigma, complex)
    assert np.isclose(sigma, 5.0 + 0j)


def test_datfile_name_generation_multiruns(tmpdir):
    # this tests issue #68
    parfiles = pylbo.generate_parfiles(
        parfile_dict={
            "number_of_runs": 10,
            "gridpoints": 50,
            "parameters": {
                "k2": 0.0,
                "k3": np.pi * np.sin(0.25 * np.pi),
                "cte_b03": np.linspace(0.001, 3, 10),
            },
            "equilibrium_type": "adiabatic_homo",
        },
        basename="parfile",
        output_dir=tmpdir,
    )
    container = [f90nml.read(file) for file in parfiles]
    for i, config in enumerate(container, start=1):
        assert config["savelist"]["basename_datfile"] == f"{i:04d}parfile"
