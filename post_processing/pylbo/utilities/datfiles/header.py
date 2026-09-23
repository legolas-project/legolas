from __future__ import annotations

from typing import Any, BinaryIO

import numpy as np
from pylbo._version import VersionHandler
from pylbo.utilities.datfiles.istream_reader import (
    SIZE_COMPLEX,
    SIZE_DOUBLE,
    SIZE_INT,
    read_boolean_from_istream,
    read_complex_from_istream,
    read_float_from_istream,
    read_int_from_istream,
    read_string_from_istream,
)
from pylbo.utilities.toolbox import transform_to_numpy
from pylbo.visualisation.utils import ensure_attr_set


class LegolasHeader:
    """Baseclass for a Legolas header"""

    def __init__(self, istream: BinaryIO, version: VersionHandler):
        self.legolas_version = version
        self.data = {}
        self._str_len = None
        self._str_len_array = None
        self._set_str_lengths(istream)
        [ensure_attr_set(self, attr) for attr in ("_str_len", "_str_len_array")]
        self.read_header_data(istream)
        self.read_data_offsets(istream)

    def __str__(self) -> str:
        keys_to_avoid = ["ef_written_flags", "ef_written_idxs", "offsets"]
        return "".join(
            [
                f"{key}: {self.data.get(key)}\n"
                for key in self.data.keys()
                if key not in keys_to_avoid
            ]
        )

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def _set_str_lengths(self, istream: BinaryIO) -> None:
        self._str_len, self._str_len_array = read_int_from_istream(istream, amount=2)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def read_header_data(self, istream: BinaryIO) -> None:
        data = {}
        data.update(self._read_physics_type_info(istream))
        data.update(self._read_grid_info(istream))
        data.update(self._read_io_info(istream))
        data.update(self._read_solver_info(istream))
        data.update(self._read_equilibrium_info(istream))
        data.update(self._read_units_info(istream))
        data.update(self._read_physics_info(istream))
        data.update(self._read_parameters(istream))
        data.update(self._read_equilibrium_names(istream))
        self.data.update(data)

    def read_data_offsets(self, istream: BinaryIO) -> None:
        offsets = {}
        # eigenvalues
        nb_eigenvals = read_int_from_istream(istream)
        self.data["nb_eigenvalues"] = nb_eigenvals
        offsets["eigenvalues"] = istream.tell()
        bytesize = nb_eigenvals * SIZE_COMPLEX
        istream.seek(istream.tell() + bytesize)
        # grid
        offsets["grid"] = istream.tell()
        bytesize = self.data["gridpoints"] * SIZE_DOUBLE
        istream.seek(istream.tell() + bytesize)
        # Gaussian grid
        offsets["grid_gauss"] = istream.tell()
        bytesize = self.data["gauss_gridpoints"] * SIZE_DOUBLE
        istream.seek(istream.tell() + bytesize)
        # equilibrium arrays
        offsets["equilibrium_arrays"] = istream.tell()
        bytesize = (
            self.data["gauss_gridpoints"]
            * len(self.data["equilibrium_names"])
            * SIZE_DOUBLE
        )
        istream.seek(istream.tell() + bytesize)

        offsets.update(self._get_eigenfunction_offsets(istream))
        offsets.update(self._get_derived_eigenfunction_offsets(istream))
        offsets.update(self._get_eigenvector_offsets(istream))
        offsets.update(self._get_residual_offsets(istream))
        offsets.update(self._get_matrices_offsets(istream))
        offsets.update(self._get_iv_snapshots_offsets(istream))

        self.data["offsets"] = offsets

    def _read_physics_type_info(self, istream: BinaryIO) -> dict:
        data = {}
        data["nb_eqs"] = read_int_from_istream(istream)
        len_type = read_int_from_istream(istream)
        data["physics_type"] = read_string_from_istream(istream, length=len_type)
        len_name, size_vector = read_int_from_istream(istream, amount=2)
        data["state_vector"] = np.asarray(
            read_string_from_istream(istream, length=len_name, amount=size_vector),
            dtype=str,
        )
        data["basis_functions"] = self._read_basis_functions(
            istream, data["state_vector"]
        )
        data["dims"] = {}
        for key in ("integralblock", "subblock", "quadblock", "matrix"):
            data["dims"][f"dim_{key}"] = read_int_from_istream(istream)
        return data

    def _read_basis_functions(
        self, istream: BinaryIO, state_vector: np.ndarray[str]
    ) -> dict:
        old_defaults = {
            "rho": "quadratic",
            "v1": "cubic",
            "v2": "quadratic",
            "v3": "quadratic",
            "T": "quadratic",
            "a1": "quadratic",
            "a2": "cubic",
            "a3": "cubic",
        }
        if self.legolas_version < "2.1.0":
            return {key: old_defaults[key] for key in state_vector}
        len_basis, size_basis = read_int_from_istream(istream, amount=2)
        basis_functions = read_string_from_istream(
            istream, length=len_basis, amount=size_basis
        )
        return {var: func for var, func in zip(state_vector, basis_functions)}

    def _read_grid_info(self, istream: BinaryIO) -> dict:
        data = {}
        len_geom = read_int_from_istream(istream)
        data["geometry"] = read_string_from_istream(istream, length=len_geom)
        for key in ("", "gauss_", "ef_"):
            data[f"{key}gridpoints"] = read_int_from_istream(istream)
        data["gauss"] = {}
        n_gauss = read_int_from_istream(istream)
        data["gauss"]["number_of_nodes"] = n_gauss
        data["gauss"]["nodes"] = np.asarray(
            read_float_from_istream(istream, amount=n_gauss), dtype=float
        )
        data["gauss"]["weights"] = np.asarray(
            read_float_from_istream(istream, amount=n_gauss), dtype=float
        )
        data["x_start"], data["x_end"] = read_float_from_istream(istream, amount=2)
        return data

    def _read_io_info(self, istream: BinaryIO) -> dict:
        data = {}
        data["has_matrices"] = read_boolean_from_istream(istream)
        data["has_eigenvectors"] = read_boolean_from_istream(istream)
        data["has_residuals"] = read_boolean_from_istream(istream)
        data["has_efs"] = read_boolean_from_istream(istream)
        data["has_derived_efs"] = read_boolean_from_istream(istream)
        data["ef_subset_used"] = read_boolean_from_istream(istream)
        data["ef_subset_radius"] = read_float_from_istream(istream)
        data["ef_subset_center"] = read_complex_from_istream(istream)
        if self.legolas_version >= "2.3.0":
            data["has_iv_snapshots"] = read_boolean_from_istream(istream)
        else:
            data["has_iv_snapshots"] = False
        return data

    def _read_solver_info(self, istream: BinaryIO) -> dict:
        data = {}
        len_solver = read_int_from_istream(istream)
        data["solver"] = read_string_from_istream(istream, length=len_solver)

        arnoldi_data = {}
        len_mode = read_int_from_istream(istream)
        arnoldi_data["arpack_mode"] = read_string_from_istream(istream, length=len_mode)
        arnoldi_data["number_of_eigenvalues"] = read_int_from_istream(istream)
        len_which = read_int_from_istream(istream)
        arnoldi_data["which_eigenvalues"] = read_string_from_istream(
            istream, length=len_which
        )
        arnoldi_data["ncv"] = read_int_from_istream(istream)
        data["maxiter"] = read_int_from_istream(istream)
        data["sigma"] = read_complex_from_istream(istream)
        data["tolerance"] = read_float_from_istream(istream)
        data["arnoldi"] = arnoldi_data if data["solver"] == "arnoldi" else None
        return data

    def _read_equilibrium_info(self, istream: BinaryIO) -> dict:
        data = {}
        len_equil_type = read_int_from_istream(istream)
        data["eq_type"] = read_string_from_istream(istream, length=len_equil_type)
        len_boundary_type = read_int_from_istream(istream)
        data["boundary_type"] = read_string_from_istream(
            istream, length=len_boundary_type
        )
        return data

    def _read_units_info(self, istream: BinaryIO) -> dict:
        n_units = read_int_from_istream(istream)
        units = {"cgs": read_boolean_from_istream(istream)}
        for _ in range(n_units):
            len_name = read_int_from_istream(istream)
            name = read_string_from_istream(istream, length=len_name)
            value = read_float_from_istream(istream)
            units[name] = value
        return {"units": units}

    def _read_physics_info(self, istream: BinaryIO) -> dict:
        data = {}
        data["gamma"] = read_float_from_istream(istream)
        data["is_incompressible"] = read_boolean_from_istream(istream)
        physics = {}
        physics["flow"] = read_boolean_from_istream(istream)
        physics["cooling"] = read_boolean_from_istream(istream)
        len_curve = read_int_from_istream(istream)
        physics["cooling_curve"] = read_string_from_istream(istream, length=len_curve)
        physics["interpolation_points"] = read_int_from_istream(istream)
        physics["external_gravity"] = read_boolean_from_istream(istream)
        physics["resistivity"] = read_boolean_from_istream(istream)
        physics["has_fixed_resistivity"] = read_boolean_from_istream(istream)
        physics["viscosity"] = read_boolean_from_istream(istream)
        physics["has_viscous_heating"] = read_boolean_from_istream(istream)
        physics["conduction"] = read_boolean_from_istream(istream)
        physics["has_parallel_conduction"] = read_boolean_from_istream(istream)
        physics["has_fixed_tc_para"] = read_boolean_from_istream(istream)
        physics["has_perpendicular_conduction"] = read_boolean_from_istream(istream)
        physics["has_fixed_tc_perp"] = read_boolean_from_istream(istream)
        physics["Hall"] = read_boolean_from_istream(istream)
        physics["Hall_uses_substitution"] = read_boolean_from_istream(istream)
        physics["has_electron_inertia"] = read_boolean_from_istream(istream)
        data["physics"] = physics
        return data

    def _read_parameters(self, istream: BinaryIO) -> dict:
        parameters = {}
        nb_params, len_name = read_int_from_istream(istream, amount=2)
        for _ in range(nb_params):
            name = read_string_from_istream(istream, length=len_name)
            parameters[name] = read_float_from_istream(istream)
        parameters = {k: v for k, v in parameters.items() if not np.isnan(v)}
        return {"parameters": parameters}

    def _read_equilibrium_names(self, istream: BinaryIO) -> dict:
        nb_names, len_name = read_int_from_istream(istream, amount=2)
        self.data["has_background"] = nb_names > 0
        names = (
            read_string_from_istream(istream, length=len_name, amount=nb_names)
            if self.data["has_background"]
            else []
        )
        # due to a typo older datfiles contain 'db03' instead of 'dB03'
        names = [name.replace("db03", "dB03") for name in names]
        return {"equilibrium_names": names}

    def _get_eigenfunction_offsets(self, istream: BinaryIO) -> dict:
        if not self.data["has_efs"]:
            return {}
        self.data["ef_names"] = self.data["state_vector"]
        # eigenfunction grid
        ef_gridsize = read_int_from_istream(istream)
        offsets = self._get_ef_grid_offset(ef_gridsize, istream)
        # flags
        self._get_ef_written_flags(istream)
        # eigenfunction block offsets
        offsets.update(self._get_ef_block_offsets(istream))
        return offsets

    def _get_ef_grid_offset(self, ef_gridsize: int, istream: BinaryIO) -> dict:
        offsets = {"ef_grid": istream.tell()}
        bytesize = ef_gridsize * SIZE_DOUBLE
        istream.seek(istream.tell() + bytesize)
        return offsets

    def _get_ef_written_flags(self, istream: BinaryIO) -> None:
        # eigenfunction flags
        ef_flags_size = read_int_from_istream(istream)
        self.data["ef_written_flags"] = np.asarray(
            read_int_from_istream(istream, amount=ef_flags_size), dtype=bool
        )
        ef_idxs_size = read_int_from_istream(istream)
        self.data["ef_written_idxs"] = transform_to_numpy(
            np.asarray(read_int_from_istream(istream, amount=ef_idxs_size), dtype=int)
            - 1
        )  # -1 corrects for Fortran 1-based indexing
        # do a sanity check
        assert np.all(
            self.data["ef_written_idxs"] == np.where(self.data["ef_written_flags"])[0]
        )

    def _get_ef_block_offsets(self, istream: BinaryIO) -> dict:
        # eigenfunction offsets
        offsets = {"ef_arrays": istream.tell()}
        # bytesize of a single eigenfunction block (all efs for 1 state vector variable)
        bytesize_block = (
            self.data["ef_gridpoints"]
            * len(self.data["ef_written_idxs"])
            * SIZE_COMPLEX
        )
        offsets["ef_block_bytesize"] = bytesize_block
        offsets["ef_bytesize"] = self.data["ef_gridpoints"] * SIZE_COMPLEX
        istream.seek(istream.tell() + bytesize_block * self.data["nb_eqs"])
        return offsets

    def _get_derived_eigenfunction_offsets(self, istream: BinaryIO) -> dict:
        if not self.data["has_derived_efs"]:
            return {}
        nb_names, size_names = read_int_from_istream(istream, amount=2)
        return self._get_derived_ef_names_and_offsets(nb_names, size_names, istream)

    def _get_derived_ef_names_and_offsets(
        self, nb_names, size_names, istream: BinaryIO
    ) -> dict:
        self.data["derived_ef_names"] = np.asarray(
            read_string_from_istream(istream, length=size_names, amount=nb_names),
            dtype=str,
        )
        offsets = {"derived_ef_arrays": istream.tell()}
        bytesize_block = (
            self.data["ef_gridpoints"]
            * len(self.data["ef_written_idxs"])
            * nb_names
            * SIZE_COMPLEX
        )
        istream.seek(istream.tell() + bytesize_block)
        return offsets

    def _get_eigenvector_offsets(self, istream: BinaryIO) -> dict:
        if not self.data["has_eigenvectors"]:
            return {}
        len_eigvecs, nb_eigvecs = read_int_from_istream(istream, amount=2)
        offsets = {
            "eigenvectors": istream.tell(),
            "eigenvector_length": len_eigvecs,
            "nb_eigenvectors": nb_eigvecs,
        }
        bytesize = len_eigvecs * nb_eigvecs * SIZE_COMPLEX
        istream.seek(istream.tell() + bytesize)
        return offsets

    def _get_residual_offsets(self, istream: BinaryIO) -> dict:
        if not self.data["has_residuals"]:
            return {}
        nb_residuals = read_int_from_istream(istream)
        offsets = {"residuals": istream.tell(), "nb_residuals": nb_residuals}
        bytesize = nb_residuals * SIZE_DOUBLE
        istream.seek(istream.tell() + bytesize)
        return offsets

    def _get_matrices_offsets(self, istream: BinaryIO) -> dict:
        if not self.data["has_matrices"]:
            return {}
        nonzero_B_elements, nonzero_A_elements = read_int_from_istream(
            istream, amount=2
        )
        # B matrix is written as (row, column, real value)
        byte_size = (2 * SIZE_INT + SIZE_DOUBLE) * nonzero_B_elements
        offsets = {"matrix_B": istream.tell()}
        istream.seek(istream.tell() + byte_size)
        self.data["nonzero_B_elements"] = nonzero_B_elements
        # A matrix is written as (row, column, complex value)
        byte_size = (
            2 * SIZE_INT + 2 * SIZE_DOUBLE
        ) * nonzero_A_elements  # complex = 2 * SIZE_DOUBLE
        offsets["matrix_A"] = istream.tell()
        self.data["nonzero_A_elements"] = nonzero_A_elements
        istream.seek(istream.tell() + byte_size)
        return offsets

    def _get_iv_snapshots_offsets(self, istream: BinaryIO) -> dict:
        """
        Attempt to read the IV snapshot metadata (if present).
        Return dict of offsets and metadata. If not present, return empty dict.
        """
        if not self.data["has_iv_snapshots"]:
            return {}

        # Read 3 integers: number of snapshots / points / components
        nsnap, npts, ncomp = read_int_from_istream(istream, amount=3)

        self.data["iv_snapshots_count"] = nsnap
        self.data["iv_snapshots_npoints"] = npts
        self.data["iv_snapshots_ncomp"] = ncomp

        # Data block start
        offset_iv_data = istream.tell()

        # Compute total bytes: nsnap * ncomp * npts * SIZE_DOUBLE
        total_values = nsnap * ncomp * npts
        bytes_to_skip = total_values * SIZE_DOUBLE

        # skip ahead
        istream.seek(offset_iv_data + bytes_to_skip)

        return {"iv_snapshots": offset_iv_data}
