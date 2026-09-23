from __future__ import annotations

from typing import BinaryIO

import numpy as np
from pylbo._version import VersionHandler
from pylbo.utilities.datfiles.header import LegolasHeader
from pylbo.utilities.datfiles.istream_reader import (
    SIZE_COMPLEX,
    SIZE_DOUBLE,
    read_boolean_from_istream,
    read_complex_from_istream,
    read_float_from_istream,
    read_int_from_istream,
    read_string_from_istream,
    requires_version,
)


class LegolasLegacyHeader(LegolasHeader):
    def __init__(self, istream: BinaryIO, version: VersionHandler) -> None:
        super().__init__(istream, version)

    def read_header_data(self, istream: BinaryIO) -> None:
        data = {}

        data["geometry"] = read_string_from_istream(istream, length=self._str_len)
        data["x_start"], data["x_end"] = read_float_from_istream(istream, amount=2)

        for key in ("", "gauss_", "matrix_", "ef_"):
            data[f"{key}gridpoints"] = read_int_from_istream(istream)

        data["gamma"] = read_float_from_istream(istream)
        data["eq_type"] = read_string_from_istream(istream, length=self._str_len)

        data["has_efs"] = read_boolean_from_istream(istream)
        data["has_derived_efs"] = self._read_has_derived_efs(istream)
        data["has_matrices"] = read_boolean_from_istream(istream)
        data["has_eigenvectors"] = self._read_has_eigenvectors(istream)
        data["has_residuals"] = self._read_has_residuals(istream)
        (
            data["ef_subset_used"],
            data["ef_subset_center"],
            data["ef_subset_radius"],
        ) = self._read_ef_subset_properties(istream)
        data["parameters"] = self._read_parameters(istream)
        data["equilibrium_names"] = self._read_equilibrium_names(istream)

        data["units"] = self._read_units(istream)
        data["nb_eigenvalues"] = (
            read_int_from_istream(istream)
            if self.legolas_version >= "1.0.2"
            else data["matrix_gridpoints"]
        )
        data["offsets"] = {}
        self.data.update(data)

        self._handle_entries_not_in_datfile_for_compatibility()

    def read_data_offsets(self, istream: BinaryIO) -> None:
        offsets = {}
        # eigenvalue offset
        offsets["eigenvalues"] = istream.tell()
        bytesize = self.data["nb_eigenvalues"] * SIZE_COMPLEX
        istream.seek(istream.tell() + bytesize)
        # grid offset
        offsets["grid"] = istream.tell()
        bytesize = self.data["gridpoints"] * SIZE_DOUBLE
        istream.seek(istream.tell() + bytesize)
        # grid gauss offset
        offsets["grid_gauss"] = istream.tell()
        bytesize = self.data["gauss_gridpoints"] * SIZE_DOUBLE
        istream.seek(istream.tell() + bytesize)
        # equilibrium arrays offset
        offsets["equilibrium_arrays"] = istream.tell()
        bytesize = (
            self.data["gauss_gridpoints"]
            * len(self.data["equilibrium_names"])
            * SIZE_DOUBLE
        )
        istream.seek(istream.tell() + bytesize)

        offsets.update(self._get_eigenfunction_offsets(istream))
        offsets.update(self._get_derived_eigenfunction_offsets(istream))
        offsets.update(super()._get_eigenvector_offsets(istream))
        offsets.update(super()._get_residual_offsets(istream))
        offsets.update(super()._get_matrices_offsets(istream))

        self.data["offsets"].update(offsets)

    @requires_version("1.1.3", default=False)
    def _read_has_derived_efs(self, istream: BinaryIO) -> bool:
        return read_boolean_from_istream(istream)

    @requires_version("1.3.0", default=False)
    def _read_has_eigenvectors(self, istream: BinaryIO) -> bool:
        return read_boolean_from_istream(istream)

    @requires_version("1.3.0", default=False)
    def _read_has_residuals(self, istream: BinaryIO) -> bool:
        return read_boolean_from_istream(istream)

    @requires_version("1.1.4", default=(False, None, None))
    def _read_ef_subset_properties(
        self, istream: BinaryIO
    ) -> tuple(bool, complex, float):
        used = read_boolean_from_istream(istream)
        center = read_complex_from_istream(istream)
        radius = read_float_from_istream(istream)
        return (used, center, radius)

    def _read_parameters(self, istream: BinaryIO) -> dict:
        nb_params = read_int_from_istream(istream)
        len_param_name = (
            read_int_from_istream(istream)
            if self.legolas_version >= "1.0.2"
            else self._str_len_array
        )
        parameter_names = read_string_from_istream(
            istream, length=len_param_name, amount=nb_params
        )
        parameter_values = read_float_from_istream(istream, amount=nb_params)
        return {
            name: value
            for name, value in zip(parameter_names, parameter_values)
            if not np.isnan(value)
        }

    def _read_equilibrium_names(self, istream: BinaryIO) -> list[str]:
        nb_names = read_int_from_istream(istream)
        len_name = (
            read_int_from_istream(istream)
            if self.legolas_version >= "1.0.2"
            else self._str_len_array
        )
        names = read_string_from_istream(istream, length=len_name, amount=nb_names)
        names = [name.replace("grav", "gravity") for name in names]
        return names

    def _read_units(self, istream: BinaryIO) -> dict:
        units = {"cgs": read_boolean_from_istream(istream)}
        if self.legolas_version >= "1.0.2":
            nb_units, len_unit_name = read_int_from_istream(istream, amount=2)
            unit_names = read_string_from_istream(
                istream, length=len_unit_name, amount=nb_units
            )
        else:
            unit_names = [
                "unit_length",
                "unit_time",
                "unit_density",
                "unit_velocity",
                "unit_temperature",
                "unit_pressure",
                "unit_magneticfield",
                "unit_numberdensity",
                "unit_lambdaT",
                "unit_conduction",
                "unit_resistivity",
            ]
            nb_units = len(unit_names)
        unit_values = read_float_from_istream(istream, amount=nb_units)
        for name, value in zip(unit_names, unit_values):
            units[name] = value
        # mean molecular weight is added in 1.1.2, before this it defaults to 1
        units.setdefault("mean_molecular_weight", 1.0)
        return units

    def _get_eigenfunction_offsets(self, istream: BinaryIO) -> dict:
        if not self.data["has_efs"]:
            return {}
        # eigenfunction names
        nb_efs = read_int_from_istream(istream)
        self.data["ef_names"] = read_string_from_istream(
            istream, length=self._str_len_array, amount=nb_efs
        )
        offsets = super()._get_ef_grid_offset(self.data["ef_gridpoints"], istream)
        self._get_ef_written_flags(istream)
        offsets.update(super()._get_ef_block_offsets(istream))
        return offsets

    def _get_ef_written_flags(self, istream: BinaryIO) -> None:
        if self.legolas_version < "1.1.4":
            self.data["ef_written_flags"] = np.asarray(
                [True] * self.data["nb_eigenvalues"], dtype=bool
            )
            self.data["ef_written_idxs"] = np.arange(0, self.data["nb_eigenvalues"])
            return
        super()._get_ef_written_flags(istream)

    def _get_derived_eigenfunction_offsets(self, istream: BinaryIO) -> dict:
        if not self.data["has_derived_efs"]:
            return {}
        nb_defs = read_int_from_istream(istream)
        return super()._get_derived_ef_names_and_offsets(
            nb_defs, self._str_len_array, istream
        )

    def _handle_entries_not_in_datfile_for_compatibility(self) -> None:
        # default block dimensions
        self.data["nb_eqs"] = 8
        self.data["physics_type"] = "mhd"
        self.data["state_vector"] = ["rho", "v1", "v2", "v3", "T", "a1", "a2", "a3"]
        self.data["basis_functions"] = {
            "rho": "quadratic",
            "v1": "cubic",
            "v2": "quadratic",
            "v3": "quadratic",
            "T": "quadratic",
            "a1": "quadratic",
            "a2": "cubic",
            "a3": "cubic",
        }
        self.data["dims"] = {
            "dim_integralblock": 2,
            "dim_subblock": 8 * 2,
            "dim_quadblock": 2 * 8 * 2,
            "dim_matrix": self.data["gridpoints"] * 8 * 2,
        }
        self.data["has_background"] = True
