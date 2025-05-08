from __future__ import annotations

from os import PathLike
from typing import BinaryIO

import numpy as np
from pylbo._version import VersionHandler
from pylbo.utilities.datfiles.header import LegolasHeader
from pylbo.utilities.datfiles.header_legacy import LegolasLegacyHeader
from pylbo.utilities.datfiles.istream_reader import (
    read_complex_from_istream,
    read_float_from_istream,
    read_int_from_istream,
    read_mixed_from_istream,
    read_string_from_istream,
)
from pylbo.utilities.logger import pylboLogger
from pylbo.utilities.toolbox import transform_to_list, transform_to_numpy


class LegolasFileReader:
    def __init__(self, datfile: PathLike, byte_order: str = "native"):
        self._byte_order = byte_order
        self.datfile = datfile
        with open(self.datfile, "rb") as istream:
            istream.seek(0)
            self.legolas_version = self._read_legolas_version(istream)
            self._offset = istream.tell()

    def _read_legolas_version(self, istream: BinaryIO) -> VersionHandler:
        version_name = read_string_from_istream(istream, length=len("legolas_version"))
        if version_name == "legolas_version":
            # formatted version is character of length 10
            version = read_string_from_istream(istream, length=10)
        elif version_name == "datfile_version":
            # old numbering, single integer
            version = f"0.{str(read_int_from_istream(istream))}.0"
        else:
            # very old numbering
            istream.seek(0)
            version = "0.0.0"
        return VersionHandler(version)

    def get_header(self) -> LegolasHeader:
        with open(self.datfile, "rb") as istream:
            istream.seek(self._offset)
            if self.legolas_version < "2.0":
                return LegolasLegacyHeader(istream, self.legolas_version)
            else:
                return LegolasHeader(istream, self.legolas_version)

    def read_grid(self, header: LegolasHeader) -> np.ndarray:
        with open(self.datfile, "rb") as istream:
            grid = read_float_from_istream(
                istream, amount=header["gridpoints"], offset=header["offsets"]["grid"]
            )
        return np.asarray(grid, dtype=float)

    def read_gaussian_grid(self, header: LegolasHeader) -> np.ndarray:
        with open(self.datfile, "rb") as istream:
            grid = read_float_from_istream(
                istream,
                amount=header["gauss_gridpoints"],
                offset=header["offsets"]["grid_gauss"],
            )
        return np.asarray(grid, dtype=float)

    def read_ef_grid(self, header: LegolasHeader) -> np.ndarray:
        with open(self.datfile, "rb") as istream:
            grid = read_float_from_istream(
                istream,
                amount=header["ef_gridpoints"],
                offset=header["offsets"]["ef_grid"],
            )
        return np.asarray(grid, dtype=float)

    def read_equilibrium_arrays(self, header: LegolasHeader) -> dict:
        arrays = {}
        with open(self.datfile, "rb") as istream:
            istream.seek(header["offsets"]["equilibrium_arrays"])
            for name in header["equilibrium_names"]:
                arrays[name] = np.asarray(
                    read_float_from_istream(istream, amount=header["gauss_gridpoints"]),
                    dtype=float,
                )
        return arrays

    def read_eigenvalues(self, header: LegolasHeader) -> np.ndarray:
        with open(self.datfile, "rb") as istream:
            eigenvalues = read_complex_from_istream(
                istream,
                amount=header["nb_eigenvalues"],
                offset=header["offsets"]["eigenvalues"],
            )
        # ensure we have a list, also for 1 eigenvalue
        return np.asarray(transform_to_list(eigenvalues), dtype=complex)

    def read_eigenvectors(self, header: LegolasHeader) -> np.ndarray:
        with open(self.datfile, "rb") as istream:
            offsets = header["offsets"]
            eigvec_length = offsets["eigenvector_length"]
            nb_eigvecs = offsets["nb_eigenvectors"]
            eigenvectors = read_complex_from_istream(
                istream,
                amount=eigvec_length * nb_eigvecs,
                offset=offsets["eigenvectors"],
            )
        return np.reshape(
            np.asarray(eigenvectors, dtype=complex),
            (eigvec_length, nb_eigvecs),
            order="F",
        )

    def read_residuals(self, header: LegolasHeader) -> np.ndarray:
        with open(self.datfile, "rb") as istream:
            residuals = read_float_from_istream(
                istream,
                amount=header["offsets"]["nb_residuals"],
                offset=header["offsets"]["residuals"],
            )
        return np.asarray(residuals, dtype=float)

    def read_matrix_A(
        self, header: LegolasHeader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with open(self.datfile, "rb") as istream:
            hdr = read_mixed_from_istream(
                istream,
                fmt=(2 * "i" + 2 * "d"),
                amount=header["nonzero_A_elements"],
                offset=header["offsets"]["matrix_A"],
            )
            rows, cols, reals, imags = hdr[::4], hdr[1::4], hdr[2::4], hdr[3::4]
            vals = [complex(x, y) for x, y in zip(reals, imags)]
        return (
            np.asarray(rows, dtype=int),
            np.asarray(cols, dtype=int),
            np.asarray(vals, dtype=complex),
        )

    def read_matrix_B(
        self, header: LegolasHeader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with open(self.datfile, "rb") as istream:
            hdr = read_mixed_from_istream(
                istream,
                fmt=(2 * "i" + "d"),
                amount=header["nonzero_B_elements"],
                offset=header["offsets"]["matrix_B"],
            )
            rows, cols, vals = hdr[::3], hdr[1::3], hdr[2::3]
        return (
            np.asarray(rows, dtype=int),
            np.asarray(cols, dtype=int),
            np.asarray(vals, dtype=float),
        )

    def read_eigenfunction(self, header: LegolasHeader, ev_index: int) -> dict:
        # extract corresponding index in the array with written indices
        ef_index = self._get_ef_index(header, ev_index)
        if ef_index is None:
            return None
        return self._read_eigenfunction_like(
            header,
            offset=header["offsets"]["ef_arrays"],
            ef_index=ef_index,
            state_vector=header["ef_names"],
        )

    def read_derived_eigenfunction(self, header: LegolasHeader, ev_index: int) -> dict:
        ef_index = self._get_ef_index(header, ev_index)
        if ef_index is None:
            return None
        return self._read_eigenfunction_like(
            header,
            offset=header["offsets"]["derived_ef_arrays"],
            ef_index=ef_index,
            state_vector=header["derived_ef_names"],
        )

    def _read_eigenfunction_like(
        self,
        header: LegolasHeader,
        offset: int,
        ef_index: int,
        state_vector: np.ndarray,
    ) -> dict:
        state_vector = transform_to_numpy(state_vector)
        eigenfunctions = {}
        with open(self.datfile, "rb") as istream:
            for name_idx, name in enumerate(state_vector):
                # get offset of particular eigenfunction block
                block_offset = name_idx * header["offsets"]["ef_block_bytesize"]
                # get offset of requested eigenfunction in block
                ef_offset = ef_index * header["offsets"]["ef_bytesize"]
                eigenfunctions[name] = np.asarray(
                    read_complex_from_istream(
                        istream,
                        amount=header["ef_gridpoints"],
                        offset=offset + block_offset + ef_offset,
                    ),
                    dtype=complex,
                )
        return eigenfunctions

    def _get_ef_index(self, header: LegolasHeader, ev_index: int) -> int:
        # extract eigenfunction index from the array with written indices
        try:
            ((ef_index,),) = np.where(header["ef_written_idxs"] == ev_index)
        except ValueError:
            pylboLogger.warning("selected eigenvalue has no eigenfunctions!")
            return None
        return ef_index
