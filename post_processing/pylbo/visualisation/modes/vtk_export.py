from __future__ import annotations

import struct
from pathlib import Path
from typing import Union

import numpy as np
from pylbo.utilities.logger import pylboLogger
from pylbo.utilities.toolbox import transform_to_list, transform_to_numpy
from pylbo.visualisation.modes.mode_data import ModeVisualisationData
from pylbo.visualisation.utils import validate_ef_name
from tqdm import tqdm


class VTKDataExporter:
    """
    Main class to prepare data for export to VTK files.

    Parameters
    ----------
    data : ModeVisualisationData
        The data for the visualisation
    u1 : np.ndarray
        The 1D :math:`u_1` coordinate array.
    u2 : np.ndarray
        The 1D :math:`u_2` coordinate array.
    u3 : np.ndarray
        The 1D :math:`u_3` coordinate array.

    Attributes
    ----------
    _u1 : ndarray
        The 1D :math:`u_1` coordinates.
    _u2 : ndarray
        The 1D :math:`u_2` coordinates.
    _u3 : ndarray
        The 1D :math:`u_3` coordinates.
    u1_data : ndarray
        The 3D :math:`u_1` coordinate data.
    u2_data : ndarray
        The 3D :math:`u_2` coordinate data.
    u3_data : ndarray
        The 3D :math:`u_3` coordinate data.
    dims : tuple
        The grid dimensions.
    _vtk_dtype : str
        The VTK data type, defaults to "float".
    _vtk_byte_order : str
        The VTK byte order, defaults to ">" (big endian).
    _vtk_fmt : str
        The VTK data format, defaults to ">f".
    """

    def __init__(
        self,
        data: ModeVisualisationData,
        u1: np.ndarray,
        u2: np.ndarray,
        u3: np.ndarray,
    ) -> None:
        self.data = data

        self._vtk_dtype = None
        self._vtk_byte_order = ">"  # big endian
        self._vtk_fmt = None
        self._pbar = None

        self.dims = None
        for i in "123":
            setattr(self, f"_u{i}", None)
            setattr(self, f"u{i}_data", None)
        self._set_coordinate_data(u1, u2, u3)

    def _set_coordinate_data(self, u1: np.ndarray, u2: np.ndarray, u3: np.ndarray):
        """
        Sets the coordinate data.

        Parameters
        ----------
        u1 : np.ndarray
            The 1D :math:`u_1` coordinate array.
        u2 : np.ndarray
            The 1D :math:`u_2` coordinate array.
        u3 : np.ndarray
            The 1D :math:`u_3` coordinate array.
        """
        self._u1 = u1
        self._u2 = u2
        self._u3 = u3
        self.dims = (len(u1), len(u2), len(u3))
        self.u1_data, self.u2_data, self.u3_data = np.meshgrid(
            self._u1, self._u2, self._u3, indexing="ij"
        )

    def get_coordinate_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the coordinate data. This should always return the data in a Cartesian
        reference frame (u1, u2, u3), so coordinate transformations should be
        implemented in subclassed if necessary.

        Returns
        -------
        u1_data : ndarray
            The 3D :math:`u_1` coordinate data.
        u2_data : ndarray
            The 3D :math:`u_2` coordinate data.
        u3_data : ndarray
            The 3D :math:`u_3` coordinate data.
        """
        raise NotImplementedError()

    def broadcast_to_3d(self, array: np.ndarray) -> np.ndarray:
        """
        Broadcasts a 1D array to a 3D array with the same shape as the coordinate data.

        Parameters
        ----------
        array : np.ndarray
            The 1D array to broadcast.

        Returns
        -------
        np.ndarray
            The broadcasted array.
        """
        return np.broadcast_to(array, shape=reversed(self.dims)).transpose()

    def get_solution(self, name: str, time: float) -> np.ndarray:
        """
        Returns the eigenmode solution for a given time.

        Parameters
        ----------
        name : str
            The name of the eigenfunction.
        time : float
            The time at which to get the solution.

        Returns
        -------
        np.ndarray
            The eigenmode solution.
        """
        name = validate_ef_name(self.data.ds_bg, name)
        solution = 0
        for all_efs, factors, k2, k3 in zip(
            self.data._all_efs, self.data.complex_factor, self.data.k2, self.data.k3
        ):
            for ef, factor in zip(all_efs, factors):
                solution += self.data.get_mode_solution(
                    ef=self.broadcast_to_3d(ef.get(name)),
                    omega=ef.get("eigenvalue"),
                    complex_factor=factor,
                    u2=self.u2_data,
                    u3=self.u3_data,
                    t=time,
                    k2=k2,
                    k3=k3,
                )
        return solution

    def _log_info(self, msg: str) -> None:
        """
        Logs an info message only if the progress bar is inactive.
        """
        if self._pbar is not None:
            return
        pylboLogger.info(msg)

    def _validate_and_set_dtype(self, dtype: str) -> None:
        """
        Validates and sets the VTK data type.

        Parameters
        ----------
        dtype : str
            The VTK data type. Valid values are "float32" and "float64".

        Raises
        ------
        ValueError
            If the VTK data type is not valid.
        """
        if dtype == "float32":
            self._vtk_dtype = "float"
            self._vtk_fmt = f"{self._vtk_byte_order}f"
        elif dtype == "float64":
            self._vtk_dtype = "double"
            self._vtk_fmt = f"{self._vtk_byte_order}d"
        else:
            raise ValueError(f"dtype {dtype} not supported.")

    def _write_vtk_header(self, vtkfile):
        """
        Writes the VTK file header. This includes the VTK file version, the
        data type, the grid dimensions and the number of points.

        Parameters
        ----------
        vtkfile : str
            The name of the VTK file to write to.
        """
        self._log_info("writing VTK file header...")
        with open(vtkfile, "w") as ostream:
            vtktitle = vtkfile.stem if len(vtkfile.stem) < 256 else "vtk output"
            ostream.write("# vtk DataFile Version 3.0 \n")
            ostream.write(f"{vtktitle} \n")
            ostream.write("BINARY \n")
            ostream.write("DATASET STRUCTURED_GRID \n")
            ostream.write(f"DIMENSIONS {self.dims[0]} {self.dims[1]} {self.dims[2]} \n")
            ostream.write(f"POINTS {np.prod(self.dims)} {self._vtk_dtype} \n")

    def _write_vtk_coordinate_data(self, vtkfile):
        """
        Writes the VTK grid coordinates.

        Parameters
        ----------
        vtkfile : str
            The name of the VTK file to write to.
        """
        self._log_info("writing VTK coordinate data...")
        u1_data, u2_data, u3_data = self.get_coordinate_data()
        with open(vtkfile, "ab") as ostream:
            for k in range(self.dims[2]):
                for j in range(self.dims[1]):
                    for i in range(self.dims[0]):
                        ostream.write(struct.pack(self._vtk_fmt, u1_data[i, j, k]))
                        ostream.write(struct.pack(self._vtk_fmt, u2_data[i, j, k]))
                        ostream.write(struct.pack(self._vtk_fmt, u3_data[i, j, k]))

    def _write_vtk_point_data_start(self, vtkfile):
        """
        Writes the VTK point data start marker, i.e. the 'POINT_DATA' statement and
        the number of points.

        Parameters
        ----------
        vtkfile : str
            The name of the VTK file to write to.
        """
        with open(vtkfile, "a") as ostream:
            ostream.write(f"\nPOINT_DATA {np.prod(self.dims)} \n")

    def _write_vtk_scalar_field(self, vtkfile, fieldname, fielddata):
        """
        Writes a 3D VTK scalar field with a given fieldname. If `fielddata` is
        smaller than `1e-12` everywhere the field is not written to the VTK file.

        Parameters
        ----------
        vtkfile : str
            The name of the VTK file to write to.
        fieldname : str
            The name of the field.
        fielddata : ndarray
            The field data.
        """
        if np.all(np.isclose(fielddata, 0, atol=1e-12)):
            pylboLogger.warning(
                f"field {fieldname} is zero everywhere and thus not written to VTK."
            )
            return
        # note: spaces are NOT supported in fieldnames (parentheses should be fine)
        # see https://gitlab.kitware.com/paraview/paraview/-/issues/19769
        fieldname = fieldname.replace(" ", "_")
        with open(vtkfile, "a") as ostream:
            ostream.write(f"SCALARS {fieldname} {self._vtk_dtype} \n")
            ostream.write("LOOKUP_TABLE default \n")
        with open(vtkfile, "ab") as ostream:
            for k in range(self.dims[2]):
                for j in range(self.dims[1]):
                    for i in range(self.dims[0]):
                        ostream.write(struct.pack(self._vtk_fmt, fielddata[i, j, k]))

    def _write_vtk_auxiliary_coordinates(self, vtkfile):
        """
        Writes auxiliary coordinate data to the VTK file, for example the theta values
        in cylindrical geometry. These are needed for appropriate transformations
        to draw vector fields in e.g. ParaView.

        Parameters
        ----------
        vtkfile : str
            The name of the VTK file to write to.
        """
        pass

    def export_to_vtk(
        self,
        filename: str,
        time: Union[float, np.ndarray],
        names: Union[str, list[str]] = None,
        bg_names: Union[str, list[str]] = None,
        dtype: str = "float32",
        starting_index: int = 0,
    ) -> None:
        """
        Exports the mode solution to a VTK file.

        Parameters
        ----------
        filename : str
            The name of the VTK file to write to.
        time : Union[float, np.ndarray]
            The time(s) at which to export the mode solution.
        names : Union[str, list[str]], optional
            The name(s) of the mode(s) to export.
        bg_names : Union[str, list[str]], optional
            The name(s) of the equilibrium background(s) to export.
        dtype : str, optional
            The VTK data type, defaults to "float32" (32 bit floating point).
            Can be set to "float64" (64 bit floating point) but uses more memory.
        starting_index : int, optional
            The starting index for filenames, defaults to 0.
        """
        time = transform_to_numpy(time)
        names = [] if names is None else transform_to_list(names)
        bg_names = [] if bg_names is None else transform_to_list(bg_names)
        self._validate_and_set_dtype(dtype)
        filename = Path(filename).with_suffix("")  # remove extension
        self._log_info("exporting eigenmode(s) to VTK file...")
        if len(time) > 1:
            self._pbar = tqdm(total=len(time), desc="writing VTK files", unit="file")
            self.data._print_bg_info = False
        for it, t in enumerate(time, start=starting_index):
            vtkfile = Path(f"{filename}_t{it:04d}.vtk")
            self._write_vtk_header(vtkfile)
            self._write_vtk_coordinate_data(vtkfile)
            self._write_vtk_point_data_start(vtkfile)
            self._log_info("writing VTK scalar field data...")
            for name in names:
                solution = self.get_solution(name, t)
                self._write_vtk_scalar_field(vtkfile, name, solution)
            for bg_name in bg_names:
                bg = self.data.get_background(shape=self.dims, name=bg_name)
                self._write_vtk_scalar_field(vtkfile, bg_name, bg)
            self._write_vtk_auxiliary_coordinates(vtkfile)
            if self._pbar is not None:
                self._pbar.update()
            self._log_info(f"done. File exported to {vtkfile.resolve()}")


class VTKCartesianData(VTKDataExporter):
    def __init__(
        self,
        data: ModeVisualisationData,
        u2: np.ndarray,
        u3: np.ndarray,
    ) -> None:
        super().__init__(data=data, u1=data.ds_bg.ef_grid, u2=u2, u3=u3)

    def get_coordinate_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.u1_data, self.u2_data, self.u3_data


class VTKCylindricalData(VTKDataExporter):
    def __init__(
        self,
        data: ModeVisualisationData,
        u2: np.ndarray,
        u3: np.ndarray,
    ) -> None:
        super().__init__(data=data, u1=data.ds_bg.ef_grid, u2=u2, u3=u3)

    def get_coordinate_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.u1_data * np.cos(self.u2_data),
            self.u1_data * np.sin(self.u2_data),
            self.u3_data,
        )

    def _write_vtk_auxiliary_coordinates(self, vtkfile):
        self._write_vtk_scalar_field(vtkfile, "thetas", self.u2_data)
