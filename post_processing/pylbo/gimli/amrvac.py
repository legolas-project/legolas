import numpy as np
from scipy.io import FortranFile

from pylbo.utilities.datfiles.file_loader import load
from pylbo.visualisation.modes.mode_data import ModeVisualisationData

class Amrvac:
    """
    Class to prepare Legolas data for use in MPI-AMRVAC (https://amrvac.org).

    Parameters
    ----------
    config : dict
        The configuration dictionary detailing which Legolas file and selection of eigenmodes to use.
    """
    def __init__(self, config):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """
        Validates the presence and value of `physics_type` in the configuration dictionary.

        Raises
        ------
        ValueError
            If `physics_type` is missing or invalid.
        """
        if not 'physics_type' in self.config.keys():
            raise ValueError('"physics_type" ("hd" / "mhd") not specified.')
        elif self.config['physics_type'] == 'mhd':
            self.ef_list = ['rho', 'v1', 'v2', 'v3', 'p', 'B1', 'B2', 'B3']
            self.eq_list = ['rho0', None, 'v02', 'v03', 'rho0 * T0', None, 'B02', 'B03']
            self.units = ['unit_length', 'unit_numberdensity', 'unit_temperature', 'unit_density', 
                      'unit_pressure', 'unit_velocity', 'unit_magneticfield', 'unit_time']
        elif self.config['physics_type'] == 'hd':
            self.ef_list = ['rho', 'v1', 'v2', 'v3', 'p']
            self.eq_list = ['rho0', None, 'v02', 'v03', 'rho0 * T0']
            self.units = ['unit_length', 'unit_numberdensity', 'unit_temperature', 'unit_density', 
                      'unit_pressure', 'unit_velocity', 'unit_time']
        else:
            raise ValueError('Unknown physics type.')
        return
    
    def _validate_datfile(self):
        """
        Validates whether a valid Legolas data file was specified in the configuration.
        Further checks whether all necessary parameters are present in the configuration to prepare Legolas data for use with MPI-AMRVAC.

        Raises
        ------
        ValueError
            If no datfile is specified or if the datfile is invalid; if no initial guess for the eigenvalue is specified;
            if the initial guess for the eigenvalue is not a single float/complex number or a list/NumPy array of float/complex numbers;
            if `ev_time` for the eigenvalue is not a float or an integer;
            if the length of `weights` is not equal to the number of eigenvalues or if the elements of the weights do not add up to 1;
            if `ef_factor` is not a list with length equal to the number of eigenvalues or if `ef_factor` does not have modulus 1;
            if `quantity` for normalisation is not specified or is not a string;
            if `quantity` is not in the list of equilibrium quantities;
            if `percentage` is not a float.
        """
        if not 'datfile' in self.config.keys():
            raise ValueError('No datfile specified.')
        else:
            try:
                self.ds = load(self.config['datfile'])
            except:
                raise ValueError('Invalid datfile specified.')
            
        if not 'ev_guess' in self.config.keys():
            raise ValueError('Initial guess for eigenvalue not specified.')
        elif not isinstance(self.config['ev_guess'], (float, complex, list, np.ndarray)):
            raise ValueError('"ev_guess" must be a single float/complex number or a list/NumPy array of float/complex numbers.')
        elif isinstance(self.config['ev_guess'], (float, complex)):
            self.config['ev_guess'] = [self.config['ev_guess']]
        
        if not 'ev_time' in self.config.keys():
            self.config['ev_time'] = 0
            print('No "ev_time" specified, defaulting to 0.')
        elif not isinstance(self.config['ev_time'], (float, int)):
            raise ValueError('"ev_time" must be a float or an integer.')

        if 'weights' in self.config.keys():
            if len(self.config['ev_guess']) > 1 and not isinstance(self.config['weights'], (list, np.ndarray)):
                raise ValueError('"weights" must be a list with length equal to the number of eigenvalues and elements adding up to 1.')
            elif len(self.config['ev_guess']) != len(self.config['weights']):
                raise ValueError('Length of "weights" must be equal to the number of eigenvalues.')
            elif abs(np.sum(self.config['weights'])) > 1e-12:
                raise ValueError('Elements of "weights" must add up to 1.')
        else:
            print('No "weights" specified, defaulting to equal weights.')
            self.config['weights'] = np.ones(len(self.config['ev_guess'])) / len(self.config['ev_guess'])
        
        if 'ef_factor' in self.config.keys():
            if len(self.config['ev_guess']) > 1 and not isinstance(self.config['ef_factor'], (list, np.ndarray)):
                raise ValueError('"ef_factor" must be a list with length equal to the number of eigenvalues.')
            elif not isinstance(self.config['ef_factor'], (float, int, complex)):
                raise ValueError('"ef_factor" must be an integer, a float, or a complex number.')
            elif abs(self.config['ef_factor'] - 1) > 1e-12:
                raise ValueError('"ef_factor" must have modulus 1.')
            else:
                self.config['ef_factor'] = [self.config['ef_factor']]
        else:
            print('No "ef_factor" specified, defaulting to 1 for all eigenvalues.')
            self.config['ef_factor'] = np.ones(len(self.config['ev_guess']))

        if not 'quantity' in self.config.keys():
            print('No "quantity" specified for normalisation, defaulting to "B02".')
            self.config['quantity'] = 'B02'
        elif not isinstance(self.config['quantity'], str):
            raise ValueError('"quantity" must be a string.')
        elif not self.config['quantity'] in self.eq_list:
            raise ValueError(f'Unknown quantity "{self.config["quantity"]}" specified.')
        
        if not 'percentage' in self.config.keys():
            print('No "percentage" specified, defaulting to 0.01.')
            self.config['percentage'] = 0.01
        elif not isinstance(self.config['percentage'], float):
            raise ValueError('"percentage" must be a float.')
        
        if 'norm_range' in self.config.keys():
            if not isinstance(self.config['norm_range'], (list, np.ndarray)) or len(self.config['norm_range']) != 2:
                raise ValueError('"norm_range" must be a list or NumPy array with two elements.')
            elif self.config['norm_range'][0] >= self.config['norm_range'][1]:
                raise ValueError('First element of "norm_range" must be smaller than the second element.')

        return
    
    def _get_combined_perturbation(self, ef):
        """
        Takes Legolas's perturbations of different eigenvalues and adds them up to a single perturbation.

        Parameters
        ----------
        ef : str
            The eigenfunction to combine.
        
        Returns
        -------
        np.ndarray
            The combined perturbation.
        """
        ef_data = self.ds.get_eigenfunctions(ev_guesses=self.config['ev_guess'])
        perturbation = np.zeros(self.ds.ef_gridpoints, dtype=np.complex128)
        for ii in range(len(ef_data)):
            fac = self.config['ef_factor'][ii]
            w = self.config['weights'][ii]
            expfac = np.exp(-1j * ef_data[ii]['eigenvalue'] * self.config['ev_time'])
            raw = ef_data[ii][ef]
            perturbation += w * fac * (raw / np.nanmax(np.abs(raw))) * expfac
        return perturbation

    def _get_total_perturbation(self, ef_type):
        """
        Combines the perturbations of different eigenvalues into a single perturbation.
        Derives the pressure perturbation from the density and temperature perturbations.

        Parameters
        ----------
        ef_type : str
            The eigenfunction to calculate.

        Returns
        -------
        np.ndarray
            The total perturbation.
        """
        if ef_type == 'p':
            rho1 = self._get_combined_perturbation('rho')
            T1 = self._get_combined_perturbation('T')
            data1 = ModeVisualisationData(self.ds, self.config['ev_guess'], ef_name='rho', add_background=True)
            rho0 = data1.get_background(rho1.shape, 'rho0')
            data2 = ModeVisualisationData(self.ds, self.config['ev_guess'], ef_name='T', add_background=True)
            T0 = data2.get_background(T1.shape, 'T0')
            perturbation = rho1 * T0 + rho0 * T1
        else:
            perturbation = self._get_combined_perturbation(ef_type)
        return perturbation
    
    def _get_normalisation(self):
        """
        Normalises the perturbation of the specified quantity by the maximum background value.

        Returns
        -------
        float
            The normalisation factor.
        """
        ef_match = self.config['quantity'].replace('0', '')
        max_bg = np.nanmax(np.abs(self.ds.equilibria[self.config['quantity']]))
        perturbation = self._get_total_perturbation(ef_match)
        if np.nanmax(np.abs(perturbation)) < 1e-10:
            raise ValueError(f"{self.config['quantity']} is not perturbed by the specified mode(s). Select another quantity, please.")
        else:
            if 'norm_range' in self.config.keys():
                idx1 = np.where(self.ds.ef_grid > self.config['norm_range'][0])[0][0]
                idx2 = np.where(self.ds.ef_grid > self.config['norm_range'][1])[0][0]
                perturbation = perturbation[idx1:idx2]
            norm = self.config['percentage'] * max_bg / np.nanmax(np.abs(perturbation))
        return norm

    def prepare_legolas_data(self, loc='./'):
        """
        Prepares a file (.ldat) from the Legolas data for use with MPI-AMRVAC.

        Parameters
        ----------
        loc : str, optional
            The location to save the .ldat file. Default is the current directory.

        Raises
        ------
        ValueError
            If the datfile is invalid.
        
        Examples
        --------
        >>> from pylbo.gimli import Amrvac
        >>> amrvac_config = {
        >>>     "datfile": "./datfile.dat",
        >>>     "physics_type": "mhd",
        >>>     "ev_guess": [-0.1, 0.1],
        >>>     "ev_time": 0,
        >>>     "percentage": 0.01,
        >>>     "quantity": "rho0"
        >>> }
        >>> amrvac = gimli.Amrvac(amrvac_config)
        >>> amrvac.prepare_legolas_data()
        """
        self._validate_datfile()
        datfile = self.config['datfile']
        position = -1
        for index in range(len(datfile)):
            if datfile[index] == '/':
                position = index
        f = FortranFile(loc + datfile[position+1:-4] + '.ldat', 'w')
        f.write_record(np.array([self.ds.ef_gridpoints], dtype=np.int32))
        f.write_record(np.array([self.ds.parameters['k2'], self.ds.parameters['k3']], dtype=np.float64))
        f.write_record(self.ds.ef_grid)

        norm = self._get_normalisation()
        for ix in range(len(self.ef_list)):
            pert = self._get_total_perturbation(self.ef_list[ix]) * norm
            f.write_record(pert)

        u = []
        for ix in range(len(self.units)):
            u.append(self.ds.units[self.units[ix]])
        f.write_record(np.array(u, dtype=np.float64))

        f.close()
        return
