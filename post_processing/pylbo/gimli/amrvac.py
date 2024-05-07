import numpy as np
from scipy.io import FortranFile

from pylbo.utilities.datfiles.file_loader import load
from pylbo.visualisation.modes.mode_data import ModeVisualisationData

class Amrvac:
    def __init__(self, config):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
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
    
    def _validate_datfile(self):
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
    
    def _get_combined_perturbation(self, ef):
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
        ef_match = self.config['quantity'].replace('0', '')
        max_bg = np.nanmax(np.abs(self.ds.equilibria[self.config['quantity']]))
        perturbation = self._get_total_perturbation(ef_match)
        if np.nanmax(np.abs(perturbation)) < 1e-10:
            raise ValueError(f"{self.config['quantity']} is not perturbed by the specified mode(s). Select another quantity, please.")
        else:
            norm = self.config['percentage'] * max_bg / np.nanmax(np.abs(perturbation))
        return norm

    def prepare_legolas_data(self, loc='./'):
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

    def user_module(self):
        print('Not implemented yet.')
        return

    def parfile(self):
        print('Not implemented yet.')
        return
