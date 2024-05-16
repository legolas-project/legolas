from pylbo.automation.api import generate_parfiles
import sympy as sp
from sympy.printing.fortran import fcode

from pylbo.gimli.utils import create_file, write_pad, get_equilibrium_parameters, is_sympy_number, is_symbol_dependent
from pylbo.gimli._version import __version__

def write_physics_calls(file, equilibrium):
    """
    Writes the use of user-defined physics functions to the Legolas user module.

    Parameters
    ----------
    file : file
        The file object to write to.
    equilibrium : Equilibrium
        The equilibrium object containing the user-defined equilibrium and physics functions.
    """
    physics = equilibrium.get_physics()
    for key in list(physics.keys()):
        if physics[key][0] != None:
            func = physics[key][1]
            func_list = ''
            for ix in range(len(func)):
                func_list = func_list + func[ix] + '_func=' + func[ix]
                if ix < len(func)-1:
                    func_list = func_list + ', '
            write_pad(file, f'call physics%set_{key}_funcs({func_list})', 2)
    return

def fortran_function(file, expr, varname, translation, constant=False, level=0):
    """
    Writes a sympy expression to the user module as a Fortran function.

    Parameters
    ----------
    file : file
        The file object to write to.
    expr : sympy expression
        The expression to write.
    varname : str
        The name of the function.
    translation : dict
        A dictionary containing any substitution rules for sympy to Fortran expressions.
    constant : bool
        Set to `True` if the function is a constant.
    level : int
        The indentation level.
    """
    if constant:
        write_pad(file, f'real(dp) function {varname}()', level)
    else:
        write_pad(file, f'real(dp) function {varname}(x)', level)
        write_pad(file, 'real(dp), intent(in) :: x', level+1)

    if is_sympy_number(expr):
        write_pad(file, fcode(sp.sympify(float(expr)), assign_to=varname).lstrip(), level+1)
    else:
        func = fcode(expr, assign_to=varname).lstrip()
        for key in list(translation.keys()):
            func = func.replace(key, translation[key])
        func = func.replace('\n', ' &\n')
        func = func.replace('@', '')
        write_pad(file, func, level+1)
    write_pad(file, f'end function {varname}\n', level)
    return

def write_equilibrium_functions(file, equilibrium):
    """
    Iterates over all Legolas equilibrium quantities and writes them to the user module.

    Parameters
    ----------
    file : file
        The file object to write to.
    equilibrium : Equilibrium
        The equilibrium object containing the user-defined equilibrium functions.
    """
    x = equilibrium.variables.x
    varlist = {'rho0' : equilibrium.rho0,
               'v02' : equilibrium.v02,
               'v03' : equilibrium.v03,
               'T0' : equilibrium.T0,
               'B02' : equilibrium.B02,
               'B03' : equilibrium.B03
               }
    for key in list(varlist.keys()):
        expr = varlist[key]
        if expr == None:
            continue
        else:
            expr = sp.sympify(expr)
        cst = not is_symbol_dependent([x], expr)
        fortran_function(file, expr, key, equilibrium.variables.fkey, constant=cst, level=1)
        dexpr = sp.diff(expr, x)
        cst = not is_symbol_dependent([x], dexpr)
        fortran_function(file, dexpr, 'd'+key, equilibrium.variables.fkey, constant=cst, level=1)
        if key != 'rho0':
            ddexpr = sp.diff(dexpr, x)
            cst = not is_symbol_dependent([x], ddexpr)
            fortran_function(file, ddexpr, 'dd'+key, equilibrium.variables.fkey, constant=cst, level=1)    
    return

def write_physics_functions(file, equilibrium):
    """
    Iterates over all Legolas physics expressions and writes them to the user module.

    Parameters
    ----------
    file : file
        The file object to write to.
    equilibrium : Equilibrium
        The equilibrium object containing the user-defined physics functions.
    """
    varlist = equilibrium.get_physics()
    replacements = equilibrium.get_dependencies()
    for key in list(varlist.keys()):
        expr = varlist[key][0]
        if expr == None:
            continue
        else:
            expr = sp.sympify(expr)
        cst = not is_symbol_dependent(varlist[key][2], expr) ### also check for rho_0, T_0 and B_0^2
        fortran_function(file, expr, varlist[key][1][0], replacements, constant=cst, level=1)
        for ix in range(len(varlist[key][2])):
            dexpr = sp.diff(expr, varlist[key][2][ix])
            cst = not is_symbol_dependent(varlist[key][2], dexpr)
            fortran_function(file, dexpr, varlist[key][1][ix+1], replacements, constant=cst, level=1)
    return

class Legolas:
    """
    Class for generating user-defined Legolas modules and parfiles.

    Parameters
    ----------
    equilibrium : Equilibrium
        The equilibrium object containing the user-defined equilibrium and physics functions.
    config : dict
        A dictionary containing the configuration for the Legolas run (both equilibrium parameter values and technical settings).
    """
    def __init__(self, equilibrium, config):
        self.equilibrium = equilibrium
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """
        Validates the validity of the configuration dictionary.

        Raises
        ------
        ValueError
            If the configuration dictionary is missing the `physics_type` key or it contain an invalid value.
        """
        if not 'physics_type' in self.config.keys():
            raise ValueError('"physics_type" ("hd" / "mhd") not specified.')
        elif self.config['physics_type'] != 'hd' and self.config['physics_type'] != 'mhd':
            raise ValueError('Unknown physics type.')
        return

    def user_module(self, filename='smod_user_defined'):
        """
        Writes the user module for the Legolas run.

        Parameters
        ----------
        filename : str
            The name of the user module file.
        
        Examples
        --------
        The example below defines a homogeneous hydrodynamic equilibrium with constant density and temperature.
        The values of the equilibrium parameters are set in the configuration dictionary.

        >>> from pylbo.gimli import Variables, Equilibrium, Legolas
        >>> var = Variables()
        >>> eq = Equilibrium(var, rho0=var.rhoc, v02=0, v03=0, T0=var.Tc)
        >>> config = {
        >>>     "geometry": "Cartesian",
        >>>     "x_start": 0,
        >>>     "x_end": 1,
        >>>     "gridpoints": 51,
        >>>     "parameters": {
        >>>         "k2": 0.5,
        >>>         "k3": 0,
        >>>         "cte_rho0": 1,
        >>>         "cte_T0": 1
        >>>     },
        >>>     "equilibrium_type": "user_defined",
        >>>     "boundary_type": "wall_weak",
        >>>     "physics_type": "mhd"
        >>> }
        >>> legolas = Legolas(eq, config)
        >>> legolas.user_module()
        """
        name = filename + '.f08'
        create_file(name)
        file = open(name, 'a')
        write_pad(file, '!> Submodule for user-defined equilibria.', 0)
        write_pad(file, f'!! Generated with GIMLI v{__version__}.', 0)
        write_pad(file, 'submodule (mod_equilibrium) smod_user_defined', 0)
        write_pad(file, 'use mod_logging, only: logger', 1)
        eqparam = get_equilibrium_parameters(self.config)
        write_pad(file, 'use mod_equilibrium_params, only: ' + eqparam, 1)
        write_pad(file, 'implicit none', 1)
        file.write('\n')
        write_pad(file, 'contains', 0)
        file.write('\n')
        write_pad(file, 'module procedure user_defined_eq', 1)
        write_pad(file, 'if(settings%equilibrium%use_defaults) then', 2)
        write_pad(file, 'call logger%error("No default values specified.")', 3)
        write_pad(file, 'end if', 2)
        file.write('\n')
        write_pad(file, 'call background%set_density_funcs(rho0_func=rho0, drho0_func=drho0)', 2)
        write_pad(file, 'call background%set_velocity_2_funcs(v02_func=v02, dv02_func=dv02, ddv02_func=ddv02)', 2)
        write_pad(file, 'call background%set_velocity_3_funcs(v03_func=v03, dv03_func=dv03, ddv03_func=ddv03)', 2)
        write_pad(file, 'call background%set_temperature_funcs(T0_func=T0, dT0_func=dT0, ddT0_func=ddT0)', 2)
        if self.config['physics_type'] == 'mhd':
            write_pad(file, 'call background%set_magnetic_2_funcs(B02_func=B02, dB02_func=dB02, ddB02_func=ddB02)', 2)
            write_pad(file, 'call background%set_magnetic_3_funcs(B03_func=B03, dB03_func=dB03, ddB03_func=ddB03)', 2)
        file.write('\n')
        write_physics_calls(file, self.equilibrium)
        write_pad(file, 'end procedure user_defined_eq', 1)
        file.write('\n')
        write_equilibrium_functions(file, self.equilibrium)
        write_physics_functions(file, self.equilibrium)
        write_pad(file, 'end submodule', 0)
        file.close()
        return

    def parfile(self, filename='legolas_config', make_dir=False):
        """
        Writes the parameter file for the Legolas run.

        Parameters
        ----------
        filename : str
            The name of the parameter file.
        make_dir : bool
            If `True`, creates a directory for the parameter file.
        
        Returns
        -------
        parfiles : list
            A list containing the paths to the parameter files.
        
        Examples
        --------
        The example below defines a homogeneous hydrodynamic equilibrium with constant density and temperature.
        The values of the equilibrium parameters are set in the configuration dictionary and written to the parameter file.
        
        >>> from pylbo.gimli import Variables, Equilibrium, Legolas
        >>> var = Variables()
        >>> eq = Equilibrium(var, rho0=var.rhoc, v02=0, v03=0, T0=var.Tc)
        >>> config = {
        >>>     "geometry": "Cartesian",
        >>>     "x_start": 0,
        >>>     "x_end": 1,
        >>>     "gridpoints": 51,
        >>>     "parameters": {
        >>>         "k2": 0.5,
        >>>         "k3": 0,
        >>>         "cte_rho0": 1,
        >>>         "cte_T0": 1
        >>>     },
        >>>     "equilibrium_type": "user_defined",
        >>>     "boundary_type": "wall_weak",
        >>>     "physics_type": "mhd"
        >>> }
        >>> legolas = Legolas(eq, config)
        >>> legolas.parfile()
        """
        parfiles = generate_parfiles(self.config, basename=filename, subdir=make_dir)
        return parfiles