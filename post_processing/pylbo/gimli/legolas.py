from pylbo.automation.api import generate_parfiles
import sympy as sp
from sympy.printing.fortran import fcode

from pylbo.gimli.utils import create_file, write_pad, get_equilibrium_parameters, is_sympy_number, is_symbol_dependent

def write_physics_calls(file, equil):
    physics = equil.get_physics()
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
    def __init__(self, equilibrium, config):
        self.equilibrium = equilibrium
        self.configuration = config

    def user_module(self, filename='smod_user_defined'):
        name = filename + '.f08'
        create_file(name)
        file = open(name, 'a')
        write_pad(file, '!> Submodule for user-defined equilibria.', 0)
        write_pad(file, 'submodule (mod_equilibrium) smod_user_defined', 0)
        write_pad(file, 'use mod_logging, only: logger', 1)
        eqparam = get_equilibrium_parameters(self.configuration)
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
        if self.configuration['physics_type'] == 'mhd':
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

    def parfile(self, filename='legolas', make_dir=False):
        generate_parfiles(self.configuration, basename=filename, subdir=make_dir)
        return