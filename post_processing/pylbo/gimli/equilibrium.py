import sympy as sp

from pylbo.gimli.utils import is_symbol_dependent

class Variables:
    """
    Defines a set of variables and constants to be used in defining an Equilibrium object.

    Examples
    --------
    >>> from pylbo.gimli import Variables
    >>> var = Variables()
    """
    def __init__(self):
        self.x, self.y, self.z = sp.symbols('x,y,z')
        self.rho0, self.T0, self.B0sq = sp.symbols('rho_0,T_0,B_0^2')

        self.k2, self.k3 = sp.symbols('k_2,k_3')
        self.rhoc, self.Tc, self.B2c, self.B3c, self.v2c, self.v3c, self.pc = sp.symbols('rho_c,T_c,B_2,B_3,v_2,v_3,p_c')
        self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8 = sp.symbols('p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8')
        self.alpha, self.beta, self.delta, self.theta, self.tau, self.lamda, self.nu = sp.symbols('alpha,beta,delta,theta,tau,lambda,nu')
        self.r0, self.rc, self.rj, self.Bth0, self.Bz0, self.V, self.j0, self.g = sp.symbols('r_0,r_c,r_j,B_theta,B_z,V,j_0,g')
        self.fkey = {
            'rho_c' : 'cte_rho0',
            'T_c' : 'cte_T0',
            'B_2' : 'cte_B02',
            'B_3' : 'cte_B03',
            'v_2' : 'cte_v02',
            'v_3' : 'cte_v03',
            'p_c' : 'cte_p0',
            'p_1' : 'p1',
            'p_2' : 'p2',
            'p_3' : 'p3',
            'p_4' : 'p4',
            'p_5' : 'p5',
            'p_6' : 'p6',
            'p_7' : 'p7',
            'p_8' : 'p8',
            'r_0' : 'r0',
            'r_c' : 'rc',
            'r_j' : 'rj',
            'B_theta' : 'Bth0',
            'B_z' : 'Bz0',
            'j_0' : 'j0'
        }

class Equilibrium:
    """"
    Class containing all equilibrium expressions and initialisation functions.
    This object is a required argument when generating user files with the Legolas and Amrvac classes.

    Parameters
    ----------
    var : :class:`Variables`
        The Variables object containing the symbols to be used in the equilibrium expressions.
    rho0 : sympy expression
        The equilibrium density expression.
    v02, v03 : sympy expressions
        The equilibrium velocity expressions.
    T0 : sympy expression
        The equilibrium temperature expression.
    B02, B03 : sympy expressions
        The equilibrium magnetic field expressions.
    resistivity : sympy expression
        The resistivity expression.
    gravity : constant
        The gravitational acceleration.
    condpara : sympy expression
        The parallel conduction expression.
    condperp : sympy expression
        The perpendicular conduction expression.
    cooling : sympy expression
        The cooling expression.
    heating : sympy expression
        The heating expression.

    Examples
    --------
    The example below defines a homogeneous hydrodynamic equilibrium with constant density and temperature.
    Their values can be set later when passing this equilibrium to the Legolas or Amrvac class along with a dictionary.

    >>> from pylbo.gimli import Equilibrium, Variables
    >>> var = Variables()
    >>> eq = Equilibrium(var, rho0=var.rhoc, v02=0, v03=0, T0=var.Tc)
    """
    def __init__(self, var, rho0, v02, v03, T0, B02=None, B03=None,
                 resistivity=None, gravity=None, condpara=None, condperp=None,
                 cooling=None, heating=None):
        self.variables = var
        self.rho0 = rho0
        self.v02, self.v03 = v02, v03
        self.T0 = T0
        self.B02, self.B03 = B02, B03

        self._dict_phys = {
            'resistivity' : [resistivity, ['eta', 'detadT', 'detadr'], 
                             [self.variables.T0, self.variables.x]],
            'gravity' : [gravity, ['g0'], []],
            'parallel_conduction' : [condpara, ['tcpara', 'dtcparadT'], [self.variables.T0]],
            'perpendicular_conduction' : [condperp, ['tcperp', 'dtcperpdT', 'dtcperpdrho', ' dtcperpdB2'], [self.variables.T0, self.variables.rho0, self.variables.B0sq]],
            'cooling' : [cooling, ['lambdaT', 'dlambdadT'], [self.variables.T0]],
            'heating' : [heating, ['H', 'dHdT', 'dHdrho'], [self.variables.T0, self.variables.rho0]]
        }

    def get_physics(self):
        """
        Returns a dictionary containing the physics expressions and the dependencies to check for.
        """
        return self._dict_phys
    
    def get_dependencies(self):
        """
        Checks for dependencies on other equilibrium quantities.
        Returns a dictionary with the replacement expressions for use in Fortran files.
        """
        if is_symbol_dependent([self.variables.x], self.rho0):
            rho_replace = '(rho0(x))'
        else:
            rho_replace = '(rho0())'
        if is_symbol_dependent([self.variables.x], self.T0):
            T_replace = '(T0(x))'
        else:
            T_replace = '(T0())'
        if is_symbol_dependent([self.variables.x], self.B02) and is_symbol_dependent([self.variables.x], self.B03):
            B2_replace = '(B02(x)**2+B03(x)**2)'
        elif is_symbol_dependent([self.variables.x], self.B02):
            B2_replace = '(B02(x)**2+B03()**2)'
        elif is_symbol_dependent([self.variables.x], self.B03):
            B2_replace = '(B02()**2+B03(x)**2)'
        else:
            B2_replace = '(B02()**2+B03()**2)'
        
        dict_dependencies = {
            'rho_0': rho_replace,
            'T_0': T_replace,
            'B_0^2': B2_replace
        }
        return dict_dependencies