import sys, os

def is_symbol_dependent(symbols, expr):
    """
    Checks whether an expression depends on any of the symbols in a given list.

    Parameters
    ----------
    symbols : list
        The list of symbols to check for.
    expr : sympy expression
        The expression to check for the symbols in the list.
    
    Returns
    -------
    sdep : bool
        Whether the expression depends on any of the symbols.
    """
    try:
        myset = expr.free_symbols
    except:
        return False
    sdep = False
    for symb in symbols: 
        if symb in myset:
            sdep = True
    return sdep

def is_sympy_number(expr):
    """
    Checks whether an expression is a number.

    Parameters
    ----------
    expr : sympy expression
        The expression to check.
    
    Returns
    -------
    bool
        Whether the expression is a number.
    """
    myset = expr.free_symbols
    if myset == set():
        return True
    else:
        return False

def get_equilibrium_parameters(param):
    """
    Removes the wavenumbers from the equilibrium parameters.

    Parameters
    ----------
    param : dict
        The equilibrium parameters dictionary.
    
    Returns
    -------
    str
        The equilibrium parameters without the wavenumbers.
    """
    param_dict = param['parameters']
    keys = list(param_dict.keys())
    if 'k2' in keys:
        keys.remove('k2')
    if 'k3' in keys:
        keys.remove('k3')
    return ', '.join(keys)

### I/O functions
def create_file(filename):
    """
    Creates a file with a given path (or asks whether to overwrite it if it exists already).

    Parameters
    ----------
    filename : str
        The file path.
    """
    if os.path.exists(filename):
        overwrite = input('File already exists. Overwrite? (y/n): ')
        if overwrite == 'y' or overwrite == 'Y' or overwrite == 'yes' or overwrite == 'Yes':
            os.remove(filename)
        else:
            print('Cannot overwrite file. Exiting.')
            sys.exit()
    file = open(filename, "x")
    file.close()
    return

def write_pad(file, string, level):
    """
    Writes a string to a file with a given indentation level.

    Parameters
    ----------
    file : file object
        The file object to write to.
    string : str
        The string to write.
    level : int
        The indentation level.
    """
    for ix in range(level):
        string = '  ' + string
    file.write(string + '\n')
    return