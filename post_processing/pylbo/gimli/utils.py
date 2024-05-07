import sys, os

def is_symbol_dependent(symbols, expr):
    myset = expr.free_symbols
    sdep = False
    for symb in symbols: 
        if symb in myset:
            sdep = True
    return sdep

def is_sympy_number(expr):
    myset = expr.free_symbols
    if myset == set():
        return True
    else:
        return False

def get_equilibrium_parameters(param):
    param_dict = param['parameters']
    keys = list(param_dict.keys())
    if 'k2' in keys:
        keys.remove('k2')
    if 'k3' in keys:
        keys.remove('k3')
    return ', '.join(keys)

### I/O functions
def create_file(filename):
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
    for ix in range(level):
        string = '  ' + string
    file.write(string + '\n')
    return