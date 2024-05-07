def is_symbol_dependent(symbols, expr):
    myset = expr.free_symbols
    sdep = False
    for symb in symbols: 
        if symb in myset:
            sdep = True
    return sdep
