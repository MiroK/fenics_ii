from dolfin import Expression, errornorm
from sympy.printing import ccode
import sympy as sp
import numpy as np


def expr_body(expr, **kwargs):
    if not hasattr(expr, '__len__'):
        # Defined in terms of some coordinates
        xyz = set(sp.symbols('x[0], x[1], x[2]'))
        xyz_used = xyz & expr.free_symbols
        assert xyz_used <= xyz
        # Expression params which need default values
        params = (expr.free_symbols - xyz_used) & set(kwargs.keys())
        # Body
        expr = ccode(expr).replace('M_PI', 'pi')
        # Default to zero
        kwargs.update(dict((str(p), 0.) for p in params))
        # Convert
        return expr
    # Vectors, Matrices as iterables of expressions
    else:
        return [expr_body(e, **kwargs) for e in expr]


def as_expression(expr, degree=4, **kwargs):
    '''Turns sympy expressions to Dolfin expressions.'''
    return Expression(expr_body(expr), degree=degree, **kwargs)


def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        cr.next()
        return cr
    return start


@coroutine
def monitor_error(u, norms, memory, reduction=lambda x: x):
    '''
    Send in current solution to get the error size and convergence printed.
    '''
    GREEN = '\033[1;37;32m%s\033[0m'
    
    mesh_size0, error0 = None, None
    counter = 0
    while True:
        counter += 1
        uh = yield
        mesh_size = uh[0].function_space().mesh().hmin()

        error = [norm(ui, uhi) for norm, ui, uhi in zip(norms, u, uh)]
        error = np.array(reduction(error))

        ndofs = [uh_i.function_space().dim() for uh_i in uh]

        if error0 is not None:
            rate = np.log(error/error0)/np.log(mesh_size/mesh_size0)
        else:
            rate = np.nan*np.ones_like(error)
            
        msg = ' '.join(['{case %d}' % counter] +
                       ['h = %.4E' % mesh_size] +  # Resolution
                       # Error
                       ['e_(u%d) = %.2E[%.2f]' % (i, e, r)  
                        for i, (e, r) in enumerate(zip(error, rate))] +
                       # Unknowns
                       ['#(%d)=%d' % p for p in enumerate(ndofs)] +
                       # Total
                       ['|%d' % sum(ndofs)])
        # Screen
        print GREEN % msg
        
        error0, mesh_size0 = error, mesh_size
        memory.append(np.r_[mesh_size, error])


# Two arg norms
H1_norm = lambda u, uh: errornorm(u, uh, 'H1', degree_rise=2)
H10_norm = lambda u, uh: errornorm(u, uh, 'H10', degree_rise=2)
L2_norm = lambda u, uh: errornorm(u, uh, 'L2', degree_rise=2)
Hdiv_norm = lambda u, uh: errornorm(u, uh, 'Hdiv', degree_rise=2)
Hdiv0_norm = lambda u, uh: errornorm(u, uh, 'Hdiv0', degree_rise=2)
Hcurl_norm = lambda u, uh: errornorm(u, uh, 'Hcurl', degree_rise=2)
Hcurl0_norm = lambda u, uh: errornorm(u, uh, 'Hcurl0', degree_rise=2)
