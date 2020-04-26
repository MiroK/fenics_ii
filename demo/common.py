from __future__ import absolute_import
from __future__ import print_function
from dolfin import Expression, errornorm
from sympy.printing import ccode
from itertools import takewhile
import sympy as sp
import numpy as np
import re
from six.moves import zip


number = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')

first = lambda iterable: next(iter(iterable))
    
def parse_eps(eps):
    '''Parse the eps param list '''
    groups = []
    i = 0
    while i < len(eps):
        e = eps[i]
        if '[' in e:
            group = [x for x in takewhile(lambda s: ']' not in s, eps[i:])]
            # Don't forget the one that broke stuff
            group.append(eps[i+len(group)])
            i += len(group)
            # Parse it
            groups.append(parse_eps([first(number.findall(s)) for s in group]))
        else:
            groups.append(float(first(number.findall(e))))
            i += 1
    return groups


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
        foo = tuple(expr_body(e, **kwargs) for e in expr)
        # sp.Matrix flattens so we need to reshape back
        if isinstance(expr, sp.Matrix):
            if expr.is_square:
                matrix = np.array(foo).reshape(expr.shape)
                foo = tuple(tuple(row) for row in matrix)

        return foo


def as_expression(expr, degree=4, **kwargs):
    '''Turns sympy expressions to Dolfin expressions.'''
    return Expression(expr_body(expr), degree=degree, **kwargs)


def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        next(cr)
        return cr
    return start


@coroutine
def monitor_error(u, norms, memory, reduction=lambda x: x, path=''):
    '''
    Send in current solution to get the error size and convergence printed.
    '''
    GREEN = '\033[1;37;32m%s\033[0m'
    BLUE = '\033[1;37;34m%s\033[0m'

    if path:
        with open(path, 'w') as f: f.write('#\n')
    
    mesh_size0, error0 = None, None
    counter = 0
    while True:
        counter += 1
        uh, W, niters, r_norm = yield

        # Here I assume that the higher dim mesh foos come first.
        # This then better be a mesh for som LinearComnimation guys
        mesh = None
        for uhi in uh:
            if hasattr(uhi, 'function_space'):
                mesh = uhi.function_space().mesh()
                break
        assert mesh is not None
        mesh_size = mesh.hmin()
        
        error = [norm(ui, uhi) if hasattr(uhi, 'function_space') else norm(ui, uhi, mesh)
                 for norm, ui, uhi in zip(norms, u, uh)]
        error = np.array(reduction(error))

        ndofs = [Wi.dim() for Wi in W]

        if error0 is not None:
            rate = np.log(error/error0)/np.log(mesh_size/mesh_size0)
        else:
            rate = np.nan*np.ones_like(error)
            
        msg = ' '.join(['{case %d}' % counter] +
                       ['h=%.2E' % mesh_size] +  # Resolution
                       # Error
                       ['e_(u%d)=%.2E[%.2f]' % (i, e, r)  
                        for i, (e, r) in enumerate(zip(error, rate))] +
                       # Unknowns
                       ['#(%d)=%d' % p for p in enumerate(ndofs)] +
                       # Total
                       ['#(all)=%d' % sum(ndofs)] +
                       # Rnorm
                       ['|r|_l2=%g' % r_norm] +
                       ['niters=%d' % niters])
        # Screen
        print(GREEN % msg)
        # Log
        if path:
            with open(path, 'a') as f: f.write(msg + '\n')

        error0, mesh_size0 = error, mesh_size
        memory.append(np.r_[mesh_size, error])

        
# Two arg norms
H1_norm = lambda u, uh, m=None: errornorm(u, uh, 'H1', degree_rise=2, mesh=m)
H10_norm = lambda u, uh, m=None: errornorm(u, uh, 'H10', degree_rise=2, mesh=m)
L2_norm = lambda u, uh, m=None: errornorm(u, uh, 'L2', degree_rise=2, mesh=m)
Hdiv_norm = lambda u, uh, m=None: errornorm(u, uh, 'Hdiv', degree_rise=2, mesh=m)
Hdiv0_norm = lambda u, uh, m=None: errornorm(u, uh, 'Hdiv0', degree_rise=2, mesh=m)
Hcurl_norm = lambda u, uh, m=None: errornorm(u, uh, 'Hcurl', degree_rise=2, mesh=m)
Hcurl0_norm = lambda u, uh, m=None: errornorm(u, uh, 'Hcurl0', degree_rise=2, mesh=m)

linf_norm = lambda u, uh: np.linalg.norm(u - uh.vector().get_local())
