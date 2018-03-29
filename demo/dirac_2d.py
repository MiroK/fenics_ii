from dolfin import *
from xii import *
import numpy as np


def setup_problem(i, (x0, f, g), eps=None):
    '''Babuska on [0, 1]^2'''
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    volume = sum(c.volume() for c in cells(mesh))

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'R', 0)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
    # Point Constraints
    Du, Dv = PointTrace(u, x0), PointTrace(v, x0)

    a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a01 = inner(Dv, p)*dx  # Note here we use that |mesh| is 1. Otherwise
    a10 = inner(Du, q)*dx  # the integrand has to be scaled down by volume

    L0 = inner(f, v)*dx
    L1 = inner(Constant(g), q)*dx

    a = [[a00, a01], [a10, 0]]
    L = [L0, L1]

    return a, L, W


def setup_preconditioner(W, which, eps=None):
    '''This is a block diagonal preconditioner based on H1 x R^1 norm'''
    from block.algebraic.petsc import AMG
    from hsmg import HsNorm
    
    V, Q = W

    # H1
    u, v = TrialFunction(V), TestFunction(V)
    b00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    # Inverted by BoomerAMG
    B00 = AMG(ii_assemble(b00))
    # Easy 
    B11 = 1

    return block_diag_mat([B00, B11])

# --------------------------------------------------------------------

def setup_mms(eps=None):
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp
    
    x, y  = sp.symbols('x[0], x[1]')
    u = sp.cos(sp.pi*x*(1-x)*y*(1-y))

    f = -u.diff(x, 2) - u.diff(y, 2) + u
    x0 = (0.33, 0.66)
    # The desired point value is that of u in the point
    g = u.subs({x: x0[0], y: x0[1]})
    # This means that no stress is needed to enforce it :)
    p = np.array([0.])

    up = [as_expression(u), p]
    fg = [x0, as_expression(f), g]

    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and abs norm for the multiplier'''
    from common import monitor_error, H1_norm, linf_norm
    return monitor_error(true, [H1_norm, linf_norm], history, path=path)
