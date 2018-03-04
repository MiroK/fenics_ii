# The `Hello World` of fractional problems:
# H-0.5 problem
#
# -Delta u + u = f in Omega
#            u = g on boundary
#
# with the boundary conditions enforced weakly by Lagrange multiplier.
from dolfin import *
from xii import *


def solve_problem(i, (f, g)):
    '''Babuska on [0, 1]^2'''
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
    Tu = Trace(u, bmesh)
    Tv = Trace(v, bmesh)

    # The line integral
    dx_ = Measure('dx', domain=bmesh)

    a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a01 = inner(Tv, p)*dx_
    a10 = inner(Tu, q)*dx_

    L0 = inner(f, v)*dx
    L1 = inner(g, q)*dx_

    a = [[a00, a01], [a10, 0]]
    L = [L0, L1]

    # Assemble blocks
    AA, bb = map(ii_assemble, (a, L))
    # Turn into a (monolithic) PETScMatrix/Vector
    AA, bb = map(ii_convert, (AA, bb))

    return AA, bb, W

# --------------------------------------------------------------------

def setup_mms():
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp
    
    x, y  = sp.symbols('x[0], x[1]')
    u = sp.cos(sp.pi*x*(1-x)*y*(1-y))
    p = sp.S(0)  # Normal stress is the multiplier, here it is zero

    f = -u.diff(x, 2) - u.diff(y, 2) + u
    g = u

    up = map(as_expression, (u, p))
    fg = map(as_expression, (f, g))

    return up, fg


def setup_error_monitor(true, history):
    '''We measure error in H1 and L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    return monitor_error(true, [H1_norm, L2_norm], history)
