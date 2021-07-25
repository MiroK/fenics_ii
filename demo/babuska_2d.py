# The `Hello World` of fractional problems:
#
# -Delta u + u = f in Omega
#            u = g on boundary
#
# with the boundary conditions enforced weakly by Lagrange multiplier.
from dolfin import *
from xii import *


def setup_problem(i, mms, eps=None):
    '''Babuska on [0, 1]^2'''
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))
    Tu = Trace(u, bmesh)
    Tv = Trace(v, bmesh)

    # The line integral
    dx_ = Measure('dx', domain=bmesh)

    a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a01 = inner(Tv, p)*dx_
    a10 = inner(Tu, q)*dx_

    f, g = mms['data']
    L0 = inner(f, v)*dx
    L1 = inner(g, q)*dx_

    a = [[a00, a01], [a10, 0]]
    L = [L0, L1]

    return a, L, W, None


def setup_mms(eps=None):
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp
    
    x, y  = sp.symbols('x[0], x[1]')
    u = sp.cos(sp.pi*x*(1-x)*y*(1-y))
    p = sp.S(0)  # Normal stress is the multiplier, here it is zero

    f = -u.diff(x, 2) - u.diff(y, 2) + u
    g = u

    up = list(map(as_expression, (u, p)))
    fg = list(map(as_expression, (f, g)))

    return {'solution': up, 'data': fg}

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    mms = setup_mms()

    utrue, ptrue = mms['solution']
    for i in range(5):
        a, L, W, bcs = setup_problem(i, mms, eps=None)
        # Use direct solver to get the solution
        A, b = map(ii_assemble, (a, L))
        # Here A, b are cbc.block object so we need to convert them to matrices
        A, b = map(ii_convert, (A, b))
        
        wh = ii_Function(W)
        solve(A, wh.vector(), b)

        uh, ph = wh
        # Check error
        eu = errornorm(utrue, uh, 'H1')
        ep = errornorm(ptrue, ph, 'L2')

        print(f'eu = {eu}, ep = {ep}')

        
        
