# The `Hello World` of fractional problems:
#
# -Delta u + u = f in Omega
#            u = g on boundary
#
# with the boundary conditions enforced weakly by Lagrange multiplier.
from dolfin import *
from xii import *
import ulfy 


def setup_problem(i, mms, case='conforming'):
    '''Babuska on [0, 1]^2'''
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    
    if case == 'conforming':
        bmesh = BoundaryMesh(mesh, 'exterior')

    elif case == 'nested':
        n = 2*2**i
        cmesh = UnitSquareMesh(*(n, )*2)
        bmesh = BoundaryMesh(cmesh, 'exterior')

    else:
        assert case == 'non_nested'
        n = 1+2*2**i
        cmesh = UnitSquareMesh(*(n, )*2)
        bmesh = BoundaryMesh(cmesh, 'exterior')

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


def setup_mms():
    '''Simple MMS problem for UnitSquareMesh'''
    mesh = UnitSquareMesh(2, 2)
    
    x, y  = SpatialCoordinate(mesh)
    u = cos(pi*x*(1-x)*y*(1-y))
    p = Constant(0)

    f = -div(grad(u)) + u
    g = u

    to_expr = lambda f: ulfy.Expression(f, degree=4)
    
    up = tuple(to_expr(x) for x in (u, p))
    fg = tuple(to_expr(x) for x in (f, g))

    return {'solution': up, 'data': fg}

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm, L2_norm

    set_log_level(40)
    
    mms = setup_mms()

    u_true, p_true = mms['solution']

    clog = ConvergenceLog({'u': (u_true, H1_norm, '1'),
                           'p': (p_true, L2_norm, '0')})
    print(clog.header())
    for i in range(5):
        a, L, W, bcs = setup_problem(i, mms, case='non_nested')
        # Use direct solver to get the solution
        A, b = map(ii_assemble, (a, L))
        # Here A, b are cbc.block object so we need to convert them to matrices
        A, b = map(ii_convert, (A, b))
        
        wh = ii_Function(W)
        solve(A, wh.vector(), b)

        uh, ph = wh
        clog.add((uh, ph))

        print(clog.report_last(with_name=False))
