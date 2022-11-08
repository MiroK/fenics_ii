# The `Hello World` of fractional problems 3d eddition
#
# -Delta u + u = f in Omega = (0, 1)^3
#            u = g on bottom by LM
#            u = g strongly elsewhere

from dolfin import *
from xii import *
import ulfy 


def setup_problem(i, mms):
    '''Babuska on [0, 1]^3'''
    n = 2*2**i
    mesh = UnitCubeMesh(*(n, )*3)

    facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    DomainBoundary().mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 0)').mark(facet_f, 2)
    
    bmesh = EmbeddedMesh(facet_f, 2)

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
    Tu, Tv = Trace(u, bmesh), Trace(v, bmesh)

    # The line integral
    dx_ = Measure('dx', domain=bmesh)
    # We now define the system as
    a = block_form(W, 2)
    a[0][0] = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a[0][1] = inner(Tv, p)*dx_
    a[1][0] = inner(Tu, q)*dx_

    f, g = mms['data']
    # And the rhs
    L = block_form(W, 1)
    L[0] = inner(f, v)*dx
    L[1] = inner(g, q)*dx_

    bcs = [[DirichletBC(V, mms['solution'][0], facet_f, 1)],
           [DirichletBC(Q, Constant(0), 'on_boundary')]]

    return a, L, W, bcs


def setup_mms():
    '''Simple MMS problem for UnitSquareMesh'''
    mesh = UnitCubeMesh(1, 1, 1)
    
    x, y, z = SpatialCoordinate(mesh)
    u = cos(pi*x*(1-x)*y*(1-y)*z*(1-z))
    p = Constant(0)

    f = -div(grad(u)) + u
    g = u

    to_expr = lambda f: ulfy.Expression(f, degree=4)
    
    up = tuple(to_expr(x) for x in (u, p))
    fg = tuple(to_expr(x) for x in (f, g))

    return {'solution': up, 'data': fg}

# --------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm, L2_norm
    import sys

    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    mms = setup_mms()
    u_true, p_true = mms['solution']
    # ... and will monitor the error in H1 norm and L2 norm for the bulk
    # variable and the multiplier respectively
    clog = ConvergenceLog({'u': (u_true, H1_norm, '1'), 'p': (p_true, L2_norm, '0')})

    print(clog.header())
    for i in range(5):
        a, L, W, bcs = setup_problem(i, mms)

        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, bcs)
        A, b = map(ii_convert, (A, b))

        wh = ii_Function(W)
        solve(A, wh.vector(), b)

        uh, ph = wh
        
        clog.add((uh, ph))
        print(clog.report_last(with_name=False))

    iru, fru = clog['u'].get_rate()
    irp, frp = clog['p'].get_rate()
    # Compared to poisson_babuska_bc cases now (accidentaly) setting
    # Constant(0) in DirichletBC for LM space is the tru value and hence
    # we get improved convergence. We're also helped by the fact that the
    # true LM is 0 everywhere
    passed = all((iru > 0.95, irp > 1.95))
    # NOTE: discard lstsq fit because we're not in assymptotic regime yet
    print(passed)
    sys.exit(int(passed))
