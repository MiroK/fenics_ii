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
    # There are different way in which the boundary mesh where the 
    # multiplier lives can be defined. In the most straighforward
    # way we take it as a boundary mesh of the background.
    if case == 'conforming':
        bmesh = BoundaryMesh(mesh, 'exterior')
    # The same can be accomplished by picking the LM mesh from marked
    # facets
    elif case == 'conforming_facetf':
        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        DomainBoundary().mark(facet_f, 1)
        bmesh = EmbeddedMesh(facet_f, 1)
    # We can also have a misalignment between trace mesh of the background
    # and the LM mesh. Here we combine 2 cells of trace mesh into one LM
    # mesh cell meaning that each LM mesh vertex is found in the trace mesh.
    elif case == 'nested':
        n = 2*2**i
        cmesh = UnitSquareMesh(*(n, )*2)
        bmesh = BoundaryMesh(cmesh, 'exterior')
    # However this nesting is not necessary
    else:
        assert case == 'non_nested'
        n = 1+2*2**i
        cmesh = UnitSquareMesh(*(n, )*2)
        bmesh = BoundaryMesh(cmesh, 'exterior')

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
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide LM mesh construction
    parser.add_argument('--conformity', type=str, default='conforming',
                        choices=['conforming', 'nested', 'non_nested', 'conforming_facetf'])
    args, _ = parser.parse_known_args()

    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    mms = setup_mms()
    u_true, p_true = mms['solution']
    # ... and will monitor the error in H1 norm and L2 norm for the bulk
    # variable and the multiplier respectively
    clog = ConvergenceLog({'u': (u_true, H1_norm, '1'), 'p': (p_true, L2_norm, '0')})

    print(clog.header())
    for i in range(6):
        a, L, W, bcs = setup_problem(i, mms, case=args.conformity)
        # Use direct solver to get the solution
        A, b = map(ii_assemble, (a, L))
        # Here A, b are cbc.block object so we need to convert them to matrices
        A, b = map(ii_convert, (A, b))
        # Now solve with direct solver
        wh = ii_Function(W)
        solve(A, wh.vector(), b)
        # Components of the wh in V x Q are dolfin Funcitons
        uh, ph = wh
        
        clog.add((uh, ph))
        print(clog.report_last(with_name=False))

    # In this case we use P1 elements so orders 1 and 2 are expected
    # NOTE: the exit code here is for `test_demo.py`
    iru, fru = clog['u'].get_rate()
    irp, frp = clog['p'].get_rate()

    passed = all((iru > 0.95, fru > 0.95, irp > 1.95, frp > 1.95))
    sys.exit(int(passed))
