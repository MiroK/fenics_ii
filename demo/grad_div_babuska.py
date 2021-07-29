# We solve
# 
#       -grad div u + u = f (0, 1)^2
#                   u.n = 0 on the boundary
#
# Solve with RT_0-P_0 or RT_1-P_1 to showcase higher order
from dolfin import *
from xii import *
import ulfy 


def setup_mms():
    '''Simple MMS problem for UnitSquareMesh'''
    mesh = UnitSquareMesh(2, 2)
    # The following labeling of edges will be enforced
    #   4
    # 1   2
    #   3
    subdomains = {1: CompiledSubDomain('near(x[0], 0)'),
                  2: CompiledSubDomain('near(x[0], 1)'),
                  3: CompiledSubDomain('near(x[1], 0)'),
                  4: CompiledSubDomain('near(x[1], 1)')}
    
    x, y  = SpatialCoordinate(mesh)
    u = as_vector((sin(pi*(x+y)),
                   cos(pi*(x-y))))
    
    # LM is defined as -div(u), in addition when specifying bcs Hdiv uses
    # sigma, so no need for normal
    p = -div(u)

    f = -grad(div(u)) + u
    g_u = [u]*4  

    to_expr = lambda f: ulfy.Expression(f, degree=4)
    
    w = tuple(to_expr(x) for x in (u, p))

    return {'solution': w,
            'force': to_expr(f),
            'u': dict(enumerate(map(to_expr, g_u), 1)),
            'subdomains': subdomains}


def setup_problem(facet_f, mms, RT_deg=1):
    '''Grad-div on [0, 1]^2'''
    mesh = facet_f.mesh()

    lm_tags = (1, 2, 3, 4)
        
    bmesh = EmbeddedMesh(facet_f, lm_tags)
    bmesh_subd = bmesh.marking_function

    V = FunctionSpace(mesh, 'RT', RT_deg)
    Q = FunctionSpace(bmesh, 'DG', RT_deg-1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
    
    Tu, Tv = Trace(u, bmesh), Trace(v, bmesh)

    dx_ = Measure('dx', domain=bmesh, subdomain_data=bmesh_subd)
    # For dx_ integral use xii normal instead of FacetNormal
    n_ = OuterNormal(bmesh, [0.5, 0.5])

    # We now define the system as
    a = block_form(W, 2)
    a[0][0] = inner(u, v)*dx + inner(div(u), div(v))*dx
    a[0][1] = inner(p, dot(Tv, n_))*dx_
    a[1][0] = inner(q, dot(Tu, n_))*dx_   

    # And the rhs
    L = block_form(W, 1)
    L[0] = inner(mms['force'], v)*dx
    # On multiplier we are given Dirichlet data
    L[1] = sum(inner(dot(mms['u'][tag], n_), q)*dx_(tag) for tag in lm_tags)

    return a, L, W, None

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, Hdiv_norm, L2_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # RT degree
    parser.add_argument('--RT_degree', type=int, default=1, choices=[1, 2])
    args, _ = parser.parse_known_args()

    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    mms = setup_mms()
    # Flux, pressure, multiplier
    u_true, p_true = mms['solution']

    clog = ConvergenceLog({
        'u': (u_true, Hdiv_norm, 'div'),
        'p': (p_true, L2_norm, '0')
    })

    print(clog.header())
    for i in range(5):
        n = 2*2**i
        mesh = UnitSquareMesh(n, n)
        facet_f = MeshFunction('size_t', mesh, 1, 0)
        [subdomain.mark(facet_f, tag) for tag, subdomain in mms['subdomains'].items()]
        
        a, L, W, bcs = setup_problem(facet_f, mms, RT_deg=args.RT_degree)

        A, b = map(ii_assemble, (a, L))
        A, b = map(ii_convert, (A, b))

        wh = ii_Function(W)
        solve(A, wh.vector(), b)

        uh, ph = wh

        clog.add((uh, ph))
        print(clog.report_last(with_name=False))
        
    eoc = 0.95 if args.RT_degree == 1 else 1.95
    # Match increamental and lstsq rate
    passed = all(clog[var].get_rate()[0] > eoc for var in ('u', 'p'))
    passed = passed and all(clog[var].get_rate()[1] > eoc for var in ('u', 'p'))

    sys.exit(int(passed))
