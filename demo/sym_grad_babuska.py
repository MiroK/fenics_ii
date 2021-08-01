# We solve
#
# -div(sym(grad(u)) + u = f in Omega = (0, 1)^2
#                    Bu = g on boundary 
#
# where Bu=u Bu = u.n or Bu = u.t. Note that based on Bu additional boundary
# conditions need to be enforced to get a well posed problem
from utils import PieceWiseExpression
from dolfin import *
from xii import *
import ulfy 


def setup_mms(Bop):
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
    
    x, y = SpatialCoordinate(mesh)
    u = as_vector((sin(pi*(x-2*y)),
                   cos(2*pi*(x+y))))

    sigma = sym(grad(u))
    f = -div(sigma) + u

    # For LM We have -sigma.n as LM  when Bu = u
    #                -sigma.n.n      when Bu = u.n
    #                -sigma.n.t      when Bu = u.t
    normals = (Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1)))    
    R = Constant(((0, 1), (-1, 0)))

    lm = [-dot(sigma, n) for n in normals]
    lm_n = [dot(l, n) for l, n in zip(lm, normals)]
    lm_t = [dot(l, dot(R, n)) for l, n in zip(lm, normals)] 

    to_expr = lambda f: ulfy.Expression(f, degree=4)
    
    # Multiplier is traction
    if Bop == 'full':
        lm_ = dict(enumerate(map(to_expr, lm), 1))
    # LM is traction components
    elif Bop == 'normal':
        lm_ = dict(enumerate(map(to_expr, lm_n), 1))
    else:
        assert Bop == 'tangent'
        lm_ = dict(enumerate(map(to_expr, lm_t), 1))

    up = (to_expr(u), PiecewiseExpression(subdomains, lm_))
    
    return {'solution': up,
            'f': to_expr(f),
            'dirichlet': dict(enumerate(map(to_expr, [u]*4), 1)),
            # Will select proper component from traction
            'neumann': {tag: to_expr(dot(sigma, n)) for tag, n in enumerate(normals, 1)},
            'subdomains': subdomains}


def setup_problem(facet_f, mms, Bop):
    '''Babuska on [0, 1]^2'''
    Rot = Constant(((0, 1), (-1, 0)))
    
    mesh = facet_f.mesh()
    n = FacetNormal(mesh)
    t = dot(Rot, n)
    ds = Measure('ds', domain=mesh, subdomain_data=facet_f)
    
    lm_tags = (1, 2, 3, 4)
    bmesh = EmbeddedMesh(facet_f, lm_tags)
    n_ = OuterNormal(bmesh, [0.5, 0.5])
    t_ = dot(Rot, n_)
    dx_ = Measure('dx', domain=bmesh, subdomain_data=bmesh.marking_function)

    V = VectorFunctionSpace(mesh, 'CG', 1)
    # With full we need a vector valued multiplier
    if Bop == 'full':
        Q = VectorFunctionSpace(bmesh, 'CG', 1)
    else:
        Q = FunctionSpace(bmesh, 'CG', 1)

    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
    Tu, Tv = Trace(u, bmesh), Trace(v, bmesh)
    
    f = mms['f']
    g_u, g_sigma = mms['dirichlet'], mms['neumann']
    # We now define the system as
    a, L = block_form(W, 2), block_form(W, 1)
    
    a[0][0] = inner(sym(grad(u)), sym(grad(v)))*dx + inner(u, v)*dx
    L[0] = inner(f, v)*dx
    # From integration by parts we have inner(dot(sigma, n), v)*ds
    # and the decompositions inner(dot(n, dot(sigma, n)), dot(n, v))*ds
    # inner(dot(t, dot(sigma, n)), dot(t, v))*ds. The "other" term de
    # composition goes to the rhs
    if Bop == 'full':
        a[0][1] = inner(Tv, p)*dx_
        a[1][0] = inner(Tu, q)*dx_
        # No contribs to rhs in L[0]        
        L[1] = sum(inner(g_u[tag], q)*dx_(tag) for tag in lm_tags)

    elif Bop == 'normal':
        a[0][1] = inner(dot(Tv, n_), p)*dx_
        a[1][0] = inner(dot(Tu, n_), q)*dx_

        L[0] += sum(inner(dot(g_sigma[tag], t), dot(v, t))*ds(tag) for tag in lm_tags)
        L[1] = sum(inner(dot(g_u[tag], n_), q)*dx_(tag) for tag in lm_tags)

    else:
        a[0][1] = inner(dot(Tv, t_), p)*dx_
        a[1][0] = inner(dot(Tu, t_), q)*dx_

        L[0] += sum(inner(dot(g_sigma[tag], n), dot(v, n))*ds(tag) for tag in lm_tags)
        L[1] = sum(inner(dot(g_u[tag], t_), q)*dx_(tag) for tag in lm_tags)
        
    return a, L, W, None

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm, L2_norm, broken_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide on bcs setup
    parser.add_argument('--Bop', type=str, default='full',
                        choices=['full', 'normal', 'tangent'])
    args, _ = parser.parse_known_args()
    
    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    mms = setup_mms(args.Bop)
    u_true, p_true = mms['solution']
    # ... and will monitor the error in H1 norm and L2 norm for the bulk
    # variable and the multiplier respectively. Note that natural space
    # for LM is H^{-1/2} so L^2 convergence is a simplification here
    clog = ConvergenceLog({
        'u': (u_true, H1_norm, '1'),
        'p': (p_true.expressions, broken_norm(p_true.subdomains, L2_norm), '0')
    })

    print(clog.header())
    for i in range(8):
        n = 2*2**i
        mesh = UnitSquareMesh(n, n)
        facet_f = MeshFunction('size_t', mesh, 1, 0)
        [subdomain.mark(facet_f, tag) for tag, subdomain in mms['subdomains'].items()]
        
        a, L, W, bcs = setup_problem(facet_f, mms, Bop=args.Bop)

        A, b = map(ii_assemble, (a, L))
        A, b = map(ii_convert, (A, b))

        wh = ii_Function(W)
        solve(A, wh.vector(), b)

        uh, ph = wh
        
        clog.add((uh, ph))
        print(clog.report_last(with_name=False))

    iru, _ = clog['u'].get_rate()
    irp, _ = clog['p'].get_rate()

    passed = all((iru > 0.95, irp > 0.45))
    sys.exit(int(passed))
