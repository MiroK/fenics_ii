# Mixed Poisson problem with flux bcs enforced weakly
# 
#       div(sigma) = f (0, 1)^2
#  sigma + grad(u) = 0
#          sigma.n = g_sigma enforced strongly on bottom edge
#          sigma.n = g_sigma enforced by LM on top and right edge
#                u = g_u     enforced weakly
#
# Solve with RT_0-P_0-P_0 or RT_1-P_1-P_1 to showcase higher order
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
    u = sin(pi*(x+y))
    sigma = -grad(u)
    # LM is defined as u, in addition when specifying bcs Hdiv uses
    # sigma, so no need for normal
    p = u

    f = div(sigma)
    g_u = [u]*4  
    g_sigma = [sigma]*4

    to_expr = lambda f: ulfy.Expression(f, degree=4)
    
    w = tuple(to_expr(x) for x in (sigma, u, p))

    return {'solution': w,
            'force': to_expr(f),
            'flux': dict(enumerate(map(to_expr, g_sigma), 1)),
            'pressure': dict(enumerate(map(to_expr, g_u), 1)),
            'subdomains': subdomains}


def setup_problem(facet_f, mms, flux_deg=1):
    '''Babuska on [0, 1]^2'''
    mesh = facet_f.mesh()

    lm_tags, pressure_tags, flux_tags = (2, 4), (1, ), (3, )
        
    bmesh = EmbeddedMesh(facet_f, lm_tags)
    bmesh_subd = bmesh.marking_function

    S = FunctionSpace(mesh, 'RT', flux_deg)
    V = FunctionSpace(mesh, 'DG', flux_deg-1)
    Q = FunctionSpace(bmesh, 'DG', flux_deg-1)
    W = [S, V, Q]

    sigma, u, p = map(TrialFunction, W)
    tau, v, q = map(TestFunction, W)
    
    Tsigma, Ttau = Trace(sigma, bmesh), Trace(tau, bmesh)

    # The line integral; we want subdomains because the data are
    # for piece of boundary
    dx_ = Measure('dx', domain=bmesh, subdomain_data=bmesh_subd)
    ds = Measure('ds', domain=mesh, subdomain_data=facet_f)    
    # For dx_ integral use xii normal instead of FacetNormal
    n_ = OuterNormal(bmesh, [0.5, 0.5])
    n = FacetNormal(mesh)

    # We now define the system as
    a = block_form(W, 2)
    a[0][0] = inner(sigma, tau)*dx
    a[0][1] = -inner(u, div(tau))*dx
    a[0][2] = inner(p, dot(Ttau, n_))*dx_
    a[1][0] = -inner(v, div(sigma))*dx
    a[2][0] = inner(q, dot(Tsigma, n_))*dx_

    # And the rhs
    L = block_form(W, 1)
    # Weakly pressure
    L[0] = -sum(inner(mms['pressure'][tag], dot(tau, n))*ds(tag) for tag in pressure_tags)
    
    L[1] = -inner(mms['force'], v)*dx
    # On multiplier we are given Dirichlet data
    L[2] = sum(inner(dot(mms['flux'][tag], n_), q)*dx_(tag) for tag in lm_tags)

    # Boundary conditions; a list for each subspace
    S_bcs = [DirichletBC(S, mms['flux'][tag], facet_f, tag) for tag in flux_tags]
    V_bcs = []
    Q_bcs = []
    # For W the position indicates subspace
    W_bcs = [S_bcs, V_bcs, Q_bcs]

    return a, L, W, W_bcs

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, Hdiv_norm, L2_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # RT degree
    parser.add_argument('--flux_degree', type=int, default=1, choices=[1, 2])
    args, _ = parser.parse_known_args()

    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    mms = setup_mms()
    # Flux, pressure, multiplier
    sigma_true, u_true, p_true = mms['solution']

    clog = ConvergenceLog({
        'sigma': (sigma_true, Hdiv_norm, 'div'),
        'u': (u_true, L2_norm, '0'),
        'p': (p_true, L2_norm, '0')
    })

    print(clog.header())
    for i in range(5):
        n = 4*2**i
        mesh = UnitSquareMesh(n, n)
        facet_f = MeshFunction('size_t', mesh, 1, 0)
        [subdomain.mark(facet_f, tag) for tag, subdomain in mms['subdomains'].items()]
        
        a, L, W, bcs = setup_problem(facet_f, mms, flux_deg=args.flux_degree)

        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, bcs)
        # Here A, b are cbc.block object so we need to convert them to matrices
        A, b = map(ii_convert, (A, b))
        # Now solve with direct solver
        wh = ii_Function(W)
        solve(A, wh.vector(), b)

        sigmah, uh, ph = wh

        clog.add((sigmah, uh, ph))
        print(clog.report_last(with_name=False))

        
    eoc = 0.95 if args.flux_degree == 1 else 1.95
    # Match increamental and lstsq rate
    passed = all(clog[var].get_rate()[0] > eoc for var in ('sigma', 'u', 'p'))
    passed = passed and all(clog[var].get_rate()[1] > eoc for var in ('sigma', 'u', 'p'))

    sys.exit(int(passed))
