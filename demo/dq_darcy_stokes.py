# This example solves the coupled Darcy-Stokes problem where the
# Darcy part is formulated only in terms of pressure (see works
# of Discacciati and Quarteroni)
#
# Darcy domain = [0.25, 0.75]^2
# Stokes domain = [0, 1]^2 \ Darcy domain
#
# On the Darcy domain we solve: uD = -K*grad(pD)
#                               div(uD) = fD
#
# On Stokes we have: -div(sigma(uS, pS)) = fS
#                     div(uS) = 0
#                     where sigma(u, p) = -p*I + 2*mu*D(u) and D = sym o grad
#
# Letting t = sigma(uD, pD).nD, there are following interface conditions:
#
# -t.nS = pD + g_n
# -t.tauS = alpha*uS.tauS - g_t  [tauS is the tangent and alpha is the BJS parameter]
# uS.nS + uD.nD = g_u
#

from utils import rotate, immersed_geometry, PiecewiseExpression
import sympy as sp
from dolfin import *
from xii import *
import ulfy


def setup_mms(parameter_values):
    '''Manufacture solution for the immersed test case'''
    # We take Stokes side as the master for normal orientation
    mesh = UnitSquareMesh(2, 2, 'crossed')

    x, y = SpatialCoordinate(mesh)
    mu, K, alpha = Constant(1), Constant(1), Constant(1)

    phi = sin(pi*(x-2*y))
    uS = as_vector((phi.dx(1), -phi.dx(0)))
    pS = cos(2*pi*(3*x-y))

    pD = sin(2*pi*(x+y))
    uD = -K*grad(pD)

    # Stokes ...
    sigma = 2*mu*sym(grad(uS)) - pS*Identity(2)
    fS = -div(sigma)
    # ... we need for standard boudaries velocity and traction.
    # NOTE: normals are assumed to be labeled as in immersed_geometry
    normalsS = tuple(map(Constant, ((1, 0), (-1, 0), (0, 1), (0, -1), (-1, 0), (1, 0), (0, -1), (0, 1))))
    traction = tuple(dot(sigma, n) for n in normalsS)
    
    # Darcy ...
    fD = div(uD)
    # ... we need pD and uD
    normalsD = tuple(map(Constant, ((-1, 0), (1, 0), (0, -1), (0, 1))))
    
    # Interface
    normalsI = tuple(map(Constant, ((1, 0), (-1, 0), (0, 1), (0, -1))))    

    g_u = tuple(dot(uS, n) - dot(uD, n) for n in normalsI)
    g_n = tuple(-dot(n, dot(sigma, n)) - pD for n in normalsI)
    g_t = tuple(-dot(rotate(n), dot(sigma, n)) - alpha*dot(rotate(n), uS) for n in normalsI)

    # Multiplier is -normal part of traction
    lms = [-dot(n, dot(sigma, n)) for n in normalsI]

    # Don't want to trigger compiler on parameter change
    mu_, alpha_, K_ = sp.symbols('mu, alpha, K')
    subs = {mu: mu_, alpha: alpha_, K: K_}

    # Check coefs
    assert parameter_values['mu'] > 0 and parameter_values['K'] > 0 and parameter_values['alpha'] >= 0

    to_expr = lambda f: ulfy.Expression(f, degree=4, subs=subs,
                                        mu=parameter_values['mu'],
                                        K=parameter_values['K'],
                                        alpha=parameter_values['alpha'])

    # As tagged in utils.immersed_geometry
    lm_subdomains = {
        1: CompiledSubDomain('near(x[0], 0.25) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))'),
        2: CompiledSubDomain('near(x[0], 0.75) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))'),
        3: CompiledSubDomain('near(x[1], 0.25) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS))'),
        4: CompiledSubDomain('near(x[1], 0.75) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS))')
    }
    
    return {
        'solution': {'uS': to_expr(uS), 'uD': to_expr(uD), 'pS': to_expr(pS), 'pD': to_expr(pD),
                     'lm': PiecewiseExpression(lm_subdomains, dict(enumerate(map(to_expr, lms), 1)))},
        'fS': to_expr(fS),
        'fD': to_expr(fD),
        # Standard boundary data
        'velocity_S': dict(enumerate(map(to_expr, [uS]*len(normalsS)), 1)),
        'traction_S': dict(enumerate(map(to_expr, traction), 1)),
        'pressure_D': dict(enumerate(map(to_expr, [pD]*len(normalsD)), 1)),
        'flux_D': dict(enumerate(map(to_expr, [uD]*len(normalsD)), 1)),
        # Interface boundary conditions
        'g_u': dict(enumerate(map(to_expr, g_u), 1)),
        'g_n': dict(enumerate(map(to_expr, g_n), 1)),
        'g_t': dict(enumerate(map(to_expr, g_t), 1)),
        # Geometry setup
        'get_geometry': immersed_geometry
    }
    

def setup_problem(i, mms, pS_deg, pD_deg, parameters):
    '''Solver for the Darcy-emmersed-Stokes test case'''
    meshD, boundariesD = mms['get_geometry'](i, 'inner')
    meshS, boundariesS = mms['get_geometry'](i, 'outer')
    interface, subdomainsI = mms['get_geometry'](i, 'interface')

    dsD = Measure('ds', domain=meshD, subdomain_data=boundariesD)
    nD = FacetNormal(meshD)

    dsS = Measure('ds', domain=meshS, subdomain_data=boundariesS)
    nS = FacetNormal(meshS)
    tS = rotate(nS)

    dx_ = Measure('dx', domain=interface, subdomain_data=subdomainsI)
    nD_ = OuterNormal(interface, [0.5, 0.5])
    nS_ = -nD_   # We have nS as master
    tS_ = rotate(nS_)
    
    # And now for the fun stuff
    VS = VectorFunctionSpace(meshS, 'CG', 2)
    QS = {1: FunctionSpace(meshS, 'CG', 1),
          0: FunctionSpace(meshS, 'DG', 0)}[pS_deg]
    QD = FunctionSpace(meshD, 'CG', pD_deg)
    W = [VS, QS, QD]

    uS, pS, pD = map(TrialFunction, W)
    vS, qS, qD = map(TestFunction, W)
    
    TuS, TvS = (Trace(x, interface) for x in (uS, vS))
    TpD, TqD = (Trace(x, interface) for x in (pD, qD))

    # Material parameters
    mu, K, alpha = (Constant(parameters[key]) for key in ('mu', 'K', 'alpha'))
    
    a = block_form(W, 2)
    a[0][0] = (Constant(2*mu)*inner(sym(grad(uS)), sym(grad(vS)))*dx +
               alpha*inner(dot(TuS, tS_), dot(TvS, tS_))*dx_)
    a[0][1] = -inner(pS, div(vS))*dx
    a[0][2] = inner(TpD, dot(TvS, nS_))*dx_

    a[1][0] = -inner(qS, div(uS))*dx
    a[2][0] = inner(TqD, dot(TuS, nS_))*dx_
    a[2][2] = -inner(K*grad(pD), grad(qD))*dx

    # We will have 7, 8 as Neumann boundaries for Stokes  and 5, 6 for Dirichlet
    lm_tags = (1, 2, 3, 4)

    L = block_form(W, 1)
    L[0] = (inner(mms['fS'], vS)*dx
            # Contribution from Neumann bcs on the boundary
            + sum(inner(mms['traction_S'][tag], vS)*dsS(tag) for tag in (7, 8))
            # Multiplier contrib from sigma.n.n
            - sum(inner(mms['g_n'][tag], dot(vS, nS))*dsS(tag) for tag in lm_tags)
            # and sigma.n.t
            - sum(inner(mms['g_t'][tag], dot(vS, tS))*dsS(tag) for tag in lm_tags))
                  
    L[2] = (-inner(mms['fD'], qD)*dx
            # csrv of mass contributions
            + sum(inner(mms['g_u'][tag], qD)*dsD(tag) for tag in lm_tags))

    VS_bcs = [DirichletBC(VS, mms['velocity_S'][tag], boundariesS, tag) for tag in (5, 6)]
    W_bcs = [VS_bcs, [], []]
    
    return a, L, W, W_bcs

# --------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm, L2_norm, Hdiv_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide material parameters
    parser.add_argument('--param_mu', type=float, default=1, help='Stokes viscosity')
    parser.add_argument('--param_K', type=float, default=1, help='Darcy conductivity')
    parser.add_argument('--param_alpha', type=float, default=1, help='BJS')
    # and discretization
    parser.add_argument('--pD_degree', type=int, default=2, choices=[1, 2])
    parser.add_argument('--pS_degree', type=int, default=1, choices=[0, 1])    
    args, _ = parser.parse_known_args()

    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    material_params = {k.split('_')[-1] : v for k, v in vars(args).items() if k.startswith('param_')}
    mms = setup_mms(material_params)
    uS_true, pS_true, pD_true = (mms['solution'][k] for k in ('uS', 'pS', 'pD'))

    clog = ConvergenceLog({'uS': (uS_true, H1_norm, '1'),
                           'pS': (pS_true, L2_norm, '0'),
                           'pD': (pD_true, H1_norm, '1')})

    print(clog.header())
    for i in range(6):
        a, L, W, bcs = setup_problem(i, mms, pS_deg=args.pS_degree, pD_deg=args.pD_degree,
                                     parameters=material_params)
        # Use direct solver to get the solution
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, bcs)
        A, b = map(ii_convert, (A, b))

        wh = ii_Function(W)
        LUSolver(A, 'mumps').solve(wh.vector(), b)

        uSh, pSh, pDh = wh
        
        clog.add((uSh, pSh, pDh))
        print(clog.report_last(with_name=False))
        
    ruS, rpS, rpD = (clog[var].get_rate()[0] for var in ('uS', 'pS', 'pD'))
    # NOTE: convergence of the variable is not independent so
    if args.pS_degree == 1 and args.pD_degree == 2:
        # Should be quadratic
        passed = ruS > 1.9 and rpS > 0.9*args.pS_degree and rpD > 0.9*args.pD_degree
    else:
        # The others might drag the velocity down so we settle for linear there
        passed = ruS > 0.9 and rpS > 0.9*args.pS_degree and rpD > 0.9*args.pD_degree
        
    sys.exit(int(passed))
