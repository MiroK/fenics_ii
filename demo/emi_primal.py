# This system arises when solving EMI models by operator splitting
# and using some implicit scheme for the PDE part.
# 
# With Omega1 = (0.25, 0.75)^2 and Omega0 = (0, 1)^2 \ Omega1 we consider
#
# -div(kappa0*grad(u0)) = f0 in Omega0
# -div(kappa1*grad(u1)) = f1 in Omega1
#
# kappa0*grad u0.n0 + kappa1*grad u1.n1 = g_n on Interface
# u0 - u1 = -kappa0*grad u0.n0 + g_r          on Interface
#
# On outer boundary of Omega0 we assume Dirichlet bcs. Note that
# in real EMI f0 = f1 = g_n = 0
from utils import immersed_geometry, PiecewiseExpression
import sympy as sp
from dolfin import *
from xii import *
import ulfy


def setup_mms(parameter_values):
    '''Manufacture solution for the immersed test case'''
    mesh = UnitSquareMesh(2, 2, 'crossed')

    x, y = SpatialCoordinate(mesh)
    kappa0, kappa1 = Constant(1), Constant(1)

    u0 = sin(pi*(x-2*y))
    u1 = cos(2*pi*(3*x-y))

    sigma0 = kappa0*grad(u0)
    sigma1 = kappa1*grad(u1)

    f0 = -div(sigma0)
    f1 = -div(sigma1)
    
    # Interface is oriented based on outer
    normalsI = tuple(map(Constant, ((1, 0), (-1, 0), (0, 1), (0, -1))))    
    # Neumann and Robin interface data
    g_n = tuple(dot(sigma0, n) - dot(sigma1, n) for n in normalsI)
    g_r = tuple(u0 - u1 + dot(sigma0, n) for n in normalsI)
    # True multiplier value
    lms = [-dot(sigma0, n) for n in normalsI]

    # Don't want to trigger compiler on parameter change
    kappa0_, kappa1_ = sp.symbols('kappa0 kappa1')
    subs = {kappa0: kappa0_, kappa1: kappa1_}

    # Check coefs
    assert parameter_values['kappa0'] > 0 and parameter_values['kappa1'] > 0

    to_expr = lambda f: ulfy.Expression(f, degree=4, subs=subs,
                                        kappa0=parameter_values['kappa0'],
                                        kappa1=parameter_values['kappa1'])

    # As tagged in utils.immersed_geometry
    lm_subdomains = {
        1: CompiledSubDomain('near(x[0], 0.25) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))'),
        2: CompiledSubDomain('near(x[0], 0.75) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))'),
        3: CompiledSubDomain('near(x[1], 0.25) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS))'),
        4: CompiledSubDomain('near(x[1], 0.75) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS))')
    }
    
    return {
        'solution': {'u0': to_expr(u0), 'u1': to_expr(u1),
                     'lm': PiecewiseExpression(lm_subdomains, dict(enumerate(map(to_expr, lms), 1)))},
        'f0': to_expr(f0),
        'f1': to_expr(f1),
        # Standard boundary data
        'dirichlet_0': dict(enumerate(map(to_expr, [u0]*8), 1)),
        # Interface boundary conditions
        'g_n': dict(enumerate(map(to_expr, g_n), 1)),
        'g_r': dict(enumerate(map(to_expr, g_r), 1)),
        # Geometry setup
        'get_geometry': immersed_geometry
    }


def setup_problem(i, mms, parameters, degree):
    '''Solution of the Darcy-emmersed-Stokes test case'''
    mesh0, boundaries0 = mms['get_geometry'](i, 'outer')
    mesh1, boundaries1 = mms['get_geometry'](i, 'inner')
    interface, subdomainsI = mms['get_geometry'](i, 'interface')

    ds0 = Measure('ds', domain=mesh0, subdomain_data=boundaries0)
    n0 = FacetNormal(mesh0)

    ds1 = Measure('ds', domain=mesh1, subdomain_data=boundaries1)
    n1 = FacetNormal(mesh1)

    dx_ = Measure('dx', domain=interface, subdomain_data=subdomainsI)
    n1_ = OuterNormal(interface, [0.5, 0.5])
    n0_ = -n1_   # We have 0 as master
    
    V0 = FunctionSpace(mesh0, 'CG', degree)
    V1 = FunctionSpace(mesh1, 'CG', degree)
    W = [V0, V1]

    u0, u1 = map(TrialFunction, W)
    v0, v1 = map(TestFunction, W)
    
    Tu0, Tu1 = (Trace(x, interface) for x in (u0, u1))
    Tv0, Tv1 = (Trace(x, interface) for x in (v0, v1))

    # Material parameters
    kappa0, kappa1 = (Constant(parameters[key]) for key in ('kappa0', 'kappa1'))

    a = block_form(W, 2)
    a[0][0] = kappa0*inner(grad(u0), grad(v0))*dx + inner(Tu0, Tv0)*dx_
    a[0][1] = -inner(Tu1, Tv0)*dx_
    a[1][0] = -inner(Tu0, Tv1)*dx_
    a[1][1] = kappa1*inner(grad(u1), grad(v1))*dx + inner(Tu1, Tv1)*dx_
              
    lm_tags = (1, 2, 3, 4)

    L = block_form(W, 1)
    L[0] = (inner(mms['f0'], v0)*dx
            #  Multiplier contrib from Robin
            + sum(inner(mms['g_r'][tag], v0)*ds0(tag) for tag in lm_tags))

    L[1] = (inner(mms['f1'], v1)*dx
            #  Multiplier contrib from Robin ...
            - sum(inner(mms['g_r'][tag], v1)*ds1(tag) for tag in lm_tags)
            # ... and Neumann
            + sum(inner(mms['g_n'][tag], v1)*ds1(tag) for tag in lm_tags)            
    )

    # Dirichlet on all outside
    V0_bcs = [DirichletBC(V0, mms['dirichlet_0'][tag], boundaries0, tag) for tag in (5, 6, 7, 8)]
    W_bcs = [V0_bcs, []]
    
    return a, L, W, W_bcs

# --------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide material parameters ...
    parser.add_argument('--param_kappa0', type=float, default=1, help='outer domain diffusion')
    parser.add_argument('--param_kappa1', type=float, default=1, help='inner domain diffusion')

    parser.add_argument('--degree', type=int, default=1, choices=[1, 2], help='Polynomial degree')    
    args, _ = parser.parse_known_args()

    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    material_params = {k.split('_')[-1] : v for k, v in vars(args).items() if k.startswith('param_')}
    mms = setup_mms(material_params)
    u0_true, u1_true = (mms['solution'][k] for k in ('u0', 'u1'))

    clog = ConvergenceLog({'u0': (u0_true, H1_norm, '1'),
                           'u1': (u1_true, H1_norm, '1')})

    print(clog.header())
    for i in range(6):
        a, L, W, bcs = setup_problem(i, mms, parameters=material_params,
                                     degree=args.degree)
        # Use direct solver to get the solution
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, bcs)
        A, b = map(ii_convert, (A, b))

        wh = ii_Function(W)
        LUSolver(A, 'mumps').solve(wh.vector(), b)

        u0h, u1h = wh
        
        clog.add((u0h, u1h))
        print(clog.report_last(with_name=False))
        
    rates = tuple(clog[var].get_rate()[0] for var in ('u0', 'u1'))

    expected = (args.degree, )*2
    passed = all(abs(r-e) < 0.1 for r, e in zip(rates, expected))
        
    sys.exit(int(passed))
