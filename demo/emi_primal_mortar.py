# Same system as in `emi_primal.py` Lagrange multiplier is instroduced
# for the Robin type coupling condition
import sympy as sp
from dolfin import *
from xii import *
import ulfy


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
    Q = FunctionSpace(interface, 'CG', degree)
    W = [V0, V1, Q]

    u0, u1, p = map(TrialFunction, W)
    v0, v1, q = map(TestFunction, W)
    
    Tu0, Tu1 = (Trace(x, interface) for x in (u0, u1))
    Tv0, Tv1 = (Trace(x, interface) for x in (v0, v1))

    # Material parameters
    kappa0, kappa1 = (Constant(parameters[key]) for key in ('kappa0', 'kappa1'))

    a = block_form(W, 2)
    a[0][0] = kappa0*inner(grad(u0), grad(v0))*dx
    a[0][2] = inner(p, Tv0)*dx_
    a[1][1] = kappa1*inner(grad(u1), grad(v1))*dx
    a[1][2] = -inner(p, Tv1)*dx_
    a[2][0] = inner(q, Tu0)*dx_
    a[2][1] = -inner(q, Tu1)*dx_
    a[2][2] = -inner(p, q)*dx
              
    lm_tags = (1, 2, 3, 4)

    L = block_form(W, 1)
    L[0] = inner(mms['f0'], v0)*dx

    L[1] = (inner(mms['f1'], v1)*dx
            # Multiplier contribution from 
            + sum(inner(mms['g_n'][tag], v1)*ds1(tag) for tag in lm_tags)            
    )

    #  Multiplier contrib from Robin
    L[2] = sum(inner(mms['g_r'][tag], q)*dx_(tag) for tag in lm_tags)

    # Dirichlet on all outside
    V0_bcs = [DirichletBC(V0, mms['dirichlet_0'][tag], boundaries0, tag) for tag in (5, 6, 7, 8)]
    W_bcs = [V0_bcs, [], []]
    
    return a, L, W, W_bcs

# --------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm, L2_norm, broken_norm
    from emi_primal import setup_mms
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
    u0_true, u1_true, lm_true = (mms['solution'][k] for k in ('u0', 'u1', 'lm'))

    clog = ConvergenceLog({'u0': (u0_true, H1_norm, '1'),
                           'u1': (u1_true, H1_norm, '1'),
                           'lm': (lm_true.expressions, broken_norm(lm_true.subdomains, L2_norm), '0')})

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

        u0h, u1h, lmh = wh
        
        clog.add((u0h, u1h, lmh))
        print(clog.report_last(with_name=False))
        
    urates = tuple(clog[var].get_rate()[0] for var in ('u0', 'u1'))
    prate, _ = clog['lm'].get_rate()
    
    expected = (args.degree, )*2
    passed = all(abs(r-e) < 0.1 for r, e in zip(urates, expected)) and prate > 0.4
        
    sys.exit(int(passed))
