# Same problem setup as in `dq_darcy_stokes.py` except mixed
# formulation is used to solve the Darcy subproblem and thus
# we have a Lagrange multiplier on the interface to enforce the
# coupling (mass conservation in particular)
from utils import rotate
import sympy as sp
from dolfin import *
from xii import *
import ulfy


def setup_problem(i, mms, parameters, stokes_CR):
    '''Solution of the Darcy-emmersed-Stokes test case'''
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
    if not stokes_CR:
        VS = VectorFunctionSpace(meshS, 'CG', 2)
        QS = FunctionSpace(meshS, 'CG', 1)
    else:
        VS = VectorFunctionSpace(meshS, 'CR', 1)
        QS = FunctionSpace(meshS, 'DG', 0)        
    
    VD = FunctionSpace(meshD, 'RT', 1)
    QD = FunctionSpace(meshD, 'DG', 0)
    M = FunctionSpace(interface, 'DG', 0)
    W = [VS, VD, QS, QD, M]

    uS, uD, pS, pD, l = map(TrialFunction, W)
    vS, vD, qS, qD, m = map(TestFunction, W)
    
    TuS, TvS = (Trace(x, interface) for x in (uS, vS))
    TuD, TvD = (Trace(x, interface) for x in (uD, vD))

    # Material parameters
    mu, K, alpha = (Constant(parameters[key]) for key in ('mu', 'K', 'alpha'))

    a = block_form(W, 2)
    a[0][0] = (Constant(2*mu)*inner(sym(grad(uS)), sym(grad(vS)))*dx +
               alpha*inner(dot(TuS, tS_), dot(TvS, tS_))*dx_)
    # Stabilization for CR
    if stokes_CR:
        hA = FacetArea(meshS)        
        a[0][0] += 2*mu/avg(hA)*inner(jump(uS, nS), jump(vS, nS))*dS

    a[0][2] = -inner(pS, div(vS))*dx
    a[0][4] = inner(l, dot(TvS, nS_))*dx_

    a[1][1] = (1/K)*inner(uD, vD)*dx
    a[1][3] = -inner(pD, div(vD))*dx
    a[1][4] = inner(l, dot(TvD, nD_))*dx_

    a[2][0] = -inner(qS, div(uS))*dx
    a[3][1] = -inner(qD, div(uD))*dx

    a[4][0] = inner(m, dot(TuS, nS_))*dx_
    a[4][1] = inner(m, dot(TuD, nD_))*dx_

    # We will have 7, 8 as Neumann boundaries for Stokes  and 5, 6 for Dirichlet
    lm_tags = (1, 2, 3, 4)

    L = block_form(W, 1)
    L[0] = (inner(mms['fS'], vS)*dx
            # Contribution from Neumann bcs on the boundary
            + sum(inner(mms['traction_S'][tag], vS)*dsS(tag) for tag in (7, 8))
            #  Multiplier contrib from sigma.n.t
            - sum(inner(mms['g_t'][tag], dot(vS, tS))*dsS(tag) for tag in lm_tags))
    # Multiplier contrib from sigma.n.n
    L[1] = sum(inner(mms['g_n'][tag], dot(vD, nD))*dsD(tag) for tag in lm_tags)
    
    L[3] = -inner(mms['fD'], qD)*dx
    # Interface mass conservation
    L[4] = sum(inner(mms['g_u'][tag], m)*dx_(tag) for tag in lm_tags)

    VS_bcs = [DirichletBC(VS, mms['velocity_S'][tag], boundariesS, tag) for tag in (5, 6)]
    W_bcs = [VS_bcs, [], [], [], []]
    
    return a, L, W, W_bcs

# --------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm, L2_norm, Hdiv_norm, broken_norm
    from dq_darcy_stokes import setup_mms
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide material parameters ...
    parser.add_argument('--param_mu', type=float, default=1, help='Stokes viscosity')
    parser.add_argument('--param_K', type=float, default=1, help='Darcy conductivity')
    parser.add_argument('--param_alpha', type=float, default=1, help='BJS')
    # ... and whether to use CR-P0 discretization for Stokes
    parser.add_argument('--stokes_CR', type=int, default=0, choices=[0, 1])
    args, _ = parser.parse_known_args()

    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    material_params = {k.split('_')[-1] : v for k, v in vars(args).items() if k.startswith('param_')}
    mms = setup_mms(material_params)
    uS_true, uD_true, pS_true, pD_true = (mms['solution'][k] for k in ('uS', 'uD', 'pS', 'pD'))
    lm_true = mms['solution']['lm']

    clog = ConvergenceLog({'uS': (uS_true, H1_norm, '1'),
                           'uD': (uD_true, Hdiv_norm, 'div'),
                           'pS': (pS_true, L2_norm, '0'),
                           'pD': (pD_true, L2_norm, '0'),
                           # Multiplier is defined piecewise
                           'lm': (lm_true.expressions, broken_norm(lm_true.subdomains, L2_norm), '0')
    })

    print(clog.header())
    for i in range(6):
        a, L, W, bcs = setup_problem(i, mms, parameters=material_params,
                                     stokes_CR=args.stokes_CR)
        # Use direct solver to get the solution
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, bcs)
        A, b = map(ii_convert, (A, b))

        wh = ii_Function(W)
        LUSolver(A, 'mumps').solve(wh.vector(), b)

        uSh, uDh, pSh, pDh, lmh = wh
        
        clog.add((uSh, uDh, pSh, pDh, lmh))
        print(clog.report_last(with_name=False))
        
    rates = tuple(clog[var].get_rate()[0] for var in ('uS', 'uD', 'pS', 'pD', 'lm'))

    if args.stokes_CR:
        expected = (1, )*5
    else:
        expected = (2, 1, 2, 1, 1)
    passed = all(abs(r-e) < 0.1 for r, e in zip(rates, expected))
        
    sys.exit(int(passed))
