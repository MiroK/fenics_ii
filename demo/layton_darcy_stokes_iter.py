# Same problem setup as in `dq_darcy_stokes.py` except mixed
# formulation is used to solve the Darcy subproblem and thus
# we have a Lagrange multiplier on the interface to enforce the
# coupling (mass conservation in particular)
from block.algebraic.petsc import LU
from hsmg.hseig import HsEig
from utils import rotate
import sympy as sp
from dolfin import *
from xii import *
import ulfy


def setup_preconditioner(W, i, mms, parameters, stokes_CR, hs_block):
    '''H^1 x H(div) x L2 x L2 x H^{-0.5}\cah H^{0.5}'''
    _, boundariesS_ = mms['get_geometry'](i, 'outer')
    
    [VS, VD, QS, QD, M] = W

    uS, uD, pS, pD, l = map(TrialFunction, W)
    vS, vD, qS, qD, m = map(TestFunction, W)

    interface = M.mesh()
    TuS, TvS = (Trace(x, interface) for x in (uS, vS))

    nD_ = OuterNormal(interface, [0.5, 0.5])
    nS_ = -nD_   # We have nS as master
    tS_ = rotate(nS_)
    dx_ = Measure('dx', domain=interface)

    # Material parameters
    mu, K, alpha = (Constant(parameters[key]) for key in ('mu', 'K', 'alpha'))

    # Velocity
    meshS = VS.mesh()
    
    aVS = (Constant(2*mu)*inner(sym(grad(uS)), sym(grad(vS)))*dx +
           alpha*inner(dot(TuS, tS_), dot(TvS, tS_))*dx_)
    # Stabilization for CR
    if stokes_CR:
        nS = FacetNormal(meshS)
        hA = FacetArea(meshS)        
        aVS += 2*mu/avg(hA)*inner(jump(uS, nS), jump(vS, nS))*dS

    boundariesS = MeshFunction('size_t', meshS, 1, 0)
    boundariesS.array()[:] = boundariesS_.array()
        
    VS_bcs = [DirichletBC(VS, mms['velocity_S'][tag], boundariesS, tag) for tag in (5, 6)]
    BVS = ii_assemble(aVS)
    BVS, _ = apply_bc(BVS, b=None, bcs=VS_bcs)
    BVS = ii_convert(BVS)

    # Darcy-flux
    aVD = (1/K)*inner(uD, vD)*dx + (1/K)*inner(div(uD), div(vD))*dx
    BVD = assemble(aVD)

    # Stokes pressure
    aQS = (1/mu)*inner(pS, qS)*dx
    BQS = assemble(aQS)

    # Darcy pressure
    aQD = K*inner(pD, qD)*dx
    BQD = assemble(aQD)

    # Multiplier
    if hs_block == 'eig':
        Hs0 = HsEig(M, s=-0.5, kappa=1/mu).collapse()
        Hs1 = HsEig(M, s=0.5, kappa=K).collapse()
        BM = ii_convert(Hs0 + Hs1)

    return block_diag_mat([LU(x) for x in (BVS, BVD, BQS, BQD, BM)])

# --------------------------------------------------------------------

if __name__ == '__main__':
    from layton_darcy_stokes import setup_problem
    from dq_darcy_stokes import setup_mms    
    from block.iterative import MinRes
    from common import ConvergenceLog, H1_norm, L2_norm, Hdiv_norm, broken_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide material parameters ...
    parser.add_argument('--param_mu', type=float, default=1E-2, help='Stokes viscosity')
    parser.add_argument('--param_K', type=float, default=1E-5, help='Darcy conductivity')
    parser.add_argument('--param_alpha', type=float, default=1, help='BJS')
    # ... and whether to use CR-P0 discretization for Stokes
    parser.add_argument('--stokes_CR', type=int, default=0, choices=[0, 1])

    parser.add_argument('--hs', type=str, default='eig',
                        choices=['eig', 'mg'], help='Realization of fractional preconditioner')    
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
                           'lm': (lm_true.expressions, broken_norm(lm_true.subdomains, L2_norm), '0'),
                           'niters': None
    })

    print(clog.header())
    for i in range(6):
        a, L, W, bcs = setup_problem(i, mms, parameters=material_params,
                                     stokes_CR=args.stokes_CR)
        # Use direct solver to get the solution
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, bcs)

        B = setup_preconditioner(W, i, mms, parameters=material_params,
                                 stokes_CR=args.stokes_CR, hs_block=args.hs)
        
        wh = ii_Function(W)
        Ainv = MinRes(A, precond=B, tolerance=1E-10, show=0)
        # NOTE: assigning from block_vec
        wh.vector()[:] = ii_convert(Ainv*b)

        uSh, uDh, pSh, pDh, lmh = wh
        niters = len(Ainv.residuals)
        
        clog.add((uSh, uDh, pSh, pDh, lmh, niters))
        print(clog.report_last(with_name=False))
        
    rates = tuple(clog[var].get_rate()[0] for var in ('uS', 'uD', 'pS', 'pD', 'lm'))

    if args.stokes_CR:
        expected = (1, )*5
    else:
        expected = (2, 1, 2, 1, 1)
    passed = all(r > e or abs(r-e) < 0.3 for r, e in zip(rates, expected))

    passed = passed and all(count < 75 for count in clog['niters'])        
    
    sys.exit(int(passed))
