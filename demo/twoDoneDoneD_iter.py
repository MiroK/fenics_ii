# |---|---| Omega is (0, 1)^2, the vertical line is Gamma
# |   |   |
# |   |   | 
# |---|---|
#
# With some handwaving we consider the following coupled problem 
#
#     -div(kappa*grad u) + p*delta_Gamma = f  in Omega
#     -div(kappa1*grad u1) - p           = f1 on Gamma
#     u - u1 = g                              on Gamma
#
# closed by Dirichlet boundary conditions on u, u1 on their respective
# boundaries
from block.algebraic.petsc import LU
from hsmg.hseig import Hs0Eig
import sympy as sp
from dolfin import *
from xii import *
import ulfy


def setup_preconditioner(W, i, mms, parameters, hs_block):
    '''Preconditioner is H1 x H1 x H-0.5 cap H-1'''
    _, facet_f_ = mms['get_geometry'](i)
    [V, V1, Q] = W
    # We have different meshIds
    facet_f = MeshFunction('size_t', V.mesh(), 1, 0)
    facet_f.array()[:] = facet_f_.array()
    
    u, u1, p = map(TrialFunction, W)
    v, v1, q = map(TestFunction, W)
    
    # Material parameters
    kappa, kappa1 = (Constant(parameters[key]) for key in ('kappa', 'kappa1'))

    aV = kappa*inner(grad(u), grad(v))*dx
    V_bcs = [DirichletBC(V, Constant(0), facet_f, tag)
             for tag, value in mms['dirichlet_omega'].items()]
    BV, _ = assemble_system(aV, inner(Constant(0), v)*dx, V_bcs)

    aV1 = kappa1*inner(grad(u1), grad(v1))*dx
    V1_bcs = [DirichletBC(V1, mms['dirichlet_gamma'], 'on_boundary')]
    BV1, _ = assemble_system(aV1, inner(Constant(0), v1)*dx, V1_bcs)

    bmesh = Q.mesh()
    facet_f = MeshFunction('size_t', bmesh, bmesh.topology().dim()-1, 0)
    DomainBoundary().mark(facet_f, 1)

    if hs_block == 'eig':
        Hs0 = Hs0Eig(Q, s=-0.5, kappa=1/kappa, bcs=[(facet_f, 1)]).collapse()
        Hs1 = Hs0Eig(Q, s=-1.0, kappa=1/kappa1, bcs=[(facet_f, 1)]).collapse()
        BQ = ii_convert(Hs0 + Hs1)

    return block_diag_mat([LU(BV), LU(BV1), LU(BQ)])

# --------------------------------------------------------------------

if __name__ == '__main__':
    from twoDoneDoneD import setup_mms, setup_problem
    from block.iterative import MinRes    
    from common import ConvergenceLog, H1_norm, L2_norm, broken_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide material parameters ...
    parser.add_argument('--param_kappa', type=float, default=1, help='2d diffusion')
    parser.add_argument('--param_kappa1', type=float, default=1, help='1d diffusion')
    parser.add_argument('--hs', type=str, default='eig',
                        choices=['eig', 'mg'], help='Realization of fractional preconditioner')    
    args, _ = parser.parse_known_args()

    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    material_params = {k.split('_')[-1] : v for k, v in vars(args).items() if k.startswith('param_')}
    mms = setup_mms(material_params)
    u_true, u1_true, lm_true = (mms['solution'][k] for k in ('u', 'u1', 'lm'))

    clog = ConvergenceLog({'u': (u_true.expressions, broken_norm(u_true.subdomains, H1_norm), '1'),
                           'u1': (u1_true, H1_norm, '1'),
                           'lm': (lm_true, L2_norm, '0'),
                           'niters': None})

    print(clog.header())
    for i in range(6):
        a, L, W, bcs = setup_problem(i, mms, parameters=material_params)
        # Use direct solver to get the solution
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, bcs)

        B = setup_preconditioner(W, i, mms, parameters=material_params, hs_block=args.hs)

        wh = ii_Function(W)
        Ainv = MinRes(A, precond=B, tolerance=1E-10, show=0)
        # NOTE: assigning from block_vec
        wh.vector()[:] = ii_convert(Ainv*b)

        uh, u1h, lmh = wh
        niters = len(Ainv.residuals)
        
        clog.add((uh, u1h, lmh, niters))
        print(clog.report_last(with_name=False))

    urates = tuple(clog[var].get_rate()[0] for var in ('u', 'u1'))
    prate, _ = clog['lm'].get_rate()
    
    expected = (1, 1)
    passed = all(abs(r-e) < 0.1 for r, e in zip(urates, expected)) and prate > 0.4

    passed = passed and all(count < 32 for count in clog['niters'])        
    
    sys.exit(int(passed))
