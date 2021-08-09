# Mixed Poisson problem with flux bcs enforced weakly
# 
#       div(sigma) = f (0, 1)^2
#  sigma + grad(u) = 0
#          sigma.n = g_sigma enforced strongly on bottom edge
#          sigma.n = g_sigma enforced by LM on top and right edge
#                u = g_u     enforced weakly
#
# Solve with RT_0-P_0-P_0 or RT_1-P_1-P_1 to showcase higher order
from block.algebraic.petsc import LU
from hsmg.hseig import HsEig
from dolfin import *
from xii import *
import ulfy 


def setup_preconditioner(facet_f, mms, flux_deg, hs_block):
    '''Preconditioner will be Riesz map H(div) x L^2 x H^{1/2}'''
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

    # H(div)
    aS = inner(sigma, tau)*dx + inner(div(sigma), div(tau))*dx
    S_bcs = [DirichletBC(S, mms['flux'][tag], facet_f, tag) for tag in flux_tags]
    BS, _ = assemble_system(aS, inner(Constant((0, 0)), tau)*dx, S_bcs)
    # L^2
    aV = inner(u, v)*dx
    BV = assemble(aV)
    # H^s norm
    facet_f = MeshFunction('size_t', bmesh, bmesh.topology().dim()-1, 0)    
    CompiledSubDomain('near(x[0], 1)').mark(facet_f, 1)
    Q_bcs = [(facet_f, 1)]

    if hs_block == 'eig':
        BQ = HsEig(Q, s=0.5, bcs=Q_bcs)
    else:
        raise NotImplementedError

    return block_diag_mat([LU(BS), LU(BV), BQ**-1])

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from mixed_poisson_babuska import setup_mms, setup_problem
    from block.iterative import MinRes
    from common import ConvergenceLog, Hdiv_norm, L2_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # RT degree
    parser.add_argument('--flux_degree', type=int, default=1, choices=[1])
    # FIXME: degree 2 neets to fix eig penalty
    parser.add_argument('--hs', type=str, default='eig',
                        choices=['eig', 'mg'], help='Realization of fractional preconditioner')    
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
        'p': (p_true, L2_norm, '0'),
        'niters': None,
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

        B = setup_preconditioner(facet_f, mms, flux_deg=args.flux_degree, hs_block=args.hs)
        
        wh = ii_Function(W)
        Ainv = MinRes(A, precond=B, tolerance=1E-10, show=0)
        # NOTE: assigning from block_vec
        wh.vector()[:] = ii_convert(Ainv*b)

        sigmah, uh, ph = wh
        niters = len(Ainv.residuals)

        clog.add((sigmah, uh, ph, niters))
        print(clog.report_last(with_name=False))

    eoc = 0.95 if args.flux_degree == 1 else 1.95
    # Match increamental and lstsq rate
    passed = all(clog[var].get_rate()[0] > eoc for var in ('sigma', 'u', 'p'))
    passed = passed and all(clog[var].get_rate()[1] > eoc for var in ('sigma', 'u', 'p'))
    passed = passed and all(count < 40 for count in clog['niters'])
    
    sys.exit(int(passed))
