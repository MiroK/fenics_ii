#  -Delta u = f in (0, 1)^2
#         u = g_u enforced strongly on left edge and
# grad(u).n = g_s enforce on bottom edge
#         u = g enforced by LM on top and right edges
#
# New thing here will be the bcs and also broken norm computations of
# the multiplier error. NOTE: this is `poisson_babuska_bc.py --bcs mixed` option.
from block.algebraic.petsc import LU
from hsmg.hseig import HsEig
from dolfin import *
from xii import *
import ulfy 


def setup_preconditioner(W, facet_f, mms, bc_setup, hs_block):
    '''Preconditioner will be a Riesz map wrt H1 x H^{-1/2} inner product'''
    # See the mms below for how the boundaries are marked
    # NOTE: which bcs meet LM boundary effects boundary conditions
    # on the multiplier space
    if bc_setup == 'dir_neu':
        lm_tags, dir_tags, neu_tags = (2, 4), (1, ), (3, )
        
    elif bc_setup == 'dir':
        lm_tags, dir_tags, neu_tags = (2, 4), (1, 3), ()

    else:
        lm_tags, dir_tags, neu_tags = (2, 4), (), (1, 3)

    [V, Q] = W
    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    aV = inner(grad(u), grad(v))*dx
    if not dir_tags:
        # For Neumann case we need an invertible block for V
        aV += inner(u, v)*dx
        V_bcs = None
    else:
        # Boundary conditions; a list for each subspace
        V_bcs = [DirichletBC(V, mms['dirichlet'][tag], facet_f, tag) for tag in dir_tags]
    # H1 part
    BV, _ = assemble_system(aV, inner(Constant(0), v)*dx, V_bcs)

    # For LM we want Dirichlet where we meet the Dirichlet Boundary
    bmesh = Q.mesh()
    facet_f = MeshFunction('size_t', bmesh, bmesh.topology().dim()-1, 0)
    if bc_setup == 'dir_neu':
        CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
        Q_bcs = [(facet_f, 1)]
    elif bc_setup == 'dir':  # Both endpoints
        DomainBoundary().mark(facet_f, 1)
        Q_bcs = [(facet_f, 1)]
    else:
        Q_bcs = None

    if hs_block == 'eig':
        BQ = HsEig(Q, s=-0.5, bcs=Q_bcs)
    else:
        raise NotImplementedError

    return block_diag_mat([LU(BV), BQ**-1])

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm, L2_norm, broken_norm
    from poisson_babuska_bc import setup_mms, setup_problem
    from block.iterative import MinRes    
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide on bcs setup
    parser.add_argument('--bcs', type=str, default='dir_neu',
                        choices=['dir', 'neu', 'dir_neu'])
    parser.add_argument('--hs', type=str, default='eig',
                        choices=['eig', 'mg'], help='Realization of fractional preconditioner')
    args, _ = parser.parse_known_args()
    
    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    mms = setup_mms()
    u_true, p_true = mms['solution']
    # ... and will monitor the error in H1 norm and L2 norm for the bulk
    # variable and the multiplier respectively
    clog = ConvergenceLog({
        'u': (u_true, H1_norm, '1'),
        'p': (p_true.expressions, broken_norm(p_true.subdomains, L2_norm), '0'),
        'niters': None
    })

    print(clog.header())
    for i in range(6):
        n = 4*2**i
        mesh = UnitSquareMesh(n, n)
        facet_f = MeshFunction('size_t', mesh, 1, 0)
        [subdomain.mark(facet_f, tag) for tag, subdomain in mms['subdomains'].items()]
        
        a, L, W, bcs = setup_problem(facet_f, mms, bc_setup=args.bcs)
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, bcs)

        B = setup_preconditioner(W, facet_f, mms, bc_setup=args.bcs, hs_block=args.hs)
        
        wh = ii_Function(W)
        Ainv = MinRes(A, precond=B, tolerance=1E-10, show=0)
        # NOTE: assigning from block_vec
        wh.vector()[:] = ii_convert(Ainv*b)
        
        # Components of the wh in V x Q are dolfin Fudict(enumerate(map(to_expr, g_u), 1))ncitons
        uh, ph = wh
        niters = len(Ainv.residuals)
        
        clog.add((uh, ph, niters))
        print(clog.report_last(with_name=False))

    # In this case we use P1 elements and order 1 is expected for bulk
    # Due to bcs on LM convergence of LM is suboptimal for dirichlet case
    iru, fru = clog['u'].get_rate()
    irp, frp = clog['p'].get_rate()

    if args.bcs == 'neu':
        passed = all((iru > 0.95, fru > 0.95, irp > 1.95, frp > 1.95))
    else:
        passed = all((iru > 0.95, fru > 0.95, irp > 0.45, frp > 0.45))

    passed = passed and all(count < 40 for count in clog['niters'])        
        
    sys.exit(int(passed))
