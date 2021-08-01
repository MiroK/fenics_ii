#  -Delta u = f in (0, 1)^2
#         u = g_u enforced strongly on left edge and
# grad(u).n = g_s enforce on bottom edge
#         u = g enforced by LM on top and right edges
#
# New thing here will be the bcs and also broken norm computations of
# the multiplier error. NOTE: this is `poisson_babuska_bc.py --bcs mixed` option.
from utils import PieceWiseExpression
from dolfin import *
from xii import *
import ulfy 


def setup_problem(facet_f, mms, bc_setup):
    '''Babuska on [0, 1]^2'''
    mesh = facet_f.mesh()
    # See the mms below for how the boundaries are marked
    # NOTE: which bcs meet LM boundary effects boundary conditions
    # on the multiplier space
    if bc_setup == 'dir_neu':
        lm_tags, dir_tags, neu_tags = (2, 4), (1, ), (3, )
        
    elif bc_setup == 'dir':
        lm_tags, dir_tags, neu_tags = (2, 4), (1, 3), ()

    else:
        lm_tags, dir_tags, neu_tags = (2, 4), (), (1, 3)

    bmesh = EmbeddedMesh(facet_f, lm_tags)
    bmesh_subd = bmesh.marking_function

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
    Tu, Tv = Trace(u, bmesh), Trace(v, bmesh)

    # The line integral; we want subdomains because the data are
    # for piece of boundary
    dx_ = Measure('dx', domain=bmesh, subdomain_data=bmesh_subd)
    ds = Measure('ds', domain=mesh, subdomain_data=facet_f)
    # We now define the system as
    a = block_form(W, 2)
    a[0][0] = inner(grad(u), grad(v))*dx 
    a[0][1] = inner(Tv, p)*dx_
    a[1][0] = inner(Tu, q)*dx_

    # And the rhs
    L = block_form(W, 1)
    L[0] = inner(mms['force'], v)*dx
    # Neumann contributions
    L[0] += sum(inner(mms['neumann'][tag], v)*ds(tag) for tag in neu_tags)

    # On multiplier we are given Dirichlet data
    L[1] = sum(inner(mms['dirichlet'][tag], q)*dx_(tag) for tag in lm_tags)

    # Boundary conditions; a list for each subspace
    V_bcs = [DirichletBC(V, mms['dirichlet'][tag], facet_f, tag) for tag in dir_tags]
    
    # For LM we want Dirichlet where we meet the Dirichlet Boundary
    # NOTE: the value we prescribe here is not the true solution (and
    # this is one of the reasons why convergence rate will be worse in
    # the L^2 norm). It's not the right norm for the LM anyways
    if bc_setup == 'dir_neu':
        Q_bcs = [DirichletBC(Q, Constant(0), 'near(x[0], 0)')]
    elif bc_setup == 'dir':  # Both endpoints
        Q_bcs = [DirichletBC(Q, Constant(0), 'on_boundary')]
    else:
        Q_bcs = []
    # For W the position indicates subspace
    W_bcs = [V_bcs, Q_bcs]

    return a, L, W, W_bcs


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
    # LM is defined as -grad(u).n so we need for each boundary piece
    # We assume numb
    normals = (Constant((-1, 0)), Constant((1, 0)), Constant((0, -1)), Constant((0, 1)))
    lms = [-dot(grad(u), n) for n in normals]

    f = -div(grad(u))
    g_u = [u]*4  # Dirichlet data (for each piece)
    g_sigma = [dot(grad(u), n) for n in normals]  # Neumann data for each piece

    to_expr = lambda f: ulfy.Expression(f, degree=4)
    
    up = (to_expr(u), PiecewiseExpression(subdomains, dict(enumerate(map(to_expr, lms), 1))))

    return {'solution': up,
            'force': to_expr(f),
            'dirichlet': dict(enumerate(map(to_expr, g_u), 1)),
            'neumann': dict(enumerate(map(to_expr, g_sigma), 1)),
            'subdomains': subdomains}

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm, L2_norm, broken_norm
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide on bcs setup
    parser.add_argument('--bcs', type=str, default='dir_neu',
                        choices=['dir', 'neu', 'dir_neu'])
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
        'p': (p_true.expressions, broken_norm(p_true.subdomains, L2_norm), '0')
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
        # Here A, b are cbc.block object so we need to convert them to matrices
        A, b = map(ii_convert, (A, b))
        # Now solve with direct solver
        wh = ii_Function(W)
        solve(A, wh.vector(), b)
        # Components of the wh in V x Q are dolfin Fudict(enumerate(map(to_expr, g_u), 1))ncitons
        uh, ph = wh
        
        clog.add((uh, ph))
        print(clog.report_last(with_name=False))

    # In this case we use P1 elements and order 1 is expected for bulk
    # Due to bcs on LM convergence of LM is suboptimal for dirichlet case
    iru, fru = clog['u'].get_rate()
    irp, frp = clog['p'].get_rate()

    if args.bcs == 'neu':
        passed = all((iru > 0.95, fru > 0.95, irp > 1.95, frp > 1.95))
    else:
        passed = all((iru > 0.95, fru > 0.95, irp > 0.45, frp > 0.45))
    sys.exit(int(passed))
