# Stokes problem where P1-P1 and P1-P0 elements are made to work
# by considering the pressure space on a coarse mesh. This way we
# show case Injection (coupling)
# 
#       -div(2*sym(grad(u)) - p*I) = f (0, 1)^2
#                           div(u) = 0
#                                u = g on boundary
#

from dolfin import *
from block.algebraic.petsc import AMG
from xii.meshing.refinement import centroid_refine
from xii.meshing.dual_mesh import DualMesh
from xii import *
import ulfy 


def setup_mms():
    '''Simple MMS problem for UnitSquareMesh'''
    mesh = UnitSquareMesh(2, 2)
    
    x, y  = SpatialCoordinate(mesh)
    
    phi = sin(pi*(x+y))
    u = as_vector((phi.dx(1), -phi.dx(0)))

    p = cos(pi*x)*cos(pi*y)
    sigma = 2*sym(grad(u)) - p*Identity(2)

    f = -div(sigma)
    g_u = u
    
    to_expr = lambda f: ulfy.Expression(f, degree=4)
    
    w = tuple(to_expr(x) for x in (u, p))

    return {'solution': w,
            'force': to_expr(f),
            'velocity': to_expr(g_u)}


def setup_problem(mesh_c, mms, pressure_deg, refinement):
    '''Babuska on [0, 1]^2'''
    if refinement == 'bisection':
        # Each tri to 4 by bisecting edges
        mesh = refine(mesh_c)
        
    elif refinement == 'centroid':
        # Each tri to 3 using center point of the mesh
        mesh = centroid_refine(mesh_c)

    elif refinement == 'dual':
        # Each tri yields 6 - dual mesh from FVM
        mesh = DualMesh(mesh_c)

    else:
        raise ValueError

    V = VectorFunctionSpace(mesh, 'CG', 1)
    
    if pressure_deg == 0:
        Qc = FunctionSpace(mesh_c, 'DG', 0)
    else:
        Qc = FunctionSpace(mesh_c, 'CG', 1)

    # Pressure nullspace due to boundary conditions
    R = FunctionSpace(mesh_c, 'R', 0)

    W = [V, Qc, R]

    u, pc, phi = map(TrialFunction, W)
    v, qc, psi = map(TestFunction, W)

    p, q = Injection(pc, mesh), Injection(qc, mesh)
    dxc = Measure('dx', domain=mesh_c)
    dx = Measure('dx', domain=mesh)  # On the fine mesh
    
    a = block_form(W, 2)
    a[0][0] = 2*inner(sym(grad(u)), sym(grad(v)))*dx
    a[0][1] = -inner(p, div(v))*dx
    a[1][0] = -inner(q, div(u))*dx
    a[1][2] = inner(q, phi)*dxc
    a[2][1] = inner(p, psi)*dxc

    L = block_form(W, 1)
    # Volumetric
    L[0] = inner(mms['force'], v)*dx

    V_bcs = [DirichletBC(V, mms['velocity'], 'on_boundary')]
    W_bcs = [V_bcs, [], []]

    return a, L, W, W_bcs


def setup_preconditioner(W, AA, bcs):
    '''H1 x L2'''
    # I don't want symgrad here because of AMG
    V, Q, R = W

    u, v = TrialFunction(V), TestFunction(V)
    # Seminorm due to bcs
    aV = inner(grad(u), grad(v))*dx
    AV, _ = assemble_system(aV, inner(Constant((0, 0)), v)*dx, bcs[0])

    p, q = TrialFunction(Q), TestFunction(Q)
    # Pressure mass matrix is L2
    aQ = inner(p, q)*dx
    AQ = assemble(aQ)

    AR = AA[2][2]

    return block_diag_mat([AMG(AV), AMG(AQ), AMG(AR)])

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from common import ConvergenceLog, H1_norm, L2_norm
    from block.iterative import MinRes
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # We will conside isoP2 with P1 or P0
    parser.add_argument('--pressure_degree', type=int, default=1, choices=[0, 1])
    # and the fine mesh obtained by refimenets with
    parser.add_argument('--refinement', type=str, default='bisection', choices=['bisection', 'centroid', 'dual'])
    #
    parser.add_argument('--direct_solver', type=int, default=1, choices=[0, 1])    
    args, _ = parser.parse_known_args()

    if args.pressure_degree == 0:
        assert args.refinement != 'centroid' 

    # Reduce verbosity
    set_log_level(40)
    # For checking convergence we pick the solution of the test case ...
    mms = setup_mms()
    # Flux, pressure, multiplier
    u_true, p_true = mms['solution']

    clog = ConvergenceLog({
        'u': (u_true, H1_norm, '0'),
        'p': (p_true, L2_norm, '0'),
        'niters': None,
    })

    print(clog.header())
    for i in range(5):
        n = 4*2**i
        mesh = UnitSquareMesh(n, n)  # Pressure mesh
        
        a, L, W, bcs = setup_problem(mesh, mms,
                                     pressure_deg=args.pressure_degree,
                                     refinement=args.refinement)

        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, bcs)
        
        # Now solve with direct solver
        wh = ii_Function(W)
        if args.direct_solver:
            # Here A, b are cbc.block object so we need to convert them to matrices
            A, b = map(ii_convert, (A, b))
            solve(A, wh.vector(), b)

            niters = 1
        else:
            B = setup_preconditioner(W, A, bcs)
        
            wh = ii_Function(W)
            Ainv = MinRes(A, precond=B, tolerance=1E-10, show=0)
            wh.vector()[:] = ii_convert(Ainv*b)

            niters = len(Ainv.residuals)
        uh, ph, _ = wh

        clog.add((uh, ph, niters))
        print(clog.report_last(with_name=False))

    # See for bisection
    # https://www.math.uci.edu/~chenlong/ifemdoc/fem/StokesisoP2P1femrate.html
    # https://www.math.uci.edu/~chenlong/ifemdoc/fem/StokesisoP2P0femrate.html
    passed = all((clog['u'].get_rate()[0] > 0.95,
                  clog['p'].get_rate()[0] > (0.95 if args.pressure_degree == 0 else 1.5),
                  all(count < 65 for count in clog['niters'])))        

    sys.exit(int(passed))
