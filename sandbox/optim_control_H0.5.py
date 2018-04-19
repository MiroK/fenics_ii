# |u - f|^2{H^{0.5}(\partial\Omega)} + alpha/2*|p|_{L^2(\Omega)}
# 
# subject to
#
# -\Delta u + u + f = 0 in Omega
#         grad(u).n = 0 on \partial\Omega
#
from dolfin import *
from xii import *
from hsmg import HsNorm
from block import block_transpose
from block.algebraic.petsc import AMG
from block.algebraic.petsc import LumpedInvDiag
from xii.assembler.trace_matrix import trace_mat_no_restrict
from xii.linalg.convert import collapse



def solve_problem(ncells, eps, solver_params):
    '''Optim problem on [0, 1]^2'''
    # Made up
    f = Expression('x[0] + x[1]', degree=1)
    
    mesh = UnitSquareMesh(*(ncells, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    Q = FunctionSpace(mesh, 'CG', 1)
    V = FunctionSpace(mesh, 'CG', 1)
    B = FunctionSpace(mesh, 'CG', 1)
    W = [Q, V, B]

    p, u, lmbda = map(TrialFunction, W)
    q, v, beta = map(TestFunction, W)
    Tu = Trace(u, bmesh)
    Tv = Trace(v, bmesh)

    # The line integral
    dxGamma = Measure('dx', domain=bmesh)

    a = [[0]*len(W) for _ in range(len(W))]

    a[0][0] = Constant(eps)*inner(p, q)*dx
    a[0][2] = inner(q, lmbda)*dx
    # We put zero on the diagonal temporarily and then replace by
    # the fractional laplacian
    a[1][1] = 0
    a[1][2] = inner(grad(v), grad(lmbda))*dx + inner(v, lmbda)*dx

    a[2][0] = inner(p, beta)*dx
    a[2][1] = inner(grad(u), grad(beta))*dx + inner(u, beta)*dx

    L = [inner(Constant(0), q)*dx,
         inner(Constant(0), v)*dx,  # Same replacement idea here
         inner(Constant(0), beta)*dx]
    # All but (1, 1) and 1 are final
    AA, bb = map(ii_assemble, (a, L))

    # Now I want and operator which corresponds to (Tv, (-Delta^{0.5} T_u))_bdry
    TV = FunctionSpace(bmesh, 'CG', 1)
    T = PETScMatrix(trace_mat_no_restrict(V, TV))
    # The fractional laplacian nodal->dual
    fDelta = HsNorm(TV, s=0.5)
    # This should be it
    A11 = block_transpose(T)*fDelta*T
    # Now, while here we can also comute the rhs contribution
    # (Tv, (-Delta^{0.5}, f))
    f_vec = interpolate(f, V).vector()
    b1 = A11*f_vec
    bb[1] = b1
    AA[1][1] = A11

    wh = ii_Function(W)
    # Direct solve
    if not solver_params:
        AAm, bbm = map(ii_convert, (AA, bb))
        LUSolver('umfpack').solve(AAm, wh.vector(), bbm)
        return wh, -1

    # Preconditioner like in the L2 case but with L2 bdry replaced
    b00 = Constant(eps)*inner(p, q)*dx
    B00 = LumpedInvDiag(ii_assemble(b00))

    # H2 norm with H1 elements
    H1 = ii_assemble(inner(grad(v), grad(u))*dx + inner(v, u)*dx)
    # From dual to nodal
    R = LumpedInvDiag(ii_assemble(inner(u, v)*dx))
    # The whole matrix to be inverted is then (second term is H2 approx)
    B11 = collapse(A11 + eps*H1*R*H1)
    # And the inverse
    B11 = AMG(B11, parameters={'pc_hypre_boomeramg_cycle_type': 'W'})

    b22 = Constant(1./eps)*inner(lmbda, beta)*dx
    B22 = LumpedInvDiag(ii_assemble(b22))
    
    BB = block_diag_mat([B00, B11, B22])

    # Want the iterations to start from random (iterative)
    wh.block_vec().randomize()

    # Default is minres
    if '-ksp_type' not in solver_params: solver_params['-ksp_type'] = 'minres'
            
    opts = PETSc.Options()
    for key, value in solver_params.iteritems():
        opts.setValue(key, None if value == 'none' else value)
        
    ksp = PETSc.KSP().create()
    ksp.setOperators(ii_PETScOperator(AA))
    ksp.setPC(ii_PETScPreconditioner(BB, ksp))
    
    ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)
    ksp.setFromOptions()
    
    ksp.solve(as_petsc_nest(bb), wh.petsc_vec())
    
    niters = ksp.getIterationNumber()

    return wh, niters

# -------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, sys, petsc4py, os, itertools
    from collections import defaultdict
    # Make petsc4py work with command line. This allows configuring
    # ksp (petsc krylov solver) as e.g.
    #      -ksp_rtol 1E-8 -ksp_atol none -ksp_monitor_true_residual none
    petsc4py.init(sys.argv)
    from petsc4py import PETSc

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # The demo file: runnning it defines setups*
    # Some problems are paramer dependent
    parser.add_argument('-eps', default=[1.0], nargs='+', type=float,
                        help='Regularization parameter')
    # How many uniform refinements to make
    parser.add_argument('-ncells', type=int, default=[32], nargs='+', help='Mesh size')
    # How many uniform refinements to make
    parser.add_argument('-save_dir', type=str, default='',
                        help='Path for directory storing results')
    # Iterative solver?
    parser.add_argument('-solver', type=str, default='direct', choices=['direct', 'iterative'],
                        help='Use direct solver with params as in main')
    
    args, petsc_args = parser.parse_known_args()

    if petsc_args:
        solver_params = dict((k, v) for k, v in zip(petsc_args[::2], petsc_args[1::2]))
    else:
        solver_params = {}

    results = defaultdict(list)
    dofs = set()
    for eps, ncells in itertools.product(args.eps, args.ncells):
        wh, niters = solve_problem(ncells, eps, solver_params)

        ndofs = sum(whi.function_space().dim() for whi in wh)
        print eps, ndofs, niters
        
        results[eps].append(niters)
        dofs.add(ndofs)

    cols = sorted(results.keys())
    rows = [sorted(dofs)] + [results[c] for c in cols]
    print '\t eps=', cols
    for values in itertools.izip(*rows):
        print values

    # Only send the final
    if args.save_dir:
        path = os.path.join(args.save_dir, 'fract_optimality')
        for i, wh_i in enumerate(wh):
            # Renaming to make it easier to save state in Visit/Pareview
            wh_i.rename('u', str(i))
            
            File('%s_%d.pvd' % (path, i)) << wh_i
