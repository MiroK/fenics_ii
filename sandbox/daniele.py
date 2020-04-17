from dolfin import *
from xii import *
from block import block_bc, block_mat
import numpy as np


K_CONSTANTS = (0.1, 1, 0.5, 0.2)


def create_mesh(N):
    '''[-1, 1]^3 with straight line'''
    assert N % 2 == 0
    
    mesh = BoxMesh(Point(*(-1, )*3), Point(*(1, )*3), N, N, 2*N)

    gamma = MeshFunction('size_t', mesh, 1, 0)
    # Grid
    for xi in (-0.5, 0.0, 0.5):
        for yi in (-0.5, 0.0, 0.5):
            CompiledSubDomain('near(x[0], A) && near(x[1], B)', A=xi, B=yi).mark(gamma, 1)
    # Connect the grid
    for zi in (-0.5, 0.5):
        for yi in (-0.5, 0.0, 0.5):
            CompiledSubDomain('near(x[1], A) && near(x[2], B)', A=yi, B=zi).mark(gamma, 1)

    for zi in (-0.5, 0.5):
        for xi in (-0.5, 0.5):
            CompiledSubDomain('near(x[0], A) && near(x[2], B)', A=xi, B=zi).mark(gamma, 1)


    mesh1d = EmbeddedMesh(gamma, 1)
    bdry_vertex = [0., 0., -1]

    return mesh, mesh1d, bdry_vertex


def setup_problem(radius, mesh_gen):
    '''This is 3d-1d system.'''
    # FIXME: Dirichlet bcs
    
    mesh3d, mesh1d, bdry_vertex = mesh_gen()
    # There is a 3d/1d/iface/bdry conductiity; made up
    k3d, k1d, kG, kbdry = list(map(Constant, K_CONSTANTS))
    
    # f is made up
    f = Expression('pow(x[0]-0.5, 2)+(x[1]-0.5, 2)+(x[2]-0.5, 2)', degree=2)
    
    # g is 1 at bdry vertex and then decays with distance
    g = Expression('exp(-sqrt(pow(x[0]-x0, 2)+pow(x[1]-x1, 2)+pow(x[2]-x2, 2)))',
                   degree=4,
                   x0=bdry_vertex[0], x1=bdry_vertex[1], x2=bdry_vertex[2])

    dxG = Measure('dx', domain=mesh1d)

    V3 = FunctionSpace(mesh3d, 'CG', 1)
    V = FunctionSpace(mesh1d, 'CG', 1)

    W = (V3, V)
    u3, u = list(map(TrialFunction, W))
    v3, v = list(map(TestFunction, W))

    # Averaging surface
    cylinder = Circle(radius=radius, degree=10)

    Pi_u3, Pi_v3 = (Average(x, mesh1d, cylinder) for x in (u3, v3))

    a = [[0]*len(W) for _ in range(len(W))]

    a[0][0] = inner(k3d*grad(u3), grad(v3))*dx +\
              inner(kbdry*u3, v3)*ds + \
              inner(Constant(2*pi*radius)*kG*Pi_u3, Pi_v3)*dxG

    a[1][1] = inner(Constant(pi*radius**2)*k1d*grad(u), grad(v))*dxG +\
              inner(u, v)*ds +\
              inner(Constant(2*pi*radius)*kG*u, v)*dxG

    a[0][1] = -inner(Constant(2*pi*radius)*kG*u, Pi_v3)*dxG
    a[1][0] = -inner(Constant(2*pi*radius)*kG*v, Pi_u3)*dxG

    L = [inner(kbdry*f, v3)*ds,
         inner(kbdry*g, v)*dxG]

    AA, bb = list(map(ii_assemble, (a, L)))

    # Return the block system
    return AA, bb, W


def setup_preconditioner(W, radius, which):
    '''Wishful thinking using mostly H1 norms on the spaces'''
    from block.algebraic.petsc import AMG

    # For radius = 0.1 both preconditioner perform almost the same.
    # However, they get worse when radius -> 0 so the name of the game
    # will be to make things robust w.r.t radius.
    
    u3, u = list(map(TrialFunction, W))
    v3, v = list(map(TestFunction, W))

    # There is a 3d/1d/iface/bdry conductiity; made up
    k3d, k1d, kG, kbdry = list(map(Constant, K_CONSTANTS))
    
    b00 = inner(grad(u3), grad(v3))*dx + inner(u3, v3)*dx
    # For H1 norm we are done
    if which == 'H1':
        B00 = AMG(assemble(b00))

        b11 = Constant(pi*radius**2)*inner(grad(u), grad(v))*dx + inner(u, v)*dx
        B11 = AMG(assemble(b11))

        return block_diag_mat([B00, B11])

    # Add the coupling term
    if which == 'A':
        gamma = W[-1].mesh()
        dxG = Measure('dx', domain=gamma)
        
        circle = Circle(radius=radius, degree=10)

        Pi_u3, Pi_v3 = (Average(x, gamma, circle) for x in (u3, v3))

        b00 += inner(Constant(2*pi*radius)*Pi_u3, Pi_v3)*dxG
        B00 = AMG(ii_convert(ii_assemble(b00)))
        
        b11 = Constant(pi*radius**2)*inner(grad(u), grad(v))*dx + inner(u, v)*dx
        B11 = AMG(assemble(b11))

        return block_diag_mat([B00, B11])

    return None


def krylov_solve(wh, AA, bb, BB, petsc_args):
    '''The problem as hand is symmetric and by num evidence a positive definite'''
    ## AA and BB as block_mat
    ksp = PETSc.KSP().create()
            
    opts = PETSc.Options()
    for key, value in zip(petsc_args[::2], petsc_args[1::2]):
        opts.setValue(key, None if value == 'none' else value)
    
    # FIXME: we do things differently for block mat and monolithic mat
    if isinstance(AA, block_mat):
        ksp.setOperators(ii_PETScOperator(AA))
        ksp.setPC(ii_PETScPreconditioner(BB, ksp))
        # b and x in A*x = b

        b = as_petsc_nest(bb)
        x = wh.petsc_vec()
    else:
        AA = as_backend_type(AA).mat()
        # Monolithic
        ksp.setOperators(AA)
        assert BB is None  # Use AA
        
        pc = ksp.getPC()
        pc.setType('hypre')
        pc.setOperators(AA)
        
        b = as_backend_type(bb).vec()
        x = as_backend_type(wh.vector()).vec()

    ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)
    ksp.setType('cg')
    ksp.setComputeEigenvalues(1)
    
    ksp.setFromOptions()

    # Start from random guess
    wh.block_vec().randomize()

    ksp.solve(b, x)
    niters = ksp.getIterationNumber()

    eigs = ksp.computeEigenvalues()
    assert (e > 0 for e in eigs)
    eigs = np.abs(eigs)
    
    return niters, max(eigs)/min(eigs)
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, sys, petsc4py

    petsc4py.init(sys.argv)
    from petsc4py import PETSc

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Problem params
    parser.add_argument('-N', default=[4], nargs='+', type=int, help='Cube partiioned to NxNx2N')
    
    parser.add_argument('-radius', type=float, default=0.1, help='Radius of the tubes')

    parser.add_argument('-solver', type=str, default='direct', choices=['direct', 'iterative'],
                        help='Direct/iterative solver')

    parser.add_argument('-precond', type=str, default='H1', choices=['H1', 'A', 'full'],
                        help='Leading block uses H1 or (H1 + the averaging term) or the full system matrix')
    
    args, petsc_args = parser.parse_known_args()

    data = []
    for N in args.N:
        mesh_gen = lambda : create_mesh(N)
    
        timer = Timer('setup'); timer.start()
        AA, bb, W = setup_problem(args.radius, mesh_gen)
        print('\tProblem setup took %g s\n \tNumber of unknowns %d ' %  (timer.stop(), sum(Wi.dim() for Wi in W)))

        x = AA*bb
        y = AA.transpmult(bb)
        assert (x-y).norm() < 1E-14, 'AA is not symmetric!'
    
        wh = ii_Function(W)

        timer = Timer('solver'); timer.start()

        if args.solver == 'direct':
            # Convert
            AAm, bbm = list(map(ii_convert, (AA, bb)))
            niters = LUSolver('umfpack').solve(AAm, wh.vector(), bbm)
            cond = -1
        else:
            # Handle block preconditioners
            BB = setup_preconditioner(W, radius=args.radius, which=args.precond)

            if BB is None:
                # Okay let's try monolithic
                AA, bb = list(map(ii_convert, (AA, bb)))
                
            niters, cond = krylov_solve(wh, AA, bb, BB, petsc_args)

        data.append((N, (W[0].dim(), W[1].dim()), niters, cond))
        print('\tSolver took %g s. Niters %d, estim cond %g' % (timer.stop(), niters, cond))
        
    for row in data: print(row)

    for i, wh_i in enumerate(wh):
        # Renaming to make it easier to save state in Visit/Pareview
        wh_i.rename('u', str(i))
        File('coupled_rad%g_scale%d_u%d.pvd' % (args.radius, N, i)) << wh_i
