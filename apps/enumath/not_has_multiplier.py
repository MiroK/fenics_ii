from dolfin import *
from xii import *
from block import block_bc


def setup_problem(radius, mesh_gen):
    '''
    This is 3d-1d system inspired by https://arxiv.org/abs/1803.04896
    There is a Robin forcing f on the entire 3d bdry and g 1d bdry. 
    No multiplier.
    '''
    # FIXME: Dirichlet bcs
    
    mesh3d, mesh1d, bdry_vertex = mesh_gen()
    # There is a 3d/1d/iface/bdry conductiity; made up
    k3d, k1d, kG, kbdry = map(Constant, (0.1, 1, 0.5, 0.2))
    
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
    u3, u = map(TrialFunction, W)
    v3, v = map(TestFunction, W)

    averaging_shape = Circle(radius, 10)
    Pi_u3, Pi_v3 = (Average(x, mesh1d, averaging_shape) for x in (u3, v3))

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

    AA, bb = map(ii_assemble, (a, L))

    # Return the block system
    return AA, bb, W

# --------------------------------------------------------------------

if __name__ == '__main__':
    import meshing, argparse, sys, petsc4py

    petsc4py.init(sys.argv)
    from petsc4py import PETSc

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Problem params
    parser.add_argument('-scale', default=0.25, type=float,
                        help='Scale mesh size relative to definition in geo')
    
    parser.add_argument('-npoints', type=int, default=80,
                        help='Num points to draw the curve')
    
    parser.add_argument('-radius', type=float, default=0.1,
                        help='Radius of the tubes')
    
    args, petsc_args = parser.parse_known_args()

    # Let's rock
    mesh_gen = lambda: meshing.load(args.scale,
                                    lambda mesh: meshing.fun(mesh, npoints=args.npoints))

    timer = Timer('setup'); timer.start()
    AA, bb, W = setup_problem(args.radius, mesh_gen)
    print '\tProblem setup took %g s\n \tNumber of unknowns %d ' %  (timer.stop(), sum(Wi.dim() for Wi in W))

    x = AA*bb
    y = AA.transpmult(bb)
    assert (x-y).norm() < 1E-14, 'AA is not symmetric!'
    
    wh = ii_Function(W)

    timer = Timer('solver'); timer.start()
    # Convert
    AAm, bbm = map(ii_convert, (AA, bb))
    niters = LUSolver('umfpack').solve(AAm, wh.vector(), bbm)
    
    print '\tSolver took %g s. Niters %d' % (timer.stop(), niters)
    
    for i, wh_i in enumerate(wh):
        # Renaming to make it easier to save state in Visit/Pareview
        wh_i.rename('u', str(i))
        File('nothas_rad%g_scale%g_u%d.pvd' % (args.radius, args.scale, i)) << wh_i
