# We have a cube [-2, 2]^3 with a monifold inside it which contains a curve.
# Let ui, i=3, 2, 1 be quantities on the respetive i-dim domains and f
# some function on the curve. Then we are after the solution of
#
# min_{u3, u2, u1} of sum_i |grad(ui)|_{Omegai}^3 subject to 
#
# u3 - u2 = 0 on omega2
# u2 - u1 = f on omega1
#
# Here the manifold is taken as (part of) surface of the cube [-inner, inner]^3


from xii import *
from dolfin import *


def main(f, mesh_generator):
    '''Coupled 3d-2d-1d Poisson problems driven only by the 1d force f'''
    t = Timer('mesh'); t.start()
    mesh3d, mesh2d, mesh1d = mesh_generator()
    print '\tGot mesh in %g s' % t.stop()

    t = Timer('system'); t.start()
    # Temperatures
    V3, V2, V1 = [FunctionSpace(mesh, 'CG', 1) for mesh in (mesh3d, mesh2d, mesh1d)]
    # Multipliers
    Q2, Q1 = FunctionSpace(mesh2d, 'CG', 1), FunctionSpace(mesh1d, 'CG', 1)

    W = (V3, V2, V1, Q2, Q1)

    print 'Total unknowns', sum(Wi.dim() for Wi in W), 'in 1d', V1.dim()
    
    u3, u2, u1, p2, p1 = map(TrialFunction, W)
    v3, v2, v1, q2, q1 = map(TestFunction, W)
    # I will refer to the 2d surface as S and the curve as G
    dxS = Measure('dx', domain=mesh2d)
    TS_u3, TS_v3 = (Trace(x, mesh2d) for x in (u3, v3)) 

    dxG = Measure('dx', domain=mesh1d)
    TG_u2, TG_v2 = (Trace(x, mesh1d) for x in (u2, v2))

    # We will have a 5x5 block system
    a = [[0]*5 for _ in range(5)]
    # Subdomain physics
    a[0][0] = inner(grad(u3), grad(v3))*dx
    a[1][1] = inner(grad(u2), grad(v2))*dxS
    a[2][2] = inner(grad(u1), grad(v1))*dxG
    # 3-2 coupling
    a[0][3] = inner(p2, TS_v3)*dxS
    a[1][3] = -inner(p2, v2)*dxS
    # 2-1 coupling
    a[1][4] = inner(p1, TG_v2)*dxG
    a[2][4] = -inner(p1, v1)*dxG
    # 1-2 coupling
    a[4][1] = inner(q1, TG_u2)*dxG
    a[4][2] = -inner(q1, u1)*dxG
    # 2-3 coupling
    a[3][0] = inner(q2, TS_u3)*dxS
    a[3][1] = -inner(q2, u2)*dxS

    L = [inner(Constant(0), v3)*dx,
         inner(Constant(0), v2)*dxS,
         inner(Constant(0), v1)*dxG,
         inner(Constant(0), q2)*dxS,
         inner(f, q1)*dxG]

    AA, bb = map(ii_assemble, (a, L))
    print '\tPerformed ii_assemble in %g s' % t.stop()

    wh = ii_Function(W)

    t = Timer('convert'); t.start()
    AAm, bbm = map(ii_convert, (AA, bb))
    print '\tPerformed ii_convert in %g s' % t.stop()

    t = Timer('solve'); t.start()        
    LUSolver('umfpack').solve(AAm, wh.vector(), bbm)
    print '\tSolved in %g s' % t.stop()
    
    for i, wh_i in enumerate(wh):
        wh_i.rename('u', str(i))
        File('./u%d.pvd' % i) << wh_i

# --------------------------------------------------------------------

if __name__ == '__main__':
    f = Expression('sin(pi*(x[0]+x[1]+x[2]))', degree=3)

    # Setup the mesh_generator for the random curve
    import meshing
    # Using 1/4 of the default mesh sizes, the inner box is [-1/2, 1/2]^3
    # and thee random curve will be made using 40 vertices
    mesh_generator = lambda :meshing.load(scale=2./8,
                                          inner_size=0.5,
                                          curve_gen=lambda mesh: meshing.fun(mesh, 40))
    
    main(f, mesh_generator)
