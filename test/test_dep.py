from block.block_bc import block_rhs_bc
from block.algebraic.petsc import LU
from dolfin import *
from xii import *
import sympy as sp


def test(dt, ncells, w_exact):
    '''Want to solve a heat equation'''
    # Rest
    for f in w_exact: f.t = 0.
    
    # On a unit square
    left = CompiledSubDomain('near(x[0], 0)')
    rest = CompiledSubDomain('near(x[0], 1) || near(x[1]*(1-x[1]), 0)')

    mesh = UnitSquareMesh(ncells, ncells)
    boundaries = MeshFunction('size_t', mesh, 1, 0)
    left.mark(boundaries, 1)
    rest.mark(boundaries, 2)

    bmesh = EmbeddedMesh(boundaries, 1)

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))
    Tu, Tv = Trace(u, bmesh), Trace(v, bmesh)
    dx_ = Measure('dx', domain=bmesh)

    wh = ii_Function(W)

    a = block_form(W, 2)
    a[0][0] = inner(u, v)*dx + dt*inner(grad(u), grad(v))*dx
    a[0][1] = inner(p, Tv)*dx_
    a[1][0] = inner(q, Tu)*dx_

    # Unpack data
    u_exact, p_exact, f_exact = w_exact
    p_exact.k = dt(0)
    
    L = block_form(W, 2)
    L[0] = inner(wh[0], v)*dx + dt*inner(f_exact, v)*dx
    L[1] = inner(u_exact, q)*dx_

    bcs = [[DirichletBC(V, u_exact, boundaries, 2)],
           [DirichletBC(Q, Constant(0), 'on_boundary')]]

    A, b = list(map(ii_assemble, (a, L)))
    b_bc = block_rhs_bc(bcs, A)
    # Apply bcs to the system
    A_bc, _ = apply_bc(A, b, bcs)
    # Solver is setup based on monolithic
    A_mono = ii_convert(A_bc)

    print(('Setting up solver %d' % sum(Wi.dim() for Wi in W)))
    time_Ainv = Timer('Ainv')
    A_inv = PETScLUSolver(A_mono, 'umfpack')  # Setup once
    print(('Done in %g s' % time_Ainv.stop()))

    time = 0;
    while time < 0.5:
        time += dt(0)
        
        f_exact.t = time
        u_exact.t = time
        # We don't need to reasamble the system
        #aux, b = map(ii_assemble, (a, L))
        #_, b = apply_bc(aux, b, bcs)
        b = ii_assemble(L)
        b_bc.apply(b)
        b_ = ii_convert(b)
        
        x = wh.vector()
        A_inv.solve(x, b_)
        
        uh, ph = wh
    eu = errornorm(u_exact, uh, 'H1')
    ep = errornorm(p_exact, ph, 'L2')

    return mesh.hmin(), eu, ep

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    
    # Test case on unit square: left weakly; rest has Dirichlet
    t, x, y, k = sp.symbols('t, x[0], x[1], k')
    u = sp.sin(2*t)*sp.sin(5*(x**2 + y**2))
    f = u.diff(t, 1) - (u.diff(x, 2) + u.diff(y, 2))
    p = k*u.diff(x, 1)

    u_exact = Expression(sp.printing.ccode(u), degree=4, t=0)
    p_exact = Expression(sp.printing.ccode(p), degree=4, t=0, k=0)
    f_exact = Expression(sp.printing.ccode(f), degree=4, t=0)

    w_exact = (u_exact, p_exact, f_exact)

    dts = (0.01, 0.005, 0.0025, 0.00125)
    sizes = (8, 16, 32, 64, 128, 256)

    u_errors, p_errors = np.zeros((len(sizes), len(dts))), np.zeros((len(sizes), len(dts)))
    
    for row, ncells in enumerate(sizes):
        for col, dt in enumerate(dts):
            dt = Constant(dt)

            h, eu, ep = test(dt, ncells, w_exact)
            u_errors[row][col] = eu
            p_errors[row][col] = ep


    print(u_errors)
    print()
    print(p_errors)
