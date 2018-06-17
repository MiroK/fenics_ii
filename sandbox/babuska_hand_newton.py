# Nonlinear Babuska
# -div(nl(u)*grad(u)) = f in [0, 1]^2
#               T(u) = h on boundary
#

from dolfin import *
from xii import *
from ulfy import Expression
from petsc4py import PETSc


def nonlinear_babuska(N, u_exact, p_exact):
    '''MMS for the problem'''
    
    mesh = UnitSquareMesh(N, N)
    bmesh = BoundaryMesh(mesh, 'exterior')

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = (V, Q)

    up = ii_Function(W)
    u, p = up  # Split

    v, q = map(TestFunction, W)
    Tu, Tv = (Trace(x, bmesh) for x in (u, v))

    dxGamma = Measure('dx', domain=bmesh)

    # Nonlinearity
    nl = lambda u: (1+u)**2

    f_exact = -div(nl(u)*grad(u))
    g_exact = u

    f = interpolate(Expression(f_exact, subs={u: u_exact}, degree=1), V)
    h = interpolate(Expression(g_exact, subs={u: u_exact}, degree=4), Q)

    # The nonlinear functional
    F = [inner(nl(u)*grad(u), grad(v))*dx + inner(p, Tv)*dxGamma - inner(f, v)*dx,
         inner(Tu, q)*dxGamma - inner(h, q)*dxGamma]

    dF = block_jacobian(F, up)

    # Newton
    eps = 1.0
    tol = 1.0E-10
    niter = 0
    maxiter = 25

    dup = ii_Function(W)
    while eps > tol and niter < maxiter:
        niter += 1
    
        A, b = (ii_convert(ii_assemble(x)) for x in (dF, F))

        # FIXME: gmres
        #        iterative with preconditioner
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        ksp.setType('gmres')
        ksp.getPC().setType('lu')
        ksp.setOperators(as_backend_type(A).mat())
        ksp.setFromOptions()
        
        ksp.solve(as_backend_type(dup.vector()).vec(),
                  as_backend_type(b).vec())

        # solve(A, dup.vector(), b)
        
        eps = sqrt(sum(x.norm('l2')**2 for x in dup.vectors()))
        
        print '\t%d |du| = %.6E |A|= %.6E |b| = %.6E' % (niter, eps, A.norm('linf'), b.norm('l2'))

        # FIXME: Update
        for i in range(len(W)):
            up[i].vector().axpy(-1, dup[i].vector())

    return up

# --------------------------------------------------------------------

if __name__ == '__main__':
    import sympy as sp

    # Setup the test case
    x, y = sp.symbols('x y')
    u_exact = sp.cos(sp.pi*x*(1-x)*y*(1-y))
    p_exact = sp.S(0)

    u_expr = Expression(u_exact, degree=4)
    p_expr = Expression(p_exact, degree=4)

    eu0, ep0, h0 = -1, -1, -1
    for N in (8, 16, 32, 64, 128, 256):
        uh, ph = nonlinear_babuska(N, u_exact, p_exact)
        Vh = uh.function_space()
        Qh = ph.function_space()

        eu = errornorm(uh, u_expr, 'H1', degree_rise=0)
        ep = errornorm(ph, p_expr, 'L2', degree_rise=0)
        h = Vh.mesh().hmin()

        if eu0 > 0:
            rate_u = ln(eu/eu0)/ln(h/h0)
            rate_p = ln(ep/ep0)/ln(h/h0)
        else:
            rate_u, rate_p = -1, -1
            
        eu0, ep0, h0 = eu, ep, h
        
        data = (eu, rate_u, ep, rate_p, Vh.dim() + Qh.dim())
        
        print('|e|_1 = %.4E[%.2f] |p|_0 = %.4E[%.2f] | ndofs = %d' % data)
    
    File('./nl_results/babuska_uh.pvd') << uh
    File('./nl_results/babuska_ph.pvd') << ph

