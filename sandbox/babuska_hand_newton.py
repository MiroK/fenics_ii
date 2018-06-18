# Nonlinear Babuska
# -div(nl(u)*grad(u)) = f in [0, 1]^2
#               T(u) = h on boundary
#
# Solved either with DOLFIN.solver [calling direct solver]
# or with ksp('preonly') together with LU preconditioner
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

    OptDB = PETSc.Options()

    # This is way to use inv(A) for the solution
    OptDB.setValue('ksp_type', 'preonly')  # Only appy preconditiner
    # Which is lu factorization with umfpack. NOTE: the package matters
    # a lot; e.g. superlu_dist below blows up
    OptDB.setValue('pc_type', 'lu')
    OptDB.setValue('pc_factor_mat_solver_package', 'umfpack')

    dup = ii_Function(W)
    x_vec = as_backend_type(dup.vector()).vec()
    while eps > tol and niter < maxiter:
        niter += 1
    
        A, b = (ii_convert(ii_assemble(x)) for x in (dF, F))

        # PETSc solver
        A_mat = as_backend_type(A).mat()
        b_vec = as_backend_type(b).vec()

        ksp = PETSc.KSP().create()
        ksp.setOperators(A_mat)
        ksp.setFromOptions()

        ksp.solve(b_vec, x_vec)
        niters = ksp.getIterationNumber()

        # DOLFIN solver
        # niters = solve(A, dup.vector(), b)
        
        eps = sqrt(sum(x.norm('l2')**2 for x in dup.vectors()))
        
        print '\t%d |du| = %g |A|= %g |b| = %g | niters %d' % (
            niter, eps, A.norm('linf'), b.norm('l2'), niters
        )

        # FIXME: Update
        for i in range(len(W)):
            up[i].vector().axpy(-1, dup[i].vector())

    return up

# --------------------------------------------------------------------

if __name__ == '__main__':
    import sympy as sp

    RED = '\033[1;37;31m%s\033[0m'

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
        
        msg = '|e|_1 = %.4E[%.2f] |p|_0 = %.4E[%.2f] | ndofs = %d' % data
        print(RED % msg)
    
    File('./nl_results/babuska_uh.pvd') << uh
    File('./nl_results/babuska_ph.pvd') << ph

