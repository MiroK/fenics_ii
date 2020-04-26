# Nonlinear Babuska
# -div(nl(u)*grad(u)) = f in [0, 1]^2
#               T(u) = h on boundary
#
# Each Newton step is solver with preconditioned MinRes where the preconditioner
# is assembled once. We define it based on H1 x H^{-0.5} norm

from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from xii import *
from xii.nonlin.jacobian import ii_derivative
from ulfy import Expression
from petsc4py import PETSc
from hsmg.hseig import HsNorm
from block.algebraic.petsc import AMG
import numpy as np
from six.moves import map
from six.moves import range


def nonlinear_babuska(N, u_exact, p_exact):
    '''MMS for the problem'''
    mesh = UnitSquareMesh(N, N)
    bmesh = BoundaryMesh(mesh, 'exterior')

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = (V, Q)

    up = ii_Function(W)
    u, p = up  # Split

    v, q = list(map(TestFunction, W))
    Tu, Tv = (Trace(x, bmesh) for x in (u, v))

    dxGamma = Measure('dx', domain=bmesh)

    # Nonlinearity
    nl = lambda u: (Constant(2) + u)**2

    f_exact = -div(nl(u)*grad(u))
    g_exact = u

    f = interpolate(Expression(f_exact, subs={u: u_exact}, degree=1), V)
    h = interpolate(Expression(g_exact, subs={u: u_exact}, degree=4), Q)

    # The nonlinear functional
    F = [inner(nl(u)*grad(u), grad(v))*dx + inner(p, Tv)*dxGamma - inner(f, v)*dx,
         inner(Tu, q)*dxGamma - inner(h, q)*dxGamma]

    dF = block_jacobian(F, up)

    # Setup H1 x H-0.5 preconditioner only once
    B0 = ii_derivative(inner(grad(u), grad(v))*dx + inner(u, v)*dx, u)
    # The Q norm via spectral
    B1 = inverse(HsNorm(Q, s=-0.5))  # The norm is inverted exactly
    # Preconditioner
    B = block_diag_mat([AMG(ii_assemble(B0)), B1])

    # Newton
    eps = 1.0
    tol = 1.0E-10
    niter = 0.
    maxiter = 25

    OptDB = PETSc.Options()
    # Since later b gets very small making relative too much work
    OptDB.setValue('ksp_atol', 1E-6)  
    OptDB.setValue('ksp_monitor_true_residual', None)

    dup = ii_Function(W)
    x_vec = as_backend_type(dup.vector()).vec()

    n_kspiters = []
    while eps > tol and niter < maxiter:
        niter += 1.
    
        A, b = (ii_assemble(x) for x in (dF, F))

        ksp = PETSc.KSP().create()
        ksp.setType('minres')
        ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)

        ksp.setOperators(ii_PETScOperator(A))
        ksp.setPC(ii_PETScPreconditioner(B, ksp))

        ksp.setFromOptions()

        ksp.solve(as_petsc_nest(b), x_vec)
        niters = ksp.getIterationNumber() + 1

        n_kspiters.append(niters)
        
        eps = sqrt(sum(x.norm('l2')**2 for x in dup.vectors()))
        
        print('\t%d |du| = %g | niters %d' % (niter, eps, niters))

        # FIXME: Update
        for i in range(len(W)):
            up[i].vector().axpy(-1, dup[i].vector())

    return up, n_kspiters

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
    for N in (8, 16, 32, 64, 128, 256, 512, 1024):
        (uh, ph), niters = nonlinear_babuska(N, u_exact, p_exact)
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
        data = data + (len(niters), min(niters), np.mean(niters), max(niters))
        
        msg = '|e|_1 = %.4E[%.2f] |p|_0 = %.4E[%.2f] | ndofs = %d | newton_iters = %d | inner_iters = (%d, %.2f %d)' % data
        print((RED % msg))
    
    File('./nl_results/babuska_uh.pvd') << uh
    File('./nl_results/babuska_ph.pvd') << ph
