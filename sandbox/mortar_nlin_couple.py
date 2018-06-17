# Nonlinear Babuska
# -div(nl1(u1)*grad(u1)) = f1 in Omega_1
# -div(nl2(u2)*grad(u2)) = f2 in Omega_2
#
# nl1(u1)*grad(u1).n1 + nl2(u2)*grad(u2).n2 = 0
# nl3(u1 - u2) + nl1(u1)*grad(u1).n1 = h

from dolfin import *
from xii import *
from ulfy import Expression


def mortar_lin_couple(N):
    '''...'''
    interior = CompiledSubDomain('std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25')
    outer_mesh = UnitSquareMesh(N, N)
    
    subdomains = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim(), 0)
    # Awkward marking
    for cell in cells(outer_mesh):
        x = cell.midpoint().array()            
        subdomains[cell] = int(interior.inside(x, False))
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    inner_mesh = SubMesh(outer_mesh, subdomains, 1)
    outer_mesh = SubMesh(outer_mesh, subdomains, 0)
    # Interior boundary
    surfaces = MeshFunction('size_t', inner_mesh, inner_mesh.topology().dim()-1, 0)
    DomainBoundary().mark(surfaces, 1)
    
    gamma_mesh = EmbeddedMesh(surfaces, 1)

    # ---

    V1 = FunctionSpace(inner_mesh, 'CG', 1)
    V2 = FunctionSpace(outer_mesh, 'CG', 1)
    Q = FunctionSpace(gamma_mesh, 'CG', 1)
    W = (V1, V2, Q)

    w = ii_Function(W)
    u1, u2, p = w  # Split

    v1, v2, q = map(TestFunction, W)
    Tu1, Tv1 = (Trace(x, gamma_mesh) for x in (u1, v1))
    Tu2, Tv2 = (Trace(x, gamma_mesh) for x in (u2, v2))

    dxGamma = Measure('dx', domain=gamma_mesh)

    # Nonlinearity
    nl1 = lambda u: (1+u)**2
    nl2 = lambda u: (2+u)**4

    f1 = interpolate(Expression('x[0]*x[0]+x[1]*x[1]', degree=1), V1)
    f2 = interpolate(Constant(1), V2)
    h = interpolate(Expression('x[0] + x[1]', degree=1), Q)

    # The nonlinear functional
    F = [inner(nl1(u1)*grad(u1), grad(v1))*dx + inner(u1, v1)*dx + inner(p, Tv1)*dxGamma - inner(f1, v1)*dx,
         inner(nl2(u2)*grad(u2), grad(v2))*dx + inner(u2, v2)*dx - inner(p, Tv2)*dxGamma - inner(f2, v2)*dx,
         inner((Tu1-Tu2)**2, q)*dxGamma - inner(p, q)*dxGamma - inner(h, q)*dxGamma]

    dF = block_jacobian(F, w)

    # Newton
    eps = 1.0
    tol = 1.0E-10
    niter = 0
    maxiter = 25

    dw = ii_Function(W)
    while eps > tol and niter < maxiter:
        niter += 1
    
        A, b = map(ii_assemble, (dF, F))
        A, b = map(ii_convert, (A, b))
        
        solve(A, dw.vector(), b)
        
        eps = sqrt(sum(x.norm('l2')**2 for x in dw.vectors()))
        
        print '\t%d |du| = %.6E |A|= %.6E |b| = %.6E' % (niter, eps, A.norm('linf'), b.norm('l2'))

        # FIXME: Update
        for i in range(len(W)):
            w[i].vector().axpy(-1, dw[i].vector())

    return w

# --------------------------------------------------------------------

if __name__ == '__main__':
    import sympy as sp

    u1h, u2h, ph = mortar_lin_couple(N=16)

    # # Setup the test case
    # x, y = sp.symbols('x y')
    # u_exact = sp.cos(sp.pi*x*(1-x)*y*(1-y))
    # p_exact = sp.S(0)

    # u_expr = Expression(u_exact, degree=4)
    # p_expr = Expression(p_exact, degree=4)

    # eu0, ep0, h0 = -1, -1, -1
    # for N in (8, 16, 32, 64, 128, 256):
    #     uh, ph = nonlinear_babuska(N, u_exact, p_exact)
    #     Vh = uh.function_space()
    #     Qh = ph.function_space()

    #     eu = errornorm(uh, u_expr, 'H1', degree_rise=0)
    #     ep = errornorm(ph, p_expr, 'L2', degree_rise=0)
    #     h = Vh.mesh().hmin()
        
    #     if eu0 > 0:
    #         rate_u = ln(eu/eu0)/ln(h/h0)
    #         rate_p = ln(ep/ep0)/ln(h/h0)
    #     else:
    #         rate_u, rate_p = -1, -1
            
    #     eu0, ep0, h0 = eu, ep, h
        
    #     data = (eu, rate_u, ep, rate_p, Vh.dim() + Qh.dim())
        
    #     print('|e|_1 = %.4E[%.2f] |p|_0 = %.4E[%.2f] | ndofs = %d' % data)
    
    File('./nl_results/mortar_nlin_u1h.pvd') << u1h
    File('./nl_results/mortar_nlin_u2h.pvd') << u2h
    File('./nl_results/mortar_nlin_ph.pvd') << ph

