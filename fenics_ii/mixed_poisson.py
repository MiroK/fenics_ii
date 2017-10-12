from dolfin import *
from block import block_mat, block_vec
from block.algebraic.petsc import AMG, ILU, Jacobi, Cholesky, LU
from block.iterative import MinRes

# We're solving min (sigma, sigma)*dx/2 subject to -div(sigma) = f

u_exact = Expression('sin(k*pi*x[0])*sin(l*pi*x[1])', k=3, l=2, degree=4)
sigma_exact = Expression(('k*pi*cos(k*pi*x[0])*sin(l*pi*x[1])',
                          'l*pi*sin(k*pi*x[0])*cos(l*pi*x[1])'),
                          k=u_exact.k, l=u_exact.l, degree=4)
f = Expression('((k*pi)*(k*pi)+(l*pi)*(l*pi))*u', u=u_exact, k=u_exact.k,
               l=u_exact.l, degree=4)

def main(n, Vdeg, Qdeg, prec):
    '''Mixed Poisson checking.'''
    mesh = UnitSquareMesh(n, n)

    V = FunctionSpace(mesh, 'BDM', Vdeg)  # sigma
    Q = FunctionSpace(mesh, 'DG', Qdeg)   # u
    W = [V, Q]

    sigma, u = map(TrialFunction, W)
    tau, v = map(TestFunction, W)

    # System
    a00 = inner(sigma, tau)*dx
    a01 = inner(u, div(tau))*dx
    a10 = inner(div(sigma), v)*dx
    L0 = inner(Constant((0, 0)), tau)*dx
    L1 = inner(-f, v)*dx

    A00 = assemble(a00)
    A01 = assemble(a01)
    A10 = assemble(a10)
    b0 = assemble(L0)
    b1 = assemble(L1)

    AA = block_mat([[A00, A01], [A10, 0]])
    bb = block_vec([b0, b1])

    # Preconditioner with Hdiv, L2
    if prec == 'Hdiv':
        p00 = inner(sigma, tau)*dx + inner(div(sigma), div(tau))*dx
        p11 = inner(u, v)*dx

        P00 = assemble(p00)
        P11 = assemble(p11)
        # Approx. inverses
        # See also
        # http://www.firedrakeproject.org/demos/saddle_point_systems.py.html 
        # NOTE: Does not really work for large meshes
        P00 = LU(P00)  
        P11 = ILU(P11)  
    # Preconditioner with L^2 and H^1
    else:
        p00 = inner(sigma, tau)*dx

        n = FacetNormal(mesh)
        h = CellSize(mesh)
        h_avg = avg(h)
        # DG0
        if Qdeg == 0:
            p11 = h_avg**(-1)*dot(jump(v, n), jump(u, n))*dS +\
                  h**(-1)*dot(u, v)*ds +\
                  inner(u, v)*dx
        # Higher order
        else:
            alpha = Constant(1)

            p11 = dot(grad(v), grad(u))*dx \
                  + alpha*h_avg**(-1)*dot(jump(v, n), jump(u, n))*dS \
                  + alpha*h**(-1)*dot(u, v)*ds \
                  + inner(u, v)*dx

        P00 = assemble(p00)
        P11 = assemble(p11)
        # Approx. inverses
        P00 = ILU(P00)  # Jacobi(P00)
        P11 = AMG(P11)  # AMG(P11)

    BB = block_mat([[P00, 0], [0, P11]])

    # Solve
    x = AA.create_vec()
    x.randomize()
    AAinv = MinRes(AA, precond=BB, show=2, tolerance=1E-10, 
                    initial_guess=x, maxiter=500)
    Sigma, U = AAinv * bb

    sigmah = Function(V, Sigma)
    uh = Function(Q, U)

    e_sigma = errornorm(sigma_exact, sigmah, 'Hdiv')
    e_u = errornorm(u_exact, uh, 'L2')
    h = mesh.hmin()

    niters = len(AAinv.residuals) - 1
    dims = (V.dim(), Q.dim())

    return niters, dims, e_sigma, e_u, h

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    Vdeg, Qdeg = 1, 0
    prec = 'Hdiv'

    temp = 'niters = %d, dim = %d+%d, sigma=%.4E(%.2f) u=%.4E(%.2f)'
    e0_sigma, e0_u, h0 = -1, -1, -1
    for n in [8, 16, 32, 64, 128, 256, 512]:
        niters, dims, e_sigma, e_u, h = main(n, Vdeg, Qdeg, prec)
        if e0_sigma > 0:
            rate_sigma = ln(e_sigma/e0_sigma)/ln(h/h0)
            rate_u = ln(e_u/e0_u)/ln(h/h0)
            msg = temp % (niters, dims[0], dims[1], e_sigma, rate_sigma, e_u, rate_u)
            print '\033[1;37;31m%s\033[0m' % msg
        e0_sigma, e0_u, h0 = e_sigma, e_u, h
