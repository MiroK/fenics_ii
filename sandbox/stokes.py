from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from xii import *
from six.moves import map
from six.moves import zip

# We solve the Stokes problem on a unit quare
#
# -div(sigma) = f where sigma = grad(u) - pI
#      div(u) = 0
#
# sigma.n = h on {x = 1}
#       u = u0 on {y = 1 | y = 0}
#
# u.n = u0.n and u.t + t.sigma.n = g on {x = 0}
#
# The final bc is not so standard. We bake the tangential part into the
# weak form. Condition on u.n is enforced by Lagrange multiplier

GREEN = '\033[1;37;32m%s\033[0m'
RED = '\033[1;37;31m%s\033[0m'


def test(n, data, check_stab=0):
    '''Return solution'''
    omega = UnitSquareMesh(n, n)
    boundaries = MeshFunction('size_t', omega, 1, 0)
    DomainBoundary().mark(boundaries, 1)
    # Dirichlet Domain
    CompiledSubDomain('near(x[1]*(1-x[1]), 0)').mark(boundaries, 1)
    # Neumann
    CompiledSubDomain('near(x[0], 1)').mark(boundaries, 2)    
    # LM domain
    CompiledSubDomain('near(x[0], 0)').mark(boundaries, 3)
    gamma = EmbeddedMesh(boundaries, 3)

    V = VectorFunctionSpace(omega, 'CG', 2)
    Q = FunctionSpace(omega, 'CG', 1)
    # This should be a stable tripplet: DG0, P1, P2
    Y = FunctionSpace(gamma, 'DG', 0)  
    W = [V, Q, Y]

    u, p, x = list(map(TrialFunction, W))
    v, q, y = list(map(TestFunction, W))

    Tu, Tv = Trace(u, gamma), Trace(v, gamma)
    # Normal and volume measure of LM
    n = Constant((-1, 0))
    tau = Constant((0, 1))  # Tangent
    dx_ = Measure('dx', domain=gamma)
    # For Neumann term
    ds_ = Measure('ds', domain=omega, subdomain_data=boundaries)
    
    a = block_form(W, 2)
    a[0][0] = inner(grad(u), grad(v))*dx + inner(dot(Tu, tau), dot(Tv, tau))*dx_
    a[0][1] = inner(-p, div(v))*dx
    a[0][2] = inner(dot(Tv, n), x)*dx_
    a[1][0] = inner(-q, div(u))*dx
    a[2][0] = inner(dot(Tu, n), y)*dx_

    L = block_form(W, 1)
    L[0] = inner(data['f'], v)*dx + inner(dot(Tv, tau), data['g'])*dx_
    # Neumann bit
    L[0] += inner(data['h'], v)*ds_(2)
    L[2] = inner(dot(data['u'], n), y)*dx

    if Y.ufl_element().family() == 'Discontinuous Lagrange':
        assert Y.ufl_element().degree() == 0
        Y_bcs = []
    else:
        # NOTE: this are necessary to obtain a non-singular problem.
        # However, the bdry values for LM are in general not part of the
        # problem data so this bcs hurts convergence
        Y_bcs = [DirichletBC(Y, Constant(0), 'on_boundary')]
        
    W_bcs = [[DirichletBC(V, data['u'], boundaries, 1)],
             [],
             Y_bcs]

    A, b = list(map(ii_assemble, (a, L)))
    A, b = apply_bc(A, b, W_bcs)

    # Check inf-sub stability
    # We do it directly so this can get costly
    if check_stab and sum(Wi.dim() for Wi in W) < 8000:
        # Get the norms for stable problem
        B00 = A[0][0]
        B11 = assemble(inner(p, q)*dx)

        from hsmg.hseig import Hs0Norm
        from scipy.linalg import eigvalsh
        
        if Y_bcs:
            B22 = Hs0Norm(Y, s=-0.5, bcs=Y_bcs)
        else:
            B22 = Hs0Norm(Y, s=-0.5, bcs=True)
        B22*(B22.create_vec())
        B22 = B22.matrix

        B = block_diag_mat([B00, B11, B22])

        A, B = list(map(ii_convert, (A, B)))
        print(RED % ('Solving for %d eigenvalues' % A.size(0)))
        lmin, lmax = np.sort(np.abs(eigvalsh(A.array(), B.array())))[[0, -1]]
        print(GREEN % ('lmin = %g, lmax = %g, cond = %g' % (lmin, lmax, lmax/lmin)))

    wh = ii_Function(W)
    solve(ii_convert(A), wh.vector(), ii_convert(b))
    
    return omega.hmin(), wh

# -------------------------------------------------------------------

if __name__ == '__main__':
    import sympy as sp
    from calculus import Grad, Div, asExpr
    from itertools import repeat
    import numpy as np
    
    # Setup mms    
    x, y = sp.symbols('x[0], x[1]')
    # Divergence free velocity
    u = sp.Matrix([sp.sin(y),
                   sp.cos(x)])
    # Some pressure
    p = sp.sin(x**2 + y**2)
    
    sigma = Grad(u) - p*sp.eye(2)
    f = -Div(sigma)
    # Bdry data on {x=0} with normal and tangent
    normal, tangent = sp.Matrix([-1, 0]), sp.Matrix([0, 1])
    # The condition u.n = g has
    # The one on tangent is
    g = u.dot(tangent) +  tangent.dot(sigma.dot(normal))
    # Full neumann
    h = sp.Matrix(sigma.dot(sp.Matrix([1, 0])))

    data = {'f': asExpr(f),  # Forcing
            'g': asExpr(g),  # Constraint on u.t
            'h': asExpr(h),  # Neumann
            'u': asExpr(u),  # Dirichlet; exact solution u
            'p': asExpr(p),  # Exact pressure and LM
            'x': asExpr(-normal.dot(sigma.dot(normal)))}

    table = []
    h0, errors0 = None, None
    for n in (1, 2, 4, 8, 16, 32):
        h, (uh, ph, xh) = test(n, data, check_stab=8000)
        errors = np.array([errornorm(data['u'], uh, 'H1'),
                           errornorm(data['p'], ph, 'L2'),
                           errornorm(data['x'], xh, 'L2')])

        if errors0 is not None:
            rates = np.log(errors/errors0)/np.log(h/h0)
        else:
            rates = repeat(-1)

        table.append(sum(list(zip(errors, rates)), ()))
        errors0, h0 = errors, h

    # Review
    print()
    fmt = '\t'.join(['%.2E(%.2f)']*3)
    for row in table:
        print(fmt % tuple(row))
