#
# -div(sigma(u)) = f in Omega
#              u = g on boundary
#
# where sigma(u) = 2*mu*eps(u) + lmbda*div(u)*I
#
# with the boundary conditions enforced weakly by Lagrange multiplier.
from dolfin import *
from xii import *


def setup_problem(i, (f, g), eps=None):
    '''Elasticity on [0, 1]^2'''
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    V = VectorFunctionSpace(mesh, 'CG', 1)
    Q = VectorFunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
    Tu = Trace(u, bmesh)
    Tv = Trace(v, bmesh)

    # The line integral
    dx_ = Measure('dx', domain=bmesh)

    mu = Constant(eps)
    lmbda = Constant(2*eps)  # This is just some choice. Must be respected in MMS!

    sigma = lambda u: 2*mu*sym(grad(u)) + lmbda*div(u)*Identity(2)
    
    a00 = inner(sigma(u), sym(grad(v)))*dx
    a01 = inner(Tv, p)*dx_
    a10 = inner(Tu, q)*dx_

    L0 = inner(f, v)*dx
    L1 = inner(g, q)*dx_

    a = [[a00, a01], [a10, 0]]
    L = [L0, L1]

    return a, L, W


def setup_preconditioner(W, which, eps=None):
    '''
    This is a block diagonal preconditioner based on 
    
        H1 x H-0.5 or (H1 \cap H0.5) x L2
    '''
    from block.algebraic.petsc import AMG
    from block.algebraic.petsc import LumpedInvDiag
    from hsmg import HsNorm
    
    V, Q = W

    # H1
    u, v = TrialFunction(V), TestFunction(V)
    b00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    # Inverted by BoomerAMG
    B00 = AMG(ii_assemble(b00))
    # The Q norm via spectral
    Qi = Q.sub(0).collapse()
    B11 = inverse(VectorizedOperator(HsNorm(Qi, s=-0.5), Q))

    return block_diag_mat([B00, B11])


# --------------------------------------------------------------------


def setup_mms(eps=None):
    '''Simple MMS problem for UnitSquareMesh'''
    import sympy as sp
    import ulfy
    
    mesh = UnitSquareMesh(2, 2)

    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = Function(V)

    mu = Constant(eps)
    lmbda = Constant(2*eps)  # This is just some choice. Must be respected in MMS!

    sigma = lambda u: 2*mu*sym(grad(u)) + lmbda*div(u)*Identity(2)

    # The form
    f = -div(sigma(u))

    # What we want to substitute
    x, y  = sp.symbols('x, y')
    # I chose u purposely such that sigma(u).n is zero on the boundary
    u_ = sp.Matrix([0.01*sp.cos(sp.pi*x*(1-x)*y*(1-y)),
                    -0.01*sp.cos(2*sp.pi*x*(1-x)*y*(1-y))])
    lmbda_ = sp.Matrix([0, 0])

    # As expressions
    up = (ulfy.Expression(u_, degree=5),
          ulfy.Expression(lmbda_, degree=1))
    # Note lmbda_ being a constant is compiled into constant so errornorm
    # will complaing about the Expressions's degree being too low
    fg = (ulfy.Expression(f, subs={u: u_, mu: mu(0), lmbda: lmbda(0)}, degree=4),
          ulfy.Expression(u_, degree=4))

    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    return monitor_error(true, [H1_norm, L2_norm], history, path=path)
