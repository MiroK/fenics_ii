# H0.5 problem   
#    -grad(div(sigma)) + sigma = f in Omega
#                      sigma.n = g on the boundary
#   
# To be solved with Lagrange multiplier to enforce bcs rather then
# enforcing them on the function space level.

from dolfin import *
from xii import *


def solve_problem(i, (f, g)):
    '''grad-div on [0, 1]^2'''
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    S = FunctionSpace(mesh, 'RT', 1)
    Q = FunctionSpace(bmesh, 'DG', 0)
    W = [S, Q]
    
    sigma, p = map(TrialFunction, W)
    tau, q = map(TestFunction, W)

    dxGamma = dx(domain=bmesh)        
    n_gamma = OuterNormal(bmesh, [0.5, 0.5])

    Tsigma = Trace(sigma, bmesh, '+', n_gamma)
    Ttau = Trace(tau, bmesh, '+', n_gamma)
    
    a00 = inner(div(sigma), div(tau))*dx + inner(sigma, tau)*dx
    a01 = inner(dot(Ttau, n_gamma), p)*dxGamma
    a10 = inner(dot(Tsigma, n_gamma), q)*dxGamma

    L0 = inner(f, tau)*dx
    L1 = inner(g, q)*dxGamma

    a = [[a00, a01], [a10, 0]]
    L = [L0, L1]
    
    return a, L, W


def setup_preconditioner(W, which):
    '''
    This is a block diagonal preconditioner based on Hdiv x H0.5'''
    from block.algebraic.petsc import LU
    from hsmg import HsNorm
    
    S, Q = W

    sigma, tau = TrialFunction(S), TestFunction(S)
    # Hdiv
    b00 = inner(div(sigma), div(tau))*dx + inner(sigma, tau)*dx
    B00 = LU(ii_assemble(b00))  # Exact
    # H0.5
    B11 = HsNorm(Q, s=0.5)**-1  # The norm is inverted exactly

    return block_diag_mat([B00, B11])


# --------------------------------------------------------------------


def setup_mms():
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp
    
    x, y = sp.symbols('x[0] x[1]')

    sigma = sp.Matrix([sp.sin(sp.pi*x*(1-x)*y*(1-y)),
                       sp.sin(2*sp.pi*x*(1-x)*y*(1-y))])

    sp_div = lambda f: f[0].diff(x, 1) + f[1].diff(y, 1)

    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])

    f = -sp_grad(sp_div(sigma)) + sigma
    g = sp.S(0)

    sigma_exact = as_expression(sigma)
    # It's quite interesting that you get surface divergence as the extra var
    p_exact = as_expression(sp_div(-sigma)) 
    f_rhs, g_rhs = map(as_expression, (f, g))

    return (sigma_exact, p_exact), (f_rhs, g_rhs)


def setup_error_monitor(true, history):
    '''We measure error in Hdiv and L2 for simplicity'''
    from common import monitor_error, Hdiv_norm, L2_norm
    return monitor_error(true, [Hdiv_norm, L2_norm], history)
