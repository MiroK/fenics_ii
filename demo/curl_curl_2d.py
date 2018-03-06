# H0.5 problem
# rot(curl(sigma)) + sigma = f in [0, 1]^2
#                  sigma.t = g on the boundary
#
# To be solved with Lagrange multiplier to enforce bcs rather then
# enforcing them on the function space level.

from dolfin import *
from xii import *


def solve_problem(i, (f, g)):
    '''curl-curl on [0, 1]^2'''
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    S = FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)    
    Q = FunctionSpace(bmesh, 'DG', 0)
    W = [S, Q]
    
    sigma, p = map(TrialFunction, W)
    tau, q = map(TestFunction, W)

    dxGamma = dx(domain=bmesh)        
    n_gamma = OuterNormal(bmesh, [0.5, 0.5])
    t_gamma = as_matrix(((0, 1), (-1, 0)))*n_gamma

    Tsigma = Trace(sigma, bmesh, '+', n_gamma)
    Ttau = Trace(tau, bmesh, '+', n_gamma)

    a00 = inner(curl(sigma), curl(tau))*dx + inner(sigma, tau)*dx
    a01 = inner(dot(Ttau, t_gamma), p)*dxGamma
    a10 = inner(dot(Tsigma, t_gamma), q)*dxGamma
    
    L0 = inner(f, tau)*dx
    L1 = inner(g, q)*dxGamma

    a = [[a00, a01], [a10, 0]]
    L = [L0, L1]

    return a, L, W

# --------------------------------------------------------------------

def setup_mms():
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp

    x, y = sp.symbols('x[0] x[1]')

    sigma = sp.Matrix([sp.sin(sp.pi*x*(1-x)*y*(1-y)),
                       sp.sin(2*sp.pi*x*(1-x)*y*(1-y))])

    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])

    sp_div = lambda f: f[0].diff(x, 1) + f[1].diff(y, 1)

    # This is a consistent with FEniCS definition
    ROT_MAT = sp.Matrix([[sp.S(0), sp.S(1)], [sp.S(-1), sp.S(0)]])

    # Maps vector to scalar: 
    sp_curl = lambda f: sp_div(ROT_MAT*f)

    # Maps scalar to vector
    sp_rot = lambda f: ROT_MAT*sp_grad(f)

    f = sp_rot(sp_curl(sigma)) + sigma
    g = sp.S(0)

    sigma_exact = as_expression(sigma)
    # It's quite nice that you get surface curl as the extra varp
    p_exact = as_expression(sp_curl(sigma))
    f_rhs, g_rhs = map(as_expression, (f, g))

    return (sigma_exact, p_exact), (f_rhs, g_rhs)


def setup_error_monitor(true, history):
    '''We measure error in Hcurl and L2 for simplicity'''
    from common import monitor_error, Hcurl_norm, L2_norm
    return monitor_error(true, [Hcurl_norm, L2_norm], history)
