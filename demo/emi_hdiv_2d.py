# On [0, 1] 
#    -Delta u1 + u1 = f1
#    -Delta u2 + u2 = f2
#    u1 = 0 on outer boundary
#
# On \partial [0.25, 0.75]^2
#    grad(u1).n1 + grad(u2).n2 = 0
#    eps*(u1 - u2) + grad(u1).n1 = g
#
# Solved in mixed form with sigma_i = -grad(u_i)
from __future__ import absolute_import
from dolfin import *
from xii import *
from six.moves import map


def setup_domains(n):
    '''On [0, 1] we want to mark domain inside and outside wrt interface'''    
    # Interface between interior and exteriorn domains
    gamma = CompiledSubDomain('near(std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)), 0.25)')
    # Marking interior domains
    interior = CompiledSubDomain('std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25 ? 1: 0')

    mesh = UnitSquareMesh(n, n)
    
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    # Interior is 1
    for cell in cells(mesh):
        subdomains[cell] = interior.inside(cell.midpoint().array(), False)
    mesh.subdomains = subdomains

    # Now the interface
    markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    gamma.mark(markers, 1)
    assert sum(1 for _ in SubsetIterator(markers, 1)) > 0

    bmesh = EmbeddedMesh(markers, 1)

    return mesh, bmesh


def setup_problem(i, xxx_todo_changeme, eps):
    '''EMI like problem without mortaring'''
    (f1, f2, g) = xxx_todo_changeme
    n = 4*2**i
    mesh, bmesh = setup_domains(n)

    S = FunctionSpace(mesh, 'RT', 1)        # sigma
    V = FunctionSpace(mesh, 'DG', 0)        # u
    Q = FunctionSpace(bmesh, 'DG', 0)       # p
    W = [S, V, Q]

    sigma, u, p = list(map(TrialFunction, W))
    tau, v, q = list(map(TestFunction, W))

    dX = Measure('dx', domain=mesh, subdomain_data=mesh.subdomains)
    dxGamma = dx(domain=bmesh)
        
    n_gamma = InnerNormal(bmesh, [0.5, 0.5])
    
    Tsigma = Trace(sigma, bmesh, '+', n_gamma)
    Ttau = Trace(tau, bmesh, '+', n_gamma)

    a00 = inner(sigma, tau)*dX(0) + inner(sigma, tau)*dX(1)
    a01 = -inner(u, div(tau))*dX
    a02 = inner(dot(Ttau, n_gamma), p)*dxGamma

    a10 = -inner(div(sigma), v)*dX
    a11 = -inner(u, v)*dX
    a12 = 0

    a20 = inner(dot(Tsigma, n_gamma), q)*dxGamma
    a21 = 0
    a22 = -Constant(eps)*inner(p, q)*dxGamma   

    L0 = inner(Constant((0, 0)), tau)*dx
    L1 = inner(-f1, v)*dX(0) + inner(-f2, v)*dX(1)
    L2 = inner(-g, q)*dxGamma

    a = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    L = [L0, L1, L2]

    return a, L, W


def setup_transform(i, wh):
    '''Break sigma, u, p into pieces on subdomains'''
    sigma_h, u_h, p_h = wh

    n = 4*2**i
    mesh, _ = setup_domains(n)
    outer_mesh = SubMesh(mesh, mesh.subdomains, 0)  # 1
    inner_mesh = SubMesh(mesh, mesh.subdomains, 1)  # 2

    # Reconstruct sigma on the subdomains
    sigma_elm = sigma_h.function_space().ufl_element()
    S1 = FunctionSpace(outer_mesh, sigma_elm)
    S2 = FunctionSpace(inner_mesh, sigma_elm)
    sigma1_h = interpolate(sigma_h, S1)
    sigma2_h = interpolate(sigma_h, S2)

    # Same for u
    u_elm = u_h.function_space().ufl_element()
    V1 = FunctionSpace(outer_mesh, u_elm)
    V2 = FunctionSpace(inner_mesh, u_elm)
    u1_h = interpolate(u_h, V1)
    u2_h = interpolate(u_h, V2)

    return [sigma1_h, sigma2_h, u1_h, u2_h, p_h]


def setup_preconditioner(W, which, eps):
    '''
    This is a block diagonal preconditioner based on 
    
        Hdiv x L2 x (H0.5 \cap sqrt(eps)*L2) or ? 
    '''
    from block.algebraic.petsc import LU
    from block.algebraic.petsc import LumpedInvDiag
    from hsmg import HsNorm
    
    S, V, Q = W

    # Hdiv
    sigma, tau = TrialFunction(S), TestFunction(S)
    b00 = inner(div(sigma), div(tau))*dx + inner(sigma, tau)*dx
    # Inverted exactly
    B00 = LU(ii_assemble(b00))

    # L2
    u, v = TrialFunction(V), TestFunction(V)
    b11 = inner(u, v)*dx
    # Inverted by BoomerAMG
    B11 = LumpedInvDiag(ii_assemble(b11))

    # The Q norm via spectral
    B22 = inverse(HsNorm(Q, s=0.5) + eps*HsNorm(Q, s=0.0))# The norm is inverted exactly

    return block_diag_mat([B00, B11, B22])


# --------------------------------------------------------------------

def setup_mms(eps):
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp
    
    pi = sp.pi    
    x, y, EPS = sp.symbols('x[0] x[1] EPS')
    
    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])

    u1 = sp.sin(2*pi*x)*sp.sin(2*pi*y)  # Zero at bdry, zero grad @ iface
    u2 = u1 + 1  # Zero grad @iface

    sigma1 = -sp_grad(u1)
    sigma2 = -sp_grad(u2)
    
    f1 = -u1.diff(x, 2) - u1.diff(y, 2) + u1
    f2 = -u2.diff(x, 2) - u2.diff(y, 2) + u2

    g = EPS*(u1 - u2) # + grad(u1).n1 # But the flux is 0

    up = list(map(as_expression, (sigma1, sigma2, u1, u2, u1 - u2)))
    # The last gut is the u1 trace value but here is is 0
    fg = list(map(as_expression, (f1, f2))) + [as_expression(g, EPS=eps)]
    
    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in Hdiv and L2 for simplicity'''
    from common import monitor_error, Hdiv_norm, L2_norm
    from math import sqrt
    # Note we produce sigma1, sigma2 u1, u2, and p error. But this is just
    # so that it is easier to compute error as the true solution is
    # discontinuous. So we compute on subdomain and then reduce
    reduction = lambda e: None if e is None else [sqrt(e[0]**2 + e[1]**2),  # Hdiv
                                                  sqrt(e[2]**2 + e[3]**2),  # L2
                                                  e[-1]]

    return monitor_error(true,
                         [Hdiv_norm, Hdiv_norm, L2_norm, L2_norm, L2_norm],
                         history,
                         reduction,
                         path=path)
