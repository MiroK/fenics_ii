from dolfin import *
from xii import *


def setup_domain(n):
    '''
    Inner is [0.25, 0.75]^2, inner is [0, 1]^2 \ [0.25, 0.75]^2 and 
    \partial [0.25, 0.75]^2 is the interface
    '''
    interior = CompiledSubDomain('std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25')
    outer_mesh = UnitSquareMesh(n, n)
    
    subdomains = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim(), 0)
    # Awkward marking
    for cell in cells(outer_mesh):
        x = cell.midpoint().array()            
        subdomains[cell] = int(interior.inside(x, False))
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    return mortar_meshes(subdomains, (0, 1))


def setup_problem(i, (f1, f2), eps=1.):
    '''TODO'''
    n = 16*2**i

    (stokes_domain, darcy_domain), iface_domain, _ = setup_domain(n)
    # And now for the fun stuff
    V1 = VectorFunctionSpace(stokes_domain, 'CG', 2)
    Q1 = FunctionSpace(stokes_domain, 'CG', 1)
    V2 = FunctionSpace(darcy_domain, 'RT', 1)
    Q2 = FunctionSpace(darcy_domain, 'DG', 0)
    M = FunctionSpace(iface_domain, 'DG', 0)
    W = [V1, Q1, V2, Q2, M]

    u1, p1, u2, p2, lambda_ = map(TrialFunction, W)
    v1, q1, v2, q2, beta_ = map(TestFunction, W)
    
    dxGamma = Measure('dx', domain=iface_domain)
    # We will need traces of the functions on the boundary
    Tu1, Tu2 = map(lambda x: Trace(x, iface_domain), (u1, u2))
    Tv1, Tv2 = map(lambda x: Trace(x, iface_domain), (v1, v2))

    n2 = OuterNormal(iface_domain, [0.5, 0.5])  # Outer of Darcy
    n1 = -n2                                  # Outer of Stokes
    # Get tangent vector
    tau1 = Constant(((0, 1), (-1, 0)))*n1

    a = [[0]*5 for i in range(len(W))]
    
    a[0][0] = (inner(sym(grad(u1)), sym(grad(v1)))*dx +
               inner(u1, v1)*dx +
               inner(dot(Tu1, tau1), dot(Tv1, tau1))*dxGamma)
    a[0][1] = -inner(p1, div(v1))*dx
    a[0][4] = inner(lambda_, dot(Tv1, n1))*dxGamma

    a[1][0] = -inner(q1, div(u1))*dx

    a[2][2] = inner(u2, v2)*dx
    a[2][3] = -inner(p2, div(v2))*dx
    a[2][4] = inner(lambda_, dot(Tv2, n2))*dxGamma

    a[3][2] = -inner(q2, div(u2))*dx

    a[4][0] = inner(beta_, dot(Tu1, n1))*dxGamma 
    a[4][2] = inner(beta_, dot(Tu2, n2))*dxGamma

    L = [0]*len(W)
    L[0] = inner(f1, v1)*dx
    L[1] = inner(Constant(0), p1)*dx
    L[2] = inner(Constant((0, 0)), v2)*dx
    L[3] = inner(f2, q2)*dx
    L[4] = inner(Constant(0), beta_)*dxGamma
    
    return a, L, W


def setup_preconditioner(W, which, eps):
    from block.algebraic.petsc import AMG
    from block.algebraic.petsc import LumpedInvDiag, LU
    from hsmg import HsNorm

    u1, p1, u2, p2, lambda_ = map(TrialFunction, W)
    v1, q1, v2, q2, beta_ = map(TestFunction, W)

    b00 = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx
    B00 = AMG(ii_assemble(b00))

    b11 = inner(p1, q1)*dx
    B11 = AMG(ii_assemble(b11))

    b22 = inner(div(u2), div(v2))*dx + inner(u2, v2)*dx
    B22 = LU(ii_assemble(b22))

    b33 = inner(p2, q2)*dx
    B33 = LumpedInvDiag(ii_assemble(b33))

    B44 = inverse(HsNorm(W[-1], s=0.5))
    
    return block_diag_mat([B00, B11, B22, B33, B44])

# --------------------------------------------------------------------

def setup_mms(eps):
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp
    
    pi = sp.pi
    x, y, EPS = sp.symbols('x[0] x[1] EPS')
    
    u1 = sp.cos(4*pi*x)*sp.cos(4*pi*y)
    u2 = 2*u1

    f1 = -u1.diff(x, 2) - u1.diff(y, 2) + u1
    f2 = -u2.diff(x, 2) - u2.diff(y, 2) + u2
    g = (u1 - u2)*EPS  
    # NOTE: the multiplier is grad(u).n and with the chosen data this
    # means that it's zero on the interface
    up = map(as_expression, (u1, u2, sp.S(0)))  # The flux
    f = map(as_expression, (f1, f2))
    g = as_expression(g, EPS=eps)  # Prevent recompilation

    return up, [Constant((0, 0)), as_expression((x-0.5)**2+(y-0.5)**2)]


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    # Note we produce u1, u2, and p error. It is more natural to have
    # broken H1 norm so reduce the first 2 errors to single number
    reduction = lambda e: None if e is None else [sqrt(e[0]**2 + e[1]**2), e[-1]]
    
    return monitor_error([], [], history, path=path)
