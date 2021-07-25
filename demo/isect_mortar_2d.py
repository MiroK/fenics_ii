# --------------------------
# |2                       |
# |    -------------       |
# |    |1          |       |
# |    |           |       |
# |    -------------       |
# --------------------------
#
# Consider
# In domain 1: -eps\Delta u1 + u1 = f1
# In domain 2: -\Delta u2 = f2
#
# On outer of 2 there is Neumann bcs  with grad(u2) = h2
# On the interface we consider
#
# u1 + eps*grad(u1).n1 + u2 + grad(u2).n2 = h
# u1 - u2 = g
#
# This is formulated with Lagrange multiplier p = u1 + eps*grad(u1).n1
# and the constraint u1 - u2 = g is enforced weakly
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

    inner_mesh = SubMesh(outer_mesh, subdomains, 1)
    outer_mesh = SubMesh(outer_mesh, subdomains, 0)

    # Outer boundary
    surfaces = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)    
    CompiledSubDomain('near(x[0]*(1-x[0]), 0) || near(x[1]*(1-x[1]), 0)').mark(surfaces, 1)
    outer_mesh.facet_f = surfaces
    
    # Interior boundary
    surfaces = MeshFunction('size_t', inner_mesh, inner_mesh.topology().dim()-1, 0)
    DomainBoundary().mark(surfaces, 1)
    
    gamma_mesh = EmbeddedMesh(surfaces, 1)

    return outer_mesh, inner_mesh, gamma_mesh


def setup_problem(i, xxx_todo_changeme, eps):
    '''EMI like problem with mortaring'''
    (f1, f2, h2, h, g) = xxx_todo_changeme
    n = 4*2**i

    outer_mesh, inner_mesh, gamma_mesh = setup_domain(n)
    # And now for the fun stuff
    V1 = FunctionSpace(inner_mesh, 'CG', 1)
    V2 = FunctionSpace(outer_mesh, 'CG', 1)
    Q = FunctionSpace(gamma_mesh, 'CG', 1)
    W = [V1, V2, Q]

    u1, u2, p = list(map(TrialFunction, W))
    v1, v2, q = list(map(TestFunction, W))

    dxGamma = Measure('dx', domain=gamma_mesh)
    ds_out = Measure('ds', domain=outer_mesh,
                           subdomain_data=outer_mesh.facet_f,
                           subdomain_id=1)
    n2 = FacetNormal(outer_mesh)
    
    # We will need traces of the functions on the boundary
    Tu1, Tu2 = [Trace(x, gamma_mesh) for x in (u1, u2)]
    Tv1, Tv2 = [Trace(x, gamma_mesh) for x in (v1, v2)]

    eps = eps[0]
    
    a00 = (Constant(eps)*inner(grad(u1), grad(v1))*dx +
           inner(u1, v1)*dx +
           inner(Tu1, Tv1)*dxGamma)
    a01 = 0
    a02 = -inner(p, Tv1)*dxGamma

    a10 = 0
    a11 = inner(grad(u2), grad(v2))*dx + inner(Tu2, Tv2)*dxGamma
    a12 = inner(p, Tv2)*dxGamma

    a20 = -inner(q, Tu1)*dxGamma
    a21 = inner(q, Tu2)*dxGamma
    a22 = 0

    L0 = inner(f1, v1)*dx
    L1 = inner(f2, v2)*dx + inner(dot(h2, n2), v2)*ds_out + inner(h, Tv2)*dxGamma
    L2 = -inner(g, q)*dxGamma

    a = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    L = [L0, L1, L2]
    
    return a, L, W


def setup_preconditioner(W, which, eps):
    '''
    TODO
    '''
    from block.algebraic.petsc import AMG
    from block.algebraic.petsc import LumpedInvDiag
    from hsmg import HsNorm
    
    V1, V2, Q = W

    gamma_mesh = Q.mesh()
    dxGamma = Measure('dx', domain=gamma_mesh)

    u1, u2 = list(map(TrialFunction, (V1, V2)))
    v1, v2 = list(map(TestFunction, (V1, V2)))
        
    Tu1, Tu2 = [Trace(x, gamma_mesh) for x in (u1, u2)]
    Tv1, Tv2 = [Trace(x, gamma_mesh) for x in (v1, v2)]

    if eps > 1:
        b00 = Constant(eps)*inner(grad(u1), grad(v1))*dx+inner(u1, v1)*dx
        B00 = ii_assemble(b00)
    else:
        b00 = Constant(eps)*inner(grad(u1), grad(v1))*dx+inner(u1, v1)*dx+inner(Tu1, Tv1)*dxGamma
        B00 = ii_convert(ii_assemble(b00))
    B00 = AMG(B00)

    b11 = inner(grad(u2), grad(v2))*dx + inner(u2, v2)*dx
    # Inverted by BoomerAMG
    B11 = AMG(ii_assemble(b11))

    if eps > 1:
        B22 = inverse(HsNorm(Q, s=-0.5))
    else:
        B22 = inverse(HsNorm(Q, s=0.5))
    
    return block_diag_mat([B00, B11, B22])

# --------------------------------------------------------------------

def setup_mms(eps):
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp
    
    pi = sp.pi
    x, y, EPS = sp.symbols('x[0] x[1] EPS')
    
    u1 = sp.cos(2*pi*(x-0.25)*(x-0.75)*(y-0.25)*(y-0.75))
    u2 = 2*sp.cos(pi*(x-0.25)*(x-0.75)*(y-0.25)*(y-0.75))
    # Note that u1 is such that grad(u1) is zero on the boudnary so
    # p = u1 only
    p = u1
    
    f1 = EPS*(-u1.diff(x, 2) - u1.diff(y, 2)) + u1
    f2 = -u2.diff(x, 2) - u2.diff(y, 2)

    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])
    # The neumann term
    h2 = sp_grad(u2)
    # The 'Robin' boundary condition simplifies as grad(u1).n1 = grad(u2).n2 = 0
    h = u1 + u2
    g = u1 - u2

    up = list(map(as_expression, (u1, u2, p)))
    fg = [as_expression(f1, EPS=eps[0])] + list(map(as_expression, (f2, h2, h, g)))

    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    # Note we produce u1, u2, and p error. It is more natural to have
    # broken H1 norm so reduce the first 2 errors to single number
    reduction = lambda e: None if e is None else [sqrt(e[0]**2 + e[1]**2), e[-1]]
    
    return monitor_error(true, [H1_norm, H1_norm, L2_norm], history, reduction, path=path)
