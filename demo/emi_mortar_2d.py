# The problem used for Trygve's paper
# H-0.5 problem
# With \Omega = [0, 1]^2 and \Omega_2 = [1/4, 3/4]^d the problem reads
#
#  -\Delta u_1 + u_1 = f_1 in \Omega \ \Omega_2=\Omega_1
#  -\Delta u_2 + u_2 = f_2 in \Omega_2
#  n1.grad(u_1) + n2.grad(u_2) = 0 on \partial\Omega_2=Gamma
#  eps(u1 - u2) + grad(u1).n1 = g on \Gamma
#  grad(u1).n1 = 0 in \partial\Omega_1

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
    # Interior boundary
    surfaces = MeshFunction('size_t', inner_mesh, inner_mesh.topology().dim()-1, 0)
    DomainBoundary().mark(surfaces, 1)
    
    gamma_mesh = EmbeddedMesh(surfaces, 1)

    return outer_mesh, inner_mesh, gamma_mesh


def setup_problem(i, xxx_todo_changeme, eps):
    '''EMI like problem with mortaring'''
    (f1, f2, g) = xxx_todo_changeme
    n = 4*2**i

    outer_mesh, inner_mesh, gamma_mesh = setup_domain(n)
    # And now for the fun stuff
    V1 = FunctionSpace(outer_mesh, 'CG', 1)
    V2 = FunctionSpace(inner_mesh, 'CG', 1)
    Q = FunctionSpace(gamma_mesh, 'CG', 1)
    W = [V1, V2, Q]

    u1, u2, p = list(map(TrialFunction, W))
    v1, v2, q = list(map(TestFunction, W))

    dxGamma = Measure('dx', domain=gamma_mesh)
    # We will need traces of the functions on the boundary
    Tu1, Tu2 = [Trace(x, gamma_mesh) for x in (u1, u2)]
    Tv1, Tv2 = [Trace(x, gamma_mesh) for x in (v1, v2)]

    a00 = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx
    a01 = 0
    a02 = inner(p, Tv1)*dxGamma

    a10 = 0
    a11 = inner(grad(u2), grad(v2))*dx + inner(u2, v2)*dx
    a12 = -inner(p, Tv2)*dxGamma

    a20 = inner(q, Tu1)*dxGamma
    a21 = -inner(q, Tu2)*dxGamma
    a22 = Constant(-1./eps)*inner(p, q)*dxGamma

    L0 = inner(f1, v1)*dx
    L1 = inner(f2, v2)*dx
    L2 = inner(g*Constant(1./eps), q)*dxGamma

    a = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    L = [L0, L1, L2]
    
    return a, L, W


def setup_preconditioner(W, which, eps):
    '''
    This is a block diagonal preconditioner based on 
    
        H1 x H1 x (H-0.5 \cap \sqrt{eps-1}*L2) or ...
    '''
    # NOTE for eps large the fractional term is expected to dominte
    from block.algebraic.petsc import AMG
    from block.algebraic.petsc import LumpedInvDiag
    from hsmg import HsNorm
    
    V1, V2, Q = W

    if which == 0:
        # H1
        print('\tUsing H1 x H1 x (sqrt(1./%g)*L2 \cap H-0.5) preconditioner' % eps)
        u1, v1 = TrialFunction(V1), TestFunction(V1)
        b00 = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx
        # Inverted by BoomerAMG
        B00 = AMG(ii_assemble(b00))

        u2, v2 = TrialFunction(V2), TestFunction(V2)
        b11 = inner(grad(u2), grad(v2))*dx + inner(u2, v2)*dx
        # Inverted by BoomerAMG
        B11 = AMG(ii_assemble(b11))
        
        # The Q norm via spectral, the norm is inverted exactly
        B22 = inverse((HsNorm(Q, s=-0.5) + (eps**-1)*HsNorm(Q, s=0.0)))
    else:
        print('\tUsing (H1 \cap H0.5) x (H1 \cap H0.5) x sqrt(%g)*L2 preconditioner' % eps)

        iface = Q.mesh()
        dxGamma = dx(domain=iface)
        # H1
        u1, v1 = TrialFunction(V1), TestFunction(V1)
        Tu1, Tv1 = Trace(u1, iface), Trace(v1, iface)
        
        b00 = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx + inner(Tu1, Tv1)*dxGamma
        # Inverted by BoomerAMG
        B00 = AMG(ii_convert(ii_assemble(b00)))

        u2, v2 = TrialFunction(V2), TestFunction(V2)
        Tu2, Tv2 = Trace(u2, iface), Trace(v2, iface)
        
        b11 = inner(grad(u2), grad(v2))*dx + inner(u2, v2)*dx + inner(Tu2, Tv2)*dxGamma
        # Inverted by BoomerAMG
        B11 = AMG(ii_convert(ii_assemble(b11)))
        
        # The Q norm via spectral
        p, q = TrialFunction(Q), TestFunction(Q)
        b22 = Constant(1./eps)*inner(p, q)*dxGamma
        B22 = LumpedInvDiag(ii_assemble(b22))

    return block_diag_mat([B00, B11, B22])

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
    up = list(map(as_expression, (u1, u2, sp.S(0))))  # The flux
    f = list(map(as_expression, (f1, f2)))
    g = as_expression(g, EPS=eps)  # Prevent recompilation

    return up, f+[g]


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    # Note we produce u1, u2, and p error. It is more natural to have
    # broken H1 norm so reduce the first 2 errors to single number
    reduction = lambda e: None if e is None else [sqrt(e[0]**2 + e[1]**2), e[-1]]
    
    return monitor_error(true, [H1_norm, H1_norm, L2_norm], history, reduction, path=path)
