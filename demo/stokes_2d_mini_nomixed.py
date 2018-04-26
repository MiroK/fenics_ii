# On [0, 1]^2 I consider
#
# -Delta u + u - grad(p) = f
#  div(u)                = 0
#
# With bcs
#
#  u = u0              on left
#  grad(u).n + p.n = 0 on top and bottom (A)
#  grad(u).n + p.n = h on right
#
# The latter bcs are to avoid singular problem is u = u0 was presribed
# on the entire boundary
#
# NOTE: The MMS here can be modified to use with Dirichlet on |_| and
# (A) condition on the top. This then becomes a much harded probelm to
# converged on - I suspect this is because the multiplier then is big
# time discontinuous in the corners! Still, regardless of the stup 

from dolfin import *
from xii import *


def setup_problem(i, (f, h, u0), eps=1.):
    '''Just showcase, no MMS (yet)'''
    # I setup the constants arbitraily
    n = 16*(2**i)

    mesh = UnitSquareMesh(n, n)

    gamma = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[1], 1)').mark(gamma, 0)
    CompiledSubDomain('near(x[0], 0)').mark(gamma, 1)
    CompiledSubDomain('near(x[1], 0)').mark(gamma, 2)
    CompiledSubDomain('near(x[0], 1)').mark(gamma, 3)

    bmesh = EmbeddedMesh(gamma, 1)

    normal = OuterNormal(bmesh, [0.5, 0.5])

    # Mini
    MiniElm = VectorElement('Lagrange', triangle, 1) + \
              VectorElement('Bubble', triangle, 3)
    V = FunctionSpace(mesh, MiniElm)
    Q = FunctionSpace(mesh, 'CG', 1)
    # NOTE: DG0 okay for convergence but iterations blow up
    #       CG1 seems okay for both
    M = VectorFunctionSpace(bmesh, 'CG', 1, 2)
    W = (V, Q, M)

    u, p, lambda_ = map(TrialFunction, W)
    v, q, beta_ = map(TestFunction, W)
    # Not trace of bubbles; that's zero anyways
    T_u, T_v = Trace(u, bmesh), Trace(v, bmesh)

    dxGamma = Measure('dx', domain=bmesh)

    a = [[0]*len(W) for _ in range(len(W))]
    a[0][0] = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a[0][1] = inner(p, div(v))*dx
    a[0][2] = inner(T_v, lambda_)*dxGamma

    a[1][0] = inner(q, div(u))*dx

    a[2][0] = inner(T_u, beta_)*dxGamma

    n = FacetNormal(mesh)
    # NOTE: here I use the fact that on top and bottom the stress is 0
    L = [inner(f, v)*dx + inner(h, v)*ds(domain=mesh, subdomain_data=gamma, subdomain_id=3),
         inner(Constant(0), q)*dx,
         inner(u0, beta_)*dxGamma]

    return a, L, W


def setup_preconditioner(W, which, eps):
    '''The preconditioner'''
    from block.algebraic.petsc import AMG, LumpedInvDiag
    from hsmg import HsNorm
    
    u, p, lambda_ = map(TrialFunction, W)
    v, q, beta_ = map(TestFunction, W)

    b00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    B00 = AMG(ii_assemble(b00))

    b11 = inner(p, q)*dx
    B11 = AMG(ii_assemble(b11))

    M = W[-1]
    Mi = M.sub(0).collapse()
    B22 = inverse(VectorizedOperator(HsNorm(Mi, s=-0.5), M))

    return block_diag_mat([B00, B11, B22])

# --------------------------------------------------------------------

def setup_mms(eps):
    '''Simple MMS...'''
    from common import as_expression
    import sympy as sp
    
    pi = sp.pi    
    x, y = sp.symbols('x[0] x[1]')
    
    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])

    sp_Grad = lambda f: sp.Matrix([[f[0].diff(x, 1), f[0].diff(y, 1)],
                                   [f[1].diff(x, 1), f[1].diff(y, 1)]])

    sp_div = lambda f: f[0].diff(x, 1) + f[1].diff(y, 1)
    
    sp_Div = lambda f: sp.Matrix([sp_div(f[0, :]), sp_div(f[1, :])])

    u = sp.Matrix([sp.cos(pi*y)*sp.sin(pi*x), -sp.cos(pi*x)*sp.sin(pi*y)])

    p = pi*sp.cos(pi*x)*sp.cos(pi*y)

    X = -sp_Grad(u)
    # The multiplier is -Grad(u).n - p.n
    I = sp.eye(2)
    
    lambda_ = ((X - p*I)*sp.Matrix([-1, 0])).subs(x, 0)
    #           ((X - p*I)*sp.Matrix([0, -1])).subs(y, 0),
    #           ((X - p*I)*sp.Matrix([1, 0])).subs(x, 1))

    h = -((X - p*I)*sp.Matrix([1, 0])).subs(x, 1)
    f = -sp_Div(sp_Grad(u)) + u - sp_grad(p)
    u0 = u
    
    up = map(as_expression, (u, p, lambda_))
    fg = map(as_expression, (f, h, u0))
    
    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2, L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    return monitor_error(true, [H1_norm, L2_norm, L2_norm], history, path=path)
