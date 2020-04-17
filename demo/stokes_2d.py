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


def setup_problem(i, xxx_todo_changeme, eps=1.):
    '''Just showcase, no MMS (yet)'''
    (f, h, u0) = xxx_todo_changeme
    n = 16*(2**i)

    mesh = UnitSquareMesh(n, n)

    gamma = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[1], 1)').mark(gamma, 0)
    CompiledSubDomain('near(x[0], 0)').mark(gamma, 1)
    CompiledSubDomain('near(x[1], 0)').mark(gamma, 2)
    CompiledSubDomain('near(x[0], 1)').mark(gamma, 3)

    bmesh = EmbeddedMesh(gamma, 1)

    normal = OuterNormal(bmesh, [0.5, 0.5])

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    # NOTE: CG1 here and fine mesh does not converge with umfpack
    M = VectorFunctionSpace(bmesh, 'CG', 1, 2)
    W = (V, Q, M)

    u, p, lambda_ = list(map(TrialFunction, W))
    v, q, beta_ = list(map(TestFunction, W))

    T_u, T_v = Trace(u, bmesh), Trace(v, bmesh)

    dxGamma = Measure('dx', domain=bmesh)

    a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a01 = inner(p, div(v))*dx
    a02 = inner(T_v, lambda_)*dxGamma
    
    a10 = inner(q, div(u))*dx
    a11 = 0
    a12 = 0

    a20 = inner(T_u, beta_)*dxGamma
    a21 = 0
    a22 = 0

    n = FacetNormal(mesh)
    # NOTE: here I use the fact that on top and bottom the stress is 0
    L0 = inner(f, v)*dx + inner(h, v)*ds(domain=mesh, subdomain_data=gamma, subdomain_id=3)
    L1 = inner(Constant(0), q)*dx
    L2 = inner(u0, beta_)*dxGamma

    a = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    L = [L0, L1, L2]

    return a, L, W


def setup_preconditioner(W, which, eps):
    '''The preconditioner'''
    from block.algebraic.petsc import AMG
    from hsmg import HsNorm
    
    u, p, lambda_ = list(map(TrialFunction, W))
    v, q, beta_ = list(map(TestFunction, W))

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
    
    up = list(map(as_expression, (u, p, lambda_)))
    fg = list(map(as_expression, (f, h, u0)))
    
    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2, L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    return monitor_error(true, [H1_norm, L2_norm, L2_norm], history, path=path)
