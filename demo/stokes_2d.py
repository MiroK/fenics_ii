# On [0, 1]^2 I consider
#
# -Delta u + u - grad(p) = f
#  div(u)                = 0
#
# With bcs
#
#  u = u0              on |_|
#  grad(u).n + p.n = 0 on top
#
# The latter bc is to avoid singular problem is u = u0 was presribed
# on the entire boundary

from dolfin import *
from xii import *


def setup_problem(i, (f, u0), eps=1.):
    '''Just showcase, no MMS (yet)'''
    # I setup the constants arbitraily
    n = 16*(2**i)

    mesh = UnitSquareMesh(n, n)

    gamma = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0]*(1-x[0]), 0) || near(x[1], 0)').mark(gamma, 1)
    bmesh = EmbeddedMesh(gamma, 1)

    normal = OuterNormal(bmesh, [0.5, 0.5])

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    # NOT CG1
    M = VectorFunctionSpace(bmesh, 'DG', 0, 2)
    W = (V, Q, M)

    u, p, lambda_ = map(TrialFunction, W)
    v, q, beta_ = map(TestFunction, W)

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
    L0 = inner(f, v)*dx
    L1 = inner(Constant(0), q)*dx
    L2 = inner(u0, beta_)*dxGamma

    a = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    L = [L0, L1, L2]

    return a, L, W


def setup_preconditioner(W, which, eps):
    '''The preconditioner'''
    from block.algebraic.petsc import AMG
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
    
    lambda_ = (((X - p*I)*sp.Matrix([-1, 0])).subs(x, 0),
               ((X - p*I)*sp.Matrix([0, -1])).subs(y, 0),
               ((X - p*I)*sp.Matrix([1, 0])).subs(x, 1))

    f = -sp_Div(sp_Grad(u)) + u - sp_grad(p)
    u0 = u
    
    up = map(as_expression, (u, p))
    fg = map(as_expression, (f, u0))
    
    return up + [map(as_expression, lambda_)], fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2, L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    # Now for the multiplier, since we really know the exact solution
    # only on pieces of the mesh, we will do things manually
    #
    def foo(u, uh):
        mesh = uh.function_space().mesh()
        cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        # Mark pieces
        CompiledSubDomain('near(x[1], 0)').mark(cell_f, 1)
        CompiledSubDomain('near(x[0], 1)').mark(cell_f, 2)

        dX = Measure('dx', domain=mesh, subdomain_data=cell_f)
        error = sum(inner(uj-uh, uj-uh)*dX(j) for j, uj in enumerate(u))
        return sqrt(assemble(error))

    return monitor_error(true, [H1_norm, L2_norm, foo], history, path=path)
