# On [0, 1]^2 I consider
#
# -eps*Delta u + u - grad(p) = f
#  div(u)                = 0
#
# With bcs
#
#  u x n = 0 and p = p0
#

from __future__ import absolute_import
from dolfin import *
from xii import *
from six.moves import map


def setup_problem(i, xxx_todo_changeme, eps):
    '''Just showcase, no MMS (yet)'''
    (f, p0) = xxx_todo_changeme
    n = 16*(2**i)

    mesh = UnitSquareMesh(n, n)

    gamma = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    DomainBoundary().mark(gamma, 1)
    bmesh = EmbeddedMesh(gamma, 1)

    normal = OuterNormal(bmesh, [0.5, 0.5])
    tangent = Constant(((0, 1), (-1, 0)))*normal

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = FunctionSpace(bmesh, 'CG', 1)
    W = (V, Q, M)

    u, p, lambda_ = list(map(TrialFunction, W))
    v, q, beta_ = list(map(TestFunction, W))

    T_u, T_v = Trace(u, bmesh), Trace(v, bmesh)

    dxGamma = Measure('dx', domain=bmesh)

    a00 = Constant(eps)*inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a01 = inner(p, div(v))*dx
    a02 = inner(T_v, lambda_*tangent)*dxGamma
    
    a10 = inner(q, div(u))*dx
    a11 = 0
    a12 = 0

    a20 = inner(T_u, beta_*tangent)*dxGamma
    a21 = 0
    a22 = 0

    n = FacetNormal(mesh)
    L0 = inner(f, v)*dx + inner(p0, dot(v, n))*ds
    L1 = inner(Constant(0), q)*dx
    L2 = inner(Constant(0), beta_)*dxGamma

    a = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    L = [L0, L1, L2]

    return a, L, W


def setup_preconditioner(W, which, eps):
    '''The preconditioner'''
    from block.algebraic.petsc import AMG
    from hsmg import HsNorm
    
    u, p, lambda_ = list(map(TrialFunction, W))
    v, q, beta_ = list(map(TestFunction, W))

    b00 = Constant(eps)*inner(grad(u), grad(v))*dx + inner(u, v)*dx
    B00 = AMG(ii_assemble(b00))

    b11a = Constant(1./eps)*inner(p, q)*dx

    b11b = inner(grad(p), grad(q))*dx
    bcs = DirichletBC(W[1], Constant(0), 'on_boundary')
    B11b, _ = assemble_system(b11b, inner(Constant(0), q)*dx, bcs)
    
    B11 = AMG(ii_assemble(b11a)) + AMG(B11b)

    B22 = inverse((1./eps)*HsNorm(W[-1], s=-0.5))

    return block_diag_mat([B00, B11, B22])

# --------------------------------------------------------------------

def setup_mms(eps):
    '''Simple MMS...'''
    from common import as_expression
    import sympy as sp
    
    pi = sp.pi    
    x, y, EPS = sp.symbols('x[0] x[1] EPS')
    
    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])

    sp_Grad = lambda f: sp.Matrix([[f[0].diff(x, 1), f[0].diff(y, 1)],
                                   [f[1].diff(x, 1), f[1].diff(y, 1)]])

    sp_div = lambda f: f[0].diff(x, 1) + f[1].diff(y, 1)
    
    sp_Div = lambda f: sp.Matrix([sp_div(f[0, :]), sp_div(f[1, :])])

    u = sp.Matrix([sp.sin(pi*y)*sp.cos(pi*x), -sp.cos(pi*y)*sp.sin(pi*x)])

    X = -EPS*sp_Grad(u)
    # I think the multiplier is the tangential component of X.n
    # so now we circulate the boundaries       0
    #                                         1 3
    #                                          2
    lambda_ = ((X.subs(y, 1)*sp.Matrix([0, 1]))[0],
               (X.subs(x, 0)*sp.Matrix([-1, 0]))[1],
               (X.subs(y, 0)*sp.Matrix([0, 1]))[0],
               -(X.subs(x, 1)*sp.Matrix([1, 0]))[1])
    
    p = sp.sin(pi*((x-0.5)**2 + (y-0.5)**2))
    # EPS * 1./EPS
    f = -EPS*sp_Div(sp_Grad(u)) + u - sp_grad(p)
    p0 = p
    
    up = [as_expression(u, EPS=eps), as_expression(p, EPS=eps)]
    fg = [as_expression(f, EPS=eps), as_expression(p0)]
    
    return up + [[as_expression(l, EPS=eps) for l in lambda_]], fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in L2 and L2, L2 for simplicity'''
    # L2 on velocity because eps -> 0 H1 does not make sense
    from common import monitor_error, H1_norm, L2_norm
    # Now for the multiplier, since we really know the exact solution
    # only on pieces of the mesh, we will do things manually
    #
    def foo(u, uh):
        mesh = uh.function_space().mesh()
        cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        # Mark pieces
        # CompiledSubDomain('near(x[1], 1)').mark(cell_f, 0)
        CompiledSubDomain('near(x[0], 0)').mark(cell_f, 1)
        CompiledSubDomain('near(x[1], 0)').mark(cell_f, 2)
        CompiledSubDomain('near(x[0], 1)').mark(cell_f, 3)

        dX = Measure('dx', domain=mesh, subdomain_data=cell_f)
        error = sum(inner(uj-uh, uj-uh)*dX(j) for j, uj in enumerate(u))
        return sqrt(assemble(error))

    return monitor_error(true, [L2_norm, L2_norm, foo], history, path=path)
