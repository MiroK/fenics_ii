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

from __future__ import absolute_import
from dolfin import *
from xii import *
from six.moves import map
from six.moves import range


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

    # Mini
    V = VectorFunctionSpace(mesh, 'CG', 1)
    Vb = VectorFunctionSpace(mesh, 'Bubble', 3)
    Q = FunctionSpace(mesh, 'CG', 1)
    # NOTE: DG0 okay for convergence but iterations blow up
    #       CG1 seems okay for both
    M = VectorFunctionSpace(bmesh, 'CG', 1, 2)
    W = (V, Vb, Q, M)

    u, ub, p, lambda_ = list(map(TrialFunction, W))
    v, vb, q, beta_ = list(map(TestFunction, W))
    # Not trace of bubbles; that's zero anyways
    T_u, T_v = Trace(u, bmesh), Trace(v, bmesh)

    dxGamma = Measure('dx', domain=bmesh)

    a = [[0]*len(W) for _ in range(len(W))]
    a[0][0] = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a[0][1] = inner(grad(ub), grad(v))*dx + inner(ub, v)*dx
    a[0][2] = inner(p, div(v))*dx
    a[0][3] = inner(T_v, lambda_)*dxGamma

    a[1][0] = inner(grad(u), grad(vb))*dx + inner(u, vb)*dx
    a[1][1] = inner(grad(ub), grad(vb))*dx + inner(ub, vb)*dx
    a[1][2] = inner(p, div(vb))*dx

    a[2][0] = inner(q, div(u))*dx
    a[2][1] = inner(q, div(ub))*dx
    
    a[3][0] = inner(T_u, beta_)*dxGamma

    n = FacetNormal(mesh)
    # NOTE: here I use the fact that on top and bottom the stress is 0
    L = [inner(f, v)*dx + inner(h, v)*ds(domain=mesh, subdomain_data=gamma, subdomain_id=3),
         inner(f, vb)*dx,
         inner(Constant(0), q)*dx,
         inner(u0, beta_)*dxGamma]

    return a, L, W


def setup_preconditioner(W, which, eps):
    '''The preconditioner'''
    from block.algebraic.petsc import AMG, LumpedInvDiag
    from block import block_transpose
    from hsmg import HsNorm
    
    u, ub, p, lambda_ = list(map(TrialFunction, W))
    v, vb, q, beta_ = list(map(TestFunction, W))

    # A block diagonal preconditioner
    if which == 0:
        b00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
        B00 = AMG(ii_assemble(b00))

        b11 = inner(grad(ub), grad(vb))*dx + inner(ub, vb)*dx
        B11 = LumpedInvDiag(ii_assemble(b11))

        b22 = inner(p, q)*dx
        B22 = AMG(ii_assemble(b22))

        M = W[-1]
        Mi = M.sub(0).collapse()
        B33 = inverse(VectorizedOperator(HsNorm(Mi, s=-0.5), M))

        return block_diag_mat([B00, B11, B22, B33])

    # Preconditioner monolithic in Stokes velocity
    b = [[0, 0], [0, 0]]
    b[0][0] = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    b[0][1] = inner(grad(ub), grad(v))*dx + inner(ub, v)*dx
    b[1][0] = inner(grad(u), grad(vb))*dx + inner(u, vb)*dx
    b[1][1] = inner(grad(ub), grad(vb))*dx + inner(ub, vb)*dx
    # Make into a monolithic matrix
    B00 = AMG(ii_convert(ii_assemble(b)))

    b11 = inner(p, q)*dx
    B11 = AMG(ii_assemble(b11))

    M = W[-1]
    Mi = M.sub(0).collapse()
    B22 = inverse(VectorizedOperator(HsNorm(Mi, s=-0.5), M))
    # So this is a 3x3 matrix 
    BB = block_diag_mat([B00, B11, B22])
    # But the preconditioner has to be 4x4
    R = ReductionOperator([2, 3, 4], W)

    return (R.T)*BB*R

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


def setup_transform(i, wh):
    '''Combine velocity with bubbles'''
    uh, ub_h, ph, lmbda_h = wh
    # NOTE: we don't look at the bubble contribution
    return [uh, ph, lmbda_h]
