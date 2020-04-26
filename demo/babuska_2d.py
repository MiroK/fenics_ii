# The `Hello World` of fractional problems:
# H-0.5 problem
#
# -Delta u + u = f in Omega
#            u = g on boundary
#
# with the boundary conditions enforced weakly by Lagrange multiplier.
from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from xii import *
from six.moves import map


def setup_problem(i, xxx_todo_changeme, eps=None):
    '''Babuska on [0, 1]^2'''
    (f, g) = xxx_todo_changeme
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = [V, Q]

    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))
    Tu = Trace(u, bmesh)
    Tv = Trace(v, bmesh)

    # The line integral
    dx_ = Measure('dx', domain=bmesh)

    a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a01 = inner(Tv, p)*dx_
    a10 = inner(Tu, q)*dx_

    L0 = inner(f, v)*dx
    L1 = inner(g, q)*dx_

    a = [[a00, a01], [a10, 0]]
    L = [L0, L1]

    return a, L, W


def setup_preconditioner(W, which, eps=None):
    '''
    This is a block diagonal preconditioner based on 
    
        H1 x H-0.5 or (H1 \cap H0.5) x L2
    '''
    from block.algebraic.petsc import AMG
    from block.algebraic.petsc import LumpedInvDiag
    from hsmg.hseig import HsNorm
    
    V, Q = W

    if which == 0:
        print('Using H1 x H-0.5 preconditioner')
        # H1
        u, v = TrialFunction(V), TestFunction(V)
        b00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
        # Inverted by BoomerAMG
        B00 = AMG(ii_assemble(b00))
        # The Q norm via spectral
        B11 = inverse(HsNorm(Q, s=-0.5))  # The norm is inverted exactly
    else:
        print('Using (H1 \cap H0.5) x L2 preconditioner')
        bdry = Q.mesh()
        dxGamma = dx(domain=bdry)
        # Cap space
        u, v = TrialFunction(V), TestFunction(V)
        Tu, Tv = Trace(u, bdry), Trace(v, bdry)
        b00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx + inner(Tu, Tv)*dxGamma
        # Inverted by BoomrAMG
        B00 = AMG(ii_convert(ii_assemble(b00)))

        # We don't have to work so hard for multiplier
        p, q = TrialFunction(Q), TestFunction(Q)
        B11 = LumpedInvDiag(ii_assemble(inner(p, q)*dx))

    return block_diag_mat([B00, B11])


# --------------------------------------------------------------------


def setup_mms(eps=None):
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp
    
    x, y  = sp.symbols('x[0], x[1]')
    u = sp.cos(sp.pi*x*(1-x)*y*(1-y))
    p = sp.S(0)  # Normal stress is the multiplier, here it is zero

    f = -u.diff(x, 2) - u.diff(y, 2) + u
    g = u

    up = list(map(as_expression, (u, p)))
    fg = list(map(as_expression, (f, g)))

    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    return monitor_error(true, [H1_norm, L2_norm], history, path=path)
