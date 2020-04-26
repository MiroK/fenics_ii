#
# -div(sigma(u)) = f in Omega
#              u = g on boundary
#
# where sigma(u) = 2*mu*eps(u) + lmbda*div(u)*I
# 
# with the boundary conditions enforced weakly by Lagrange multiplier.
#
# Here solved in mixed formulation with lmbda*div(u) = p
from __future__ import absolute_import
from dolfin import *
from xii import *
from six.moves import map
from six.moves import range


def setup_problem(i, xxx_todo_changeme, eps=None):
    '''Elasticity on [0, 1]^2'''
    (f, g) = xxx_todo_changeme
    assert len(eps) == 2
    
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    P1 = FiniteElement('Lagrange', triangle, 1)
    B3 = FiniteElement('Bubble', triangle, 3)
    MINI = VectorElement(P1 + B3, 2)
    # Displacement
    V = FunctionSpace(mesh, MINI)
    # Pressure
    Q = FunctionSpace(mesh, P1)
    # Multiplier
    Y = VectorFunctionSpace(bmesh, 'CG', 1)
    W = [V, Q, Y]

    u, p, x = list(map(TrialFunction, W))
    v, q, y = list(map(TestFunction, W))
    Tu = Trace(u, bmesh)
    Tv = Trace(v, bmesh)

    # The line integral
    dx_ = Measure('dx', domain=bmesh)

    mu = Constant(eps[0])
    lmbda = Constant(eps[1])  # This is just some choice. Must be respected in MMS!

    a = [[0]*len(W) for _ in range(len(W))]
    a[0][0] = 2*mu*inner(sym(grad(u)), sym(grad(v)))*dx
    a[0][1] = inner(div(v), p)*dx
    a[0][2] = inner(Tv, x)*dx_
    
    a[1][0] = inner(div(u), q)*dx
    a[1][1] = -Constant(1./lmbda)*inner(p, q)*dx
    a[2][0] = inner(Tu, y)*dx_

    L0 = inner(f, v)*dx
    L1 = inner(Constant(0), q)*dx
    L2 = inner(g, y)*dx_

    L = [L0, L1, L2]

    return a, L, W


def setup_preconditioner(W, which, eps=None):
    '''
    This is a block diagonal preconditioner based on H1 x L2 x H^{-0.5}
    '''
    from block.algebraic.petsc import LumpedInvDiag
    from xii.linalg.matrix_utils import as_petsc
    from numpy import hstack
    from petsc4py import PETSc
    from hsmg import HsNorm
    
    V, Q, Y = W
    
    # H1
    u, v = TrialFunction(V), TestFunction(V)
    b00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    # NOTE: since interpolation is broken with MINI I don't interpolate
    # here the RM basis to attach the vectros to matrix
    A = assemble(b00)

    A = as_petsc(A)
    # Setup the preconditioner in petsc
    pc = PETSc.PC().create()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setOperators(A)
    # Other options
    opts = PETSc.Options()
    opts.setValue('pc_hypre_boomeramg_cycle_type', 'V')
    opts.setValue('pc_hypre_boomeramg_relax_type_all',  'symmetric-SOR/Jacobi')
    opts.setValue('pc_hypre_boomeramg_coarsen_type', 'Falgout')  
    pc.setFromOptions()         
    # Wrap for cbc.block
    B00 = BlockPC(pc)

    p, q = TrialFunction(Q), TestFunction(Q)
    B11 = LumpedInvDiag(assemble(inner(p, q)*dx))
    
    # The Y norm via spectral
    Yi = Y.sub(0).collapse()
    B22 = inverse(VectorizedOperator(HsNorm(Yi, s=-0.5), Y))

    return block_diag_mat([B00, B11, B22])


# --------------------------------------------------------------------


def setup_mms(eps=None):
    '''Simple MMS problem for UnitSquareMesh'''
    import sympy as sp
    import ulfy
    
    mesh = UnitSquareMesh(2, 2)

    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = Function(V)

    # Define as function to allow ufly substition
    S = FunctionSpace(mesh, 'DG', 0)
    mu = Function(S)
    lmbda = Function(S)

    sigma = lambda u: 2*mu*sym(grad(u)) + lmbda*div(u)*Identity(2)

    # The form
    f = -div(sigma(u))
    p = lmbda*div(u)  # Solid pressure

    # What we want to substitute
    x, y, mu_, lambda_  = sp.symbols('x y mu lmbda')
    # I chose u purposely such that sigma(u).n is zero on the boundary
    u_ = sp.Matrix([0.01*sp.cos(sp.pi*x*(1-x)*y*(1-y)),
                    -0.01*sp.cos(2*sp.pi*x*(1-x)*y*(1-y))])
    x_ = sp.Matrix([0, 0])

    subs = {u: u_, mu: mu_, lmbda: lambda_}  # Function are replaced by symbols
    # As expressions
    up = (ulfy.Expression(u_, degree=5),
          ulfy.Expression(p, subs=subs, degree=4, mu=eps[0], lmbda=eps[1]),
          ulfy.Expression(x_, degree=1))
    # Note lmbda_ being a constant is compiled into constant so errornorm
    # will complaing about the Expressions's degree being too low
    fg = (ulfy.Expression(f, subs=subs, degree=4, mu=eps[0], lmbda=eps[1]),
          ulfy.Expression(u_, degree=4, mu=eps[0], lmbda=eps[1]))  # mu, lmbda are are given value

    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    return monitor_error(true, [H1_norm, L2_norm, L2_norm], history, path=path)
