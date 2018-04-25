# |u - f|^2{L^2(\partial\Omega)} + alpha/2*|p|_{L^2(\Omega)}
# 
# subject to
#
# -\Delta u + u + f = 0 in Omega
#         grad(u).n = 0 on \partial\Omega
#
from dolfin import *
from xii import *


def setup_problem(i, f, eps=None):
    '''Optim problem on [0, 1]^2'''
    n = 4*2**i
    mesh = UnitSquareMesh(*(n, )*2)
    bmesh = BoundaryMesh(mesh, 'exterior')

    Q = FunctionSpace(mesh, 'DG', 0)
    V = FunctionSpace(mesh, 'CG', 1)
    B = FunctionSpace(mesh, 'CG', 1)
    W = [Q, V, B]

    p, u, lmbda = map(TrialFunction, W)
    q, v, beta = map(TestFunction, W)
    Tu = Trace(u, bmesh)
    Tv = Trace(v, bmesh)

    # The line integral
    dxGamma = Measure('dx', domain=bmesh)

    a = [[0]*len(W) for _ in range(len(W))]

    a[0][0] = Constant(eps)*inner(p, q)*dx
    a[0][2] = inner(q, lmbda)*dx

    a[1][1] = inner(Tu, Tv)*dxGamma
    a[1][2] = inner(grad(v), grad(lmbda))*dx + inner(v, lmbda)*dx

    a[2][0] = inner(p, beta)*dx
    a[2][1] = inner(grad(u), grad(beta))*dx + inner(u, beta)*dx

    L = [inner(Constant(0), q)*dx,
         inner(Tv, f)*dxGamma,
         inner(Constant(0), beta)*dx]
    
    return a, L, W


def setup_preconditioner(W, which, eps=None):
    '''
    Mirror the structure of preconditioner proposed in 
    
      Robust preconditioners for PDE-constrained optimization with limited
      observations; Mardal and Nielsen and Nordaas, BIT 2017

    or Schoeberl and Zulehner's
      
      Symmetric indefinite preconditioners for saddle point problems with 
      applications to PDE constrained optimization problems
    '''
    from block.algebraic.petsc import AMG
    from block.algebraic.petsc import LumpedInvDiag
    from xii.linalg.convert import collapse
    print 'WHICH is', which
    (Q, V, B) = W

    p, u, lmbda = map(TrialFunction, W)
    q, v, beta = map(TestFunction, W)

    bmesh = BoundaryMesh(Q.mesh(), 'exterior')
    Tu = Trace(u, bmesh)
    Tv = Trace(v, bmesh)

    # The line integral
    dxGamma = Measure('dx', domain=bmesh)

    # Nielsen
    if which == 0:
        b00 = Constant(eps)*inner(p, q)*dx
        B00 = LumpedInvDiag(ii_assemble(b00))
        
        M_bdry = ii_convert(ii_assemble(inner(Tu, Tv)*dxGamma))
        # H2 norm with H1 elements
        A = ii_assemble(inner(grad(v), grad(u))*dx + inner(v, u)*dx)
        # From dual to nodal
        M = LumpedInvDiag(ii_assemble(inner(u, v)*dx))
        # The whole matrix to be inverted is then (second term is H2 approx)
        B11 = collapse(M_bdry + eps*A*M*A)
        # And the inverse
        B11 = AMG(B11, parameters={'pc_hypre_boomeramg_cycle_type': 'W'})
        
        b22 = Constant(1./eps)*inner(lmbda, beta)*dx
        B22 = LumpedInvDiag(ii_assemble(b22))
    # SZ
    else:
        print 'X'
        # eps*L2
        b00 = Constant(eps)*inner(p, q)*dx
        B00 = LumpedInvDiag(ii_assemble(b00))

        # L2 \cap eps H1
        a = inner(u, v)*ds + Constant(eps)*(inner(grad(v), grad(u))*dx + inner(v, u)*dx)
        B11 = AMG(assemble(a))

        # (1./eps)*L2 \cap 1./sqrt(eps)H1
        a = Constant(1./eps)*inner(beta, lmbda)*dx + \
            Constant(1./sqrt(eps))*(inner(grad(beta), grad(lmbda))*dx + inner(beta, lmbda)*dx)
        B22 = AMG(assemble(a))
    
    return block_diag_mat([B00, B11, B22])


# --------------------------------------------------------------------


def setup_mms(eps=None):
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp
    
    x, y  = sp.symbols('x[0], x[1]')
    u = sp.cos(sp.pi*x*(1-x)*y*(1-y))
    p = sp.S(0)  # Normal stress is the multiplier, here it is zero

    f = x + y
    g = u

    up = map(as_expression, (u, p))
    f = as_expression(f)

    return up, f


def setup_error_monitor(true, history, path=''):
    '''TODO'''
    from common import monitor_error, H1_norm, L2_norm
    return monitor_error(true, [], history, path=path)
