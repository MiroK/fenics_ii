# This example solves the coupled Darcy-Stokes problem analyzed in
# thesis of Marco Discacciati (with Quarteroni)
#
# As usual in the demos we have:
# Darcy domain = [0.25, 0.75]^2
# Stokes domain = [0, 1]^2 \ Darcy domain
#
# On the Darcy domain we solve: 
#                               -div(grad(u2)) = f2
#
# On Stokes we have: -div(T(u1, p1)) = f1
#                     div(u1) = 0
#                     where T(u1, p1) = -p1*I + 2*D(u1) and D = sym o grad
#
# Unlike with darcy_stokes.py we solve here only for pressure in the 
# Darcy domain. Moreover, there is no multiplier for the coupling
#
# The Stokes problem is considered with Neumann bcs on the outer boundary
# [These are specified using expression for the stress tensor]. Further,
# letting t = T(u1, p1).n1, there are following interface conditions:
#
# -t.n1 = p + f
# -t.tau1 = u1.tau1 - g  [tau1 is the tangent]
# u1.n1 - grad(p).n2 = h 
#
# NOTE: normally f,g, h are zero. Here they are not in order to make
# the exact solution easier to find.


from dolfin import *
from xii import *


def setup_domain(n):
    '''
    Inner is [0.25, 0.75]^2, inner is [0, 1]^2 \ [0.25, 0.75]^2 and 
    \partial [0.25, 0.75]^2 is the interface
    '''
    # Avoiding mortar meshes here because of speed 
    interior = CompiledSubDomain('std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25')
    outer_mesh = UnitSquareMesh(n, n)
    
    subdomains = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim(), 0)
    # Awkward marking
    for cell in cells(outer_mesh):
        x = cell.midpoint().array()            
        subdomains[cell] = int(interior.inside(x, False))
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    stokes_domain = EmbeddedMesh(subdomains, 0)
    darcy_domain = EmbeddedMesh(subdomains, 1)

    # Interior boundary
    surfaces = MeshFunction('size_t', darcy_domain, darcy_domain.topology().dim()-1, 0)
    DomainBoundary().mark(surfaces, 1)
    iface_domain = EmbeddedMesh(surfaces, 1)

    # Mark the outiside for Stokes
    facet_f = MeshFunction('size_t', stokes_domain, stokes_domain.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0]*(1-x[0]), 0) || near(x[1]*(1-x[1]), 0)').mark(facet_f, 1)
    stokes_domain.subdomains = facet_f

    return stokes_domain, darcy_domain, iface_domain


def setup_problem(i, data, eps=1.):
    '''TODO'''
    n = 16*2**i

    stokes_domain, darcy_domain, iface_domain = setup_domain(n)
    # And now for the fun stuff
    V1 = VectorFunctionSpace(stokes_domain, 'CG', 2)
    Q = FunctionSpace(darcy_domain, 'CG', 1)
    Q1 = FunctionSpace(stokes_domain, 'CG', 1)

    M = FunctionSpace(iface_domain, 'DG', 0)
    W = [V1, Q1, Q]

    u1, p1, p = list(map(TrialFunction, W))
    v1, q1, q = list(map(TestFunction, W))
    
    dxGamma = Measure('dx', domain=iface_domain)
    # We will need traces of the functions on the boundary
    Tu1, Tv1 = [Trace(x, iface_domain) for x in (u1, v1)]
    Tp, Tq = [Trace(x, iface_domain) for x in (p, q)]

    n = OuterNormal(iface_domain, [0.5, 0.5])  # Outer of Darcy
    n1 = -n                                  # Outer of Stokes
    # Get tangent vector
    tau1 = Constant(((0, -1),
                     (1, 0)))*n1

    a = [[0]*len(W) for i in range(len(W))]
    
    a[0][0] = Constant(2)*inner(sym(grad(u1)), sym(grad(v1)))*dx +\
              inner(u1, v1)*dx +\
              inner(dot(Tu1, tau1), dot(Tv1, tau1))*dxGamma
    a[0][1] = -inner(p1, div(v1))*dx
    a[0][2] = inner(Tp, dot(Tv1, n1))*dxGamma

    a[2][0] = -inner(Tq, dot(Tu1, n1))*dxGamma
    a[2][2] = inner(grad(p), grad(q))*dx

    a[1][0] = -inner(q1, div(u1))*dx

    ####
    n_outer = FacetNormal(stokes_domain)
    dsOuter = Measure('ds',
                      domain=stokes_domain,
                      subdomain_data=stokes_domain.subdomains,
                      subdomain_id=1)

    L = [0]*len(W)
    L[0] = inner(data['expr_f1'], v1)*dx + \
           inner(v1, dot(data['expr_stokes_stress'], n_outer))*dsOuter + \
           inner(dot(Tv1, tau1), dot(data['expr_u1'] + dot(data['expr_stokes_stress'], n1), tau1))*dxGamma + \
           inner(-dot(Tv1, n1), data['expr_f'])*dxGamma
    L[1] = inner(Constant(0), p1)*dx
    
    L[2] = inner(data['expr_f2'], q)*dx +\
           -inner(dot(data['expr_u1'], n1) + dot(data['expr_u2'], -n1), Tq)*dxGamma

    
    return a, L, W

# This is adopted from weak_bcs
# <---
from block.block_base import block_base

def block_op_from_action(action, create_vec):
    '''block_base object with the methods'''
    return type('BlockOpDummy',
                (block_base, ),
                {'matvec': lambda self, b, f=action: f(b),
                 'create_vec': lambda self, i, f=create_vec: f(i)})()


def UMFPACK_LU(A):
    '''Dolfin solver as preconditioner'''
    solver = LUSolver(A, 'umfpack')
    solver.set_operator(A)
    solver.parameters['reuse_factorization'] = True

    x = PETScVector(as_backend_type(A).mat().createVecLeft())

    solve_ = lambda b, x=x, solver=solver: (solver.solve(x, b), x)[1]
    create_vec_ = lambda i, x=x: x

    return block_op_from_action(solve_, create_vec_)
# --->


def setup_preconditioner(W, which, eps):
        
    from block.algebraic.petsc import AMG
    from xii.linalg.block_utils import ReductionOperator, RegroupOperator
    
    # This is best on H1 x H1 x L2 spaces where the Discacciati proces
    # well posedness
    if which == 0:
        # The following settings seem not so bad for GMRES
        # -ksp_rtol 1E-6
        # -ksp_monitor_true_residual none
        # -ksp_type gmres
        # -ksp_gmres_restart 30
        # -ksp_gmres_modifiedgramschmidt 1
    
        u1, p1, p = list(map(TrialFunction, W))
        v1, q1, q = list(map(TestFunction, W))

        b00 = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx
        B00 = AMG(ii_assemble(b00))

        b11 = inner(p1, q1)*dx
        B11 = AMG(ii_assemble(b11))

        b22 = inner(grad(p), grad(q))*dx + inner(p, q)*dx
        B22 = AMG(ii_assemble(b22))
    
        return block_diag_mat([B00, B11, B22])
    
    # System without coupling: Solve Poisson and Stokes individually
    iface_domain = BoundaryMesh(W[-1].mesh(), 'exterior')
        
    M = FunctionSpace(iface_domain, 'DG', 0)

    u1, p1, p = list(map(TrialFunction, W))
    v1, q1, q = list(map(TestFunction, W))
    
    dxGamma = Measure('dx', domain=iface_domain)
    # We will need traces of the functions on the boundary
    Tu1, Tv1 = [Trace(x, iface_domain) for x in (u1, v1)]
    Tp, Tq = [Trace(x, iface_domain) for x in (p, q)]

    n = OuterNormal(iface_domain, [0.5, 0.5])  # Outer of Darcy
    n1 = -n                                  # Outer of Stokes
    # Get tangent vector
    tau1 = Constant(((0, -1),
                     (1, 0)))*n1

    stokes = [[0]*2 for i in range(2)]
    
    stokes[0][0] = Constant(2)*inner(sym(grad(u1)), sym(grad(v1)))*dx +\
                   inner(u1, v1)*dx +\
                   inner(dot(Tu1, tau1), dot(Tv1, tau1))*dxGamma
    stokes[0][1] = -inner(p1, div(v1))*dx
    stokes[1][0] = -inner(q1, div(u1))*dx

    if which == 1:
        B0 = UMFPACK_LU(ii_convert(ii_assemble(stokes)))

        poisson = inner(p, q)*dx + inner(grad(p), grad(q))*dx
        B1 = AMG(assemble(poisson))

        # 2x2
        B = block_diag_mat([B0, B1])
        # Need an adapter for 3 vectors
        R = ReductionOperator([2, 3], W)

        return R.T*B*R        

    from block.iterative import MinRes
    # Solve stokes with Minres
    A0 = ii_assemble(stokes)
    # The preconditioner
    B = block_diag_mat([AMG(ii_assemble(inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx)),
                        AMG(ii_assemble(inner(p1, q1)*dx))])
    # Approx
    A0_inv = MinRes(A0, precond=B, relativeconv=True, tolerance=1E-5)
    B0 = block_op_from_action(action=lambda b, A0_inv=A0_inv: A0_inv*b,
                              create_vec=lambda i, A0=A0: A0.create_vec(i))
        
    poisson = inner(p, q)*dx + inner(grad(p), grad(q))*dx
    B1 = AMG(assemble(poisson))

    # 2x2
    B = block_diag_mat([B0, B1])
    # Need an adapter for 3 vectors
    R = RegroupOperator([2, 3], W)

    return R.T*B*R

# --------------------------------------------------------------------

def setup_mms(eps):
    '''Simple MMS problem for UnitSquareMesh'''
    from common import as_expression
    import sympy as sp
    
    pi = sp.pi
    x, y, EPS = sp.symbols('x[0] x[1] EPS')
    
    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])

    sp_Grad = lambda f: sp.Matrix([[f[0].diff(x, 1), f[0].diff(y, 1)],
                                   [f[1].diff(x, 1), f[1].diff(y, 1)]])

    sp_div = lambda f: f[0].diff(x, 1) + f[1].diff(y, 1)
    
    sp_Div = lambda f: sp.Matrix([sp_div(f[0, :]), sp_div(f[1, :])])

    # Stokes velocity
    u1 = sp.Matrix([sp.sin(2*pi*y), sp.sin(2*pi*x)])
    # Stokes pressure
    p1 = sp.cos(2*pi*x)*sp.cos(2*pi*y)

    sym = lambda A: (A + A.T)/2

    # Assuming all the constants are unity, here's the stress tensor
    T = lambda u, p: -p*sp.eye(2) + 2*sym(sp_Grad(u))

    # And the rhs for Stokes
    f1 = -sp_Div(T(u1, p1)) + u1

    # Darcy pressure
    p2 = p1 - sp.S(1.0)
    # Again assuming all constants are 1 here's the Darcy velocity
    u2 = -sp_grad(p2)
    # The Darcy rhs
    f2 = sp_div(u2)

    # Define a vector T(u1, p1).n1 as t, then lambda_ = -t.n1
    lambda_f = lambda n: n.dot(T(u1, p1)*n)
    assert lambda_f(sp.Matrix([1, 0])).subs(x, 0.25) == 0
    assert lambda_f(sp.Matrix([-1, 0])).subs(x, 0.75) == 0
    assert lambda_f(sp.Matrix([0, 1])).subs(y, 0.25) == 0
    assert lambda_f(sp.Matrix([0, -1])).subs(y, 0.75) == 0
    # Cool so the multiplier is easy
    lambda_ = sp.S(0)

    # And this makes the f easy as well
    f = lambda_ - p2

    # NOTE: the multiplier is grad(u).n and with the chosen data this
    # means that it's zero on the interface
    up = list(map(as_expression, (u1, p1, p2)))  # The flux
    fg = list(map(as_expression, (f, f1, f2, u1, u2, T(u1, p1))))
    fg = dict(list(zip(['expr_%s' % s for s in ('f', 'f1', 'f2', 'u1', 'u2', 'stokes_stress')],
                  fg)))
    
    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error V1 x Q1, V2 x Q2, L2(instead of fractional)'''
    from common import monitor_error, H1_norm, L2_norm, Hdiv_norm
    # First stokes, then darcy pressure
    reduction = lambda e: None if e is None else [sqrt(e[0]**2 + e[1]**2), e[2]]

    return monitor_error(true,
                         # u1, p, p1
                         [H1_norm, L2_norm, H1_norm],
                         history, path=path, reduction=reduction)
