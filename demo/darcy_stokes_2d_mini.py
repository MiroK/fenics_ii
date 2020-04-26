# This example solves the coupled Darcy-Stokes problem analyzed in
#   Layton et al: Coupling fluid flow with porous media flow
#
# As usual in the demos we have:
# Darcy domain = [0.25, 0.75]^2
# Stokes domain = [0, 1]^2 \ Darcy domain
#
# On the Darcy domain we solve: u2 = -grad(p2)
#                               div(u2) = f2
#
# On Stokes we have: -div(T(u1, p1)) = f1
#                     div(u1) = 0
#                     where T(u1, p1) = -p1*I + 2*D(u1) and D = sym o grad
#
# The Stokes problem is considered with Neumann bcs on the outer boundary
# [These are specified using expression for the stress tensor]. Further,
# letting t = T(u1, p1).n1, there are following interface conditions:
#
# -t.n1 = p2 + f
# -t.tau1 = u1.tau1 - g  [tau1 is the tangent]
# u1.n1 + u2.n2 = h 
#
# NOTE: normally f,g, h are zero. Here they are not in order to make
# the exact solution easier to find.
#
# As a Lagrange multiplier I chose lambda = -t.n1.
# NOTE: below the solution is such that lambda = 0.

from __future__ import absolute_import
from dolfin import *
from xii import *
from six.moves import map
from six.moves import range
from six.moves import zip


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
    V1 = VectorFunctionSpace(stokes_domain, 'CG', 1)
    Vb = VectorFunctionSpace(stokes_domain, 'Bubble', 3)
    Q1 = FunctionSpace(stokes_domain, 'CG', 1)
    V2 = FunctionSpace(darcy_domain, 'RT', 1)
    Q2 = FunctionSpace(darcy_domain, 'DG', 0)
    # With CG1 in M the preconditioner digs
    M = FunctionSpace(iface_domain, 'DG', 0) 
    W = [V1, Vb, Q1, V2, Q2, M]

    u1, ub, p1, u2, p2, lambda_ = list(map(TrialFunction, W))
    v1, vb, q1, v2, q2, beta_ = list(map(TestFunction, W))
    
    dxGamma = Measure('dx', domain=iface_domain)
    # We will need traces of the functions on the boundary
    Tu1, Tu2 = [Trace(x, iface_domain) for x in (u1, u2)]
    Tv1, Tv2 = [Trace(x, iface_domain) for x in (v1, v2)]

    n2 = OuterNormal(iface_domain, [0.5, 0.5])  # Outer of Darcy
    n1 = -n2                                  # Outer of Stokes
    # Get tangent vector
    tau1 = Constant(((0, -1),
                     (1, 0)))*n1

    a = [[0]*len(W) for i in range(len(W))]
    # V - Vb
    a[0][0] = Constant(2)*inner(sym(grad(u1)), sym(grad(v1)))*dx +\
              inner(u1, v1)*dx +\
              inner(dot(Tu1, tau1), dot(Tv1, tau1))*dxGamma

    a[1][1] = Constant(2)*inner(sym(grad(ub)), sym(grad(vb)))*dx +\
              inner(ub, vb)*dx

    a[0][1] = Constant(2)*inner(sym(grad(ub)), sym(grad(v1)))*dx +\
              inner(ub, v1)*dx
    
    a[1][0] = Constant(2)*inner(sym(grad(u1)), sym(grad(vb)))*dx +\
              inner(u1, vb)*dx
    # V and Vb - Q
    a[0][2] = -inner(p1, div(v1))*dx
    a[1][2] = -inner(p1, div(vb))*dx
    # Q - V and Vb
    a[2][0] = -inner(q1, div(u1))*dx
    a[2][1] = -inner(q1, div(ub))*dx
    # Coupling to Darcy
    a[0][5] = inner(lambda_, dot(Tv1, n1))*dxGamma
    # Darcy
    a[3][3] = inner(u2, v2)*dx
    a[3][4] = -inner(p2, div(v2))*dx
    a[4][3] = -inner(q2, div(u2))*dx
    # Coupling to Stokes
    a[3][5] = inner(lambda_, dot(Tv2, n2))*dxGamma
    # Couplings
    a[5][0] = inner(beta_, dot(Tu1, n1))*dxGamma 
    a[5][3] = inner(beta_, dot(Tu2, n2))*dxGamma

    ####
    n_outer = FacetNormal(stokes_domain)
    dsOuter = Measure('ds',
                      domain=stokes_domain,
                      subdomain_data=stokes_domain.subdomains,
                      subdomain_id=1)

    L = [0]*len(W)
    L[0] = inner(data['expr_f1'], v1)*dx + \
           inner(v1, dot(data['expr_stokes_stress'], n_outer))*dsOuter + \
           inner(dot(Tv1, tau1), dot(data['expr_u1'] + dot(data['expr_stokes_stress'], n1), tau1))*dxGamma
    L[1] = inner(data['expr_f1'], vb)*dx  # Bubble contributes nothing to ds, dxG
    L[2] = inner(Constant(0), q1)*dx
    L[3] = inner(data['expr_f'], dot(Tv2, n2))*dxGamma
    L[4] = inner(-data['expr_f2'], q2)*dx
    L[5] = inner(dot(data['expr_u1'], n1) + dot(data['expr_u2'], n2), beta_)*dxGamma

    return a, L, W


def setup_preconditioner(W, which, eps):
    from block.algebraic.petsc import AMG
    from block.algebraic.petsc import LumpedInvDiag, LU
    from hsmg import HsNorm

    u1, ub, p1, u2, p2, lambda_ = list(map(TrialFunction, W))
    v1, vb, q1, v2, q2, beta_ = list(map(TestFunction, W))

    # Neither is super spectacular
    # Completely block diagonal preconditioner with H1 on the bubble space
    if which == 0:
        b00 = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx
        B00 = AMG(ii_assemble(b00))

        bbb = inner(grad(ub), grad(vb))*dx + inner(ub, vb)*dx
        Bbb = LumpedInvDiag(ii_assemble(bbb))

        b11 = inner(p1, q1)*dx
        B11 = AMG(ii_assemble(b11))

        b22 = inner(div(u2), div(v2))*dx + inner(u2, v2)*dx
        B22 = LU(ii_assemble(b22))

        b33 = inner(p2, q2)*dx
        B33 = LumpedInvDiag(ii_assemble(b33))

        B44 = inverse(HsNorm(W[-1], s=0.5))
    
        return block_diag_mat([B00, Bbb, B11, B22, B33, B44])

    # Monolithic for MINI velocity
    b = [[0, 0], [0, 0]]
    b[0][0] = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx
    b[0][1] = inner(grad(ub), grad(v1))*dx + inner(ub, v1)*dx
    b[1][0] = inner(grad(u1), grad(vb))*dx + inner(u1, vb)*dx
    b[1][1] = inner(grad(ub), grad(vb))*dx + inner(ub, vb)*dx
    # Make into a monolithic matrix
    B00 = AMG(ii_convert(ii_assemble(b)))

    b11 = inner(p1, q1)*dx
    B11 = AMG(ii_assemble(b11))

    b22 = inner(div(u2), div(v2))*dx + inner(u2, v2)*dx
    B22 = LU(ii_assemble(b22))

    b33 = inner(p2, q2)*dx
    B33 = LumpedInvDiag(ii_assemble(b33))

    B44 = inverse(HsNorm(W[-1], s=0.5))
    # So this is a 5x5 matrix 
    BB = block_diag_mat([B00, B11, B22, B33, B44])
    # But the preconditioner has to be 6x6; reduce 6 to 6 by comcat
    # first two, rest stays same
    R = ReductionOperator([2, 3, 4, 5, 6], W)

    return (R.T)*BB*R

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
    up = list(map(as_expression, (u1, p1, u2, p2, lambda_)))  # The flux
    fg = list(map(as_expression, (f, f1, f2, u1, u2, T(u1, p1))))
    fg = dict(list(zip(['expr_%s' % s for s in ('f', 'f1', 'f2', 'u1', 'u2', 'stokes_stress')],
                  fg)))
    
    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error V1 x Q1, V2 x Q2, L2(instead of fractional)'''
    from common import monitor_error, H1_norm, L2_norm, Hdiv_norm
    # Note we produce u1, u2, and p error. It is more natural to have
    # broken H1 norm so reduce the first 2 errors to single number
    reduction = lambda e: None if e is None else [sqrt(e[0]**2 + e[1]**2), e[-1]]

    return monitor_error(true,
                         [H1_norm, L2_norm, Hdiv_norm, L2_norm, L2_norm],
                         history, path=path)

def setup_transform(i, wh):
    '''Combine velocity with bubbles'''
    # NOTE: we don't look at the bubble contribution
    return [wh[i] for i in (0, 2, 3, 4, 5)]
