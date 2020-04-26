# Here we solve for the extremal points of the following lagrangian
# 
#     min 0.5||u_1||_1^2 + 0.5||u_2||_2^2 -(f1, u1)_1 - (f2, u2)_2 
#
# subject to
#
#     u1 - u1 = g in omega
#             1     omega        2
# With  [---------(-------]-------------)
#       [                               )
#       [                               )
#       [-------------------------------)
#
# This is inspired by coupling differenct physics over a layer (omega).

from __future__ import absolute_import
from dolfin import *
from xii import *
from six.moves import map


def setup_domain(n):
    '''Split the unit square appropriately'''
    mesh = UnitSquareMesh(n, n)
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 3)
    CompiledSubDomain('x[0] < 0.25+DOLFIN_EPS').mark(subdomains, 1)
    CompiledSubDomain('x[0] > 0.75-DOLFIN_EPS').mark(subdomains, 2)

    mesh1 = SubDomainMesh(subdomains, (1, 3))
    mesh2 = SubDomainMesh(subdomains, (2, 3))
    mesh12 = OverlapMesh(mesh1, mesh2)

    return mesh1, mesh2, mesh12

    
def setup_problem(i, xxx_todo_changeme, eps=None):
    '''Just showcase, no MMS (yet)'''
    (f1, f2, g) = xxx_todo_changeme
    Alpha1, Alpha2 = Constant(5), Constant(1)

    n = 32*(2**i)
    mesh1, mesh2, omega = setup_domain(n)

    V1 = FunctionSpace(mesh1, 'CG', 1)
    V2 = FunctionSpace(mesh2, 'CG', 1)
    Q = FunctionSpace(omega, 'CG', 1)
    W = (V1, V2, Q)

    u1, u2, p = list(map(TrialFunction, W))
    v1, v2, q = list(map(TestFunction, W))

    R_u1, R_v1 = [Restriction(x, omega) for x in (u1, v1)]
    R_u2, R_v2 = [Restriction(x, omega) for x in (u2, v2)]

    dxOmega = Measure('dx', domain=omega)

    a00 = Alpha1*inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx
    a01 = 0
    a02 = inner(R_v1, p)*dxOmega

    a10 = 0
    a11 = Alpha2*inner(grad(u2), grad(v2))*dx + inner(u2, v2)*dx
    a12 = -inner(R_v2, p)*dxOmega

    a20 = inner(R_u1, q)*dxOmega
    a21 = -inner(R_u2, q)*dxOmega
    a22 = 0

    L0 = inner(f1, v1)*dx
    L1 = inner(f2, v2)*dx
    L2 = inner(g, q)*dxOmega

    a = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    L = [L0, L1, L2]
    
    return a, L, W

# --------------------------------------------------------------------

def setup_mms(eps=None):
    '''Simple MMS...'''
    from common import as_expression
    import sympy as sp
    
    up = []
    fg = [Expression('x[0]+x[1]', degree=1),
          Expression('sin(pi*(x[0]+x[1]))', degree=4),
          Expression('x[0]*x[1]', degree=2)]
    
    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2 for simplicity'''
    from common import monitor_error, H1_norm, H1_norm, L2_norm
    return monitor_error(true, [], history, path=path)
