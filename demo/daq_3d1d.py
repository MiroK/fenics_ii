# The system from d'Angelo & Quarteroni paper on tissue perfusion
# With Omega a 3d domain and Gamma a 1d domain inside it we want
#
# A1(grad(u), grad(v))_3 + A0(u, v)_3 + (Pi u, Tv)_3 - beta(p, Tv)_1 = (f, Tv)_1
# -beta(q, Pi u)_1      + a1(grad(p), grad(q))_1 + (a0+beta)(p, q)_1 = (f, q)_1
#

from dolfin import *
from xii import *


def setup_problem(i, f, eps=None):
    '''Just showcase, no MMS (yet)'''
    # I setup the constants arbitraily
    Alpha1, Alpha0 = Constant(0.02), Constant(0.01)
    alpha1, alpha0 = Constant(2), Constant(0.01)
    beta = Constant(10)

    n = 10*(2**i)

    mesh = UnitCubeMesh(n, n, 2*n)
    radius = 0.01           # Averaging radius for cyl. surface
    quadrature_degree = 10  # Quadraure degree for that integration

    gamma = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(gamma, 1)
    bmesh = EmbeddedMesh(gamma, 1)

    # del bmesh.parent_entity_map

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(bmesh, 'CG', 1)
    W = (V, Q)

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    # Averaging surface
    cylinder = Cylinder(radius=radius, degree=quadrature_degree)

    Pi_u = Average(u, bmesh, cylinder)
    T_v = Average(v, bmesh, None)  # This is 3d-1d trace

    dxGamma = Measure('dx', domain=bmesh)

    a00 = Alpha1*inner(grad(u), grad(v))*dx + Alpha0*inner(u, v)*dx + beta*inner(Pi_u, T_v)*dxGamma
    a01 = -beta*inner(p, T_v)*dxGamma
    a10 = -beta*inner(Pi_u, q)*dxGamma
    a11 = alpha1*inner(grad(p), grad(q))*dxGamma + (alpha0+beta)*inner(p, q)*dxGamma
    
    L0 = inner(f, T_v)*dxGamma
    L1 = inner(f, q)*dxGamma

    a = [[a00, a01], [a10, 0]]
    L = [L0, L1]

    return a, L, W

# --------------------------------------------------------------------

def setup_mms(eps=None):
    '''Simple MMS...'''
    from common import as_expression
    import sympy as sp
    
    up = []
    fg = Expression('sin(2*pi*x[2]*(pow(x[0], 2)+pow(x[1], 2)))', degree=4)
    
    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    return monitor_error(true, [], history, path=path)
