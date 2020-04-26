from __future__ import absolute_import
from __future__ import print_function
from xii import *
from dolfin import *


def test_mat(n):
    mesh = BoxMesh(Point(-1, -1, 0), Point(1, 1, 1), n, n, n)
    radius = 0.5           # Averaging radius for cyl. surface
    quadrature_degree = 10  # Quadraure degree for that integration

    gamma = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0.0) && near(x[1], 0.0)').mark(gamma, 1)
    bmesh = EmbeddedMesh(gamma, 1)

    # del bmesh.parent_entity_map

    V = FunctionSpace(mesh, 'CG', 2)

    u = TrialFunction(V)
    v = TestFunction(V)

    Pi_u = Average(u, bmesh, radius, quadrature_degree)
    T_v = Average(v, bmesh, radius=0)  # This is 3d-1d trace

    dxGamma = Measure('dx', domain=bmesh)
    A = ii_convert(ii_assemble(inner(Pi_u, T_v)*dxGamma))

    uh = interpolate(Expression('(x[0]*x[0]+x[1]*x[1])*exp(x[2]*x[2])', degree=2), V)
    vh = interpolate(Expression('cos(pi*x[0])*cos(pi*x[1])*x[2]', degree=2), V)

    x = Function(V).vector()
    A.mult(uh.vector(), x)
    approx = x.inner(vh.vector())  # This is (Pi u_h, T v_h)_gamma
    exact = (radius**2)/2*(exp(1) - exp(0))
    return bmesh.hmin(), abs(approx - exact)
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    h0, e0 = None, None
    for n in (2, 4, 8, 16):
        h, e = test_mat(n)

        if h0 is not None:
            rate = ln(e/e0)/ln(h/h0)
        else:
            rate = -1
        h0, e0 = h, e
        
        print((h, e, rate))
