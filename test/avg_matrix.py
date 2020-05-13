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


def test_normal_mat(n):
    mesh = BoxMesh(Point(-1, -1, 0), Point(1, 1, 1), n, n, n)
    radius = 0.5           # Averaging radius for cyl. surface
    quadrature_degree = 10  # Quadraure degree for that integration

    gamma = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0.0) && near(x[1], 0.0)').mark(gamma, 1)
    bmesh = EmbeddedMesh(gamma, 1)

    # del bmesh.parent_entity_map

    U = FunctionSpace(mesh, 'CG', 2)
    V = VectorFunctionSpace(mesh, 'CG', 2)

    radius = 0.01
    quad_degree = 10

    # PI
    shape = Circle(radius=radius, degree=quad_degree)

    u = TrialFunction(U)
    v = TestFunction(V)

    NPi_v = NormalAverage(v, bmesh, shape=shape)
    Pi_u = Average(u, bmesh, shape=shape)

    dxGamma = Measure('dx', domain=bmesh)
    A = ii_assemble(inner(NPi_v, Pi_u)*dxGamma)
    # FIXME: the problem here is that NormalAverage anotates
    # and keeps the shape of the vector. To pass here we'd need
    # to anotate AND change shape of NPi_v. However, this might
    # introduce troubles in identifying the integral later for assembly
    # So I think this is finally time for a rewrite of these things in
    # UFL!!!

    
    #
    # uh = interpolate(Expression('(x[0]*x[0]+x[1]*x[1])*exp(x[2]*x[2])', degree=2), V)
    # vh = interpolate(Expression('cos(pi*x[0])*cos(pi*x[1])*x[2]', degree=2), V)

    # x = Function(V).vector()
    # A.mult(uh.vector(), x)
    # approx = x.inner(vh.vector())  # This is (Pi u_h, T v_h)_gamma
    # exact = (radius**2)/2*(exp(1) - exp(0))
    # return bmesh.hmin(), abs(approx - exact)

# --------------------------------------------------------------------

if __name__ == '__main__':
    h0, e0 = None, None
    for n in (2, 4, 8, 16):
        # h, e = test_mat(n)
        h, e = test_normal_mat(n)

        if h0 is not None:
            rate = ln(e/e0)/ln(h/h0)
        else:
            rate = -1
        h0, e0 = h, e
        
        print(h, e, rate)
