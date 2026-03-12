# JSD NO COPY NO INSPIRE
import numpy as np
from dolfin import *
from xii import *
from xii.meshing.make_mesh_cpp import make_mesh


def ball_mesh(x0, radius):
    '''Crude'''
    ncells = 10
    thetas = 2*np.pi*np.linspace(0, 1, ncells, endpoint=False)
    x = np.cos(thetas)
    y = np.sin(thetas)

    perimeter_points = np.c_[x, y, np.zeros_like(x)]
    n = len(perimeter_points)

    cells = [[i, (i+1)%n, n] for i in range(len(perimeter_points))]
    cells.extend([[i, (i+1)%n, n+1] for i in range(len(perimeter_points))])
    cells = np.array(cells)

    top = np.array([[0, 0, 1]])
    bottom = np.array([[0, 0, -1]])        
    points = np.r_[perimeter_points, top, bottom]

    transform = Ball.map_from_reference(x0, n=None, R=radius)
    points = np.array([transform(p) for p in points])

    return make_mesh(points, cells, 2, 3)

# --------------------------------------------------------------------

if __name__ == '__main__':
    mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 1), 8, 8, 16)

    segment = CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5) && x[2] < 0.5+tol',
                                tol=1E-10)
    point = CompiledSubDomain('near(x[2], 0.5)')

    edge_f = MeshFunction('size_t', mesh, 1, 0)

    segment.mark(edge_f, 1)

    line_mesh = EmbeddedMesh(edge_f, 1)
    vertex_f = MeshFunction('size_t', line_mesh, 0, 0)
    point.mark(vertex_f, 1)
    ds_ = Measure('ds', domain=line_mesh, subdomain_data=vertex_f)  # For coupling over endpoint
    dx_ = Measure('dx', domain=line_mesh)

    assert abs(assemble(Constant(1)*ds_(1))) > 0

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(line_mesh, 'CG', 1)
    M = FunctionSpace(line_mesh, 'R', 0)   # LM space for point coupling
    W = [V, Q, M]

    u, p, l = map(TrialFunction, W)
    v, q, m = map(TestFunction, W)

    a = block_form(W, 2)
    a[0][0] = inner(grad(u), grad(v))*dx
    a[1][1] = inner(grad(p), grad(q))*dx

    radius = 0.1
    ball = Ball(radius=radius, degree=10)
    Pi_u, Pi_v = (Average(arg, line_mesh, ball) for arg in (u, v))
    # The condition here related mean value over ball to 1d
    # NOTE: mean means normalized! by volume so unit u == units Pi-u
    a[2][0] = inner(Pi_u, m)*ds_(1)
    a[2][1] = -inner(p, m)*ds_(1)
    # Adjoint
    a[0][2] = inner(Pi_v, l)*ds_(1)
    a[1][2] = -inner(q, l)*ds_(1)

    L = block_form(W, 1)
    # Suppose everything is driven by 1d
    L[1] = inner(Constant(1), q)*dx
    L[2] = inner(Constant(100), m)*ds_(1)

    Wbcs = [[DirichletBC(V, Constant(0), 'near(x[1], 0)')],
            [],
            []]

    A, b = map(ii_assemble, (a, L))
    A, b = apply_bc(A, b, bcs=Wbcs)


    wh = ii_Function(W)
    A_, b_ = map(monolithic, (A, b))

    solve(A_, wh.vector(), b_)

    uh, ph, _ = wh

    File('uh.pvd') << uh
    File('ph.pvd') << ph

    line_mesh = EmbeddedMesh(edge_f, 1)
    x0 = line_mesh.coordinates()[vertex_f.where_equal(1)]

    bm = ball_mesh(x0, radius)

    File('ball.pvd') << bm
