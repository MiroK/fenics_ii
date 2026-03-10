from xii.meshing.make_mesh_cpp import make_mesh
from dolfin import *
import numpy as np

from xii import Disk, Circle


def disk_mesh(x0, normal, radius, ncells=10):
    '''Disk with radius centered at x lying in place with normal'''
    if isinstance(normal, Constant):
        normal = normal.values()
    assert len(x0) == len(normal) == 3
    
    origin = np.array([[0, 0, 0]])

    thetas = 2*np.pi*np.linspace(0, 1, ncells, endpoint=False)
    x = np.cos(thetas)
    y = np.sin(thetas)

    perimeter_points = np.c_[x, y, np.zeros_like(x)]
    n = len(perimeter_points)    
    cells = np.array([[i, (i+1)%n, n] for i in range(len(perimeter_points))])
    
    points = np.r_[perimeter_points, origin]

    transform = Disk.map_from_reference(x0, normal, radius)
    points = np.array([transform(p) for p in points])

    return make_mesh(points, cells, 2, 3)


def circle_mesh(x0, normal, radius, ncells=10):
    '''Circle with radius centered at x lying in place with normal'''
    if isinstance(normal, Constant):
        normal = normal.values()    
    assert len(x0) == len(normal) == 3
    
    thetas = 2*np.pi*np.linspace(0, 1, ncells, endpoint=False)
    x = np.cos(thetas)
    y = np.sin(thetas)

    perimeter_points = np.c_[x, y, np.zeros_like(x)]
    n = len(perimeter_points)    
    cells = np.array([[i, (i+1)%n] for i in range(len(perimeter_points))])
    
    transform = Circle.map_from_reference(x0, normal, radius)
    perimeter_points = np.array([transform(p) for p in perimeter_points])

    return make_mesh(perimeter_points, cells, 1, 3)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from xii import EmbeddedMesh, Average, ii_assemble

    mesh = BoxMesh(Point(-0.5, -0.5, -0.5), Point(0.5, 0.5, 0.5), 4, 4, 4)

    edge_f = MeshFunction('size_t', mesh, 1, 0)
    line = CompiledSubDomain('near(x[0], x[1]) && near(x[1], x[2]) && x[2] < 0.25+DOLFIN_EPS')
    line.mark(edge_f, 1)

    
    line_mesh = EmbeddedMesh(edge_f, 1)
    vertex_f = MeshFunction('size_t', line_mesh, 0, 0)
    CompiledSubDomain('near(x[2], 0.25)').mark(vertex_f, 1)
    print(vertex_f.array())
    
    ds_ = Measure('ds', domain=line_mesh, subdomain_data=vertex_f)
    print(line_mesh.coordinates())

    V = VectorFunctionSpace(mesh, 'CG', 1)
    f = Expression(('x[0]', 'x[1]', 'x[2]'), degree=1)
    uh = interpolate(f, V)

    radius = 0.1
    disk = Disk(radius=radius, degree=30, quad_scheme='simple')
    Pi_u = Average(uh, line_mesh, disk)  # This divides by area!

    Q = FunctionSpace(line_mesh, 'R', 0)
    q = TestFunction(Q)

    normal = np.array([1, 1, 1])
    # normal = normal / np.linalg.norm(normal)
    normal = Constant(normal)
    area = Constant(pi*radius**2)

    mine = ii_assemble(area*dot(Pi_u, normal)*q*ds_(1)).get_local()[0]
    
    x0 = np.array([0.25, 0.25, 0.25])
    dm = disk_mesh(x0, normal, radius=radius, ncells=40)

    dx_ = Measure('dx', domain=dm)

    true = assemble(dot(f, normal)*dx_)
    print(mine, true, mine/true, 6*pi*radius**2)

    File('x.pvd') << mesh
    File('y.pvd') << line_mesh
    File('z.pvd') << dm
