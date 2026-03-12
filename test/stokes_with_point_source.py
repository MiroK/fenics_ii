from dolfin import *

from xii.meshing.make_mesh_cpp import make_mesh
from xii import *
import numpy as np

from collections import namedtuple
# Represent flux out of the tube through a disk of radius passing through plane
# defined by x and normal.
FluxPointSource = namedtuple('FluxPointSource', ('x', 'normal', 'radius', 'value'))


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



def point_source_mesh(sources, h):
    '''Auxiliary 1d mesh for Lagrange multipliers'''
    # NOTE: the mesh is 1d because of xii.Avarage
    coordinates, cells = [], []
    for (i, source) in enumerate(sources):
        v0 = source.x
        # Now we make a short cell
        n = source.normal
        n = n/np.linalg.norm(n)
        v1 = v0*h

        dm_ = disk_mesh(v0, n, radius=source.radius)
        File(f'source{i}_mesh.pvd') << dm_

        coordinates.extend((v0, v1))
        cells.append((2*i, 2*i+1))
    coordinates, cells = map(np.array, (coordinates, cells))
    print(coordinates)
    print(cells)
    mesh = make_mesh(coordinates, cells, 1, 3)
    x = mesh.coordinates()

    vertex_f = MeshFunction('size_t', mesh, 0, 0)
    for (k, source) in enumerate(sources, 1):
        dist = np.linalg.norm(x - source.x, 2, axis=1)
        i = np.argmin(dist)
        assert dist[i] < 1E-13
        vertex_f[i] = k

    return vertex_f
    
        

def stokes_solver(mesh3d, sources):
    # We want to add to standard Stokes Lagrangian a terms like
    #
    # int_disk{(dot(Average(u), n) - sourve.value)*multiplier}
    V = VectorFunctionSpace(mesh3d, 'CG', 2)
    Q = FunctionSpace(mesh3d, 'CG', 1)

    # Multiplier will like on the auxiliary mesh
    line_facet_f = point_source_mesh(sources, h=mesh3d.hmin())
    line_mesh = line_facet_f.mesh()
    ds_ = Measure('ds', domain=line_mesh, subdomain_data=line_facet_f)
    
    dimM = len(sources)
    M = VectorFunctionSpace(line_mesh, 'R', 0, dimM)
    W = [V, Q, M]

    print('dimW', sum(Wi.dim() for Wi in W))
    
    u, p, l = map(TrialFunction, W)
    v, q, m = map(TestFunction, W)

    a, L = block_form(W, 2), block_form(W, 1)

    a[0][0] = inner(sym(grad(u)), sym(grad(v)))*dx
    a[0][1] = -inner(div(v), p)*dx
    a[1][0] = -inner(div(u), q)*dx

    for (k, source) in enumerate(sources, 1):
        disk = Disk(radius=source.radius, degree=20, quad_scheme='simple')
        Pi_u, Pi_v = (Average(arg, line_mesh, disk) for arg in (u, v))        

        n_disk = Constant(source.normal)
        area = Constant(pi*source.radius**2)

        assert assemble(Constant(1)*ds_(k)) > 0
        
        a[2][0] += area*inner(dot(Pi_u, n_disk), m[k-1])*ds_(k)
        a[0][2] += area*inner(dot(Pi_v, n_disk), l[k-1])*ds_(k)

        L[2] += area*inner(Constant(source.value), m[k-1])*ds_(k)        

    Vbcs = [DirichletBC(V, Constant((0, 0, 0)), 'on_boundary')]
    Qbcs = []
    Mbcs = []
    Wbcs = [Vbcs, Qbcs, Mbcs]

    A, b = map(ii_assemble, (a, L))
    A, b = apply_bc(A, b, bcs=Wbcs)

    wh = ii_Function(W)
    A_, b_ = map(monolithic, (A, b))
    print(A_.norm('linf'), b_.norm('l2'))
    solve(A_, wh.vector(), b_)
    print(wh.vector().norm('l2'))
    
    uh, ph, _ = wh

    return uh, ph

# --------------------------------------------------------------------

if __name__ == '__main__':

    normal = np.ones(3)/np.sqrt(3)
    
    source0 = FluxPointSource(x=np.array([-0.25, -0.25, -0.25]),
                              normal=-normal,
                              radius=0.1,
                              value=20)

    source1 = FluxPointSource(x=np.array([0.25, 0.25, 0.25]),
                              normal=normal,
                              radius=0.1,
                              value=50)

    sources = [source0, source1]
    
    v = point_source_mesh(sources, h=0.2)

    mesh3d = BoxMesh(Point(-0.5, -0.5, -0.5), Point(0.5, 0.5, 0.5), 8, 8, 8)
    uh, ph = stokes_solver(mesh3d, sources)

    File('uh.pvd') << uh
    File('ph.pvd') << ph
