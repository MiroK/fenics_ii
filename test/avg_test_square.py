from dolfin import *
from xii.meshing.make_mesh_cpp import make_mesh
from xii.assembler.average_matrix import surface_average_matrix
from xii.assembler.average_shape import Square
from xii import EmbeddedMesh
import numpy as np


def make_z_mesh(num_vertices, zmin=0, zmax=1):
    '''{(0, 0, zmin + t*(zmax - zmin))}'''

    t = zmin + np.linspace(0, 1, num_vertices)*(zmax - zmin)
    coordinates = np.c_[np.zeros_like(t), np.zeros_like(t), t]
    cells = np.c_[np.arange(num_vertices - 1), np.arange(1, num_vertices)]
    cells.dtype = 'uintp'

    mesh = Mesh(mpi_comm_world())
    make_mesh(coordinates, cells, 1, 3, mesh)

    return mesh


def test(f, n, P, degree=8):
    '''Check integrals due to averaging operator'''
    mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n, n, n)

    mf = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0.0) && near(x[1], 0.0)').mark(mf, 1)
    line_mesh = EmbeddedMesh(mf, 1)

    V = FunctionSpace(mesh, 'CG', 1)
    TV = FunctionSpace(line_mesh, 'DG', 1)

    f = interpolate(f, V)

    cylinder = Square(P, degree)

    Pi = surface_average_matrix(V, TV, cylinder)
    print '\t', Pi.norm('linf'), max(len(Pi.getrow(i)[0]) for i in range(TV.dim()))
    
    Pi_f = Function(TV)
    Pi.mult(f.vector(), Pi_f.vector())

    return Pi_f

# --------------------------------------------------------------------

if __name__ == '__main__':
    # NOTE the size for integration size!!
    size = 0.125
    P = lambda x0: np.array([-size, -size, x0[2]])

    f = Expression('2', degree=2)
    Pi_f0 = f
    
    f = Expression('x[2]', degree=1)
    Pi_f0 = f

    f = Expression('x[2]*x[2]', degree=2)
    Pi_f0 = f

    f = Expression('x[0]', degree=2)
    Pi_f0 = Constant(0)

    f = Expression('x[0]+x[1]', degree=2)
    Pi_f0 = Constant(0)


    e0, n0 = None, None
    for n in (4, 8, 16, 32):
        Pi_f = test(f, n, P)
        print Pi_f(0, 0, 0.5)
        assert Pi_f.vector().norm('l2') > 0
        e = sqrt(abs(assemble(inner(Pi_f0 - Pi_f, Pi_f0 - Pi_f)*dx)))

        if e0 is not None:
            rate = ln(e/e0)/ln(float(n0)/n)
        else:
            rate = np.inf

        print 'error %g, rate=%.2f' % (e, rate)
        
        n0, e0 = n, e
