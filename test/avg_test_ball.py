from dolfin import *
from xii.meshing.make_mesh_cpp import make_mesh
from xii.assembler.average_matrix import surface_average_matrix
from xii.assembler.average_shape import Ball
from xii import EmbeddedMesh
import numpy as np


def test(f, n, radius):
    '''Check integrals due to averaging operator'''
    mesh = BoxMesh(Point(-0.5, -0.5, -0.5), Point(0.5, 0.5, 0.5), n, n, n)

    mf = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0.0) && near(x[1], 0.0) && std::abs(x[2]) < 0.75+tol', tol=1E-8).mark(mf, 1)
    line_mesh = EmbeddedMesh(mf, 1)

    V = FunctionSpace(mesh, 'CG', 1)
    TV = FunctionSpace(line_mesh, 'DG', 1)

    f = interpolate(f, V)

    cylinder = Ball(radius, degree=17)

    Pi = surface_average_matrix(V, TV, cylinder)
    print '\t', Pi.norm('linf'), max(len(Pi.getrow(i)[0]) for i in range(TV.dim()))
    
    Pi_f = Function(TV)
    Pi.mult(f.vector(), Pi_f.vector())

    return Pi_f

# --------------------------------------------------------------------

if __name__ == '__main__':

    radius = 0.05

    f = Constant(1)
    Pi_f0 = f

    f = Expression('x[2]', degree=1)
    Pi_f0 = f

    # FIXME: something more challenging?
    e0, n0 = None, None
    for n in (4, 8, 16, 32, 64):
        Pi_f = test(f, n, radius)
        assert Pi_f.vector().norm('l2') > 0
        e = sqrt(abs(assemble(inner(Pi_f0 - Pi_f, Pi_f0 - Pi_f)*dx)))

        if e0 is not None:
            rate = ln(e/e0)/ln(float(n0)/n)
        else:
            rate = np.inf

        print 'error %g, rate=%.2f' % (e, rate)
        
        n0, e0 = n, e
