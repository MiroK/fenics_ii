from dolfin import *
from xii.meshing.make_mesh_cpp import make_mesh
from xii.assembler.average_matrix import average_matrix
from xii.assembler.average_shape import Circle
from xii import EmbeddedMesh
import numpy as np
import pytest


def worker(f, n, radius=0.3, degree=8):
    '''Check integrals due to averaging operator'''
    mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n, n, n)

    mf = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0.0) && near(x[1], 0.0)').mark(mf, 1)
    line_mesh = EmbeddedMesh(mf, 1)

    V = FunctionSpace(mesh, 'CG', 1)
    TV = FunctionSpace(line_mesh, 'DG', 1)

    f = interpolate(f, V)

    cylinder = Circle(radius, degree)

    Pi = PETScMatrix(average_matrix(V, TV, cylinder))
    # print('\t', Pi.norm('linf'), max(len(Pi.getrow(i)[0]) for i in range(TV.dim())))
    
    Pi_f = Function(TV)
    Pi.mult(f.vector(), Pi_f.vector())

    return Pi_f

radius = 0.1

fs = (Expression('x[2]', degree=1),
      Expression('x[0]*x[0] + x[1]*x[1]', degree=2),
      Expression('x[2]*(x[0]*x[0] + x[1]*x[1])', degree=1),
      Expression('x[0] + x[1]', degree=1),
      Expression('sin(k*pi*x[2])', k=0.5, degree=2))
      
Pi_fs = (Expression('x[2]', degree=1),
         Constant(radius**2),
         Expression('x[2]*r*r', r=radius, degree=1),
         Constant(0),
         Expression('sin(k*pi*x[2])', k=0.5, degree=2))

@pytest.mark.parametrize(('f', 'Pi_f0'), tuple(zip(fs, Pi_fs)))
def test_avg(f, Pi_f0, radius=radius):
    e0, n0 = None, None
    for n in (4, 8, 16, 32):
        Pi_f = worker(f, n, radius=radius)
        assert Pi_f.vector().norm('l2') > 0
        e = sqrt(abs(assemble(inner(Pi_f0 - Pi_f, Pi_f0 - Pi_f)*dx)))

        if e0 is not None:
            rate = ln(e/e0)/ln(float(n0)/n)
        else:
            rate = np.inf

        assert rate == np.inf or abs(e) < 1E-10 or rate > 1
        
        n0, e0 = n, e

# --------------------------------------------------------------------

radius = 0.1

def worker_(f, n, radius=radius, degree=8):
    '''Check integrals due to averaging operator'''
    from weak_bcs.paper_daq.utils import circle_3d1d_conform
    
    mesh, entity_fs = circle_3d1d_conform(n=n, rad=radius)

    line_mesh = EmbeddedMesh(entity_fs[1], 1)

    V = FunctionSpace(mesh, 'CG', 1)
    TV = FunctionSpace(line_mesh, 'DG', 1)

    f = interpolate(f, V)

    cylinder = Circle(radius, degree)

    Pi = PETScMatrix(average_matrix(V, TV, cylinder))
    # print('\t', Pi.norm('linf'), max(len(Pi.getrow(i)[0]) for i in range(TV.dim())))
    
    Pi_f = Function(TV)
    Pi.mult(f.vector(), Pi_f.vector())

    return Pi_f


@pytest.mark.parametrize(('f', 'Pi_f0'), tuple(zip(fs, Pi_fs)))
def test_avg_(f, Pi_f0, radius=radius):
    e0, n0 = None, None
    for n in (4, 8, 16, 32):
        Pi_f = worker_(f, n, radius=radius)
        assert Pi_f.vector().norm('l2') > 0
        e = sqrt(abs(assemble(inner(Pi_f0 - Pi_f, Pi_f0 - Pi_f)*dx)))

        if e0 is not None:
            rate = ln(e/e0)/ln(float(n0)/n)
        else:
            rate = np.inf

        assert rate == np.inf or abs(e) < 1E-10 or rate > 1
        
        n0, e0 = n, e

# --------------------------------------------------------------------

def worker2(f, n, radius=0.3, degree=8):
    '''Check integrals due to averaging operator'''
    mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n, n, n)

    mf = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[1], 0.0) && near(x[2], 0.0)').mark(mf, 1)
    line_mesh = EmbeddedMesh(mf, 1)

    V = FunctionSpace(mesh, 'CG', 1)
    TV = FunctionSpace(line_mesh, 'DG', 1)

    f = interpolate(f, V)

    cylinder = Circle(radius, degree)

    Pi = PETScMatrix(average_matrix(V, TV, cylinder))
    # print('\t', Pi.norm('linf'), max(len(Pi.getrow(i)[0]) for i in range(TV.dim())))
    
    Pi_f = Function(TV)
    Pi.mult(f.vector(), Pi_f.vector())

    return Pi_f

radius = 0.1

fs = (Expression('x[0]', degree=1),
      Expression('x[1]*x[1] + x[2]*x[2]', degree=2),
      Expression('x[0]*(x[1]*x[1] + x[2]*x[2])', degree=1),
      Expression('x[1] + x[2]', degree=1),
      Expression('sin(k*pi*x[0])', k=0.5, degree=2))
      
Pi_fs = (Expression('x[0]', degree=1),
         Constant(radius**2),
         Expression('x[0]*r*r', r=radius, degree=1),
         Constant(0),
         Expression('sin(k*pi*x[0])', k=0.5, degree=2))

@pytest.mark.parametrize(('f', 'Pi_f0'), tuple(zip(fs, Pi_fs)))
def test_avg2(f, Pi_f0, radius=radius):
    e0, n0 = None, None
    for n in (4, 8, 16, 32):
        Pi_f = worker2(f, n, radius=radius)
        assert Pi_f.vector().norm('l2') > 0
        e = sqrt(abs(assemble(inner(Pi_f0 - Pi_f, Pi_f0 - Pi_f)*dx)))

        if e0 is not None:
            rate = ln(e/e0)/ln(float(n0)/n)
        else:
            rate = np.inf

        # assert rate == np.inf or abs(e) < 1E-10 or rate > 1
        
        n0, e0 = n, e
