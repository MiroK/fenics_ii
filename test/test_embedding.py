from xii import EmbeddedMesh, ii_assemble, Trace, OuterNormal
from dolfin import *
import numpy as np


def test_2d(n=32, tol=1E-10):
    '''[|]'''
    mesh = UnitSquareMesh(n, n)
    cell_f = MeshFunction('size_t', mesh, 2, 2)
    CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cell_f, 1)

    left = EmbeddedMesh(cell_f, 1)
    right = EmbeddedMesh(cell_f, 2)

    facet_f = MeshFunction('size_t', left, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 1)

    iface = EmbeddedMesh(facet_f, 1)

    # Now suppose
    facet_f = MeshFunction('size_t', right, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 2)

    # We want to embed
    mappings = iface.compute_embedding(facet_f, 2)
    # Is it correct?

    xi, x = iface.coordinates(), right.coordinates()
    assert min(np.linalg.norm(xi[mappings[0].keys()]-x[mappings[0].values()], 2, 1)) < tol

    tdim = right.topology().dim()-1
    right.init(tdim, 0)
    f2v = right.topology()(tdim, 0)

    icells = iface.cells()

    vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)
    
    assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in mappings[tdim].items()])

    try:
        iface.compute_embedding(facet_f, 2)
    except ValueError:
        pass

    V = FunctionSpace(right, 'CG', 1)
    u = interpolate(Expression('x[1]', degree=1), V)
    dx_ = Measure('dx', domain=iface)
    assert abs(ii_assemble(Trace(u, iface)*dx_) - 0.5) < tol

    
def test_2d_enclosed(n=32, tol=1E-10):
    '''
    |----|
    | [] |
    |----|
    '''
    mesh = UnitSquareMesh(n, n)
    # Lets get the outer part
    cell_f = MeshFunction('size_t', mesh, 2, 1)
    inside = '(x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS) && (x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS)'
    CompiledSubDomain(inside).mark(cell_f, 2)

    # Stokes ---
    mesh1 = EmbeddedMesh(cell_f, 1)

    bdries1 = MeshFunction('size_t', mesh1, mesh1.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(bdries1, 10)
    CompiledSubDomain('near(x[0], 1)').mark(bdries1, 20)
    CompiledSubDomain('near(x[1], 0)').mark(bdries1, 30)
    CompiledSubDomain('near(x[1], 1)').mark(bdries1, 40)

    CompiledSubDomain('near(x[0], 0.25) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries1, 1)
    CompiledSubDomain('near(x[0], 0.75) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries1, 2)
    CompiledSubDomain('near(x[1], 0.25) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries1, 3)
    CompiledSubDomain('near(x[1], 0.75) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries1, 4)

    # Darcy ---
    mesh2 = EmbeddedMesh(cell_f, 2)
    bdries2 = MeshFunction('size_t', mesh2, mesh2.topology().dim()-1, 0)

    CompiledSubDomain('near(x[0], 0.25) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries2, 1)
    CompiledSubDomain('near(x[0], 0.75) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries2, 2)
    CompiledSubDomain('near(x[1], 0.25) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries2, 3)
    CompiledSubDomain('near(x[1], 0.75) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries2, 4)

    # -----------------

    # And interface
    bmesh = EmbeddedMesh(bdries2, (1, 2, 3, 4))
    # Embedded it viewwed from stokes
    mappings = bmesh.compute_embedding(bdries1, (1, 2, 3, 4))

    xi, x = bmesh.coordinates(), mesh1.coordinates()
    assert min(np.linalg.norm(xi[mappings[0].keys()]-x[mappings[0].values()], 2, 1)) < tol

    tdim = mesh1.topology().dim()-1
    mesh1.init(tdim, 0)
    f2v = mesh1.topology()(tdim, 0)

    icells = bmesh.cells()

    vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)
    
    assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in mappings[tdim].items()])

    try:
        bmesh.compute_embedding(bdries1, 2)
    except ValueError:
        pass

    V = VectorFunctionSpace(mesh1, 'CG', 1)
    u = interpolate(Expression(('x[1]', 'x[0]'), degree=1), V)
    
    dx_ = Measure('dx', domain=bmesh)
    n_ = OuterNormal(bmesh, [0.5, 0.5])
    # Because it is divergence free
    assert abs(ii_assemble(dot(n_, Trace(u, bmesh))*dx_)) < tol


def test_2d_incomplete(n=32, tol=1E-10):
    '''[|]'''
    mesh = UnitSquareMesh(n, n)
    cell_f = MeshFunction('size_t', mesh, 2, 2)
    CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cell_f, 1)

    left = EmbeddedMesh(cell_f, 1)
    right = EmbeddedMesh(cell_f, 2)

    facet_f = MeshFunction('size_t', left, 1, 0)
    CompiledSubDomain('near(x[0], 0.0)').mark(facet_f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 2)
    CompiledSubDomain('near(x[1], 0.0)').mark(facet_f, 3)
    CompiledSubDomain('near(x[1], 1.0)').mark(facet_f, 4)
    # More complicated iface
    iface = EmbeddedMesh(facet_f, (1, 2, 3, 4))

    # Right will interact with it in a simpler way
    facet_f = MeshFunction('size_t', right, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 2)

    # We want to embed
    mappings = iface.compute_embedding(facet_f, 2)
    # Is it correct?

    xi, x = iface.coordinates(), right.coordinates()
    assert min(np.linalg.norm(xi[mappings[0].keys()]-x[mappings[0].values()], 2, 1)) < tol

    tdim = right.topology().dim()-1
    right.init(tdim, 0)
    f2v = right.topology()(tdim, 0)

    icells = iface.cells()

    vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)
    
    assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in mappings[tdim].items()])

    try:
        iface.compute_embedding(facet_f, 2)
    except ValueError:
        pass

    V = FunctionSpace(right, 'CG', 2)
    u = interpolate(Expression('x[1]*x[1]', degree=1), V)
    # FIXME: this passes but looks weird
    dx_ = Measure('dx', domain=iface, subdomain_data=iface.marking_function)
    assert abs(ii_assemble(Trace(u, iface, tag=2)*dx_(2)) - 1./3) < tol


def test_3d(n=4, tol=1E-10):
    '''[|]'''
    mesh = UnitCubeMesh(n, n, n)
    cell_f = MeshFunction('size_t', mesh, 3, 2)
    CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cell_f, 1)

    left = EmbeddedMesh(cell_f, 1)
    right = EmbeddedMesh(cell_f, 2)

    facet_f = MeshFunction('size_t', left, 2, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 1)

    iface = EmbeddedMesh(facet_f, 1)

    # Now suppose
    facet_f = MeshFunction('size_t', right, 2, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 2)

    # We want to embed
    mappings = iface.compute_embedding(facet_f, 2)
    # Is it correct?

    xi, x = iface.coordinates(), right.coordinates()
    assert min(np.linalg.norm(xi[mappings[0].keys()]-x[mappings[0].values()], 2, 1)) < tol

    tdim = right.topology().dim()-1
    right.init(tdim, 0)
    f2v = right.topology()(tdim, 0)

    icells = iface.cells()

    vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)
    
    assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in mappings[tdim].items()])

    try:
        iface.compute_embedding(facet_f, 2)
    except ValueError:
        pass

    V = FunctionSpace(right, 'CG', 1)
    u = interpolate(Expression('x[0] + x[1]', degree=1), V)
    dx_ = Measure('dx', domain=iface)
    assert abs(ii_assemble(Trace(u, iface)*dx_) - 1.0) < tol

# --------------------------------------------------------------------

if __name__ == '__main__':
    test_2d(64)
    test_2d_incomplete(64)
    test_2d_enclosed(64)    

    test_3d(4)
