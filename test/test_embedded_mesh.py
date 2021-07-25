from xii import EmbeddedMesh, ii_assemble, Trace, OuterNormal
from dolfin import *
import numpy as np


def test_2d_cell(n=32, tol=1E-10):
    '''[|]'''
    mesh = UnitSquareMesh(n, n)
    cell_f = MeshFunction('size_t', mesh, 2, 2)
    CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cell_f, 1)

    left = EmbeddedMesh(cell_f, 1)

    # We want to embed
    mappings = left.parent_entity_map[mesh.id()]
    # Is it correct?

    xi, x = left.coordinates(), mesh.coordinates()
    assert min(np.linalg.norm(xi[list(mappings[0].keys())]-x[list(mappings[0].values())], 2, 1)) < tol

    tdim = mesh.topology().dim()
    c2v = mesh.topology()(tdim, 0)

    icells = left.cells()

    vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)
    
    assert all([vertex_match(xi[icells[key]], x[c2v(val)]) for key, val in list(mappings[tdim].items())])


def test_2d(n=32, tol=1E-10):
    '''[|]'''
    mesh = UnitSquareMesh(n, n)
    cell_f = MeshFunction('size_t', mesh, 2, 2)
    CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cell_f, 1)

    left = EmbeddedMesh(cell_f, 1)

    facet_f = MeshFunction('size_t', left, 1, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 1)

    iface = EmbeddedMesh(facet_f, 1)

    # We want to embed
    mappings = iface.parent_entity_map[left.id()]
    # Is it correct?

    xi, x = iface.coordinates(), left.coordinates()
    assert min(np.linalg.norm(xi[list(mappings[0].keys())]-x[list(mappings[0].values())], 2, 1)) < tol

    tdim = left.topology().dim()-1
    left.init(tdim, 0)
    f2v = left.topology()(tdim, 0)

    icells = iface.cells()

    vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)
    
    assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in list(mappings[tdim].items())])

    V = FunctionSpace(left, 'CG', 1)
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

    # And interface
    bmesh = EmbeddedMesh(bdries1, (1, 2, 3, 4))
    # Embedded it viewwed from stokes
    mappings = bmesh.parent_entity_map[mesh1.id()]

    xi, x = bmesh.coordinates(), mesh1.coordinates()
    assert min(np.linalg.norm(xi[list(mappings[0].keys())]-x[list(mappings[0].values())], 2, 1)) < tol

    tdim = mesh1.topology().dim()-1
    mesh1.init(tdim, 0)
    f2v = mesh1.topology()(tdim, 0)

    icells = bmesh.cells()

    vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)
    
    assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in list(mappings[tdim].items())])

    V = VectorFunctionSpace(mesh1, 'CG', 1)
    u = interpolate(Expression(('x[1]', 'x[0]'), degree=1), V)
    
    dx_ = Measure('dx', domain=bmesh)
    n_ = OuterNormal(bmesh, [0.5, 0.5])
    # Because it is divergence free
    assert abs(ii_assemble(dot(n_, Trace(u, bmesh))*dx_)) < tol


def test_2d_color(n=32, tol=1E-10):
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

    # And interface
    bmesh = EmbeddedMesh(bdries1, (1, 2, 3, 4))
    # Embedded it viewwed from stokes
    mappings = bmesh.parent_entity_map[mesh1.id()]

    xi, x = bmesh.coordinates(), mesh1.coordinates()
    assert min(np.linalg.norm(xi[list(mappings[0].keys())]-x[list(mappings[0].values())], 2, 1)) < tol

    tdim = mesh1.topology().dim()-1
    mesh1.init(tdim, 0)
    f2v = mesh1.topology()(tdim, 0)

    icells = bmesh.cells()

    vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)

    assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in list(mappings[tdim].items())])

    # Now color specific
    cf = bmesh.marking_function
    assert set(cf.array()) == set((1, 2, 3, 4))

    assert all(bdries1[v] == cf[k] for k, v in list(mappings[tdim].items()))


def test_3d_cell(n=4, tol=1E-10):
    '''[|]'''
    mesh = UnitCubeMesh(n, n, n)
    cell_f = MeshFunction('size_t', mesh, 3, 2)
    CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cell_f, 1)

    left = EmbeddedMesh(cell_f, 1)

    # We want to embed
    mappings = left.parent_entity_map[mesh.id()]
    # Is it correct?

    xi, x = left.coordinates(), mesh.coordinates()
    assert min(np.linalg.norm(xi[list(mappings[0].keys())]-x[list(mappings[0].values())], 2, 1)) < tol

    tdim = left.topology().dim()
    f2v = mesh.topology()(tdim, 0)

    icells = left.cells()

    vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)

    assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in list(mappings[tdim].items())])


def test_3d(n=4, tol=1E-10):
    '''[|]'''
    mesh = UnitCubeMesh(n, n, n)
    cell_f = MeshFunction('size_t', mesh, 3, 2)
    CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cell_f, 1)

    left = EmbeddedMesh(cell_f, 1)

    facet_f = MeshFunction('size_t', left, 2, 0)
    CompiledSubDomain('near(x[0], 0.5)').mark(facet_f, 1)

    iface = EmbeddedMesh(facet_f, 1)

    # We want to embed
    mappings = iface.parent_entity_map[left.id()]
    # Is it correct?

    xi, x = iface.coordinates(), left.coordinates()
    assert min(np.linalg.norm(xi[list(mappings[0].keys())]-x[list(mappings[0].values())], 2, 1)) < tol

    tdim = left.topology().dim()-1
    left.init(tdim, 0)
    f2v = left.topology()(tdim, 0)

    icells = iface.cells()

    vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)
    
    assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in list(mappings[tdim].items())])

    V = FunctionSpace(left, 'CG', 1)
    u = interpolate(Expression('x[0] + x[1]', degree=1), V)
    dx_ = Measure('dx', domain=iface)
    assert abs(ii_assemble(Trace(u, iface)*dx_) - 1.0) < tol


def test_3d_performance(n=4, tol=1E-10, strict=False):
    '''[|]'''
    mesh = UnitCubeMesh(n, n, n)
    bdries1 = MeshFunction('size_t', mesh, 2, 0)
    DomainBoundary().mark(bdries1, 1)
    CompiledSubDomain('near(x[0], 0.25) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries1, 1)
    CompiledSubDomain('near(x[0], 0.75) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries1, 1)
    CompiledSubDomain('near(x[1], 0.25) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries1, 1)
    CompiledSubDomain('near(x[1], 0.75) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries1, 1)

    t = Timer('embed')
    iface = EmbeddedMesh(bdries1, 1)
    dt = t.stop()

    if strict:
        # We want to embed
        mappings = iface.parent_entity_map[mesh.id()]
        # Is it correct?

        xi, x = iface.coordinates(), mesh.coordinates()
        assert min(np.linalg.norm(xi[list(mappings[0].keys())]-x[list(mappings[0].values())], 2, 1)) < tol
        
        tdim = mesh.topology().dim()-1
        mesh.init(tdim, 0)
        f2v = mesh.topology()(tdim, 0)
        
        icells = iface.cells()

        vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)
    
        assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in list(mappings[tdim].items())])

    return (dt, mesh.num_cells(), iface.num_cells())

def test_3d_performance_cell(n=4, tol=1E-10, strict=False):
    '''[|]'''
    mesh = UnitCubeMesh(n, n, n)
    cell_f = MeshFunction('size_t', mesh, 3, 0)
    CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS').mark(cell_f, 1)

    t = Timer('embed')
    iface = EmbeddedMesh(cell_f, 1)
    dt = t.stop()

    if strict:
        # We want to embed
        mappings = iface.parent_entity_map[mesh.id()]
        # Is it correct?

        xi, x = iface.coordinates(), mesh.coordinates()
        assert min(np.linalg.norm(xi[list(mappings[0].keys())]-x[list(mappings[0].values())], 2, 1)) < tol
        
        tdim = mesh.topology().dim()
        mesh.init(tdim, 0)
        f2v = mesh.topology()(tdim, 0)
        
        icells = iface.cells()

        vertex_match = lambda xs, ys: all(min(np.linalg.norm(ys - x, 2, 1)) < tol for x in xs)
    
        assert all([vertex_match(xi[icells[key]], x[f2v(val)]) for key, val in list(mappings[tdim].items())])

    return (dt, mesh.num_cells(), iface.num_cells())


# --------------------------------------------------------------------

if __name__ == '__main__':
    test_2d_cell(32)    
    test_2d(64)
    test_2d_enclosed(64)

    test_2d_color(64)

    test_3d(4)
    test_3d_cell(4)
    
    test_3d_performance(4, strict=True)    

    dt0, all_cells0, embd_cells0 = None, None, None
    for n in (4, 8, 16, 32, 64):
        dt, all_cells, embd_cells = test_3d_performance(n)
        if dt0 is not None:
            rate_all = ln(dt/dt0)/ln(all_cells/all_cells0)
            rate_embd = ln(dt/dt0)/ln(embd_cells/embd_cells0)
        else:
            rate_all, rate_embd = -1, -1
            
        print(('%.3f %d[%.2f] %d[%.2f]' % (dt, all_cells, rate_all, embd_cells, rate_embd)))
        dt0, all_cells0, embd_cells0 = dt, float(all_cells), float(embd_cells)

    print()

    test_3d_performance_cell(4, strict=True)    

    dt0, all_cells0, embd_cells0 = None, None, None
    for n in (4, 8, 16, 32, 64):
        dt, all_cells, embd_cells = test_3d_performance_cell(n)
        if dt0 is not None:
            rate_all = ln(dt/dt0)/ln(all_cells/all_cells0)
            rate_embd = ln(dt/dt0)/ln(embd_cells/embd_cells0)
        else:
            rate_all, rate_embd = -1, -1
            
        print(('%.3f %d[%.2f] %d[%.2f]' % (dt, all_cells, rate_all, embd_cells, rate_embd)))
        dt0, all_cells0, embd_cells0 = dt, float(all_cells), float(embd_cells)
    
