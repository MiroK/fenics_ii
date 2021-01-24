from xii import EmbeddedMesh
from dolfin import *
import numpy as np


def test_translate_markers_2d(n=32):
    '''
    |----|
    | [] |
    |----|
    '''
    # Let there be a background mesh with facet functions
    mesh = UnitSquareMesh(n, n)

    bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(bdries, 10)
    CompiledSubDomain('near(x[0], 1)').mark(bdries, 20)
    CompiledSubDomain('near(x[1], 0)').mark(bdries, 30)
    CompiledSubDomain('near(x[1], 1)').mark(bdries, 40)
    CompiledSubDomain('near(x[0], 0.25) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries, 1)
    CompiledSubDomain('near(x[0], 0.75) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries, 2)
    CompiledSubDomain('near(x[1], 0.25) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries, 3)
    CompiledSubDomain('near(x[1], 0.75) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries, 4)

    # It will have subdomains
    cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 1)
    inside = '(x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS) && (x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS)'
    CompiledSubDomain(inside).mark(cell_f, 2)

    inside = EmbeddedMesh(cell_f, 2)
    values = inside.translate_markers(bdries, (1, 2, 3, 4, 10, 20, 30, 40))
    assert set(np.unique(values)) == set((0, 1, 2, 3, 4))

    # And they are in the right place
    x, y = mesh.coordinates(), inside.coordinates()
    _, f2v_x = (mesh.init(1, 0), mesh.topology()(1, 0))
    _, f2v_y = (inside.init(1, 0), inside.topology()(1, 0))

    for tag in (1, 2, 3, 4):
        entities_x, = np.where(bdries.array() == tag)
        entities_y, = np.where(values.array() == tag)
        assert len(entities_x) == len(entities_y)

        nodes_x = np.unique(np.hstack([f2v_x(e) for e in entities_x]))
        nodes_y = np.unique(np.hstack([f2v_y(e) for e in entities_y]))
        assert len(nodes_x) == len(nodes_y)

        xx = sorted(map(tuple, x[nodes_x]))
        yy = sorted(map(tuple, y[nodes_y]))
        assert xx == yy

        
def test_translate_markers_3d_facet(n=4):
    '''
    |----|
    | [] |
    |----|
    '''
    # Let there be a background mesh with facet functions
    mesh = UnitCubeMesh(n, n, n)

    bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(bdries, 10)
    CompiledSubDomain('near(x[0], 1)').mark(bdries, 20)
    CompiledSubDomain('near(x[1], 0)').mark(bdries, 30)
    CompiledSubDomain('near(x[1], 1)').mark(bdries, 40)
    CompiledSubDomain('near(x[0], 0.25) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries, 1)
    CompiledSubDomain('near(x[0], 0.75) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries, 2)
    CompiledSubDomain('near(x[1], 0.25) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries, 3)
    CompiledSubDomain('near(x[1], 0.75) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries, 4)

    # It will have subdomains
    cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 1)
    inside = '(x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS) && (x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS)'
    CompiledSubDomain(inside).mark(cell_f, 2)

    inside = EmbeddedMesh(cell_f, 2)
    values = inside.translate_markers(bdries, (1, 2, 3, 4, 10, 20, 30, 40))
    assert set(np.unique(values)) == set((0, 1, 2, 3, 4))

    # And they are in the right place
    x, y = mesh.coordinates(), inside.coordinates()
    _, f2v_x = (mesh.init(1, 0), mesh.topology()(mesh.topology().dim()-1, 0))
    _, f2v_y = (inside.init(1, 0), inside.topology()(mesh.topology().dim()-1, 0))

    for tag in (1, 2, 3, 4):
        entities_x, = np.where(bdries.array() == tag)
        entities_y, = np.where(values.array() == tag)
        assert len(entities_x) == len(entities_y)

        nodes_x = np.unique(np.hstack([f2v_x(e) for e in entities_x]))
        nodes_y = np.unique(np.hstack([f2v_y(e) for e in entities_y]))
        assert len(nodes_x) == len(nodes_y)

        xx = sorted(map(tuple, x[nodes_x]))
        yy = sorted(map(tuple, y[nodes_y]))
        assert xx == yy


def test_translate_markers_3d_facet(n=4):
    '''
    |----|
    | [] |
    |----|
    '''
    # Let there be a background mesh with facet functions
    mesh = UnitCubeMesh(n, n, n)

    bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(bdries, 10)
    CompiledSubDomain('near(x[0], 1)').mark(bdries, 20)
    CompiledSubDomain('near(x[1], 0)').mark(bdries, 30)
    CompiledSubDomain('near(x[1], 1)').mark(bdries, 40)
    CompiledSubDomain('near(x[0], 0.25) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries, 1)
    CompiledSubDomain('near(x[0], 0.75) && ((x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS))').mark(bdries, 2)
    CompiledSubDomain('near(x[1], 0.25) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries, 3)
    CompiledSubDomain('near(x[1], 0.75) && ((x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS))').mark(bdries, 4)

    # It will have subdomains
    cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 1)
    inside = '(x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS) && (x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS)'
    CompiledSubDomain(inside).mark(cell_f, 2)

    inside = EmbeddedMesh(cell_f, 2)
    values = inside.translate_markers(bdries, (1, 2, 3, 4, 10, 20, 30, 40))
    assert set(np.unique(values)) == set((0, 1, 2, 3, 4))

    tdim = mesh.topology().dim()
    # And they are in the right place
    x, y = mesh.coordinates(), inside.coordinates()
    _, f2v_x = (mesh.init(tdim-1, 0), mesh.topology()(tdim-1, 0))
    _, f2v_y = (inside.init(tdim-1, 0), inside.topology()(tdim-1, 0))

    for tag in (1, 2, 3, 4):
        entities_x, = np.where(bdries.array() == tag)
        entities_y, = np.where(values.array() == tag)
        assert len(entities_x) == len(entities_y)

        nodes_x = np.unique(np.hstack([f2v_x(e) for e in entities_x]))
        nodes_y = np.unique(np.hstack([f2v_y(e) for e in entities_y]))
        assert len(nodes_x) == len(nodes_y)

        xx = sorted(map(tuple, x[nodes_x]))
        yy = sorted(map(tuple, y[nodes_y]))
        assert xx == yy


def test_translate_markers_3d_edge(n=4):
    '''
    |----|
    | [] |
    |----|
    '''
    # Let there be a background mesh with facet functions
    mesh = UnitCubeMesh(n, n, n)

    bdries = MeshFunction('size_t', mesh, mesh.topology().dim()-2, 0)
    CompiledSubDomain('near(x[0], 0.25) && near(x[1], 0.25)').mark(bdries, 1)
    CompiledSubDomain('near(x[0], 0.75) && near(x[1], 0.75)').mark(bdries, 2)    

    # It will have subdomains
    cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 1)
    inside = '(x[0] > 0.25-DOLFIN_EPS) && (x[0] < 0.75+DOLFIN_EPS) && (x[1] > 0.25-DOLFIN_EPS) && (x[1] < 0.75+DOLFIN_EPS)'
    CompiledSubDomain(inside).mark(cell_f, 2)

    inside = EmbeddedMesh(cell_f, 2)
    values = inside.translate_markers(bdries, (1, 2))
    assert set(np.unique(values)) == set((0, 1, 2))

    tdim = mesh.topology().dim()
    # And they are in the right place
    x, y = mesh.coordinates(), inside.coordinates()
    _, f2v_x = (mesh.init(tdim-2, 0), mesh.topology()(tdim-2, 0))
    _, f2v_y = (inside.init(tdim-2, 0), inside.topology()(tdim-2, 0))

    for tag in (1, 2):
        entities_x, = np.where(bdries.array() == tag)
        entities_y, = np.where(values.array() == tag)
        assert len(entities_x) == len(entities_y)

        nodes_x = np.unique(np.hstack([f2v_x(e) for e in entities_x]))
        nodes_y = np.unique(np.hstack([f2v_y(e) for e in entities_y]))
        assert len(nodes_x) == len(nodes_y)

        xx = sorted(map(tuple, x[nodes_x]))
        yy = sorted(map(tuple, y[nodes_y]))
        assert xx == yy
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    test_translate_markers_2d(n=32)
    test_translate_markers_3d_facet(n=4)
    test_translate_markers_3d_edge(n=4)
