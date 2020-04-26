from __future__ import absolute_import
from collections import defaultdict
import dolfin as df
import numpy as np

from xii.meshing.embedded_mesh import EmbeddedMesh
from six.moves import zip


class SubDomainMesh(EmbeddedMesh):
    '''Embedded mesh for cell funcions.'''
    def __init__(self, marking_function, markers):
        assert marking_function.dim() == marking_function.mesh().topology().dim()
        EmbeddedMesh.__init__(self, marking_function, markers)

        
def OverlapMesh(mesh1, mesh2, tol=1E-14):
    '''
    Given two subdomain meshes which share a single unique common tag in 
    their marking function we create here a mesh of cells corresponding 
    to that tag. The new mesh cells can be mapped to mesh1/2 cells.
    '''
    assert isinstance(mesh1, SubDomainMesh), type(mesh1)
    assert isinstance(mesh2, SubDomainMesh)

    tdim = mesh1.topology().dim()
    assert mesh2.topology().dim() == tdim
    assert mesh1.geometry().dim() == mesh2.geometry().dim()
    
    # Overlap has to be unique as well (for now)
    tags1 = set(mesh1.marking_function.array())
    tags2 = set(mesh2.marking_function.array())

    common_tags = tags1 & tags2
    assert len(common_tags) == 1
    tag = int(common_tags.pop())

    # A bit of wishful thinking here: create overlap mesh from mesh1
    # and hope it makes sense for mesh2 as well
    emesh = SubDomainMesh(mesh1.marking_function, tag)
    # Now we have a mesh from cells of omega to cells in mesh1. Let's
    # build a map for mesh2 simlar to `build_embedding_map`
    tree = mesh2.bounding_box_tree()
    # Localize first the vertex in mesh2
    mesh2.init(tdim)    # Then we want to find a cell in mesh2 which
    mesh2.init(tdim, 0)  # has the same vertices

    c2v = mesh2.topology()(tdim, 0)

    mesh_x = mesh2.coordinates()
    emesh_x = emesh.coordinates()
    # Get som idea of mesh size to make relative comparison of coords
    scale = max(emesh_x.max(axis=0) - emesh_x.min(axis=0))
    # Also build the map for vertices
    entity_map = {0: [None]*emesh.num_vertices(), tdim: [None]*emesh.num_cells()}
    vertex_map = entity_map[0]
    cell_map = entity_map[tdim]

    collided_cells = {}
    for cell in df.cells(emesh):
        # The idea is the there is exactly on the_cell which will be
        # found in every point-cell collision patches
        the_cell = set()

        for vertex in cell.entities(0):
            # Try to be less efficient by computing each vertex collision
            # only once
            if vertex in collided_cells:
                mcells = collided_cells[vertex]
            else:
                vertex_x = emesh_x[vertex]
                mcells = tree.compute_entity_collisions(df.Point(*vertex_x))
                # What is the id of vertex in the mesh
                mcell_vertices = c2v(next(iter(mcells)))
                the_vertex = min(mcell_vertices, key=lambda v: np.linalg.norm(vertex_x-mesh_x[v]))
                error = np.linalg.norm(vertex_x - mesh_x[the_vertex])/scale
                assert error < tol, 'Found a hanging node %16f' % error

                vertex_map[vertex] = the_vertex
                collided_cells[vertex] = mcells
            
            if not the_cell:
                the_cell.update(mcells)
            else:
                the_cell = the_cell & set(mcells)
        assert len(the_cell) == 1, the_cell
        # Insert
        cell_map[cell.index()] = the_cell.pop()
    # Sanity
    assert not any(v is None for v in entity_map[0])
    assert not any(v is None for v in entity_map[tdim])
    # At this point we can build add the map
    emesh.parent_entity_map[mesh2.id()] = entity_map

    return emesh


# -------------------------------------------------------------------


if __name__ == '__main__':
    mesh = df.UnitSquareMesh(4, 4)
    subdomains = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 3)
    df.CompiledSubDomain('x[0] < 0.25+DOLFIN_EPS').mark(subdomains, 1)
    df.CompiledSubDomain('x[0] > 0.75-DOLFIN_EPS').mark(subdomains, 2)

    mesh1 = SubDomainMesh(subdomains, (1, 3))
    mesh2 = SubDomainMesh(subdomains, (2, 3))
    mesh12 = OverlapMesh(mesh1, mesh2)

    # FIXME: split the file!
    map1 = mesh12.parent_entity_map[mesh1.id()][2]
    map2 = mesh12.parent_entity_map[mesh2.id()][2]
    # Cell check out
    for c, c1, c2 in zip(df.cells(mesh12), map1, map2):
        assert df.near(c.midpoint().distance(df.Cell(mesh1, c1).midpoint()), 0, 1E-14)
        assert df.near(c.midpoint().distance(df.Cell(mesh2, c2).midpoint()), 0, 1E-14)
        
    # Vertices are not that important but anyways
    x1 = mesh1.coordinates(); map1 = mesh12.parent_entity_map[mesh1.id()][0]
    x2 = mesh2.coordinates(); map2 = mesh12.parent_entity_map[mesh2.id()][0]
    for x, i1, i2 in zip(mesh12.coordinates(), map1, map2):
        assert np.linalg.norm(x - x1[i1]) < 1E-13
        assert np.linalg.norm(x - x2[i2]) < 1E-13
