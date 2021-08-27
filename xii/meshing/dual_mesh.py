from xii.meshing.cell_centers import CellCircumcenter, CellCentroid
from xii.meshing.make_mesh_cpp import make_mesh
import dolfin as df
import numpy as np
from functools import reduce


def dual_mesh(mesh, center='centroid', nrefs=1):
    '''        
    Dual mesh (from FVM) is triangles is obtained by dividing each triangle 
    into 6 new ones containing cell (mid point|circumcenter) and edge midpoints.
    '''
    # Base case
    if nrefs == 1:
        tdim, gdim = mesh.topology().dim(), mesh.geometry().dim()
        # Data for the mesh and mapping of old mesh vertex to cells of the dual patch
        coordinates, cells, mapping = dual_mesh_data(mesh, center)

        # Fill
        dmesh = make_mesh(coordinates=coordinates, cells=cells, tdim=tdim, gdim=gdim)
        # Attach data
        dmesh.parent_entity_map = {mesh.id(): mapping}

        return dmesh

    # At least 2
    root_id = mesh.id()
    tdim = mesh.topology().dim()
    
    mesh0 = dual_mesh(mesh, center, nrefs=1)
    nrefs -= 1
    # Root will stay the same for those are the vertices that
    # were originally in the mesh and only those can be traced
    ref_vertices, root_vertices = zip(*mesh0.parent_entity_map[mesh.id()][0].items())
        
    while nrefs > 0:
        nrefs -= 1
            
        mesh1 = dual_mesh(mesh0, center, nrefs=1)
        # Upda mesh1 mapping
        mapping0, = mesh0.parent_entity_map.values()
        mapping1, = mesh1.parent_entity_map.values()
            
        new_mapping = {}
        # New cells fall under some parent
        e_mapping0, e_mapping1 = mapping0[tdim], mapping1[tdim]
        new_mapping[tdim] = {k: e_mapping0[v] for k, v in e_mapping1.items()}
        # But that's not the case with vertices, we only look for root
        # ones in the new
        e_mapping1 = mapping1[0]
        ref_vertices = [ref_v for ref_v, coarse_v in e_mapping1.items()
                        if coarse_v in ref_vertices]
        assert len(ref_vertices) == len(root_vertices)
        new_mapping[0] = dict(zip(ref_vertices, root_vertices))

        mesh1.parent_entity_map = {root_id: new_mapping}
            
        mesh0 = mesh1

    return mesh0


def dual_mesh_data(mesh, center):
    '''
    Coordinates, cells of new mesh + map of old vertices to patch of 
    new cells.
    '''
    tdim = mesh.topology().dim()
    # Just dispatch
    return {1: dual_mesh_line,
            2: dual_mesh_triangle,
            3: dual_mesh_tetrahedron}[tdim](mesh, center)


def dual_mesh_line(mesh, center):
    # FIXME: the rest here
    assert mesh.ufl_cell().cellname() == 'interval'
    raise NotImplementedError


def dual_mesh_tetrahedron(mesh, center):
    # FIXME: the rest here
    assert mesh.ufl_cell().cellname() == 'tetrahedron'
    raise NotImplementedError


def dual_mesh_triangle(mesh, center):
    # FIXME: is there a generic algorithm to handle all simplices
    assert mesh.ufl_cell().cellname() == 'triangle'

    ncells = mesh.num_cells()

    x = mesh.coordinates()  # the old ones
    start, gdim = x.shape
    # In the new mesh there will as extra vertices the center points ...
    center = {'centroid': CellCentroid,
              'circumcenter': CellCircumcenter}[center]

    centers = df.interpolate(
        center(mesh), df.VectorFunctionSpace(mesh, 'DG', 0)
    ).vector().get_local().reshape((-1, gdim))

    # ... and the facet midpoints
    facet_centers = np.array([f.midpoint().array() for f in df.facets(mesh)])
    facet_centers = facet_centers[:, :gdim]

    # Vertices of the new mesh will be [x;centers;facet_centers]
    # The old cell to center map is then
    c2center = np.arange(start, start+ncells)
    start = c2center[-1]+1
    # and facet to vertex that is the facet center is ...
    f2center = np.arange(start, start+len(facet_centers))
    # Cobine new vertices to get all coords of the mesh
    vertices = np.row_stack([x, centers, facet_centers])

    # Builing cells
    _, c2f = mesh.init(2, 1), mesh.topology()(2, 1)
    _, f2v = mesh.init(1, 0), mesh.topology()(1, 0)

    # Each cell contibutes 6, (2 per facet)
    cells = []
    for cell in range(ncells):
        cell_center = c2center[cell]
        for facet in c2f(cell):
            facet_center = f2center[facet]
            cells.extend([(cell_center, facet_center, v) for v in f2v(facet)])
    cells = np.array(cells)

    cell_map = np.repeat(np.arange(ncells), 6)
    vertex_map = np.arange(mesh.num_vertices())

    mapping = {0: dict(enumerate(vertex_map)),
               2: dict(enumerate(cell_map))}

    return vertices, cells, mapping
