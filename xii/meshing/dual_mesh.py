from xii.meshing.cell_centers import CellCircumcenter, CellCentroid
from xii.meshing.make_mesh_cpp import make_mesh
import dolfin as df
import numpy as np
from functools import reduce


class DualMesh(df.Mesh):
    '''        
    Dual mesh (from FVM) is triangles is obtained by dividing each triangle 
    into 6 new ones containing cell (mid point|circumcenter) and edge midpoints.
    '''
    def __init__(self, mesh, center='centroid', nrefs=1):
        tdim, gdim = mesh.topology().dim(), mesh.geometry().dim()
        # Data for the mesh and mapping of old mesh vertex to cells of the dual patch
        coordinates, cells, mapping = DualMesh.dual_mesh_data(mesh, center)

        if nrefs == 1:
            df.Mesh.__init__(self)
            # Fill
            make_mesh(coordinates=coordinates, cells=cells, tdim=tdim, gdim=gdim, mesh=self)
            # Attach data
            self.parent_entity_map = {mesh.id(): mapping}

    @staticmethod
    def dual_mesh_data(mesh, center):
        '''
        Coordinates, cells of new mesh + map of old vertices to patch of 
        new cells.
        '''
        tdim = mesh.topology().dim()
        # Just dispatch
        return {1: DualMesh.dual_mesh_line,
                2: DualMesh.dual_mesh_triangle,
                3: DualMesh.dual_mesh_tetrahedron}[tdim](mesh, center)

    @staticmethod
    def dual_mesh_line(mesh, center):
        # FIXME: the rest here
        assert mesh.ufl_cell().cellname() == 'interval'
        raise NotImplementedError
    
    @staticmethod
    def dual_mesh_tetrahedron(mesh, center):
        # FIXME: the rest here
        assert mesh.ufl_cell().cellname() == 'tetrahedron'
        raise NotImplementedError

    @staticmethod
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
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (UnitCubeMesh, MeshFunction, File, BoundaryMesh,
                        UnitSquareMesh)
    import matplotlib.pyplot as plt
    import operator

    # mesh = BoundaryMesh(UnitCubeMesh(4, 4, 4), 'exterior')
    mesh = UnitSquareMesh(1, 1)

    dual_mesh = DualMesh(mesh)
    mapping = dual_mesh.parent_entity_map[mesh.id()][2]

    # We kept the mesh area unchanged
    old = sum(c.volume() for c in df.cells(mesh))
    dual = sum(c.volume() for c in df.cells(dual_mesh))
    assert abs(old - dual) < 1E-13, (old, dual)

    from xii.meshing.refinement import point_is_inside

    x = dual_mesh.coordinates()
    for dc in df.cells(dual_mesh):
        old_cell = mapping[dc.index()]

        mp = dc.midpoint().array()[:2]
        assert point_is_inside(mp, x[dc.entities(0)], tol=1E-10)
#     # The patch overlap vertices are unqiue
#     assert len(set(patch_vertices)) == len(patch_vertices)

#     File('old_mesh.pvd') << mesh
#     # Show how to build the patch mapping from the data of the dual mesh
#     cell_f = MeshFunction('size_t', dual_mesh, 2, 0)
#     values = cell_f.array()
#     for idx, (f, l) in enumerate(zip(mapping[:-1], mapping[1:])):
#         values[f:l] = idx

#     File('dual_mesh.pvd') << cell_f

    # Scaling
    nvertices, times = [], []
    for n in (4, 8, 16, 32, 64, 128, 256, 512):

        mesh = UnitSquareMesh(n, n)
        nvertices.append(mesh.num_vertices())
        
        timer = df.Timer('f')
        dual_mesh = DualMesh(mesh)
        time = timer.stop()
        print(n, time, mesh.num_vertices(), '->', dual_mesh.num_vertices())
        times.append(time)
    nvertices, times = list(map(np.array, (nvertices, times)))
    
    slope, shift = np.polyfit(np.log2(nvertices), np.log2(times), deg=1)
    print(slope)
    
    plt.figure()
    plt.xlabel('N (number of vertices)')
    plt.ylabel('dual mesh construction [s]')
    
    plt.loglog(nvertices, times, marker='x')
    plt.loglog(nvertices, (2**shift)*nvertices**slope, label='O(N^(%.2f))' % slope)
    plt.legend(loc='best')
    
    plt.show()
