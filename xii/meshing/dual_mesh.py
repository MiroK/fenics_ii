from make_mesh_cpp import make_mesh
import dolfin as df
import numpy as np


class DualMesh(df.Mesh):
    '''        
    Dual mesh (from FVM) is triangles is obtained by dividing each triangle 
    into 6 new ones containing cell mid point and edge midpoints.
    '''
    def __init__(self, mesh):
        tdim, gdim = mesh.topology().dim(), mesh.geometry().dim()
        # Data for the mesh and mapping of old mesh vertex to cells of the dual patch
        coordinates, cells, macro_map = DualMesh.dual_mesh_data(mesh)

        df.Mesh.__init__(self)
        # Fill
        make_mesh(coordinates=coordinates, cells=cells, tdim=tdim, gdim=gdim, mesh=self)

        # Attach data
        self.macro_map = macro_map

    @staticmethod
    def dual_mesh_data(mesh):
        '''
        Coordinates, cells of new mesh + map of old vertices to patch of 
        new cells.
        '''
        tdim = mesh.topology().dim()
        # Just dispatch
        return {1: DualMesh.dual_mesh_line,
                2: DualMesh.dual_mesh_triangle,
                3: DualMesh.dual_mesh_tetrahedron}[tdim](mesh)

    @staticmethod
    def dual_mesh_line(mesh):
        # FIXME: the rest here
        assert mesh.ufl_cell().cellname() == 'interval'
        raise NotImplementedError
    
    @staticmethod
    def dual_mesh_tetrahedron(mesh):
        # FIXME: the rest here
        assert mesh.ufl_cell().cellname() == 'tetrahedron'
        raise NotImplementedError

    @staticmethod
    def dual_mesh_triangle(mesh):
        # FIXME: is there a generic algorithm to handle all simplices
        assert mesh.ufl_cell().cellname() == 'triangle'

        nv = mesh.num_vertices()
        nf = mesh.init(1)
        # Each facet will add one new vertex and so will each cell. Suppose
        # we do the ordering as [old_vtx, new_facet_vtx, new_cell_vtx], then
        # entity idx + offset refers to new vtx index.

        # Build dual mesh by macro patches of the vertex
        mesh.init(2, 1)
        c2f = mesh.topology()(2, 1)

        mesh.init(1, 0)
        f2v = mesh.topology()(1, 0)

        mesh.init(0, 2)
        v2c = mesh.topology()(0, 2)

        cells, macro_map = [], [0]
        for vtx in xrange(nv):
            offset = macro_map[-1]
            # Each cell yields two new one
            for cell in v2c(vtx):
                cell_mid = cell + nv + nf
                # Give me the two facets of the cell connected to vertex
                f0, f1 = filter(lambda f: vtx in f2v(f), c2f(cell))

                cells.append([vtx, f0 + nv, cell_mid])
                cells.append([vtx, cell_mid, f1 + nv])
                offset += 2
            macro_map.append(offset)
            
        cells = np.array(cells, dtype='uintp')

        old_vtx = mesh.coordinates()
        facet_vtx = np.array([facet.midpoint().array() for facet in df.facets(mesh)])
        cell_vtx = np.array([cell.midpoint().array() for cell in df.cells(mesh)])
        # NOTE: even in 2d the midpoint is 3d
        if mesh.geometry().dim() == 2:
            facet_vtx = facet_vtx[:, :2]
            cell_vtx = cell_vtx[:, :2]

        vertices = np.vstack([old_vtx, facet_vtx, cell_vtx])

        return vertices, cells, macro_map
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (UnitCubeMesh, MeshFunction, File, BoundaryMesh,
                        UnitSquareMesh)
    import matplotlib.pyplot as plt
    import operator

    
    mesh = BoundaryMesh(UnitCubeMesh(4, 4, 4), 'exterior')

    dual_mesh = DualMesh(mesh)
    mapping = dual_mesh.macro_map

    # We kept the mesh area unchanged
    old = sum(c.volume() for c in df.cells(mesh))
    dual = sum(c.volume() for c in df.cells(dual_mesh))
    assert abs(old - dual) < 1E-13

    # The overlap vertex of the patch has to be the same geometrical
    # point as the corresponding old mesh vertex
    x_old = mesh.coordinates()
    x_new = dual_mesh.coordinates()

    patch_vertices = [] 
    
    dual_mesh.init(2, 0)
    c2v = dual_mesh.topology()(2, 0)
    for idx_old, (f, l) in enumerate(zip(mapping[:-1], mapping[1:])):
        # f:l are cells of the patch
        idx_new, = reduce(operator.and_, map(set, (c2v(c) for c in range(f, l))))
        patch_vertices.append(idx_new)

        assert np.linalg.norm(x_old[idx_old] - x_new[idx_new]) < 1E-13
        
    # The patch overlap vertices are unqiue
    assert len(set(patch_vertices)) == len(patch_vertices)

    File('old_mesh.pvd') << mesh
    # Show how to build the patch mapping from the data of the dual mesh
    cell_f = MeshFunction('size_t', dual_mesh, 2, 0)
    values = cell_f.array()
    for idx, (f, l) in enumerate(zip(mapping[:-1], mapping[1:])):
        values[f:l] = idx

    File('dual_mesh.pvd') << cell_f

    # Scaling
    nvertices, times = [], []
    for n in (4, 8, 16, 32, 64, 128, 256, 512):

        mesh = UnitSquareMesh(n, n)
        nvertices.append(mesh.num_vertices())
        
        timer = df.Timer('f')
        dual_mesh = DualMesh(mesh)
        time = timer.stop()
        print n, time, mesh.num_vertices(), '->', dual_mesh.num_vertices()
        times.append(time)
    nvertices, times = map(np.array, (nvertices, times))
    
    slope, shift = np.polyfit(np.log2(nvertices), np.log2(times), deg=1)
    print slope
    
    plt.figure()
    plt.xlabel('N (number of vertices)')
    plt.ylabel('dual mesh construction [s]')
    
    plt.loglog(nvertices, times, marker='x')
    plt.loglog(nvertices, (2**shift)*nvertices**slope, label='O(N^(%.2f))' % slope)
    plt.legend(loc='best')
    
    plt.show()
