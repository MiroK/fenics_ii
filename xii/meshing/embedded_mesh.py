from make_mesh_cpp import make_mesh
from collections import defaultdict
from scipy.spatial import cKDTree
from itertools import chain
import dolfin as df
import numpy as np
import operator


class EmbeddedMesh(df.Mesh):
    '''
    Construct a mesh of marked entities in marking_function.
    The output is the mesh with cell function which inherited the markers. 
    and an antribute `parent_entity_map` which is dict with a map of new 
    mesh vertices to the old ones, and new mesh cells to the old mesh entities.
    Having several maps in the dict is useful for mortaring.
    '''
    def __init__(self, marking_function, markers):
        if not isinstance(markers, (list, tuple)): markers = [markers]
        
        # Convenience option to specify only subdomains
        is_number = lambda m: isinstance(m, int)
        new_markers = []
        # Build a new list int list with facet_function marked
        if not all(map(is_number, markers)):
            
            numbers = filter(is_number, markers)
            next_int_marker = max(numbers) if numbers else 0
            for marker in markers:
                if is_number(marker):
                    new_markers.append(marker)
                else:
                    next_int_marker += 1
                    # SubDomain
                    try:
                        marker.mark(marking_function, next_int_marker)
                    except AttributeError:
                        # A string
                        try:
                            df.CompiledSubDomain(marker).mark(marking_function, next_int_marker)
                        # A lambda
                        except TypeError:
                            df.CompiledSubDomain(*marker()).mark(marking_function, next_int_marker)

                    new_markers.append(next_int_marker)
            
            markers = new_markers
            
        base_mesh = marking_function.mesh()

        assert base_mesh.topology().dim() >= marking_function.dim()

        # Work in serial only (much like submesh)
        assert df.MPI.size(base_mesh.mpi_comm()) == 1

        gdim = base_mesh.geometry().dim()
        tdim = marking_function.dim()
        assert tdim > 0, 'No Embedded mesh from vertices'

        assert markers, markers

        # We reuse a lot of Submesh capabilities if marking by cell_f
        if base_mesh.topology().dim() == marking_function.dim():
            # Submesh works only with one marker so we conform
            color_array = marking_function.array()
            color_cells = dict((m, np.where(color_array == m)[0]) for m in markers)

            # So everybody is marked as 1
            one_cell_f = df.MeshFunction('size_t', base_mesh, tdim, 0)
            for cells in color_cells.itervalues(): one_cell_f.array()[cells] = 1
            
            # The Embedded mesh now steals a lot from submesh
            submesh = df.SubMesh(base_mesh, one_cell_f, 1)

            df.Mesh.__init__(self, submesh)

            # The entity mapping attribute;
            # NOTE: At this point there is not reason to use a dict as
            # a lookup table
            mesh_key = marking_function.mesh().id()
            mapping_0 = submesh.data().array('parent_vertex_indices', 0)

            mapping_tdim = submesh.data().array('parent_cell_indices', tdim)
            self.parent_entity_map = {mesh_key: {0: dict(enumerate(mapping_0)),
                                                 tdim: dict(enumerate(mapping_tdim))}}
            # Finally it remains to preserve the markers
            f = df.MeshFunction('size_t', self, tdim, 0)
            f_values = f.array()
            if len(markers) > 1:
                old2new = dict(zip(mapping_tdim, range(len(mapping_tdim))))
                for color, old_cells in color_cells.iteritems():
                    new_cells = np.array([old2new[o] for o in old_cells], dtype='uintp')
                    f_values[new_cells] = color
            else:
                f.set_all(markers[0])
            
            self.marking_function = f
            # Declare which tagged cells are found
            self.tagged_cells = set(markers)
            # https://stackoverflow.com/questions/2491819/how-to-return-a-value-from-init-in-python            
            return None  

        # Otherwise the mesh needs to by build from scratch
        base_mesh.init(tdim, 0)
        t2v = base_mesh.topology()(tdim, 0)
        # Collect unique vertices based on their new-mesh indexing
        vertex_map, seen, = dict(), set()
        # The cells of the embedded mesh are defined in terms of their embedded-numbering
        new_cells, new_vertices = [], []
        # We want the cells to inherit the colors of entities
        color_map, cell_map = {}, []
        
        mf_array = marking_function.array()
        cell_idx = 0  # New one
        for marker in markers:
            entities_of_marker, = np.where(mf_array == marker)

            cell_map.extend(entities_of_marker)
            # Represent entities in terms of their vertices (vertices are of base mesh)
            entities_old_vertex = np.row_stack([t2v(e) for e in entities_of_marker])
            marker_vertices = set(entities_old_vertex.flat)
            # See if these have been seen with the previous color and rebuild
            # mapping of old to new vertices
            new_marker_vertices = marker_vertices - seen
            new_vertices.extend(new_marker_vertices)

            # Mapping from new mesh where entities are cell
            vertex_map.update({old: new for new, old in enumerate(new_marker_vertices, len(vertex_map))})

            # The mapping is then used to encode entities as new cells
            marker_cells = np.fromiter(map(vertex_map.__getitem__, entities_old_vertex.flat),
                                       dtype='uintp').reshape(entities_old_vertex.shape)
            new_cells.append(marker_cells)
            # For colors
            color_map[marker] = (cell_idx, cell_idx + len(marker_cells))
            # For next round
            seen.update(new_vertices)
            cell_idx += len(marker_cells)

        vertex_coordinates = base_mesh.coordinates()[new_vertices]
        new_cells = np.row_stack(new_cells)
        
        # With acquired data build the mesh
        df.Mesh.__init__(self)
        # Fill
        make_mesh(coordinates=vertex_coordinates, cells=new_cells, tdim=tdim, gdim=gdim,
                  mesh=self)

        # The entity mapping attribute
        mesh_key = marking_function.mesh().id()
        self.parent_entity_map = {mesh_key: {0: dict(enumerate(new_vertices)),
                                             tdim: dict(enumerate(cell_map))}}

        f = df.MeshFunction('size_t', self, tdim, 0)
        f_ = f.array()
        # Finally the inherited marking function
        if len(markers) > 1:
            for marker, (c0, c1) in color_map.iteritems():
                f_[np.arange(c0, c1)] = marker
        else:
            f.set_all(markers[0])

        self.marking_function = f
        # Declare which tagged cells are found
        self.tagged_cells = set(markers)

        
class OuterNormal(df.Function):
    '''Outer normal of a manifold mesh as a vector DG0 function.'''
    def __init__(self, mesh, orientation):

        # Manifold assumption
        assert 1 <= mesh.topology().dim() < mesh.geometry().dim()
        gdim = mesh.geometry().dim()

        # Orientation from inside point
        if isinstance(orientation, (list, np.ndarray, tuple)):
            assert len(orientation) == gdim

            kwargs = {'x0%d' % i: val for i, val in enumerate(orientation)}
            orientation = ['x[%d] - x0%d' % (i, i) for i in range(gdim)]
            orientation = df.Expression(orientation, degree=1, **kwargs)
        
        assert orientation.ufl_shape == (gdim, )    

        V = df.VectorFunctionSpace(mesh, 'DG', 0, gdim)
        df.Function.__init__(self, V)
        n_values = self.vector().get_local()

        X = mesh.coordinates()

        dd = X[np.argmin(X[:, 0])] - X[np.argmax(X[:, 0])] 
        
        values = []
        R = np.array([[0, -1], [1, 0]])

        for cell in df.cells(mesh):
            n = cell.cell_normal().array()[:gdim]

            x = cell.midpoint().array()[:gdim]
            # Disagree?
            if np.inner(orientation(x), n) < 0:
                n *= -1.
            values.append(n/np.linalg.norm(n))

        values = np.array(values)

        for sub in range(gdim):
            dofs = V.sub(sub).dofmap().dofs()
            n_values[dofs] = values[:, sub]
        self.vector().set_local(n_values)
        self.vector().apply('insert')


def InnerNormal(mesh, orientation):
    '''-1 * the outer normal'''
    n = OuterNormal(mesh, orientation)
    n.vector()[:] *= -1
    return n

def is_1_sequence(iterable):
    '''value mathces the index (-offset)'''
    steps = np.unique(np.diff(np.sort(np.fromiter(iterable, dtype='uintp'))))
    if not len(steps) == 1:
        return False

    return steps[0] == 1

def build_embedding_map(emesh, mesh, esubdomains=None, tags=None, tol=1E-14):
    '''
    Operating with the assumption that the emsh consists of entities 
    of mesh we find here a map from emesh vertices and cells to mesh
    vertices and entities.
    '''
    df.info('\tEmbedding map'); e_timer = df.Timer('emap')
    assert emesh.topology().dim() < mesh.topology().dim()
    edim = emesh.topology().dim()

    # We have right subdomains, i-e- a cell function
    assert esubdomains is None or esubdomains.dim() == edim

    # Let's make the inputs consistent
    if esubdomains is None:
        assert tags is None
        esubdomains = df.MeshFunction('size_t', emesh, edim, 0)

    # Meaning all the emesh cells and vertices need to be found
    all_check = tags is None
    # All the cells
    if all_check: tags = set((0, ))

    # We might be lucky and this is a boundary mesh -> extract
    if hasattr(emesh, 'entity_map'):
        # One must be careful here for it is guaranteed that emsh was
        # constructed from mesh. This has to be flagged by the user
        if hasattr(emesh, 'parent_id') and emesh.parent_id == mesh.id():
            entity_map = {0: dict(enumerate(emesh.entity_map(0).array())),
                          edim: dict(enumerate(emesh.entity_map(edim).array()))}
            df.info('\tDone (Embeddeding map by extracting) %g' % e_timer.stop())

            return entity_map
        # Otherwise we work hard
        
    # Localization will require
    tree = mesh.bounding_box_tree()
    # to find candidata cells. We zoom in on the unique entity by
    mesh.init(edim)   # Set amoong which emesh cells will be searched
    mesh.init(mesh.topology().dim(), edim)  # Via cell connectivity
    mesh.init(edim, 0)  # And coordinate comparison

    c2v = mesh.topology()(mesh.topology().dim(), 0)
    c2e = mesh.topology()(mesh.topology().dim(), edim)
    e2v = mesh.topology()(edim, 0)

    tagged_cells = chain(*[df.SubsetIterator(esubdomains, tag) for tag in tags])

    mesh_x = mesh.coordinates()
    emesh_x = emesh.coordinates()
    # Get som idea of mesh size to make relative comparison of coords
    scale = max(emesh_x.max(axis=0) - emesh_x.min(axis=0))
    # Also build the map for vertices
    entity_map = {0: dict(), edim: dict()}
    vertex_map = entity_map[0]
    cells_with_vertex = dict()
    for cell in tagged_cells:
        
        the_entity = set()
        for vertex in cell.entities(0):
            if vertex not in vertex_map:
                vertex_x = emesh_x[vertex]
                mcells = tree.compute_entity_collisions(df.Point(*vertex_x))

                # What is the id of vertex in the mesh
                mcell_vertices = c2v(mcells[0])
                the_vertex = min(mcell_vertices, key=lambda v: np.linalg.norm(vertex_x-mesh_x[v]))
                error = np.linalg.norm(vertex_x - mesh_x[the_vertex])/scale
                assert error < tol, 'Found a hanging node %16f' % error
            
                vertex_map[vertex] = the_vertex
                cells_with_vertex[vertex] = mcells
            else:
                the_vertex = vertex_map[vertex]
                mcells = cells_with_vertex[vertex]
            # For each I want to get its entities which containt the vertex
            # We are after such (UNIQUE) entity which would be in each such
            # set build for a vertex
            vertex_set = {entity
                          for mcell in mcells for entity in c2e(mcell)
                          if the_vertex in e2v(entity)}

            if not the_entity:
                the_entity.update(vertex_set)
            else:
                the_entity.intersection_update(vertex_set)
        assert len(the_entity) == 1
        # Insert
        entity_map[edim][cell.index()] = the_entity.pop()

    if all_check:
        # All and continuous
        assert len(entity_map[0]) == emesh.num_vertices()
        assert len(entity_map[edim]) == emesh.num_cells()
        # Continuity
        assert is_1_sequence(entity_map[0]) 
        assert is_1_sequence(entity_map[edim])

    df.info('\tDone (Embeddeding map) %g' % e_timer.stop())
    return entity_map
# NOTE: vertex map could be computed with KDTree but that is slower and
# scales worse!
#
#    parent = mesh.coordinates()
#    child = emesh.coordinates()

#    tree = cKDTree(parent, leafsize=24)
#    _, vertex_map = tree.query(child, k=1)


# -------------------------------------------------------------------


if __name__ == '__main__':
    # Embedding map
    n0, dt0 = None, None
    for n in []: #8, 16, 32, 64, 128, 256, 512, 1024]:

        mesh = df.UnitSquareMesh(n, n)
        emesh = df.BoundaryMesh(mesh, 'exterior')
        
        time = df.Timer('map'); time.start()        
        mapping = build_embedding_map(emesh, mesh, tol=1E-14)

        # mapping_ = build_embedding_map__(emesh, mesh, tol=1E-14)

        # for k in mapping:
        #     assert np.linalg.norm(np.array(mapping[k][0]) - np.array(mapping_[k][0])) < 1E-13
        #     assert np.linalg.norm(np.array(mapping[k][2]) - np.array(mapping_[k][2])) < 1E-13

        dt = time.stop()
        
        mesh_x = mesh.coordinates()
        emesh_x = emesh.coordinates()

        assert max(np.linalg.norm(ex - mesh_x[mapping[0][to]])
                   for ex, to in zip(emesh_x, mapping[0])) < 1E-14

        assert max(df.Facet(mesh, entity).midpoint().distance(df.Cell(emesh, cell).midpoint())
                   for cell, entity in mapping[1].items()) < 1E-14

        if n0 is not None:
            rate = np.log(dt/dt0)/np.log(float(n)/n0)
        else:
            rate = np.nan
        print n, dt, rate
        n0, dt0 = n, dt
        
    # Check creation
    df.set_log_level(df.WARNING)
    chi = df.CompiledSubDomain('near(x[i], 0.5)', i=0)
    
    for n in (2, 4, 8, 16, 32, 64):
        mesh = df.UnitCubeMesh(n, n, n)

        f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)

        for i in range(3):
            chi.i=i
            chi.mark(f, i+1)

        timer = df.Timer('embedding')
        emesh = EmbeddedMesh(f, [1, 2, 3])
        print(n, '->', timer.stop(), emesh.num_cells())

        volume = lambda c: df.Cell(emesh, c.index()).volume()

        # Crude check
        assert df.near(sum(volume(c) for c in df.SubsetIterator(emesh.marking_function, 1)), 1, 1E-10), sum(volume(c) for c in df.SubsetIterator(emesh.marking_function, 1))
        assert df.near(sum(volume(c) for c in df.SubsetIterator(emesh.marking_function, 2)), 1, 1E-10), sum(volume(c) for c in df.SubsetIterator(emesh.marking_function, 2))
        assert df.near(sum(volume(c) for c in df.SubsetIterator(emesh.marking_function, 3)), 1, 1E-10), sum(volume(c) for c in df.SubsetIterator(emesh.marking_function, 3))

        # Finer vertices
        new, old = map(list, zip(*emesh.parent_entity_map[mesh.id()][0].items()))
        # Coordinates mesh
        x = emesh.coordinates()[new]
        y = mesh.coordinates()[old]

        assert np.linalg.norm(x-y) < 1E-13

        # Finer entities
        tdim = emesh.topology().dim()
        mesh.init(tdim, 0)
        t2v = mesh.topology()(tdim, 0)
        
        new, old = map(list, zip(*emesh.parent_entity_map[mesh.id()][tdim].items()))

        # Midpoint coordinates of cells of embedded mesh
        x_mid = np.mean(emesh.coordinates()[emesh.cells()[new]], axis=1)
        # Obtain entities of embedding mesh as vertices
        entities_v = np.row_stack([t2v(e) for e in old])
        y = mesh.coordinates()[entities_v]
        # Their centroids ...
        y_mid = np.mean(y, axis=1)
        # ... match
        assert np.linalg.norm(x_mid - y_mid) < 1E-13

        mapping = emesh.parent_entity_map[mesh.id()][tdim]
        # Check colors
        color_values = emesh.marking_function.array()        
        colors = set(color_values)
        for color in colors:
            cells_new, = np.where(color_values == color)
            entities_old, = np.where(f.array() == color)

            assert set(mapping[cn] for cn in cells_new) == set(entities_old)
        
    # Check vertex
    # # Check normla computation
    # mesh = df.UnitCubeMesh(10, 10, 10)
    # bmesh = df.BoundaryMesh(mesh, 'exterior')

    # n = OuterNormal(bmesh, [0.5, 0.5, 0.5])

    # for cell in df.cells(bmesh):
    #     x = cell.midpoint().array()
    #     if df.near(x[0], 0):
    #         assert np.linalg.norm(n(x) - np.array([-1, 0, 0])) < 1E-10
    #     elif df.near(x[0], 1.):
    #         assert np.linalg.norm(n(x) - np.array([1, 0, 0])) < 1E-10
    #     elif df.near(x[1], 0.):
    #         assert np.linalg.norm(n(x) - np.array([0, -1, 0])) < 1E-10
    #     elif df.near(x[1], 1.):
    #         assert np.linalg.norm(n(x) - np.array([0, 1, 0])) < 1E-10
    #     elif df.near(x[2], 0):
    #         assert np.linalg.norm(n(x) - np.array([0, 0, -1])) < 1E-10
    #     else:
    #         assert np.linalg.norm(n(x) - np.array([0, 0, 1])) < 1E-10

    # EmbeddedMesh with cell_f
    mesh = df.BoxMesh(df.Point(*(-1, )*3), df.Point(*(1, )*3), 10, 10, 10)
    cell_f = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    for cell in df.cells(mesh):
        x, y, z = cell.midpoint().array()
        if x > 0:
            cell_f[cell] = 1 if y > 0 else 2
        else:
            cell_f[cell] = 3 if y > 0 else 4

    df.File('t.pvd') << cell_f
            
    mesh = EmbeddedMesh(cell_f, (1, 3))
    df.File('bar.pvd') << mesh
    df.File('foo.pvd') << mesh.marking_function
    
    for cell in df.SubsetIterator(mesh.marking_function, 1):
        x, y, z = cell.midpoint().array()
        assert x > 0 and y > 0

    for cell in df.SubsetIterator(mesh.marking_function, 3):
        x, y, z = cell.midpoint().array()
        assert x < 0 and y > 0
