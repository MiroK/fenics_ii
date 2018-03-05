from collections import defaultdict
import dolfin as df
import numpy as np


class EmbeddedMesh(df.Mesh):
    '''
    Construct a mesh of marked entities in marking_function.
    The output is the mesh with cell function which inherited the markers. 
    and an antribute `parent_entity_map` which is dict with a map of new 
    mesh vertices to the old ones, and new mesh cells to the old mesh entities.
    Having several maps in the dict is useful for mortating.
    '''
    def __init__(self, marking_function, markers):
        base_mesh = marking_function.mesh()
        # Prevent cell function (just not to duplicate functionality of
        # submesh; well for now)
        assert base_mesh.topology().dim() >= marking_function.dim()

        # Work in serial only (much like submesh)
        assert df.MPI.size(base_mesh.mpi_comm()) == 1

        gdim = base_mesh.geometry().dim()
        tdim = marking_function.dim()
        assert tdim > 0, 'No Embedded mesh from vertices'

        if isinstance(markers, int): markers = [markers]

        base_mesh.init(tdim, 0)
        # Collect unique vertices based on their new-mesh indexing, the cells
        # of the embedded mesh are defined in terms of their embedded-numbering
        new_vertices, new_cells = [], []
        # NOTE: new_vertices is actually new -> old vertex map
        # Map from cells of embedded mesh to tdim entities of base mesh, and
        cell_map = []
        cell_colors = defaultdict(list)  # Preserve the markers

        new_cell_index, new_vertex_index = 0, 0
        for marker in markers:
            for entity in df.SubsetIterator(marking_function, marker):
                vs = entity.entities(0)
                cell = []
                # Vertex lookup
                for v in vs:
                    try:
                        local = new_vertices.index(v)
                    except ValueError:
                        local = new_vertex_index
                        new_vertices.append(v)
                        new_vertex_index += 1
                    # Cell, one by one in terms of vertices
                    cell.append(local)
                # The cell
                new_cells.append(cell)
                # Into map
                cell_map.append(entity.index())
                # Colors
                cell_colors[marker].append(new_cell_index)

                new_cell_index += 1

        # With acquired data build the mesh
        df.Mesh.__init__(self)
        editor = df.MeshEditor()

        if df.__version__ == '2017.2.0':
            cell_type = {1: 'interval', 2: 'triangle'}[tdim]
            editor.open(self, cell_type, tdim, gdim)            
        else:
            editor.open(self, tdim, gdim)

        editor.init_vertices(len(new_vertices))
        editor.init_cells(len(new_cells))

        vertex_coordinates = base_mesh.coordinates()[new_vertices]

        for vi, x in enumerate(vertex_coordinates): editor.add_vertex(vi, x)

        for ci, c in enumerate(new_cells): editor.add_cell(ci, *c)

        editor.close()

        # The entity mapping attribute
        mesh_key = marking_function.mesh().id()
        self.parent_entity_map = {mesh_key: {0: new_vertices, tdim: cell_map}}

        f = df.MeshFunction('size_t', self, tdim, 0)
        f_ = f.array()
        # Finally the inherited marking function
        if len(markers) > 1:
            for marker, cells in cell_colors.iteritems(): f_[cells] = marker
        else:
            f.set_all(markers[0])

        self.marking_function = f

        
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

        values = []
        for cell in df.cells(mesh):
            n = cell.cell_normal().array()[:gdim]
            x = cell.midpoint().array()[:gdim]
            # Disagree?
            if np.inner(orientation(x), n) < 0:
                n *= -1
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


def build_embedding_map(emesh, mesh, tol=1E-14):
    '''
    Operating with the assumption that the emsh consists of entities 
    of mesh we find here a map from emesh vertices and cells to mesh
    vertices and entities.
    '''
    assert emesh.topology().dim() < mesh.topology().dim()
    edim = emesh.topology().dim()
    assert emesh.num_cells() < mesh.topology().size(edim)
    
    # Localization will require
    tree = mesh.bounding_box_tree()
    # to find candidata cells. We zoom in on the unique entity by
    mesh.init(edim)   # Set amoong which emesh cells will be searched
    mesh.init(mesh.topology().dim(), edim)  # Via cell connectivity
    mesh.init(edim, 0)  # And coordinate comparison

    c2v = mesh.topology()(mesh.topology().dim(), 0)
    c2e = mesh.topology()(mesh.topology().dim(), edim)
    e2v = mesh.topology()(edim, 0)

    mesh_x = mesh.coordinates()
    emesh_x = emesh.coordinates()
    # Get som idea of mesh size to make relative comparison of coords
    scale = max(emesh_x.max(axis=0) - emesh_x.min(axis=0))
    # Also build the map for vertices
    entity_map = {0: [None]*emesh.num_vertices(),
                  edim: [None]*emesh.num_cells()}
    for cell in df.cells(emesh):
        
        the_entity = set()
        for vertex in cell.entities(0):
            vertex_x = emesh_x[vertex]
            mcells = tree.compute_entity_collisions(df.Point(*vertex_x))
            
            # What is the id of vertex in the mesh
            mcell_vertices = c2v(mcells[0])
            the_vertex = min(mcell_vertices, key=lambda v: np.linalg.norm(vertex_x-mesh_x[v]))
            error = np.linalg.norm(vertex_x - mesh_x[the_vertex])/scale
            assert error < tol, 'Found a hanging node %16f' % error
            
            entity_map[0][vertex] = the_vertex
            
            # For each I want to get its entities which containt the vertex
            # We are after such (UNIQUE) entity which would be in each such
            # set build for a vertex
            vertex_set = {entity
                          for mcell in mcells for entity in c2e(mcell)
                          if the_vertex in e2v(entity)}

            if not the_entity:
                the_entity.update(vertex_set)
            else:
                the_entity = the_entity & vertex_set
        assert len(the_entity) == 1
        # Insert
        entity_map[edim][cell.index()] = the_entity.pop()
        
    assert not any(v is None for v in entity_map[0])
    assert not any(v is None for v in entity_map[edim])
        
    return entity_map


class SubDomainMesh(EmbeddedMesh):
    '''TODO'''
    def __init__(self, marking_function, markers):
        assert marking_function.dim() == marking_function.mesh().topology().dim()
        EmbeddedMesh.__init__(self, marking_function, markers)

        
def OverlapMesh(mesh1, mesh2, tol=1E-14):
    '''TODO'''
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
    mesh.init(tdim, 0)  # has the same vertices

    c2v = mesh2.topology()(tdim, 0)

    mesh_x = mesh2.coordinates()
    emesh_x = emesh.coordinates()
    # Get som idea of mesh size to make relative comparison of coords
    scale = max(emesh_x.max(axis=0) - emesh_x.min(axis=0))
    # Also build the map for vertices
    entity_map = {0: [None]*emesh.num_vertices(), tdim: [None]*emesh.num_cells()}
    
    for cell in df.cells(emesh):
        # The idea is the there is exactly on the_cell which will be
        # found in every point-cell collision patches
        the_cell = set()
        for vertex in cell.entities(0):
            
            vertex_x = emesh_x[vertex]
            mcells = set(tree.compute_entity_collisions(df.Point(*vertex_x)))
            # What is the id of vertex in the mesh
            mcell_vertices = c2v(next(iter(mcells)))
            the_vertex = min(mcell_vertices, key=lambda v: np.linalg.norm(vertex_x-mesh_x[v]))
            error = np.linalg.norm(vertex_x - mesh_x[the_vertex])/scale
            assert error < tol, 'Found a hanging node %16f' % error

            entity_map[0][vertex] = the_vertex
            
            if not the_cell:
                the_cell.update(mcells)
            else:
                the_cell = the_cell & mcells
        assert len(the_cell) == 1, the_cell
        # Insert
        entity_map[tdim][cell.index()] = the_cell.pop()
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
                
    exit()
    # Embedding map
    n0, dt0 = None, None
    for n in [8, 16, 32, 64, 128, 256, 512]:

        mesh = df.UnitSquareMesh(n, n)
        emesh = df.BoundaryMesh(mesh, 'exterior')
        
        time = df.Timer('map'); time.start()        
        mapping = build_embedding_map(emesh, mesh, tol=1E-14)
        dt = time.stop()
        
        mesh_x = mesh.coordinates()
        emesh_x = emesh.coordinates()

        assert max(np.linalg.norm(ex - mesh_x[to])
                   for ex, to in zip(emesh_x, mapping[0])) < 1E-14

        assert max(df.Facet(mesh, entity).midpoint().distance(df.Cell(emesh, cell).midpoint())
                   for cell, entity in enumerate(mapping[1])) < 1E-14

        if n0 is not None:
            rate = np.log(dt/dt0)/np.log(float(n)/n0)
        else:
            rate = np.nan
        print n, dt, rate
        n0, dt0 = n, dt
        
    # Check creation
    mesh = df.UnitCubeMesh(10, 10, 10)

    f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    chi = df.CompiledSubDomain('near(x[i], 0.5)', i=0) 
    for i in range(3):
        chi.i=i
        chi.mark(f, i+1)

    mesh = EmbeddedMesh(f, [1, 2, 3])

    volume = lambda c: df.Cell(mesh, c.index()).volume()
    
    assert df.near(sum(volume(c) for c in df.SubsetIterator(mesh.marking_function, 1)), 1, 1E-10)
    assert df.near(sum(volume(c) for c in df.SubsetIterator(mesh.marking_function, 2)), 1, 1E-10)
    assert df.near(sum(volume(c) for c in df.SubsetIterator(mesh.marking_function, 3)), 1, 1E-10)

    # Check normla computation
    mesh = df.UnitCubeMesh(10, 10, 10)
    bmesh = df.BoundaryMesh(mesh, 'exterior')

    n = OuterNormal(bmesh, [0.5, 0.5, 0.5])

    for cell in df.cells(bmesh):
        x = cell.midpoint().array()
        if df.near(x[0], 0):
            assert np.linalg.norm(n(x) - np.array([-1, 0, 0])) < 1E-10
        elif df.near(x[0], 1.):
            assert np.linalg.norm(n(x) - np.array([1, 0, 0])) < 1E-10
        elif df.near(x[1], 0.):
            assert np.linalg.norm(n(x) - np.array([0, -1, 0])) < 1E-10
        elif df.near(x[1], 1.):
            assert np.linalg.norm(n(x) - np.array([0, 1, 0])) < 1E-10
        elif df.near(x[2], 0):
            assert np.linalg.norm(n(x) - np.array([0, 0, -1])) < 1E-10
        else:
            assert np.linalg.norm(n(x) - np.array([0, 0, 1])) < 1E-10
