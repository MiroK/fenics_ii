from dolfin import *
import numpy as np


class EmbeddedMesh(Mesh):
    '''Mesh of lower topological entities of the base mesh.'''
    def __init__(self, base, domain_f, marker, normal=None):
        try:
            assert len(marker) == 2
            tdim = domain_f.dim()
            assert tdim == base.topology().dim()
            marker = set(marker)

            base.init(tdim-1, tdim)
            f2c = base.topology()(tdim-1, tdim)
            # We can avoid domain boundaries
            facet_f = FacetFunction('size_t', base, 1)
            DomainBoundary().mark(facet_f, 2)

            values = domain_f.array()

            for facet in SubsetIterator(facet_f, 1):
                fmarkers = set(values[f2c(facet.index())])
                if fmarkers == marker:
                    facet_f[facet] = 4
            domain_f = facet_f
            marker = 4
        except TypeError:
            pass

        tdim = domain_f.dim()  
        assert 0 < tdim < base.topology().dim()

        # Collect unique vertices based on their base-mesh indexing, the cells
        # of the embedded mesh are defined in terms of their embedded-numbering
        emesh_vertices, emesh_cells = [], []
        # Map from cells of embedded mesh to tdim entities of base mesh, and
        entity_map_tdim  = []

        base.init(tdim, 0)
        for entity in SubsetIterator(domain_f, marker):
            vs = entity.entities(0)
            cell = []
            for v in vs:
                try:
                    local = emesh_vertices.index(v)
                except ValueError:
                    local = len(emesh_vertices)
                    emesh_vertices.append(v)
                cell.append(local)
            emesh_cells.append(cell)

            entity_map_tdim.append(entity.index())
        assert emesh_cells
        # Add the entity map from emesh vertices to mesh vertices
        entity_map = {0: emesh_vertices, tdim: entity_map_tdim}

        # Get coordinates
        gdim = base.geometry().dim()
        emesh_vertices = base.coordinates().reshape((-1, gdim))[emesh_vertices]

        emesh = Mesh()
        editor = MeshEditor()
        editor.open(emesh, tdim, gdim)
        editor.init_vertices(len(emesh_vertices))
        editor.init_cells(len(emesh_cells))

        # Add vertices
        for vertex_index, v in enumerate(emesh_vertices): editor.add_vertex(vertex_index, v)

        # Add cells
        for cell_index, v in enumerate(emesh_cells): editor.add_cell(cell_index, *v)

        editor.close()

        self.base = base
        self.mesh = emesh
        self.entity_map = entity_map

        # If normal specifies a point inside/inside convex hull/on side we
        # compute the normal as a DG0 function
        if isinstance(normal, (list, np.ndarray, Point)): 
            normal = _compute_mesh_normal(emesh, inside=normal)

        self.normal = normal


def _compute_mesh_normal(mesh, inside):
    '''DG0 normal of embedded mesh'''
    dim = mesh.geometry().dim()
    assert dim > 1
    assert mesh.topology().dim() < dim

    if isinstance(inside, np.ndarray): inside = inside.tolist()

    if isinstance(inside, list): assert len(inside) == dim

    if not isinstance(inside, Point): 
        inside = Point(*inside)

    x = mesh.coordinates().reshape((-1, dim))

    values = []
    for cell in cells(mesh):
        n = cell.cell_normal()
        if (cell.midpoint()-inside).dot(n) < 0: n *= -1
        values.append(n/n.norm())

    V = VectorFunctionSpace(mesh, 'DG', 0)
    n = Function(V)
    n_values = n.vector().get_local()

    for sub in range(dim):
        dofmap = V.sub(sub).dofmap()
        for cell in cells(mesh):
            i = cell.index()
            dof = dofmap.cell_dofs(i)[0]
            value = values[i][sub]
            n_values[dof] = value
    n.vector().set_local(n_values)
    n.vector().apply('insert')
    
    return n

# ----------------------------------------------------------------------------

if __name__ == '__main__':

    base = UnitCubeMesh(10, 10, 10)

    gamma2 = CompiledSubDomain('near(x[2], 0.5) || near(x[0], 0.)')
    gamma1 = ['(near(x[0], x[1]) && near(x[1], x[2]))',
              '(near(x[0], 1) && near(x[1], 1))',
              '(near(x[0], x[1]) && near(x[2], 0))']
    gamma1 = ' || '.join(gamma1)
    gamma1 = CompiledSubDomain(gamma1)

    domain_f2 = FacetFunction('size_t', base, 0)
    gamma2.mark(domain_f2, 1)

    emesh2 = EmbeddedMesh(base, domain_f2, 1)
    mesh = emesh2.mesh
    c2entity = emesh2.entity_map[2]
    v2v = emesh2.entity_map[0]
    # The match of cell <-> tdim entity
    base.init(2)
    assert min(cell.midpoint().distance(Facet(base, c2entity[cell.index()]).midpoint())
               for cell in cells(mesh)) < 1E-12

    # The match of vertices
    assert min(v.point().distance(Vertex(base, v2v[v.index()]).point())
               for v in vertices(mesh)) < 1E-12

    # plot(emesh2.mesh)
    # interactive()

    # ------------------------------------------------------------------------

    domain_f1 = EdgeFunction('size_t', base, 0)
    gamma1.mark(domain_f1, 1)

    emesh1 = EmbeddedMesh(base, domain_f1, 1)
    mesh = emesh1.mesh
    c2entity = emesh1.entity_map[1]
    v2v = emesh1.entity_map[0]
    # The match of cell <-> tdim entity
    base.init(1)
    assert min(cell.midpoint().distance(Edge(base, c2entity[cell.index()]).midpoint())
               for cell in cells(mesh)) < 1E-12

    # The match of vertices
    assert min(v.point().distance(Vertex(base, v2v[v.index()]).point())
               for v in vertices(mesh)) < 1E-12

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # 
    # X = mesh.coordinates().reshape((-1, 3))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # for cell in cells(mesh):
    #    line = X[cell.entities(0)].T
    #    ax.plot(line[0], line[1], line[2], 'ko-')
    # plt.show()

    #########################
    # Normal checking
    #########################
    # 2d-1d
    gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
             'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
    gamma = map(lambda x: '('+x+')', gamma)
    gamma = ' || '.join(gamma)
    gamma = CompiledSubDomain(gamma)

    n = 2
    n *= 4
    omega_mesh = UnitSquareMesh(n, n)
    facet_f = FacetFunction('size_t', omega_mesh, 0)
    gamma.mark(facet_f, 1)
    # plot(facet_f, interactive=True)
    gamma_mesh = EmbeddedMesh(omega_mesh, facet_f, 1, Point(0.5, 0.5))
    for cell in cells(gamma_mesh.mesh):
        mp = cell.midpoint()
        x = [mp[0], mp[1]]
        n = Point(gamma_mesh.normal(*x))

        if near(x[0], 0.25, 1E-8) or near(x[0], 0.75, 1E-8):
            n0 = Point(-1 if x[0] < 0.5 else 1, 0)
        elif near(x[1], 0.25, 1E-8) or near(x[1], 0.75, 1E-8):
            n0 = Point(0, -1 if x[1] < 0.5 else 1)
        assert n.distance(n0) < 1E-14

    # 3d-2d
    n = 2
    n *= 4
    omega_mesh = UnitCubeMesh(n, n, n)
    facet_f = FacetFunction('size_t', omega_mesh, 0)
    gamma.mark(facet_f, 1)
    # plot(facet_f, interactive=True)
    gamma_mesh = EmbeddedMesh(omega_mesh, facet_f, 1, Point(0.5, 0.5, 0.5))
    for cell in cells(gamma_mesh.mesh):
        mp = cell.midpoint()
        x = [mp[0], mp[1], mp[2]]
        n = Point(gamma_mesh.normal(*x))

        if near(x[0], 0.25, 1E-8) or near(x[0], 0.75, 1E-8):
            n0 = Point(-1 if x[0] < 0.5 else 1, 0)
        elif near(x[1], 0.25, 1E-8) or near(x[1], 0.75, 1E-8):
            n0 = Point(0, -1 if x[1] < 0.5 else 1)
        assert n.distance(n0) < 1E-14


    #########################
    # Finally getting meshes from cell function
    #########################
    # 2d-1d
    gamma = lambda x, on_boundary: all([between(x[i], (0.25, 0.75)) for i in range(2)])
    gamma = AutoSubDomain(gamma)

    n = 2
    n *= 4
    omega_mesh = UnitSquareMesh(n, n)
    cell_f = CellFunction('size_t', omega_mesh, 0)
    gamma.mark(cell_f, 1)
    # plot(facet_f, interactive=True)
    gamma_mesh = EmbeddedMesh(omega_mesh, cell_f, [0, 1], Point(0.5, 0.5))
    # for cell in cells(gamma_mesh.mesh):
    #     mp = cell.midpoint()
    #     x = [mp[0], mp[1]]
    #     n = Point(gamma_mesh.normal(*x))

    #     if near(x[0], 0.25, 1E-8) or near(x[0], 0.75, 1E-8):
    #         n0 = Point(-1 if x[0] < 0.5 else 1, 0)
    #     elif near(x[1], 0.25, 1E-8) or near(x[1], 0.75, 1E-8):
    #         n0 = Point(0, -1 if x[1] < 0.5 else 1)
    #     assert n.distance(n0) < 1E-14

    # 3d-2d
    n = 2
    n *= 4
    omega_mesh = UnitCubeMesh(n, n, n)
    cell_f = CellFunction('size_t', omega_mesh, 0)
    gamma.mark(cell_f, 1)
    # plot(facet_f, interactive=True)
    gamma_mesh = EmbeddedMesh(omega_mesh, cell_f, [0, 1], Point(0.5, 0.5, 0.5))
    for cell in cells(gamma_mesh.mesh):
        mp = cell.midpoint()
        x = [mp[0], mp[1], mp[2]]
        n = Point(gamma_mesh.normal(*x))

        if near(x[0], 0.25, 1E-8) or near(x[0], 0.75, 1E-8):
            n0 = Point(-1 if x[0] < 0.5 else 1, 0)
        elif near(x[1], 0.25, 1E-8) or near(x[1], 0.75, 1E-8):
            n0 = Point(0, -1 if x[1] < 0.5 else 1)
        assert n.distance(n0) < 1E-14
    plot(gamma_mesh.mesh, interactive=True)
