from collections import defaultdict
from xii.meshing.embedded_mesh import EmbeddedMesh
import dolfin as df
import numpy as np


def mortar_meshes(subdomains, markers, ifacet_iter=None, strict=True, tol=1E-14):
    '''
    Let subdomains a cell function. We assume that domains (cells) marked 
    with the given markers are adjecent and an interface can be defined 
    between these domains which is a continuous curve. Then for each 
    domain we create a (sub)mesh and a single interface mesh which holds 
    a connectivity map of its cells to facets of the submeshes. The submeshes 
    are returned as a list. The connectivity map is 
    of the form submesh.id -> facets. The marking function f of the EmbeddedMesh
    that is the interface is colored such that f[color] is the interface 
    of meshes (submeshes[m] for m color_map[color]).
    '''
    assert len(markers) > 1
    # Need a cell function
    mesh = subdomains.mesh()
    tdim = mesh.topology().dim()
    assert subdomains.dim() == tdim

    markers = list(markers)
    # For each facet we want to know which 2 cells share it
    tagged_iface = defaultdict(dict)

    if ifacet_iter is None:
        mesh.init(tdim-1)
        ifacet_iter = df.facets(mesh)
    
    mesh.init(tdim-1, tdim)
    for facet in ifacet_iter:
        cells = map(int, facet.entities(tdim))

        if len(cells) > 1:
            c0, c1 = cells
            tag0, tag1 = subdomains[c0], subdomains[c1]
            if tag0 != tag1 and tag0 in markers and tag1 in markers:
                # A key of sorted tags
                if tag0 < tag1:
                    key = (tag0, tag1)
                    # The cells connected to facet order to match to tags
                    value = (c0, c1)
                else:
                    key = (tag1, tag0)
                    value = (c1, c0)
                # A facet to 2 cells map for the facets of tagged pair
                tagged_iface[key][facet.index()] = value

    # order -> tagged keys
    color_to_tag_map = tagged_iface.keys()
    # Set to color which won't be encounred
    interface = df.MeshFunction('size_t', mesh, tdim-1, len(color_to_tag_map))
    values = interface.array()

    # Mark facets corresponding to tagged pair by a color
    for color, tags in enumerate(color_to_tag_map):
        values[tagged_iface[tags].keys()] = color

    # Finally create an interface mesh for all the colors
    interface_mesh = EmbeddedMesh(interface, range(len(color_to_tag_map)))

    # Try to recogninze the meshes which violates assumptions by counting
    assert not strict or is_continuous(interface_mesh)
    
    # And subdomain mesh for each marker
    subdomain_meshes = {tag: EmbeddedMesh(subdomains, tag) for tag in markers}

    # Alloc the entity maps for the embedded mesh
    interface_map = {subdomain_meshes[tag].id(): [None]*interface_mesh.num_cells()
                     for tag in markers}
    
    # THe maps are filled by the following idea. Using marking function
    # of interface mesh one cat get cells of that color and useing entity
    # map for (original) mesh map the cells to mesh facet. A color also
    # corresponds to a pair of tags which identifies the two meshes which
    # share the facet - facet connected to 2 cells one for each mesh. The
    # final step is to lean to map submesh cells to mesh cells

    # local submesh <- global of parent mesh
    sub_mesh_map = lambda tag: dict(
        (mesh_c, submesh_c)
        for submesh_c, mesh_c in
        enumerate(subdomain_meshes[tag].parent_entity_map[mesh.id()][tdim])
    )

    # Thec cell-cell connectivity of each submesh
    c2c = {tag: sub_mesh_map(tag) for tag in markers}
    # A connectivity of interface mesh cells to facets of global mesh
    c2f = interface_mesh.parent_entity_map[mesh.id()][tdim-1]

    for color, tags in enumerate(color_to_tag_map):
        # Precompute for the 2 tags
        submeshes = [subdomain_meshes[tag] for tag in tags]
        
        for cell in df.SubsetIterator(interface_mesh.marking_function, color):
            cell_index = cell.index()
            # The corresponding global cell facet
            facet = c2f[cell_index]
            # The two cells in global mesh numbering
            global_cells = tagged_iface[tags][facet]
            # Let's find the facet in submesh
            for tag, gc, submesh in zip(tags, global_cells, submeshes):
                # The map uses local cell
                local_cell = c2c[tag][gc]
                mesh_id = submesh.id()
                
                found = False
                for submesh_facet in df.facets(df.Cell(submesh, local_cell)):
                    found = df.near(cell.midpoint().distance(submesh_facet.midpoint()), 0, tol)
                    if found:
                        interface_map[mesh_id][cell_index] = submesh_facet.index()
                        break

    # Collapse to list; I want list indexing
    subdomain_meshes = np.array([subdomain_meshes[m] for m in markers]) 
    color_map = [map(markers.index, tags) for tags in color_to_tag_map]

    # Parent in the sense that the colored piece of interface
    # could have been created from mesh
    interface_mesh.parent_entity_map.update(
        dict((k, {tdim-1: v}) for k, v in interface_map.items())
    )

    return subdomain_meshes, interface_mesh, color_map


def is_continuous(mesh):
    '''
    We say that the embedded mesh is continuous if for each 2 cells 
    there exists a continuous path of mesh (other) cells between 
    the two cells
    '''
    import networkx as nx
    assert mesh.topology().dim() < mesh.geometry().dim()
    # This mounts to the graph of the mesh having only one connected
    # component
    G = nx.Graph()
    if mesh.topology().dim() == 1:
        mesh.init(1, 0)
        G.add_edges_from((tuple(cell.entities(0)) for cell in df.cells(mesh)))
    else:
        tdim = mesh.topology().dim() == 2

        graph_edges = set()
        # Geome edge cell connectivity to define the graph edges
        mesh.init(tdim-1, tdim)
        e2c = mesh.topology()(tdim-1, tdim)
        for e in range(mesh.num_entities(tdim-1)):
            cells = sorted(e2c(e))

            graph_edges.update(zip(cells[:-1], cells[1:]))
        G.add_edges_from(graph_edges)
        
    return nx.number_connected_components(G) == 1

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    
    gamma = CompiledSubDomain('near(std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)), 0.25)')
    interior = CompiledSubDomain('std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25')    

    from xii.meshing.embedded_mesh import build_embedding_map

    for n in (8, 16, 32, 64, 128):
        # Compare with building embedded map. NOTE: only possible for a
        # single interface.
        outer_mesh = UnitSquareMesh(n, n)
    
        subdomains = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim(), 0)
        # Awkward marking
        for cell in cells(outer_mesh):
            x = cell.midpoint().array()            
            subdomains[cell] = int(interior.inside(x, False))
        
        foo = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
        gamma.mark(foo, 1)

        ifacet_iter = SubsetIterator(foo, 1)
        
        timer = Timer('x'); timer.start()
        submeshes, interface, colormap = mortar_meshes(subdomains, (0, 1), ifacet_iter)
        print('\t', timer.stop())
    x = interface.parent_entity_map

    maps = [build_embedding_map(interface, meshi) for meshi in submeshes]
    for i in range(len(maps)):
        assert interface.parent_entity_map[submeshes[i].id()][1] == maps[i][1]

    exit()
    
    n = 16
    # 2d Construction
    outer_mesh = RectangleMesh(Point(-1, -1), Point(1, 1), n, n)
    fs = [CompiledSubDomain('std::max(fabs(x[0] - 0.25), fabs(x[1] - 0.25)) < 0.25'),
          CompiledSubDomain('std::max(fabs(x[0] + 0.25), fabs(x[1] - 0.25)) < 0.25'),
          CompiledSubDomain('std::max(fabs(x[0] + 0.25), fabs(x[1] + 0.25)) < 0.25'),
          CompiledSubDomain('std::max(fabs(x[0] - 0.25), fabs(x[1] + 0.25)) < 0.25')]
    
    subdomains = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim(), 0)

    # Awkward marking
    for cell in cells(outer_mesh):
        x = cell.midpoint().array()
        for tag, f in enumerate(fs, 1):
            if f.inside(x, False):
                subdomains[cell] = tag
                break

    submeshes, interface, colormap = mortar_meshes(subdomains, (0, 1, 2, 3, 4))

    # Subdomains
    for cell in cells(submeshes[0]):
        assert not any(f.inside(cell.midpoint().array(), False) for f in fs)

    for f, mesh in zip(fs, submeshes[1:]):
        assert all(f.inside(cell.midpoint().array(), False) for cell in cells(mesh))

    # Interface
    for cell in cells(interface):
        x, y, _ = cell.midpoint().array()
        assert any((near(abs(x), 0.5) and between(y, (-0.5, 0.5)),
                    near(abs(y), 0.5) and between(x, (-0.5, 0.5)),
                    near(x, 0.0) and between(y, (-0.5, 0.5)),
                    near(y, 0.0) and between(x, (-0.5, 0.5))))

    # Map
    tdim = interface.topology().dim()
    for color, domains in enumerate(colormap):
        meshes = submeshes[domains]
        for mesh in meshes:
            mesh.init(tdim)
            # Each colored cell is a facet of submesh
            for icell in SubsetIterator(interface.marking_function, color):
                # Index
                mfacet = interface.parent_entity_map[mesh.id()][tdim][icell.index()]
                # Actual cell
                mfacet = Facet(mesh, mfacet)
                # Now one of the facets of mcell must be icell
                assert near(icell.midpoint().distance(mfacet.midpoint()), 0)

    # Here's a forbidden bmesh
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), n, n)
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    one = CompiledSubDomain('std::max(fabs(x[0] - 0.75), fabs(x[1] - 0.75)) < 0.25')
    two = CompiledSubDomain('std::max(fabs(x[0] + 0.75), fabs(x[1] + 0.75)) < 0.25')

    one.mark(subdomains, 1)
    two.mark(subdomains, 2)

    try:
        mortar_meshes(subdomains, (0, 1, 2))
    except AssertionError:
        pass


    n = 16
    # 3d
    outer_mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n, n, n)

    def domain(x):
        select = CompiledSubDomain('std::max(std::max(fabs(x[0]-x0), fabs(x[1]-x1)), fabs(x[2]-x2)) < 0.25',
                                   x0=0., x1=0., x2=0.)
        select.x0 = x[0]
        select.x1 = x[1]
        select.x2 = x[2]

        return select

    import itertools
    fs = map(domain, itertools.product(*[[-0.25, 0.25]]*3))
    
    subdomains = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim(), 0)
    # Awkward marking
    for cell in cells(outer_mesh):
        x = cell.midpoint().array()
        for tag, f in enumerate(fs, 1):
            if f.inside(x, False):
                subdomains[cell] = tag
                break

    submeshes, interface, colormap = mortar_meshes(subdomains, range(9))

    # Subdomains
    for cell in cells(submeshes[0]):
        assert not any(f.inside(cell.midpoint().array(), False) for f in fs)

    for f, mesh in zip(fs, submeshes[1:]):
        assert all(f.inside(cell.midpoint().array(), False) for cell in cells(mesh))

    # Interface
    for cell in cells(interface):
        x, y, z = cell.midpoint().array()
        assert any((near(abs(x), 0.5) and between(y, (-0.5, 0.5)) and between(z, (-0.5, 0.5)),
                    near(abs(y), 0.5) and between(x, (-0.5, 0.5)) and between(z, (-0.5, 0.5)),
                    near(abs(z), 0.5) and between(x, (-0.5, 0.5)) and between(y, (-0.5, 0.5)),
                    near(x, 0.0) and between(y, (-0.5, 0.5)) and between(z, (-0.5, 0.5)),
                    near(y, 0.0) and between(x, (-0.5, 0.5)) and between(z, (-0.5, 0.5)),
                    near(z, 0.0) and between(y, (-0.5, 0.5)) and between(x, (-0.5, 0.5))))
                    
    # Map
    tdim = interface.topology().dim()
    for color, domains in enumerate(colormap):
        meshes = submeshes[domains]
        for mesh in meshes:
            mesh.init(tdim)
            # Each colored cell is a facet of submesh
            for icell in SubsetIterator(interface.marking_function, color):
                # Index
                mfacet = interface.parent_entity_map[mesh.id()][tdim][icell.index()]
                # Actual cell
                mfacet = Facet(mesh, mfacet)
                # Now one of the facets of mcell must be icell
                assert near(icell.midpoint().distance(mfacet.midpoint()), 0)

    # Here's a forbidden bmesh
    mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n, n, n)
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    one = domain((-0.75, -0.75, -0.75))
    two = domain((0.75, 0.75, 0.75))

    one.mark(subdomains, 1)
    two.mark(subdomains, 2)

    try:
        mortar_meshes(subdomains, (0, 1, 2))
    except AssertionError:
        pass
