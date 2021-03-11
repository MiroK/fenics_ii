from functools import reduce
import dolfin as df
import numpy as np
import operator
import gmsh
import ufl


def build_mesh(vertices, cells, cell_type):
    '''Mesh defined by its vertices, cells'''
    cell_tdim = cell_type.topological_dimension()
    gdim = cell_type.geometric_dimension()
    # Consistency
    assert gdim == vertices.shape[1]
    assert cell_type.num_vertices() == cells.shape[1]
    
    mesh = df.Mesh()
    editor = df.MeshEditor()
    editor.open(mesh, cell_type.cellname(), cell_tdim, gdim)
    editor.init_cells(len(cells))
    editor.init_vertices(len(vertices))

    for i, v in enumerate(vertices):
        editor.add_vertex(i, v)

    for i, c in enumerate(cells):
        editor.add_cell(i, c)

    editor.close()

    return mesh


def msh_gmsh_model(model, dim, number_options=None, string_options=None):
    '''Generate dim-D mesh of model according to options'''
    if number_options:
        for opt in number_options:
            gmsh.option.setNumber(opt, number_options[opt])

    if string_options:
        for opt in string_options:
            gmsh.option.setString(opt, string_options[opt])

    model.occ.synchronize()
    model.geo.synchronize()

    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
            
    model.mesh.generate(dim)

    indices, nodes, _ = model.mesh.getNodes()
    indices -= 1
    nodes = nodes.reshape((-1, 3))

    physical_groups = model.getPhysicalGroups()
    data = {}
    for dim, tag in physical_groups:

        entities = model.getEntitiesForPhysicalGroup(dim, tag)
        # Grab entities of topological dimension which have the tag
        for entity in entities:
            element_data = model.mesh.getElements(dim, tag=entity)
            element_types, element_tags, node_tags = element_data
            # That enity is mesh by exatly one element type
            element_type, = element_types
            # The MSH type of the cells on the element
            num_el = len(element_tags[0])

            element_topology = node_tags[0].reshape((num_el, -1)) - 1
            cell_data = np.full(num_el, tag)
        
            if element_type in data:
                data[element_type]['topology'] = np.row_stack([
                    data[element_type]['topology'], element_topology
                ]
                )
                data[element_type]['cell_data'] = np.hstack([
                    data[element_type]['cell_data'], cell_data
                ]
            )
            else:
                data[element_type] = {'topology': element_topology,
                                      'cell_data': cell_data}

    # NOTE: gmsh nodes are ordered according to indices. The only control
    # we will have when making the mesh is that order in which we insert
    # the vertex is respected. So to make topology work we need to insert
    # them as gmsh would (according to indices)
    nodes = nodes[np.argsort(indices)]

    return nodes, data


def mesh_from_gmsh(nodes, element_data, TOL=1E-13):
    '''Return mesh and dict tdim -> MeshFunction over tdim-entities'''
    # We only support lines, triangles and tets
    assert set(element_data.keys()) <= set((1, 2, 4))

    elm_tdim = {1: 1, 2: 2, 4: 2}    
    # The idea is first build the mesh (and cell function) and later
    # descent over entities to tag them
    cell_elm = max(element_data.keys(), key=lambda elm: elm_tdim[elm])  # 
    # Here are cells defined in terms of incident nodes (in gmsh numbering)
    cells_as_nodes = element_data[cell_elm]['topology']
    # The unique vertices make up mesh vertices
    vtx_idx = np.unique(cells_as_nodes)
    mesh_vertices = nodes[vtx_idx]  # Now we have our numbering
    # Want to make from old to new to redefine cells
    node_map = {old: new for old, new in enumerate(vtx_idx)}

    cells_as_nodes = np.fromiter((node_map[c] for c in cells_as_nodes.ravel()),
                                 dtype=cells_as_nodes.dtype).reshape(cells_as_nodes.shape)

    print('Mesh has {} cells.'.format(len(cells_as_nodes)))
    # Cell-node-connectivity is enough to build mesh
    elm_name = {1: 'interval', 2: 'triangle', 4: 'tetrahedron'}
    cell_tdim = elm_tdim[cell_elm]
    # Since gmsh has nodes always as 3d we infer gdim from whether last
    # axis is 0
    if np.linalg.norm(mesh_vertices[:, 2]) < TOL:
        gdim = 2
        mesh_vertices = mesh_vertices[:, :gdim]
    else:
        gdim = 3

    cell = ufl.Cell(elm_name[cell_tdim], gdim)
    mesh = build_mesh(mesh_vertices, cells_as_nodes, cell)
    
    entity_functions = {}
    # Is there cell_function to build?
    if np.min(element_data[cell_elm]['cell_data']) > 0:
        f = df.MeshFunction('size_t', mesh, cell_tdim)
        f.array()[:] = element_data[cell_elm]['cell_data']
        entity_functions[cell_tdim] = f
    
    # The remainig entity functions need to looked up among entities of
    # the mesh
    for elm in element_data:
        # We dealt with cells already
        if elm == cell_elm:
            continue
        # All zeros is not interesting either
        if np.min(element_data[elm]['cell_data']) == 0:
            continue

        tdim = elm_tdim[elm]
        # Same as with gmsh we want to encode entity in terms of its
        # vertex connectivity
        mesh.init(tdim)
        _, v2e = (mesh.init(0, tdim), mesh.topology()(0, tdim))
        _, e2v = (mesh.init(tdim, 0), mesh.topology()(tdim, 0))        

        f = df.MeshFunction('size_t', mesh, tdim, 0)
        for entity, tag in zip(element_data[elm]['topology'], element_data[elm]['cell_data']):
            # Encode in fenics
            entity = [node_map[v] for v in entity]
            # When we look at entities incide to the related vertices
            mesh_entity, = map(int, reduce(operator.and_, (set(v2e(v)) for v in entity)))

            # ... there should be exactly one and
            assert set(entity) == set(e2v(mesh_entity))

            f[mesh_entity] = tag
            
        entity_functions[tdim] = f

    return mesh, entity_functions
