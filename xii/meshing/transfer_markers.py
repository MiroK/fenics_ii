from dolfin import MeshFunction


def transfer_markers(mesh, f, markers=None):
    '''
    Let mesh be a mesh embedded into mesh(f). Transfer all markers from f 
    to the mesh.
    '''
    # Check that we are embdded
    parent_mesh = f.mesh()
    assert parent_mesh.id() in mesh.parent_entity_map

    # If we are embedded then the transfer is only possible of the lower
    # dimensional entities
    assert f.dim() < mesh.topology().dim()
    cdim = mesh.topology().dim()
    edim = f.dim()  # That of entities

    # All not zeros
    if markers is None: markers = set(f.array()) - set((0, ))

    # The idea is to look up mesh.entities in parent by vertex ids
    child_parent_vertex = mesh.parent_entity_map[parent_mesh.id()][0]
    child_parent_cell = mesh.parent_entity_map[parent_mesh.id()][cdim]
    
    # child_entity -> as vertex in child -> as vertex in parent
    # \- connected child cells -> parent -> cells -> parent cell entities as vertex
    # Entity in terms of vertices
    mesh.init(edim, 0)
    ch_e2v = mesh.topology()(edim, 0)
    # The connected cell; we already have child to parent for cell
    mesh.init(edim, cdim)
    ch_e2c = mesh.topology()(edim, cdim)
    # Then parent entities
    parent_mesh.init(cdim, edim)
    p_c2e = parent_mesh.topology()(cdim, edim)
    # In terms of vertices
    parent_mesh.init(edim, 0)
    p_e2v = parent_mesh.topology()(edim, 0)
    
    child_f = MeshFunction('size_t', mesh, edim, 0)
    ch_values = child_f.array()
    # Assignee
    p_values = f.array()

    for entity in range(len(ch_values)):
        # As vertices in parent
        e_as_vertex = set(child_parent_vertex[v] for v in ch_e2v(entity))

        ch_cells = iter(ch_e2c(entity))
        # One of the cells entities should match
        for ch_cell in ch_cells:
            for p_entity in p_c2e(child_parent_cell[ch_cell]):

                found = p_values[p_entity] in markers and set(p_e2v(p_entity)) == e_as_vertex
                # Can assign color
                if found: break
            if found:
                ch_values[entity] = p_values[p_entity]
                break
            
    return child_f

# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (CompiledSubDomain, DomainBoundary, SubsetIterator,
                        UnitSquareMesh, Facet)
    from xii import EmbeddedMesh

    mesh = UnitSquareMesh(4, 4)
    surfaces = MeshFunction('size_t', mesh, 2, 0)
    CompiledSubDomain('x[0] > 0.5-DOLFIN_EPS').mark(surfaces, 1)

    # What should be trasfered
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    # Assign funky colors
    for i, e in enumerate(SubsetIterator(f, 1), 1): f[e] = i

    ch_mesh = EmbeddedMesh(surfaces, 1)
    ch_f = transfer_markers(ch_mesh, f)

    # Every color in child is found in parent and we get the midpoint right
    p_values, ch_values = list(f.array()), list(ch_f.array())

    for ch_value in set(ch_f.array()) - set((0, )):
        assert ch_value in p_values

        x = Facet(mesh, p_values.index(ch_value)).midpoint()
        y = Facet(ch_mesh, ch_values.index(ch_value)).midpoint()
        assert x.distance(y) < 1E-13
