from dolfin import DomainBoundary, MeshFunction, SubsetIterator
from .make_mesh_cpp import make_mesh
import numpy as np


def union_mesh(meshes, tol=1E-12):
    '''Glue together meshes into a big one.'''
    assert meshes
    
    num_meshes = len(meshes)
    # Nothing to do
    if num_meshes == 1:
        return meshes[0]
    # Recurse
    if num_meshes > 2:
        return union_mesh([union_mesh(meshes[:num_meshes/2+1]),
                           union_mesh(meshes[num_meshes/2+1:])])

    gdim, = set(m.geometry().dim() for m in meshes)
    tdim, = set(m.topology().dim() for m in meshes)

    fdim = tdim-1
    bdries = [MeshFunction('size_t', m, fdim, 0) for m in meshes]
    [DomainBoundary().mark(bdry, 1) for bdry in bdries]
    # We are after boundary vertices of both; NOTE that the assumption
    # here is that the meshes share only the boundary vertices
    [m.init(fdim) for m in meshes]
    [m.init(fdim, 0) for m in meshes]

    bdry_vertices0, bdry_vertices1 = list(map(list, (set(np.concatenate([f.entities(0) for f in SubsetIterator(bdry, 1)]))
                                                for bdry in bdries)))
    
    x0, x1 = [m.coordinates() for m in meshes]

    x1 = x1[bdry_vertices1]
    shared_vertices = {}
    while bdry_vertices0:
        i = bdry_vertices0.pop()
        x = x0[i]
        # Try to match it
        dist = np.linalg.norm(x1 - x, 2, axis=1)
        imin = np.argmin(dist)
        if dist[imin] < tol:
            shared_vertices[bdry_vertices1[imin]] = i
            x1 = np.delete(x1, imin, axis=0)
            del bdry_vertices1[imin]

    mesh0, mesh1 = meshes
    # We make 0 the master - it adds all its vertices
    # The other on add all but those that are not shared
    unshared = list(set(range(mesh1.num_vertices())) - set(shared_vertices.keys()))
    
    merge_x = mesh0.coordinates()
    offset = len(merge_x)
    # Vertices of the merged mesh
    merge_x = np.row_stack([merge_x, mesh1.coordinates()[unshared]])
    # Mapping for cells from meshes
    lg1 = {k: v for v, k in enumerate(unshared, offset)}
    lg1.update(shared_vertices)
    # Collapse to list
    _, lg1 = list(zip(*sorted(list(lg1.items()), key=lambda v: v[0])))
    lg1 = np.array(lg1)
    
    mapped_cells = np.fromiter((lg1[v] for v in np.concatenate(mesh1.cells())),
                               dtype='uintp').reshape((mesh1.num_cells(), -1))
    merged_cells = np.row_stack([mesh0.cells(), mapped_cells])

    merged_mesh = make_mesh(coordinates=merge_x,
                            cells=merged_cells, tdim=tdim, gdim=gdim)

    lg0 = np.arange(mesh0.num_vertices())
    # Mapping from leafs
    if not hasattr(mesh0, 'leafs'):
        merged_mesh.leafs = [(mesh0.id(), lg0)]
    else:
        merged_mesh.leafs = mesh0.leafs
        
    if not hasattr(mesh1, 'leafs'):
        merged_mesh.leafs.append([mesh1.id(), lg1])
    else:
        for id_, map_ in mesh1.leafs:
            merged_mesh.leafs.append((id_, lg1[map_]))
                
    return merged_mesh

#          [xxxxxxxx]
# [  [xxxx]      [xxxx]   ]
#  [xx]  [xx]  [xx]  [xx]
#  x  x  x  x  x  x  x  x

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import RectangleMesh, Point, File

    dx, dy = 0.5, 0.5

    meshes = []
    for j in range(3):
        for i in range(3):
            x0, y0 = dx*i, dy*j
            x1, y1 = x0 + dx, y0 + dy

            meshes.append(RectangleMesh(Point(x0, y0), Point(x1, y1), 4, 4))
    union = union_mesh(meshes)

    y = union.coordinates()
    for mesh in meshes:

        found = False
        for m_id, m_map in union.leafs:
            found = m_id == mesh.id()
            if found: break
        assert found
                
        x = mesh.coordinates()

        print(m_id, np.linalg.norm(x - y[m_map]))
    File('fff.pvd') << union
