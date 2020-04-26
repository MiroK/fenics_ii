from __future__ import absolute_import
from xii.meshing.make_mesh_cpp import make_mesh
from itertools import combinations, count
import numpy as np
from six.moves import map
from six.moves import range


def cross_grid_refine(mesh):
    '''
    Cross section refine is

       x                      x   
      / \     becomes        /|\  
     /   \                  / x \
    /     \                / / \ \  
    x------x              x-------x
    '''
    x = mesh.coordinates()
    cells = mesh.cells()

    ncells, nvtx_cell = cells.shape
    assert any((nvtx_cell == 2,
                nvtx_cell == 3,
                nvtx_cell == 4 and mesh.topology().dim() == 3))

    # Center points will be new coordinates
    xnew = np.mean(x[cells], axis=1)
    # Each cell gives rise to ...
    child2parent = np.empty(ncells*nvtx_cell, dtype='uintp')
    fine_cells = np.empty((ncells*nvtx_cell, nvtx_cell), dtype='uintp')
    # How we build new cells
    basis = list(map(list, combinations(list(range(nvtx_cell)), nvtx_cell-1)))

    fc, center = 0, len(x)
    for pc, cell in enumerate(cells):
        for base in basis:
            fine_cells[fc, :] = np.r_[cell[base], center]
            child2parent[fc] = pc
            fc += 1
        center += 1

    tdim = mesh.topology().dim()
    fine_mesh = make_mesh(np.row_stack([x, xnew]), fine_cells, tdim, mesh.geometry().dim())

    fine_mesh.set_parent(mesh)
    mesh.set_child(fine_mesh)

    fine_mesh.parent_entity_map = {mesh.id(): {0: dict(enumerate(range(mesh.num_vertices()))),
                                               tdim: dict(enumerate(child2parent))}}
       
    return fine_mesh

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, File, MeshFunction
    
    mesh = UnitSquareMesh(2, 2)

    fine_mesh = cross_grid_refine(mesh)
    assert fine_mesh.has_parent()

    mesh_f = MeshFunction('size_t', fine_mesh, 2, 0)
    for child, parent in fine_mesh.parent_entity_map[mesh.id()][2].items():
        mesh_f[child] = parent
    
    File('foo.pvd') << mesh_f
