from xii.meshing.make_mesh_cpp import make_mesh
import numpy as np


def line_mesh(branches, tol=1E-13):
    '''1d in 2d/3d mesh made up of branches. Branch = linked vertices'''
    it_branches = iter(branches)

    # In the mesh that we are building I want to put first the end nodes
    # of the branch and then the interior points of the branches as they
    # are visited.

    # Endpoints; first branch is not constrained and can add coordinates
    branch = next(it_branches)
    ext_nodes = branch[[0, -1]]

    ext_bounds = [(0, 1)]
    # For the other ones need to check collisions
    for branch in it_branches:
        endpoints = branch[[0, -1]]
        
        bounds = []
        for point in endpoints:
            dist_f = np.linalg.norm(ext_nodes - point, 2, axis=1)
            f_idx = np.argmin(dist_f)
            # Never seen this one
            if dist_f[f_idx] > tol:
                f_idx = len(ext_nodes)
                ext_nodes = np.vstack([ext_nodes, point])
            bounds.append(f_idx)
        ext_bounds.append(tuple(bounds))

    cells = []
    # For numbering the interior points we start with the next available
    fI = max(list(map(max, ext_bounds))) + 1  # Next avaiable
    for (fE, lE), branch in zip(ext_bounds, branches):
        int_nodes = branch[1:-1]

        if not len(int_nodes): continue
        
        # Get the node idices to encode cell connectivity
        lI = fI + len(int_nodes)
        vmap = np.r_[fE, np.arange(fI, lI), lE]

        # Actually add the coordinates
        ext_nodes = np.vstack([ext_nodes, int_nodes])

        # And the cells
        cells.extend(list(zip(vmap[:-1], vmap[1:])))
        
        # For next round
        fI = lI
    cells = np.array(cells, dtype='uintp').flatten()

    mesh = make_mesh(ext_nodes, cells, 1, ext_nodes.shape[1])

    return mesh


def line(A, B, npts):
    '''Uniformely seeded A-B'''
    D = B-A
    return np.array([A + D*t for t in np.linspace(0, 1, npts)])


def StraightLineMesh(A, B, ncells):
    '''Uniformely discrize segment A-B with ncells'''
    return line_mesh([line(A, B, ncells+1)])
