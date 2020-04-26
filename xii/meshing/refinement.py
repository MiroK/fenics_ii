from __future__ import absolute_import
from __future__ import print_function
from xii.meshing.make_mesh_cpp import make_mesh
from itertools import combinations, count
import numpy as np
from six.moves import map
from six.moves import range
from six.moves import zip


def inner_point_refine(mesh, new_pts, strict):
    ''''
    Mesh refine by adding point inside each celll

       x                      x   
      / \     becomes        /|\  
     /   \                  / x \
    /     \                / / \ \  
    x------x              x-------x

    Strict is the rel tol for checing whether new_pts[i] is inside cell[i]
    '''
    x = mesh.coordinates()
    cells = mesh.cells()

    ncells, nvtx_cell = cells.shape
    assert any((nvtx_cell == 2,
                nvtx_cell == 3,
                nvtx_cell == 4 and mesh.topology().dim() == 3))

    # Center points will be new coordinates
    xnew = new_pts(mesh)
    # We have to check for duplicates
    #if not unique_guarantee:
    #    pass
    if strict > 0:
        tol = mesh.hmin()*strict
        # The collision
        assert point_is_inside(xnew, x[cells], tol)
        
    # Each cell gives rise to ...
    child2parent = np.empty(ncells*nvtx_cell, dtype='uintp')
    fine_cells = np.empty((ncells*nvtx_cell, nvtx_cell), dtype='uintp')
    # How we build new cells
    basis = list(map(list, combinations(list(range(nvtx_cell)), nvtx_cell-1)))

    fine_coords = np.row_stack([x, xnew])
    
    fc, center = 0, len(x)
    for pc, cell in enumerate(cells):
        for base in basis:
            new_cell = np.r_[cell[base], center]
            # Every new cell must be non-empty
            assert simplex_area(fine_coords[new_cell]) > 1E-15

            fine_cells[fc, :] = new_cell
            child2parent[fc] = pc
            fc += 1
        center += 1

    tdim = mesh.topology().dim()
    fine_mesh = make_mesh(fine_coords, fine_cells, tdim, mesh.geometry().dim())
    
    fine_mesh.set_parent(mesh)
    mesh.set_child(fine_mesh)

    fine_mesh.parent_entity_map = {mesh.id(): {0: dict(enumerate(range(mesh.num_vertices()))),
                                               tdim: dict(enumerate(child2parent))}}
       
    return fine_mesh
    


def centroid_refine(mesh, strict=1E-10):
    '''Using centroid'''
    # How to
    new_pts = lambda mesh: np.mean(mesh.coordinates()[mesh.cells()], axis=1)
       
    return inner_point_refine(mesh, new_pts, strict)


def incenter_refine(mesh, strict=1E-10):
    '''Refine using angle bisector'''
    assert mesh.topology().dim() == 2
    # How to
    def new_pts(mesh):
        x = mesh.coordinates()
        cells = mesh.cells()

        lengths = np.column_stack([np.linalg.norm(x[cells[:, v0]] - x[cells[:, v1]], axis=1)
                                   for (v0, v1) in ((1, 2), (2, 0), (0, 1))])
        sum_lengths = np.sum(lengths, axis=1)
    
        new_coords = []
        for i in range(mesh.geometry().dim()):
            X = x[cells][:, :, i]
            # Weight each coordinate by length and do mean
            new_coords.append(np.sum(X*lengths, axis=1)/sum_lengths)
        new_coords = np.column_stack(new_coords)

        return new_coords

    return inner_point_refine(mesh, new_pts, strict)


def circumncenter_refine(mesh, strict=1E-10):
    '''Refine using othocenter'''
    assert mesh.topology().dim() == 2
    # How to
    def new_pts(mesh):
        x = mesh.coordinates()
        cells = mesh.cells()

        a, b, c = np.column_stack([np.linalg.norm(x[cells[:, v0]] - x[cells[:, v1]], axis=1)
                                   for (v0, v1) in ((1, 2), (2, 0), (0, 1))]).T
        
        lengths = np.c_[a**2*(b**2+c**2-a**2),
                        b**2*(c**2+a**2-b**2),
                        c**2*(a**2+b**2-c**2)]

        sum_lengths = np.sum(lengths, axis=1)
    
        new_coords = []
        for i in range(mesh.geometry().dim()):
            X = x[cells][:, :, i]
            # Weight each coordinate by length and do mean
            new_coords.append(np.sum(X*lengths, axis=1)/sum_lengths)
        new_coords = np.column_stack(new_coords)
        
        return new_coords

    return inner_point_refine(mesh, new_pts, strict)



def point_is_inside(x, cell, tol):
    '''Is x inside a simplex cell'''
    # Many
    if x.ndim > 1:
        return all(point_is_inside(*pair, tol=tol) for pair in zip(x, cell))
    # We say the point is inside if the area of small cells made using
    # it adds up to the area of the cell
    diff = abs(simplex_area(cell) - sum(map(simplex_area, simplices(x, cell))))
    return diff < tol


def simplices(x, cell):
    '''Insert x to cell'''
    # Subdivide interval AB -> AX, BX
    nvtx = len(cell)
    for other in map(list, combinations(list(range(nvtx)), nvtx-1)):
        yield np.row_stack([x, cell[other]])

        
def simplex_area(cell):
    '''The name says it all'''
    if len(cell) == 2:
        return np.linalg.norm(np.diff(cell, axis=0))

    if len(cell) == 3:
        A, B, C = cell
        if len(A) == 2:
            return np.abs(np.linalg.det(np.column_stack([cell, np.ones(3)]))/2)
        else:
            return np.linalg.norm(np.cross(B-A, C-A))/2.

    if len(cell) == 4:
        return np.abs(np.linalg.det(np.column_stack([cell, np.ones(4)]))/6.)
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, File, MeshFunction, facets, adapt
    from xii.meshing.tikz import tikzify_2d_mesh
    import networkx as nx
    from networkx.algorithms.coloring import greedy_color
    
    mesh = UnitSquareMesh(2, 2)

    # Graph coloring
    def color_mesh(mesh):
        '''Minimal color'''
        tdim = mesh.topology().dim()
        # Connectivity over facets
        mesh.init(tdim-1)
        mesh.init(tdim-1, tdim)

        G = nx.Graph()
        for facet in facets(mesh):
            maybe_edge = facet.entities(tdim)  # In the graph
            len(maybe_edge) == 2 and G.add_edge(*maybe_edge)

            greedy_color

        colors = greedy_color(G)

        return colors

    graph_colors = color_mesh(mesh)
    num_colors = len(set(graph_colors.values()))

    cell_style_map = {0: 'red!40!white',
                      2: 'red!80!white',
                      1: 'blue!20!white',
                      3: 'blue!80!white',
                      4: 'yellow!80!white', 
                      5: 'green!80!white'}
    
    refine_ways = {'centroid': centroid_refine,
                   'incenter': incenter_refine,
                   'plazza': adapt}

    meshes = {}
    for name, refine in refine_ways.items():
        print('>>>>>>>>>>>>>>>>>>>%s<<<<<<<<<<<<<<<<<<<' % name)
        fine_mesh = refine(mesh)
        meshes[name] = fine_mesh
        assert fine_mesh.has_parent()

        facet_f = MeshFunction('size_t', fine_mesh, 1, 0)
        facet_style_map = {0: 'very thin, black'}

        mesh_f = MeshFunction('size_t', fine_mesh, 2, 0)
        try:
            for child, parent in fine_mesh.parent_entity_map[mesh.id()][2].items():
                mesh_f[child] = graph_colors[parent]
        except AttributeError:
            for child, parent in enumerate(fine_mesh.data().array('parent_cell', 2)):
                mesh_f[child] = graph_colors[parent]
            
        # TODO: put tikz here
        File('%s.pvd' % name) << mesh_f

        code = tikzify_2d_mesh(facet_f, facet_style_map,
                               mesh_f, cell_style_map)

        with open('%s.tex' % name, 'w') as out:
            out.write(code)

    # ---
    from dolfin import *

    mesh = UnitIntervalMesh(10)
    x = mesh.coordinates()
    for cell in cells(mesh):
        assert abs(simplex_area(x[cell.entities(0)]) - cell.volume()) < 1E-15
        
        p = cell.midpoint().array()[:1]
        assert point_is_inside(p, x[cell.entities(0)], 1E-10)

    mesh = UnitSquareMesh(10, 10)
    x = mesh.coordinates()    
    for cell in cells(mesh):
        assert abs(simplex_area(x[cell.entities(0)]) - cell.volume()) < 1E-15
    
        p = cell.midpoint().array()[:2]
        assert point_is_inside(p, x[cell.entities(0)], 1E-10)

    mesh = UnitCubeMesh(3, 3, 3)
    x = mesh.coordinates()    
    for cell in cells(mesh):
        assert abs(simplex_area(x[cell.entities(0)]) - cell.volume()) < 1E-15

        p = cell.midpoint().array()
        assert point_is_inside(p, x[cell.entities(0)], 1E-10)    
