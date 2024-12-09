from dolfin import *
import numpy as np

template=r'''
\documentclass{standalone}
\usepackage[dvipsnames]{xcolor}

\usepackage{tikz}
\usetikzlibrary{calc}
\usetikzlibrary{shapes, snakes, patterns, arrows}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\usepackage{amsmath, amssymb}

%(colors)s

\begin{document}
\begin{tikzpicture}
%(body)s
\end{tikzpicture}
\end{document}
'''

import xii


def tikzify_2d_mesh(facet_info, cell_info=None, vertex_info=None, colors=None, bifurcation_nodes_for=None):
    '''Standalone Tikz representation of the mesh'''
    body, interesting_coordinates = [], []
    if cell_info is not None:
        cell_markers, cell_style_map = cell_info

        mesh = cell_markers.mesh()
        # Add midpoints for marked domains
        dx = Measure('dx', domain=mesh, subdomain_data=cell_markers)
        x = SpatialCoordinate(mesh)
        for color in map(int, np.unique(cell_markers.array())):
            volume = assemble(Constant(1)*dx(color))

            x_ = [assemble(xi*dx(color))/volume for xi in x]
            interesting_coordinates.append((np.array(x_), f'mid {color}'))

        gdim = mesh.geometry().dim()
        if cell_style_map is not None:
            x = mesh.coordinates()

            if gdim == 2:
                code = r'\fill[%s] (%g, %g) -- (%g, %g) -- (%g, %g) -- cycle;'
            else:
                code = r'\fill[%s] (%g, %g, %g) -- (%g, %g, %g) -- (%g, %g, %g) -- cycle;'
            for idx, cell in enumerate(mesh.cells()):
                style = cell_style_map[cell_markers[idx]]

                body.append(code % ((style, ) + tuple(x[cell].flatten())))

    if isinstance(facet_info, tuple):
        facet_info = [facet_info]

    # Here are corners meaning intersects of different colored facets
    for fi in facet_info:
        facet_markers, facet_style_map = fi
        
        mesh = facet_markers.mesh()
        x = mesh.coordinates()
        gdim = mesh.geometry().dim()
        
        dim = facet_markers.dim()
        assert dim == 1
        mesh.init(dim)
        mesh.init(dim, 0)
        f2v = mesh.topology()(dim, 0)

        facet_colors = tuple(set(np.unique(facet_markers.array())) - {0})
        for (i, ci) in enumerate(facet_colors):
            ci_vertices = set(np.hstack([f2v(f) for f in facet_markers.where_equal(ci)]))
            for cj in facet_colors[i+1:]:
                cj_vertices = set(np.hstack([f2v(f) for f in facet_markers.where_equal(cj)]))

                intersects = ci_vertices & cj_vertices
                for isect in intersects:
                    interesting_coordinates.append((x[isect], f'isect of {ci} and {cj}'))

        if gdim == 2:
            line = r'\draw[%s] (%g, %g) -- (%g, %g);'
        else:
            line = r'\draw[%s] (%g, %g, %g) -- (%g, %g, %g);'
        for color, facet_style in facet_style_map.items():

            if facet_style is None:
                continue
            
            for facet in SubsetIterator(facet_markers, color):
                xs = x[facet.entities(0)]
                body.append(line % ((facet_style, ) + tuple(xs.flatten())))

    interesting_coordinates.extend([(mesh.coordinates().min(axis=0), 'll'),
                                    (mesh.coordinates().max(axis=0), 'ur')])
    
    if vertex_info is not None:
        if isinstance(vertex_info, tuple):
            vertex_info = [vertex_info]

        for vi in vertex_info:
        
            vertex_markers, vertex_style_map = vi
            assert vertex_style_map is not None

            mesh = vertex_markers.mesh()
            x = mesh.coordinates()
            gdim = mesh.geometry().dim()

            if gdim == 2:
                code = r'\node[%s] at (%g, %g) {%s};'
            else:
                code = r'\node[%s] at (%g, %g, %g) {%s};'
            for idx, vtx in enumerate(mesh.coordinates()):
                style, marker = vertex_style_map[vertex_markers[idx]]
                if style is not None:
                    body.append(code % ((style,) + tuple(vtx) + (marker, )))

    if bifurcation_nodes_for is not None:
        iface = xii.EmbeddedMesh(facet_markers, bifurcation_nodes_for)
        vertices_iface = iface.coordinates()
        
        iface.init(0, 1)
        for facet in facets(iface):
            if len(facet.entities(1)) > 2:
                interesting_coordinates.append((vertices_iface[facet.index()], 'bif'))

    bifs = []
    # Define some useful coordinates
    for (i, (point, comment)) in enumerate(interesting_coordinates):
        j = i % 26
        k = i // 26
        if gdim == 2:
            body.append(rf'\coordinate (P{chr(65+j)}{chr(65+k)}) at ({point[0]}, {point[1]});  % {comment}')
        else:
            body.append(rf'\coordinate (P{chr(65+j)}{chr(65+k)}) at ({point[0]}, {point[1]}, {point[2]});  % {comment}')

        if comment == 'bif':
            body.append(r'\node[red, mark size=1pt] at (P%s%s) {$\pgfuseplotmark{*}$};' % (chr(65+j), chr(65+k)))
            # \node[mark size=1pt] at (XXX) {$\pgfuseplotmark{*}$};            
    body = '\n'.join(body)

    if colors is None:
        colors = ''
    else:
        print(colors)
        color_code = '\definecolor{%(color)s}{rgb}{%(RED)s, %(BLUE)s, %(GREEN)s}'
        colors = '\n'.join([color_code % {'color': color, 'RED': str(RGB[0]), 'BLUE': str(RGB[1]), 'GREEN': str(RGB[2])}
                            for color, RGB in colors.items()])
        print(colors)
    return template % {'body': body, 'colors': colors}


def load_mesh(h5_file, data_sets):
    '''
    Read in mesh and mesh functions from the data set in HDF5File.
    Data set is a tuple of (topological dim of entities, data-set-name)
    '''
    h5 = HDF5File(MPI.comm_world, h5_file, 'r')
    mesh = Mesh()
    h5.read(mesh, 'mesh', False)

    
    mesh_functions = []
    for dim, ds in data_sets:
        if h5.has_dataset(ds):
            f = MeshFunction('size_t', mesh, dim, 0)
            h5.read(f, ds)
        else:
            f = None
        mesh_functions.append(f)

    return mesh, mesh_functions

# -------------------------------------------------------------------

if __name__ == '__main__':
    from itertools import repeat
    
    path = './round_bdry.geo_d0.03125_0.5.h5'
    dim = 2
    mesh, [subdomains, bdries] = load_mesh(path, data_sets=((dim, 'volumes'), (dim-1, 'surfaces'), ))

    # style_map = dict(zip(set(bdries.array()), repeat('black!50!white, very thin')))
    style_map = dict(list(zip(set(bdries.array()), repeat(None))))
    style_map[1] = 'red, very thin'

    code = tikzify_2d_mesh(bdries, style_map)

    with open('mesh_2d.tex', 'w') as f: f.write(code)        


    mesh = UnitSquareMesh(2, 2)
    facet_f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(facet_f, 1)
    facet_style_map = {0: 'black', 1: 'black'}
    
    cell_f = MeshFunction('size_t', mesh, 2, 0)
    cell_f[0] = 1

    cell_style_map = {0: 'red', 1: 'blue'}
    
    code = tikzify_2d_mesh((facet_f, facet_style_map)
                           (cell_f, cell_style_map))
    with open('mesh_2d.tex', 'w') as f: f.write(code)        
