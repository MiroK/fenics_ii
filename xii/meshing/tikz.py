from dolfin import *

template=r'''
\documentclass{standalone}
\usepackage{tikz}
\usetikzlibrary{calc}
\usetikzlibrary{shapes, snakes, patterns, arrows}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\usepackage{amsmath, amssymb}

\begin{document}
\begin{tikzpicture}
%(body)s
\end{tikzpicture}
\end{document}
'''

def tikzify_2d_mesh(facet_info, cell_info=None, vertex_info=None):
    '''Standalone Tikz representation of the mesh'''
    body = []
    if cell_info is not None:
        cell_markers, cell_style_map = cell_info
        assert cell_style_map is not None

        mesh = cell_markers.mesh()
        x = mesh.coordinates()

        code = r'\fill[%s] (%g, %g) -- (%g, %g) -- (%g, %g) -- cycle;'
        for idx, cell in enumerate(mesh.cells()):
            style = cell_style_map[cell_markers[idx]]
            
            body.append(code % ((style, ) + tuple(x[cell].flatten())))

    if isinstance(facet_info, tuple):
        facet_info = [facet_info]

    for fi in facet_info:
        facet_markers, facet_style_map = fi
        
        mesh = facet_markers.mesh()
        assert mesh.geometry().dim() == 2
        x = mesh.coordinates()

        dim = facet_markers.dim()
        assert dim == 1
        mesh.init(dim)
        mesh.init(dim, 0)

        line = r'\draw[%(style)s] (%(x00)g, %(x01)g) -- (%(x10)g, %(x11)g);'
        for facet in facets(mesh):
            style = facet_style_map[facet_markers[facet]]
            if style is not None:
                x0, x1 = x[facet.entities(0)]

                body.append(line % {'style': style, 'x00': x0[0], 'x01': x0[1],
                                    'x10': x1[0], 'x11': x1[1]})


    if vertex_info is not None:
        if isinstance(vertex_info, tuple):
            vertex_info = [vertex_info]

        for vi in vertex_info:
        
            vertex_markers, vertex_style_map = vi
            assert vertex_style_map is not None

            mesh = vertex_markers.mesh()
            x = mesh.coordinates()

            code = r'\node[%s] at (%g, %g) {%s};'
            for idx, vtx in enumerate(mesh.coordinates()):
                style, marker = vertex_style_map[vertex_markers[idx]]
                if style is not None:
                    body.append(code % (style, vtx[0], vtx[1], marker))
                
    body = '\n'.join(body)

    return template % {'body': body}


def load_mesh(h5_file, data_sets):
    '''
    Read in mesh and mesh functions from the data set in HDF5File.
    Data set is a tuple of (topological dim of entities, data-set-name)
    '''
    h5 = HDF5File(mpi_comm_world(), h5_file, 'r')
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
    style_map = dict(zip(set(bdries.array()), repeat(None)))
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
