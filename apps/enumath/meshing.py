from __future__ import absolute_import
from __future__ import print_function
import subprocess, os
import dolfin as df
import numpy as np
from xii import EmbeddedMesh
import six
from six.moves import map
from six.moves import range
from six.moves import zip

TOL = 1E-8


def convert(msh_file, h5_file):
    '''Convert from msh to h5'''
    root, _ = os.path.splitext(msh_file)
    assert os.path.splitext(msh_file)[1] == '.msh'
    assert os.path.splitext(h5_file)[1] == '.h5'

    # Get the xml mesh
    xml_file = '.'.join([root, 'xml'])
    subprocess.call(['dolfin-convert %s %s' % (msh_file, xml_file)], shell=True)
    # Success?
    assert os.path.exists(xml_file)

    cmd = '''from dolfin import Mesh, HDF5File;\
             mesh=Mesh('%(xml_file)s');\
             assert mesh.topology().dim() == 3;\
             out=HDF5File(mesh.mpi_comm(), '%(h5_file)s', 'w');\
             out.write(mesh, 'mesh');''' % {'xml_file': xml_file,
                                             'h5_file': h5_file}

    for region in ('facet_region.xml', 'physical_region.xml'):
        name, _ = region.split('_')
        r_xml_file = '_'.join([root, region])
        if os.path.exists(r_xml_file):
            cmd_r = '''from dolfin import MeshFunction;\
                       f = MeshFunction('size_t', mesh, '%(r_xml_file)s');\
                       out.write(f, '%(name)s');\
                       ''' % {'r_xml_file': r_xml_file, 'name': name}
        
            cmd = ''.join([cmd, cmd_r])

    cmd = 'python -c "%s"' % cmd

    status = subprocess.call([cmd], shell=True)
    assert status == 0
    # Sucess?
    assert os.path.exists(h5_file)

    # Cleanup
    list(map(os.remove, [f for f in os.listdir('.')
                    if f.endswith('.xml') and f.startswith('domain')]))
    
    list(map(os.remove, [f for f in os.listdir('.')
                    if f.endswith('.msh') and f.startswith('domain')]))
    
    return True


def h5_file_name(scale):
    return 'mesh_%g.h5' % scale


def generate(scale=1):
    '''Generate mesh from geo file storing it as H5File'''
    h5_file = h5_file_name(scale)

    # Reuse
    if os.path.exists(h5_file): return h5_file
    
    # New
    cmd = 'gmsh -3 -clscale %g domain.geo' % scale
    status = subprocess.call([cmd], shell=True)
    assert status == 0
    assert os.path.exists('domain.msh')

    # xml
    assert convert('domain.msh', h5_file)
    return h5_file


def load(scale, curve_gen):
    '''Load meshes for 3d-1d problems'''
    h5_file = generate(scale)

    comm = df.mpi_comm_world()
    h5 = df.HDF5File(comm, h5_file, 'r')
    mesh3d = df.Mesh()
    h5.read(mesh3d, 'mesh', False)

    # 1d mesh from tagged edges
    edge_f, bc_vertex = curve_gen(mesh3d)
    mesh1d = EmbeddedMesh(edge_f, 1)
    # Check thet we really have only on surface point
    bdry = df.CompiledSubDomain('near(std::max(std::max(std::abs(x[0]), std::abs(x[1])), std::abs(x[2])), 1, tol)', tol=TOL)
    e = sum(int(bdry.inside(x, False)) for x in mesh1d.coordinates())
    assert e == 1, e

    return mesh3d, mesh1d, bc_vertex


def fun(mesh3d, npoints):
    '''A random curve starting close to (-1, -1, -1) and continueing inside'''
    import networkx as nx
    import random

    edge_f = df.MeshFunction('size_t', mesh3d, 1, 0)    
    mesh3d.init(1, 0)
    # Init the graph
    G = nx.Graph()

    edge_indices = {tuple(sorted(e.entities(0).tolist())): e_index
                    for e_index, e in enumerate(df.edges(mesh3d))}
    G.add_edges_from(six.iterkeys(edge_indices))

    # Let's find boundary vertices
    V = df.FunctionSpace(mesh3d, 'CG', 1)
    bdry = df.CompiledSubDomain('near(std::max(std::max(std::abs(x[0]), std::abs(x[1])), std::abs(x[2])), 1, tol)', tol=TOL)
    bc = df.DirichletBC(V, df.Constant(0), bdry, 'pointwise')

    bc_vertices = set(df.dof_to_vertex_map(V)[list(bc.get_boundary_values().keys())])
    # Start at the boundary at (-1, -1, 1)
    X = mesh3d.coordinates()
    start = np.argmin(np.sum((X - np.array([-1, -1, -1]))**2, axis=1))
    
    # All vertices
    vertices = list(range(mesh3d.num_vertices()))
    first = None
    while npoints:
        # Pick the next vertex, inside
        while True:
            stop = random.choice(vertices)
            if start != stop and stop not in bc_vertices: break

        # The path is a shortest path between vertices
        path = nx.shortest_path(G, source=start, target=stop)

        # Here it can happen that the path will have surface points
        if first is None:
            # So we walk back (guaranteed to be in) until we hit the surface
            clean_path = []
            for p in reversed(path):
                clean_path.append(p)
                if p in bc_vertices:
                    print('Shifted start to', X[p])
                    first = p
                    break
            path = clean_path
            start = first
        # Start in, must end in and stay in
        else:
            if set(path) & bc_vertices: continue
        
        for v0, v1 in zip(path[:-1], path[1:]):
            edge = (v0, v1) if v0 < v1 else (v1, v0)
            edge_f[edge_indices[edge]] = 1
        start = stop
        npoints -= 1

    # df.File('x.pvd') << edge_f
        
    return edge_f, X[first]
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    load(0.25, lambda mesh: fun(mesh, 80))
