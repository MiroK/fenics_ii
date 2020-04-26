from __future__ import absolute_import
import subprocess, os
import dolfin as df
from xii import EmbeddedMesh
import six
from six.moves import map
from six.moves import range
from six.moves import zip


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


def h5_file_name(scale, inner_size):
    return 'mesh_%g_%g.h5' % (scale, inner_size)


def generate(scale=1, inner_size=1):
    '''Generate mesh from geo file storing it as H5File'''
    h5_file = h5_file_name(scale, inner_size)

    # Reuse
    if os.path.exists(h5_file): return h5_file
    
    # New
    cmd = 'gmsh -3 -setnumber inner %g -clscale %g domain.geo' % (inner_size, scale)
    status = subprocess.call([cmd], shell=True)
    assert status == 0
    assert os.path.exists('domain.msh')

    # xml
    assert convert('domain.msh', h5_file)
    return h5_file


def load(scale, inner_size, curve_gen):
    '''Load meshes for 3d-2d-1d problems'''
    h5_file = generate(scale, inner_size)

    comm = df.mpi_comm_world()
    h5 = df.HDF5File(comm, h5_file, 'r')
    mesh = df.Mesh()
    h5.read(mesh, 'mesh', False)

    surfaces = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    h5.read(surfaces, 'facet')

    volumes = df.MeshFunction('size_t', mesh, mesh.topology().dim())
    h5.read(volumes, 'physical')

    # The 3d mesh is just mesh
    mesh_3d = mesh

    # Mesh for 2d is EmbeddedMesh using tags (1, 2, 3, 4)
    mesh_2d = EmbeddedMesh(surfaces, (1, 2, 3, 4))

    # 1d mesh from tagged facets of 2d
    facet_f = curve_gen(mesh_2d)

    mesh_1d = EmbeddedMesh(facet_f, 1)

    return mesh_3d, mesh_2d, mesh_1d


def boring(mesh_2d, inner_size):
    '''
    A mesh2d is assumed to be be a cube [-inner_size, inner_size]^2.
    The curve is mostly a collection of boundary edges.
    '''
    facet_f = df.MeshFunction('size_t', mesh_2d, 1, 0)
    mesh_2d.init(2, 1)
    
    # Mesh for the curve is tricky as we need to find the line in the faces
    def union(domains, A=inner_size, tol=1E-10):
        def body(domains):
            if isinstance(domains, str):
                if domains:
                    return '( %s )' % domains
                else:
                    return ''
            else:
                return ' || '.join(map(body, domains))
        return df.CompiledSubDomain(body(domains), A=A, tol=tol)

    lines = {4: union('near(x[1], A, tol) && near(x[2], A, tol)'),
             3: union('near(x[2], -x[0], tol)'),
             2: union('near(x[2], x[1], tol)'),
             1: union(['near(x[0], -A, tol) && near(x[2], -A, tol)',
                       'near(x[1], A, tol) && near(x[0], -A, tol)',
                       'near(x[1], -A, tol) && near(x[0], -A, tol)'])}
             
    for tag, line in six.iteritems(lines):
        # Get candidates
        facets = set(sum((cell.entities(1).tolist()
                          for cell in df.SubsetIterator(mesh_2d.marking_function, tag))
                         , []))
        for facet in facets:
            if line.inside(df.Facet(mesh_2d, facet).midpoint().array(), True):
                facet_f[int(facet)] = 1

    return facet_f


def fun(mesh2d, nselects):
    '''A random curve from the edges'''
    import networkx as nx
    import random

    facet_f = df.MeshFunction('size_t', mesh2d, 1, 0)    
    mesh2d.init(1, 0)
    # Init the graph
    G = nx.Graph()

    edge_indices = {tuple(sorted(facet.entities(0).tolist())): f_index
                    for f_index, facet in enumerate(df.facets(mesh2d))}
    G.add_edges_from(six.iterkeys(edge_indices))

    vertices = list(range(mesh2d.num_vertices()))
    for _ in range(nselects):
        v0, v1 = random.sample(vertices, 2)
        if v0 == v1: continue

        # THe path is a shortest path between 2 random vertices
        path = nx.shortest_path(G, source=v0, target=v1)
        for v0, v1 in zip(path[:-1], path[1:]):
            edge = (v0, v1) if v0 < v1 else (v1, v0)
            facet_f[edge_indices[edge]] = 1

    return facet_f
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    load(0.5, 0.5, lambda mesh: fun(mesh, 20))
