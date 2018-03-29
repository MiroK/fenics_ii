from dolfin import *
from xii import *

# mesh_file = 'geometry.h5'

# comm = mpi_comm_world()
# h5 = HDF5File(comm, mesh_file, 'r')
# mesh = Mesh()
# h5.read(mesh, 'mesh', False)

# volumes = MeshFunction('size_t', mesh, mesh.topology().dim())
# h5.read(volumes, 'physical')
mesh = Mesh('geometry.xml')
volumes = MeshFunction('size_t', mesh, 'geometry_physical_region.xml')

one = EmbeddedMesh(volumes, 1)
two = EmbeddedMesh(volumes, 2)

File('one.pvd') << one
File('two.pvd') << two

a, b, _ = mortar_meshes(volumes, (1, 2))

File('iface.pvd') << b

#f = FacetFunction('size_t', mesh, 0)
#DomainBoundary().mark(f, 1)

#File('x.pvd') << f

x = map(tuple, mesh.coordinates().tolist())
y = set(x)

print len(x), len(y)
