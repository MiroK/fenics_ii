from __future__ import absolute_import
from dolfin import *
from xii import *

# mesh_file = 'geometry.h5'

# comm = mpi_comm_world()
# h5 = HDF5File(comm, mesh_file, 'r')
# mesh = Mesh()
# h5.read(mesh, 'mesh', False)

# volumes = MeshFunction('size_t', mesh, mesh.topology().dim())
# h5.read(volumes, 'physical')
mesh = Mesh('geometry_old.xml')
surfaces = MeshFunction('size_t', mesh, 'geometry_old_facet_region.xml')

one = EmbeddedMesh(surfaces, 1)
two = EmbeddedMesh(surfaces, 4)

File('iface0_old.pvd') << one
File('ifacae1_old.pvd') << two

