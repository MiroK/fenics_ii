from dolfin import *
from xii import *

f = Expression(('x[0]', '2*x[1]', '2*x[2]-x[0]'), degree=1)

mesh = UnitCubeMesh(4, 4, 4)
facet_f = MeshFunction('size_t', mesh, 2, 0)
DomainBoundary().mark(facet_f, 1)

n = FacetNormal(mesh)
true = assemble(inner(n, f)*ds)

surface_mesh = EmbeddedMesh(facet_f, 1)
n = OuterNormal(surface_mesh, orientation=mesh)

dx_ = Measure('dx', domain=surface_mesh)
mine = ii_assemble(inner(n, f)*dx_)

print(abs(true-mine))

File('normal.pvd') << n
