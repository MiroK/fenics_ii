from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from xii import *

mesh = UnitSquareMesh(32, 32)
cf = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cf, 1)

mesh1 = EmbeddedMesh(cf, 1)

bmesh = BoundaryMesh(mesh, 'exterior')

V = FunctionSpace(mesh1, 'CG', 1)
Q = FunctionSpace(bmesh, 'CG', 1)

u = TrialFunction(V)
q = TestFunction(Q)

cf = MeshFunction('size_t', bmesh, bmesh.topology().dim(), 0)
CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cf, 1)
dxGamma = Measure('dx', domain=bmesh, subdomain_data=cf)

print(ii_assemble(inner(Trace(u, bmesh), q)*dxGamma(1)))
