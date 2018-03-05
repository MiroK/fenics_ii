# The system from d'Angelo & Quarteroni paper on tissue perfusion
from dolfin import *
from xii import *

mesh = UnitCubeMesh(10, 10, 10)
radius = 0.01
quadrature_degree = 10


f = EdgeFunction('size_t', mesh, 0)
CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(f, 1)
bmesh = EmbeddedMesh(f, 1)

V = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(bmesh, 'CG', 1)
W = (V, Q)

u, p = map(TrialFunction, W)
v, q = map(TestFunction, W)

Pi_u = Average(u, bmesh, radius, quadrature_degree)
Pi_v = Average(v, bmesh, radius, quadrature_degree)

dxGamma = Measure('dx', domain=bmesh)


a10 = inner(Pi_u, q)*dxGamma
a01 = inner(p, Pi_v)*dxGamma

print ii_assemble(a10)
print ii_assemble(a01)


