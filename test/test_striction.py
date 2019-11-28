from dolfin import *
from xii import *


mesh = UnitSquareMesh(32, 32)
cell_f = MeshFunction('size_t', mesh, 2, 0)
CompiledSubDomain('x[0] < 0.5+DOLFIN_EPS').mark(cell_f, 1)

mesh1 = EmbeddedMesh(cell_f, 1)
mesh0 = EmbeddedMesh(cell_f, 0)

V1 = FunctionSpace(mesh1, 'CG', 1)
V0 = FunctionSpace(mesh0, 'CG', 1)

u0, u1 = map(TrialFunction, (V0, V1))

Q = FunctionSpace(mesh, 'CG', 1)
q = TestFunction(Q)

dx0 = Measure('dx', domain=mesh0)
a = inner(u0, Restriction(q, mesh0))*dx0
A = ii_assemble(a)

f = Expression('x[0]+x[1]', degree=1)
g = Expression('2*x[0]-x[1]', degree=1)

fh, gh = interpolate(f, V0), interpolate(g, Q)

true = assemble(inner(f, g)*dx0)

mine = gh.vector().inner(A*fh.vector())

print abs(true-mine), abs(true)
