from dolfin import *
from xii import *


mesh_c = UnitSquareMesh(32, 32)  # Coarse (for pressure)
mesh_f = adapt(mesh_c)


Vc = FunctionSpace(mesh_c, 'CG', 1)
Vf = FunctionSpace(mesh_f, 'CG', 1)

u = TrialFunction(Vf)
q = TestFunction(Vc)

dx_c = Measure('dx', domain=mesh_c)
a = inner(q, Restriction(u, mesh_c))*dx_c
A = ii_assemble(a)

# f = Expression('x[0]+x[1]', degree=1)
# g = Expression('2*x[0]-x[1]', degree=1)

# fh, gh = interpolate(f, V0), interpolate(g, Q)

# true = assemble(inner(f, g)*dx0)

# mine = gh.vector().inner(A*fh.vector())

# print abs(true-mine), abs(true)
