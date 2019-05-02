from xii import *
from dolfin import *


n = 16
mesh = UnitSquareMesh(*(n, )*2)
mesh_fine = UnitSquareMesh(*(2*n, )*2)
bmesh = BoundaryMesh(mesh_fine, 'exterior')

V = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(bmesh, 'CG', 1)

v = TestFunction(V)
p = TrialFunction(Q)
Tv = Trace(v, bmesh)

# The line integral
dx_ = Measure('dx', domain=bmesh)
a = inner(Tv, p)*dx_
A = ii_assemble(a)

foo = Expression('1+x[0]-2*x[1]', degree=1)
bar = Expression('1-3*x[0]-2*x[1]', degree=1)

true = assemble(foo*bar*ds(domain=mesh_fine))

num = interpolate(bar, V).vector().inner(A*interpolate(foo, Q).vector())

print abs(true - num)
