from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from xii import *

mesh = UnitSquareMesh(2, 2)
bmesh = BoundaryMesh(mesh, 'exterior')

V = FunctionSpace(mesh, 'Real', 0)
Q = FunctionSpace(bmesh, 'CG', 1)
dx_ = Measure('dx', domain=bmesh)


u = TrialFunction(V)
Tu = Trace(u, bmesh)
q = TestFunction(Q)

a = inner(Tu, q)*dx_
A = ii_assemble(a)
print(ii_convert(A).array())


v = TestFunction(V)
Tv = Trace(v, bmesh)
p = TrialFunction(Q)

a = inner(Tv, p)*dx_
A = ii_assemble(a)
print(ii_convert(A).array())
