from dolfin import *
from xii.assembler.trace_matrix import trace_mat
from xii import EmbeddedMesh


mesh = UnitSquareMesh(10, 10)

bdry = MeshFunction('size_t', mesh, 1, 0)
DomainBoundary().mark(bdry, 1)
bmesh = EmbeddedMesh(bdry, 1)

V = FunctionSpace(mesh, 'CG', 2)
TV = FunctionSpace(bmesh, 'CG', 2)

Trace = trace_mat(V, TV, bmesh, {'restriction': '', 'normal': None})

f = Expression('sin(pi*(x[0]+x[1]))', degree=3)

v = interpolate(f, V)
Tv0 = interpolate(f, TV)

Tv = Function(TV)
Trace.mult(v.vector(), Tv.vector())

Tv0.vector().axpy(-1, Tv.vector())
print(Tv0.vector().norm('linf'))


V = VectorFunctionSpace(mesh, 'CG', 2)
TV = VectorFunctionSpace(bmesh, 'CG', 2)

Trace = trace_mat(V, TV, bmesh, {'restriction': '', 'normal': None})

f = Expression(('sin(pi*(x[0]+x[1]))', 'x[0]+x[1]'), degree=3)

v = interpolate(f, V)
Tv0 = interpolate(f, TV)

Tv = Function(TV)
Trace.mult(v.vector(), Tv.vector())

Tv0.vector().axpy(-1, Tv.vector())
print(Tv0.vector().norm('linf'))
