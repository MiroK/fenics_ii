# What happens when we have trace request points outside of mesh

from dolfin import *
from xii import *

n = 32

mesh0 = RectangleMesh(Point(0, 1), Point(1, 2), n, n)
ff0 = MeshFunction('size_t', mesh0, 1, 0)
CompiledSubDomain('near(x[0], 1)').mark(ff0, 1)

mesh1 = RectangleMesh(Point(1, 2), Point(2, 3), n, n)
ff1 = MeshFunction('size_t', mesh1, 1, 0)
CompiledSubDomain('near(x[1], 2)').mark(ff1, 1)

mesh2 = RectangleMesh(Point(2, 1), Point(3, 2), n, n)
ff2 = MeshFunction('size_t', mesh2, 1, 0)
CompiledSubDomain('near(x[0], 2)').mark(ff2, 1)

mesh3 = RectangleMesh(Point(1, 0), Point(2, 1), n, n)
ff3 = MeshFunction('size_t', mesh3, 1, 0)
CompiledSubDomain('near(x[1], 1)').mark(ff3, 1)

mesh = RectangleMesh(Point(1, 1), Point(2, 2), n, n)
cell_f = MeshFunction('size_t', mesh, 1, 0)

CompiledSubDomain('near(x[0], 1)').mark(cell_f, 1)
CompiledSubDomain('near(x[1], 2)').mark(cell_f, 2)
CompiledSubDomain('near(x[0], 2)').mark(cell_f, 3)
CompiledSubDomain('near(x[1], 1)').mark(cell_f, 4)

iface_mesh = EmbeddedMesh(cell_f, [1, 2, 3, 4])

V = FunctionSpace(iface_mesh, 'CG', 1)
u, v = TrialFunction(V), TestFunction(V)

dx = Measure('dx', domain=iface_mesh, subdomain_data=iface_mesh.marking_function)

f_ = Expression('2*x[0]-x[1]', degree=1)
f = interpolate(f_,
                FunctionSpace(RectangleMesh(Point(0, 0), Point(3, 3), 3*n, 3*n),
                              'CG',
                              1))

a = inner(u, v)*dx()
L = sum(inner(Trace(f, iface_mesh), v)*dx(i) for i in range(1, 5))

A, b = map(ii_assemble, (a, L))

uh = Function(V)
solve(A, uh.vector(), b)

print sqrt(abs(assemble(inner(f_ - uh, f_ - uh)*dx()))), sqrt(abs(assemble(inner(uh, uh)*dx())))
