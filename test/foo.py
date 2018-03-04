from dolfin import *
from xii import *

n = 32

mesh = UnitSquareMesh(*(n, )*2)
gamma = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
CompiledSubDomain('on_boundary || near(x[0], 0.5) || near(x[1], 0.5)').mark(gamma, 1)
bmesh = EmbeddedMesh(gamma, 1)

V = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(bmesh, 'CG', 1)
W = [V, Q]

u, p = map(TrialFunction, W)
v, q = map(TestFunction, W)
Tu = Trace(u, bmesh)
Tv = Trace(v, bmesh)

dx_ = Measure('dx', domain=bmesh)

a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx + inner(Tu, Tv)*dx_
a01 = inner(Tv, p)*dx_
a10 = inner(Tu, q)*dx_

f = Expression('x[0]+x[1]', degree=1)
x, y = SpatialCoordinate(bmesh)

L0 = inner(f, v)*dx
L1 = inner(x*(1-x)*y*(1-y), q)*dx_

a = [[a00, a01], [a10, 0]]
L = [L0, L1]


AA, bb = map(ii_assemble, (a, L))

AA, bb = map(ii_convert, (AA, bb))

# w = ii_Function(W)
#print solve(AA, w.vector(), bb)
#x = bb.copy()
uh, ph = map(Function, W)

from petsc4py import PETSc
x = PETSc.Vec().createNest([as_backend_type(uh.vector()).vec(),
                            as_backend_type(ph.vector()).vec()])
solve(AA, PETScVector(x), bb)


File('foo0.pvd') << uh
File('foo1.pvd') << ph
