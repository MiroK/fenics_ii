# What happens when we have trace request points outside of mesh

from __future__ import absolute_import
from __future__ import print_function
from dolfin import *
from xii import *
from six.moves import map
from six.moves import range

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

Q = FunctionSpace(iface_mesh, 'CG', 1)
p, q = TrialFunction(Q), TestFunction(Q)

dx = Measure('dx', domain=iface_mesh, subdomain_data=iface_mesh.marking_function)

f_ = Expression('2*x[0]-x[1]', degree=1)
f = interpolate(f_, FunctionSpace(mesh, 'CG', 1))

a = inner(p, q)*dx()
L = sum(inner(Trace(f, iface_mesh, tag=i), q)*dx(i) for i in range(1, 5))

A, b = list(map(ii_assemble, (a, L)))

uh = Function(Q)
solve(A, uh.vector(), b)

print(sqrt(abs(assemble(inner(f_ - uh, f_ - uh)*dx()))), sqrt(abs(assemble(inner(uh, uh)*dx()))))

# --------------------------------------------------------------------
# What if the interface is not shared
# --------------------------------------------------------------------
#  |---|---|
#  |0  |2__|
#  |  /1   |
#  | /-----|

domain0 = '(x[0] < 0.5 + tol) && (x[1] > x[0] - tol)'
domain1 = '(x[1] < 0.5 + tol) && (x[1] < x[0] + tol)'
domain2 = '(x[0] > 0.5-tol) && (x[1] > 0.5-tol)'

iface = ['near(x[0], 0.5) && x[1] > 0.5 - tol',
         'near(x[0], x[1]) && x[0] < 0.5 + tol',
         'near(x[1], 0.5) && x[0] > 0.5 - tol']

mesh = UnitSquareMesh(n, n)
# Iface
facet_f = MeshFunction('size_t', mesh, 1, 0)
for idx, subd in enumerate(iface, 1):
    CompiledSubDomain(subd, tol=1E-10).mark(facet_f, idx)

iface = EmbeddedMesh(facet_f, (1, 2, 3))
dx_ = Measure('dx', domain=iface, subdomain_data=iface.marking_function)

# Subdomains
cell_f = MeshFunction('size_t', mesh, 2, 0)
for idx, subd in enumerate((domain0, domain1, domain2)):
    CompiledSubDomain(subd, tol=1E-10).mark(cell_f, idx)

f = Expression('2*x[1]-3*x[0]', degree=1)
# Now imagine that each subdomain has a piece of interfomation and we
# want to glue it together
meshes = [SubMesh(mesh, cell_f, tag) for tag in range(3)]
Vs = [FunctionSpace(mesh, 'CG', 1) for mesh in meshes]
vs = [Trace(interpolate(f, Vi), iface, tag=tag) for tag, Vi in enumerate(Vs, 1)]

Q = FunctionSpace(iface, 'CG', 1)
p, q = TrialFunction(Q), TestFunction(Q)

a = inner(p, q)*dx_()
L = inner(vs[0], q)*dx_(1) + inner(vs[1], q)*dx_(2) + inner(vs[2], q)*dx_(3)

A, b = list(map(ii_assemble, (a, L)))

uh = Function(Q)
solve(A, uh.vector(), b)

print(sqrt(abs(assemble(inner(f - uh, f - uh)*dx_()))), sqrt(abs(assemble(inner(uh, uh)*dx_()))))
