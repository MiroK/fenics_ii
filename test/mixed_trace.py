from dolfin import *
from xii import *

n = 4

mesh = UnitSquareMesh(*(n, )*2)
gamma = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
CompiledSubDomain('on_boundary || near(x[0], 0.5) || near(x[1], 0.5)').mark(gamma, 1)

bmesh = EmbeddedMesh(gamma, 1)
dx_ = Measure('dx', domain=bmesh)

# The test cases
cases = []

# Common H0.5
elm0 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, elm0)

elm0 = FiniteElement('Lagrange', bmesh.ufl_cell(), 1)
Q = FunctionSpace(bmesh, elm0)

f = Expression('x[0]+x[1]', degree=1)
g = Expression('2*x[0]-3*x[1]', degree=1)

cases.append(((V, f), (Q, g)))

# Mixed trace
elm0 = FiniteElement('Lagrange', mesh.ufl_cell(), 2)
elm1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([elm0, elm1]))

elm0 = FiniteElement('Lagrange', bmesh.ufl_cell(), 2)
elm1 = FiniteElement('Lagrange', bmesh.ufl_cell(), 1)
Q = FunctionSpace(bmesh, MixedElement([elm0, elm1]))

f = Expression(('x[0]*x[1]', 'x[0]+x[1]'), degree=2)
g = Expression(('x[0]*x[0] + x[1]*x[1]', '2*x[0]-3*x[1]'), degree=2)

cases.append(((V, f), (Q, g)))

for (V, f), (Q, g) in cases:
    u = TrialFunction(V)
    q = TestFunction(Q)

    A = ii_convert(ii_assemble(inner(Trace(u, bmesh), q)*dx_))

    fV = interpolate(f, V)
    gQ = interpolate(g, Q)

    foo = assemble(inner(f, g)*dx_)

    x = Function(Q).vector()
    A.mult(fV.vector(), x)

    bar = gQ.vector().inner(x)

    assert abs(foo - bar)

# ------------------------------------------------------------------------------

n = 16

mesh = UnitSquareMesh(*(n, )*2)
gamma = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
DomainBoundary().mark(gamma, 1)

bmesh = EmbeddedMesh(gamma, 1)
n = OuterNormal(bmesh, [0.5, 0.5])
dx_ = Measure('dx', domain=bmesh)

# The test cases
cases = []

# Hdiv case
elm0 = FiniteElement('Raviart-Thomas', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, elm0)

elm0 = FiniteElement('Lagrange', bmesh.ufl_cell(), 1)
Q = FunctionSpace(bmesh, elm0)

f = Expression(('x[0]+2*x[1]', 'x[0]+x[1]'), degree=1)
g = Expression('x[0]-x[1]', degree=1)

cases.append(((V, f), (Q, g)))


for (V, f), (Q, g) in cases:
    u = TrialFunction(V)
    q = TestFunction(Q)

    A = ii_convert(ii_assemble(dot(inner(Trace(u, bmesh, '-', n), n), q)*dx_))

    fV = interpolate(f, V)
    gQ = interpolate(g, Q)

    foo = assemble(inner(dot(f, n), g)*dx_)

    x = Function(Q).vector()
    A.mult(fV.vector(), x)

    bar = gQ.vector().inner(x)

    assert abs(foo - bar)
