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

# Mixed trace
elm0 = FiniteElement('Lagrange', mesh.ufl_cell(), 2)
elm1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([elm0, elm1]))

elm0 = FiniteElement('Lagrange', bmesh.ufl_cell(), 2)
elm1 = FiniteElement('Lagrange', bmesh.ufl_cell(), 1)
Q = FunctionSpace(bmesh, MixedElement([elm0, elm1]))

f = Expression(('x[0]*x[1]', 'x[0]+x[1]'), degree=2)
g = Expression(('x[0]*x[0] + x[1]*x[1]', '2*x[0]-3*x[1]'), degree=2)


u = TrialFunction(V)
qs = TestFunctions(Q)

Tu = Trace(u, bmesh)

for i in range(2):
    for j in range(2):
        q = qs[i]
        A = ii_convert(ii_assemble(inner(Tu[j] + 2*Tu[i], q)*dx_))

        fV = interpolate(f, V)
        gQ = interpolate(g, Q)

        foo = assemble(inner(f[j] + 2*f[i], g[i])*dx_)

        x = Function(Q).vector()
        A.mult(fV.vector(), x)
        
        bar = gQ.vector().inner(x)

        assert abs(foo - bar) < 1E-13
