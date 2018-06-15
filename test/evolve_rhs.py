from dolfin import *
from xii import *

n = 4

mesh = UnitSquareMesh(*(n, )*2)
gamma = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
CompiledSubDomain('on_boundary || near(x[0], 0.5) || near(x[1], 0.5)').mark(gamma, 1)

bmesh = EmbeddedMesh(gamma, 1)
dx_ = Measure('dx', domain=bmesh)

elm0 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, elm0)

elm0 = FiniteElement('Lagrange', bmesh.ufl_cell(), 1)
Q = FunctionSpace(bmesh, elm0)

f = Expression('t*(x[0]+x[1])', degree=1, t=0)

q = TestFunction(Q)
v = Function(V)
L = inner(Trace(v, bmesh), q)*dx_

for t in range(5):
    f.t = t
    # Don't point v to a different vector, rather fill the values
    # v.assign(interpolate(f, V))
    v.vector()[:] = interpolate(f, V).vector()

    b = ii_convert(ii_assemble(L))
    ans = b.inner(interpolate(f, Q).vector())
    
    truth = assemble(inner(f, f)*dx_)

    assert abs(ans - truth) < 1E-13, abs(ans - truth)
