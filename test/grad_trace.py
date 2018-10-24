from dolfin import *
from xii import *
from xii.assembler.trace_matrix import trace_mat_no_restrict
import block

mesh = UnitSquareMesh(10, 10)
bmesh = BoundaryMesh(mesh, 'exterior')

V = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(bmesh, 'CG', 1)

u = TrialFunction(V)
q = TestFunction(Q)

n = FacetNormal(mesh)
# Want grad(u).n * q
v = TestFunction(V)

a = inner(dot(grad(u), n), v)*ds
A = assemble(a)
# Now take it from testV to test Q

T = PETScMatrix(trace_mat_no_restrict(V, Q, bmesh))

B = ii_convert(T*A)

# Now we should be able to integrate linear function exactly
x, y = SpatialCoordinate(mesh)

f = 2*x + y
g = x - 3*y

truth = assemble(inner(dot(grad(f), n), g)*ds)

fV = interpolate(Expression('x[1] + 2*x[0]', degree=1), V)
gQ = interpolate(Expression('x[0] - 3*x[1]', degree=1), Q)

me = gQ.vector().inner(B*fV.vector())

print abs(truth - me)
