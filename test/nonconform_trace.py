from dolfin import *
from xii import *

# D mesh
mesh = UnitSquareMesh(32, 32)
# Now construct mesh for the multiplier
bmesh = BoundaryMesh(UnitSquareMesh(16, 16), 'exterior')

V = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(bmesh, 'CG', 1)

u = TrialFunction(V)
p = TestFunction(Q)

dxGamma = Measure('dx', domain=bmesh)

B = ii_convert(ii_assemble(inner(Trace(u, bmesh), p)*dxGamma))

# Sanity check
f, p = Constant(1), Constant(2)
true = assemble(f*p*dxGamma)

fh = interpolate(f, V)
ph = interpolate(p, Q)

Tf = Function(Q).vector()
B.mult(fh.vector(), Tf)
print(true, ph.vector().inner(Tf))

# One more
f, p = Constant(1), Expression('x[0]+x[1]', degree=1)
true = assemble(f*p*dxGamma)

fh = interpolate(f, V)
ph = interpolate(p, Q)

Tf = Function(Q).vector()
B.mult(fh.vector(), Tf)
print(true, ph.vector().inner(Tf))

# And one more
f, p = Expression('2*x[0]-x[1]', degree=1), Expression('x[0]+x[1]', degree=1)
true = assemble(f*p*dxGamma)

fh = interpolate(f, V)
ph = interpolate(p, Q)

Tf = Function(Q).vector()
B.mult(fh.vector(), Tf)
print(true, ph.vector().inner(Tf))
