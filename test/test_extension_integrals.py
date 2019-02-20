from dolfin import *
from xii import EmbeddedMesh, ii_assemble, Extension, Trace
import numpy as np


mesh = UnitCubeMesh(16, 16, 16)

f = MeshFunction('size_t', mesh, 1, 0)
CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(f, 1)
# Mesh to extend from
Vmesh = EmbeddedMesh(f, 1)

# Get the tube as submesh of the full cube boundary
Emesh = BoundaryMesh(mesh, 'exterior')
f = MeshFunction('size_t', Emesh, 2, 0)
CompiledSubDomain('!(near(x[2], 0) || near(x[2], 1))').mark(f, 1)
# Mesh to extend to
Emesh = SubMesh(Emesh, f, 1)

# We look first at integrals (Ev1d, p)*dxLM so coupling to the multiplier
V3d = FunctionSpace(mesh, 'CG', 1)
V1d = FunctionSpace(Vmesh, 'CG', 1)
Q = FunctionSpace(Emesh, 'CG', 1)
    
W = [V3d, V1d, Q]

u3d, u1d, p = map(TrialFunction, W)
v3d, v1d, q = map(TestFunction, W)

Tu3d, Tv3d = (Trace(f, Emesh) for f in (u3d, v3d))
Eu1d, Ev1d = (Extension(f, Emesh, type='uniform') for f in (u1d, v1d))

# Cell integral of Qspace
dxLM = Measure('dx', domain=Emesh)

a = inner(Ev1d, p)*dxLM
A = ii_assemble(a)
# Let's use the matrix to perform the following integral
p_expr = Expression('x[0]-2*x[1]+3*x[2]', degree=1)
# Something which can be extended exactly
v_expr = Expression('2*x[2]', degree=1)

true = assemble(inner(p_expr, v_expr)*dxLM)

p_func = interpolate(p_expr, Q)
v_func = interpolate(v_expr, V1d)
# Quadrature
num = v_func.vector().inner(A*p_func.vector())
print '>', abs(num - true), (true, num)

# Check the transpose as well
a = inner(Eu1d, q)*dxLM
A = ii_assemble(a)
# Quadrature
num = p_func.vector().inner(A*v_func.vector())
print '>>', abs(num - true), (true, num)

# Maybe at some point we want something like this
a = inner(Tu3d, Ev1d)*dxLM
A = ii_assemble(a)

u_func = interpolate(p_expr, V3d)
num = v_func.vector().inner(A*u_func.vector())
print '>>>', abs(num - true), (true, num)
