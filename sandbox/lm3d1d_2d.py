from dolfin import *
from xii import EmbeddedMesh

mesh = UnitCubeMesh(32, 32, 32)

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

# V = FunctionSpace(Vmesh, 'CG', 1)
# EV = FunctionSpace(Emesh, 'CG', 1)

# E = uniform_extension_matrix(V, EV)

# f = Expression('x[2]', degree=1)

# fV = interpolate(f, V)

# Ef = Function(EV)
# # Ef.vector() = E * fV.vector()
# E.mult(fV.vector(), Ef.vector())

# e = inner(Ef - f, Ef - f)*dx(domain=Emesh)
# n = inner(Ef, Ef)*dx(domain=Emesh)
# print sqrt(abs(assemble(e))), sqrt(abs(assemble(n))) 

#########################################

from xii import *

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

a = [[0]*len(W) for _ in range(len(W))]

a[0][0] = inner(grad(u3d), grad(v3d))*dx
a[0][2] = -inner(Tv3d, p)*dxLM

a[1][1] = inner(grad(u1d), grad(v1d))*dx
a[1][2] = inner(Ev1d, p)*dxLM

a[2][0] = -inner(Tu3d, q)*dxLM
a[2][1] = inner(Eu1d, q)*dxLM

#print ii_assemble(a[0][0])
#print ii_assemble(a[1][1])
#print ii_assemble(a[2][0])
#print ii_assemble(a[0][2])

A12 = ii_assemble(a[1][2])
A21 = ii_assemble(a[2][1])

print Q.dim(), V1d.dim()

# FIXME: extension for tensors
#        memoization in extension
#        correctness of blocks
