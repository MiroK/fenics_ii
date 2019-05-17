from dolfin import *
from xii import EmbeddedMesh
from xii.assembler.extension_matrix import uniform_extension_matrix


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

# Check scalar
V = FunctionSpace(Vmesh, 'CG', 1)
EV = FunctionSpace(Emesh, 'CG', 1)

E = uniform_extension_matrix(V, EV)

f = Expression('x[2]', degree=1)

fV = interpolate(f, V)

Ef = Function(EV)
# Ef.vector() = E * fV.vector()
E.mult(fV.vector(), Ef.vector())

e = inner(Ef - f, Ef - f)*dx(domain=Emesh)
n = inner(Ef, Ef)*dx(domain=Emesh)
print sqrt(abs(assemble(e))), sqrt(abs(assemble(n))) 

# Check vector
V = VectorFunctionSpace(Vmesh, 'CG', 1)
EV = VectorFunctionSpace(Emesh, 'CG', 1)

E = uniform_extension_matrix(V, EV)

f = Expression(('x[2]', '-2*x[2]', '3*x[2]'), degree=1)

fV = interpolate(f, V)

Ef = Function(EV)
# Ef.vector() = E * fV.vector()
E.mult(fV.vector(), Ef.vector())

e = inner(Ef - f, Ef - f)*dx(domain=Emesh)
n = inner(Ef, Ef)*dx(domain=Emesh)
print sqrt(abs(assemble(e))), sqrt(abs(assemble(n)))


print 
# 2d test idea: if the extension mounts to shift it should be exact
n = 4

mesh = UnitSquareMesh(n, n)

# We will have the 1d guy live in the middle and the multiplier slighly off
middle = CompiledSubDomain('near(x[0], 0.5)')
off_middle = CompiledSubDomain('near(x[0], A)', A=(n/2+1.)/n)

facet_f = MeshFunction('size_t', mesh, 1, 0)
middle.mark(facet_f, 1)
off_middle.mark(facet_f, 2)

# Mesh to extend from, (so the d1)
mesh_1d = EmbeddedMesh(facet_f, 1)
# And thhe multiplier one
mesh_lm = EmbeddedMesh(facet_f, 2)

V_1d = FunctionSpace(mesh_1d, 'CG', 1)  # Extend from here
Q = FunctionSpace(mesh_lm, 'CG', 1)  # To here

E = uniform_extension_matrix(V_1d, Q)

# The extended function will practically be shifted so I want invariance
# in that case
f1d = Expression('2-x[1]', degree=1)
# LM is free
fLM = Expression('4-2*x[1]+x[0]', degree=1)

# What we are after
dxLM = Measure('dx', domain=mesh_lm)
true = assemble(inner(f1d, fLM)*dxLM)

# How do we get it?
f1d_ = interpolate(f1d, V_1d)
fLM_ = interpolate(fLM, Q)

x = Function(Q, E*f1d_.vector())
# Check invariance
print '>>>>', assemble(inner(x-f1d, x-f1d)*dxLM)
# Then this should be automatic
mine = assemble(inner(fLM_, x)*dxLM)

print abs(true - mine)
