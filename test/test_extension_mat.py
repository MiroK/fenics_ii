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
